from typing import Any, TypedDict

from src.agents.data_analyst_agent import (
    answer_clarification,
    run_data_analyst,
)
from src.agents.research_writer_agent import (
    create_clarification_request,
    generate_final_report,
    run_research_tools,
)
from src.models.research_models import (
    ClarificationRequest,
    ClarificationResponse,
    DataAnalystOutput,
    ResearchReport,
)


class MultiAgentState(TypedDict):
    ticker: str
    user_query: str

    # Agent A
    analyst_output: DataAnalystOutput | None

    # Agent B
    news_results: list[dict[str, Any]]
    web_results: list[dict[str, Any]]

    # Critique loop
    clarification_request: ClarificationRequest | None
    clarification_response: ClarificationResponse | None

    # Final result
    final_report: ResearchReport | None


def run_task_3b(
    ticker: str,
    user_query: str,
    sentiment_headlines: list[str] | None = None,
) -> ResearchReport:

    state: MultiAgentState = {
        "ticker": ticker.upper().strip(),
        "user_query": user_query,
        "analyst_output": None,
        "news_results": [],
        "web_results": [],
        "clarification_request": None,
        "clarification_response": None,
        "final_report": None,
    }

    # =========================================================
    # STEP 1 — AGENT A
    # =========================================================

    print("\n[1/5] Agent A — Data Analyst")

    analyst_output = run_data_analyst(
        ticker=state["ticker"],
        user_query=state["user_query"],
        sentiment_headlines=sentiment_headlines,
    )

    state["analyst_output"] = analyst_output

    print("Agent A completed structured analysis.")

    # =========================================================
    # STEP 2 — AGENT B RESEARCH
    # =========================================================

    print("\n[2/5] Agent B — External Research")

    news_results, web_results = run_research_tools(
        ticker=state["ticker"],
        user_query=state["user_query"],
    )

    state["news_results"] = news_results
    state["web_results"] = web_results

    print(
        f"Agent B gathered "
        f"{len(news_results)} news result group(s) and "
        f"{len(web_results)} web result group(s)."
    )

    # =========================================================
    # STEP 3 — AGENT B REQUESTS CLARIFICATION
    # =========================================================

    print("\n[3/5] Agent B — Clarification Request")

    clarification_request = create_clarification_request(
        analyst_output=state["analyst_output"],
        user_query=state["user_query"],
    )

    state["clarification_request"] = clarification_request

    print(
        "Agent B asks Agent A:\n"
        f"  {clarification_request.question}"
    )

    # =========================================================
    # STEP 4 — AGENT A RESPONDS
    # =========================================================

    print("\n[4/5] Agent A — Clarification Response")

    raw_response = answer_clarification(
        analyst_output=state["analyst_output"],
        question=clarification_request.question,
    )

    clarification_response = ClarificationResponse.model_validate(
        raw_response
    )

    state["clarification_response"] = clarification_response

    print(
        "Agent A responds:\n"
        f"  {clarification_response.answer}"
    )

    # =========================================================
    # STEP 5 — AGENT B FINAL REPORT
    # =========================================================

    print("\n[5/5] Agent B — Final Report")

    final_report = generate_final_report(
        user_query=state["user_query"],
        analyst_output=state["analyst_output"],
        clarification_request=state["clarification_request"],
        clarification_response=state["clarification_response"].answer,
        news_results=state["news_results"],
        web_results=state["web_results"],
    )

    state["final_report"] = final_report

    print("Agent B completed the final report.")

    return final_report


if __name__ == "__main__":

    QUERY = (
        "Analyse the current financial health and market sentiment "
        "of AAPL. Identify the top three risks to its share price "
        "over the next 90 days and suggest one data-driven hedge "
        "strategy."
    )

    # These headlines are supplied to Agent A only for sentiment
    # analysis. Agent A does not retrieve news itself.
    SENTIMENT_HEADLINES = [
        "Apple shares rise as investors assess new AI developments.",
        "Apple faces higher costs from memory and component prices.",
    ]

    report = run_task_3b(
        ticker="AAPL",
        user_query=QUERY,
        sentiment_headlines=SENTIMENT_HEADLINES,
    )

    print("\n")
    print("=" * 70)
    print("TASK 3B — MULTI-AGENT FINANCIAL RESEARCH REPORT")
    print("=" * 70)

    print("\nFINANCIAL HEALTH SUMMARY")
    print(report.financial_health_summary)

    print("\nMARKET SENTIMENT")
    print(
        f"{report.market_sentiment} "
        f"(score={report.sentiment_score:.2f})"
    )

    print("\nTOP THREE RISKS")
    for index, risk in enumerate(
        report.top_three_risks,
        start=1,
    ):
        print(f"\n{index}. {risk.risk}")
        print(f"Severity: {risk.severity}")

        for evidence in risk.evidence:
            print(f"   - {evidence}")

    print("\nHEDGE STRATEGY RECOMMENDATION")
    print(
        f"Strategy: "
        f"{report.hedge_strategy_recommendation.strategy}"
    )
    print(
        f"Rationale: "
        f"{report.hedge_strategy_recommendation.rationale}"
    )

    print("\n" + "=" * 70)