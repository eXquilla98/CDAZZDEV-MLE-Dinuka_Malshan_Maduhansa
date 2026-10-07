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

    analyst_output: DataAnalystOutput | None

    news_results: list[dict[str, Any]]
    web_results: list[dict[str, Any]]

    clarification_request: ClarificationRequest | None
    clarification_response: ClarificationResponse | None

    final_report: ResearchReport | None


def extract_headlines(
    news_results: list[Any],
) -> list[str]:
    """
    Extract headline titles from Agent B's news-tool observations.

    Agent B owns news retrieval.
    Agent A receives only the resulting headlines for sentiment analysis.
    """

    headlines: list[str] = []

    for observation in news_results:
        if not isinstance(observation, list):
            continue

        for item in observation:
            if not isinstance(item, dict):
                continue

            title = item.get("title")

            if isinstance(title, str) and title.strip():
                headlines.append(title.strip())

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(headlines))


def run_task_3b(
    ticker: str,
    user_query: str,
) -> ResearchReport:

    print("\n=== TASK 3B MULTI-AGENT WORKFLOW ===\n")

    # ---------------------------------------------------------
    # STEP 1 — Agent B retrieves external research
    # ---------------------------------------------------------

    print("[1/6] Agent B — External Research")

    news_results, web_results = run_research_tools(
        ticker=ticker,
        user_query=user_query,
    )

    print(f"      News observations: {len(news_results)}")
    print(f"      Web observations: {len(web_results)}")

    # ---------------------------------------------------------
    # STEP 2 — Automatically pass news headlines to Agent A
    # ---------------------------------------------------------

    sentiment_headlines = extract_headlines(news_results)

    print(
        f"      Headlines passed to Agent A: "
        f"{len(sentiment_headlines)}"
    )

    if not sentiment_headlines:
        print(
            "      Warning: No headlines were available "
            "for sentiment analysis."
        )

    # ---------------------------------------------------------
    # STEP 3 — Agent A performs quantitative analysis
    # ---------------------------------------------------------

    print("\n[2/6] Agent A — Data Analysis")

    analyst_output = run_data_analyst(
        ticker=ticker,
        user_query=user_query,
        sentiment_headlines=sentiment_headlines,
    )

    print(
        f"      Latest price: "
        f"{analyst_output.latest_price}"
    )

    print(
        f"      Volatility: "
        f"{analyst_output.annualized_volatility:.4f}"
    )

    print(
        f"      Sentiment: "
        f"{analyst_output.market_sentiment} "
        f"({analyst_output.sentiment_score:.3f})"
    )

    # ---------------------------------------------------------
    # STEP 4 — Agent B requests clarification
    # ---------------------------------------------------------

    print("\n[3/6] Agent B — Clarification Request")

    clarification_request = create_clarification_request(
        analyst_output=analyst_output,
        user_query=user_query,
    )

    print(
        f"      Question: "
        f"{clarification_request.question}"
    )

    # ---------------------------------------------------------
    # STEP 5 — Agent A responds to clarification
    # ---------------------------------------------------------

    print("\n[4/6] Agent A — Clarification Response")

    raw_response = answer_clarification(
        analyst_output=analyst_output,
        question=clarification_request.question,
    )

    clarification_response = ClarificationResponse.model_validate(
        raw_response
    )

    print(
        f"      Answer: "
        f"{clarification_response.answer}"
    )

    # ---------------------------------------------------------
    # STEP 6 — Agent B writes final report
    # ---------------------------------------------------------

    print("\n[5/6] Agent B — Final Research Report")

    final_report = generate_final_report(
        user_query=user_query,
        analyst_output=analyst_output,
        clarification_request=clarification_request,
        clarification_response=clarification_response.answer,
        news_results=news_results,
        web_results=web_results,
    )

    print("\n[6/6] Workflow Complete")

    return final_report


if __name__ == "__main__":

    QUERY = (
        "Analyse the current financial health and market sentiment "
        "of AAPL. Identify the top three risks to its share price "
        "over the next 90 days and suggest one data-driven hedge strategy."
    )

    report = run_task_3b(
        ticker="AAPL",
        user_query=QUERY,
    )

    print("\n=== FINAL REPORT ===\n")
    print(report.model_dump_json(indent=2))