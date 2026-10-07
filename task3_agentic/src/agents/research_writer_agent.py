import time
from typing import Any
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Alternative provider — uncomment when testing Groq.
# from langchain_groq import ChatGroq

from src.models.research_models import (
    ClarificationRequest,
    DataAnalystOutput,
    ResearchReport,
)
from src.observability.trace import write_trace
from src.tools.news import get_news
from src.tools.web_search import web_search


@tool
def news_tool(
    ticker: str,
    n: int = 10,
) -> list[dict[str, Any]]:
    """Retrieve recent relevant news headlines for a stock."""
    return get_news(ticker, n)


@tool
def web_search_tool(
    query: str,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Search the web for financial research and commentary."""
    return web_search(query, max_results)


TOOLS = [
    news_tool,
    web_search_tool,
]

TOOL_MAP = {
    tool_.name: tool_
    for tool_ in TOOLS
}


SYSTEM_PROMPT = """
You are Agent B, the Research Writer in a multi-agent
financial research workflow.

Your responsibility is ONLY external research and report writing.

You may use ONLY these capabilities:

1. news_tool
   - retrieve recent company news

2. web_search_tool
   - research analyst commentary and external financial information

You MUST NOT use:
- price_data_tool
- volatility_tool
- sentiment_tool

Those capabilities belong exclusively to Agent A.

You receive structured quantitative findings from Agent A.

Your responsibilities are:

1. Review the user's request.
2. Review Agent A's structured quantitative findings.
3. Gather relevant external evidence using your permitted tools.
4. Identify gaps or ambiguities in Agent A's findings.
5. Ask Agent A exactly ONE clarification question.
6. Incorporate Agent A's clarification.
7. Produce the final research report.

Do not invent facts.
Distinguish quantitative evidence from external research.
Use the evidence available to you.
"""


def build_llm() -> ChatOpenAI:
    # ---------------------------------------------------------
    # ACTIVE PROVIDER — OPENAI
    # ---------------------------------------------------------
    return ChatOpenAI(
        model="gpt-5.4-mini",
        temperature=0,
    )

    # ---------------------------------------------------------
    # ALTERNATIVE PROVIDER — GROQ
    # Uncomment when performing the final Groq test.
    # ---------------------------------------------------------
    # return ChatGroq(
    #     model="openai/gpt-oss-120b",
    #     temperature=0,
    # )


def run_research_tools(
    ticker: str,
    user_query: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Agent B gathers external research using only its
    permitted tools.

    The agent follows a tool -> observation -> decision loop
    so it can decide whether additional research is required.
    """

    llm = build_llm().bind_tools(TOOLS)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"""
Ticker:
{ticker}

User request:
{user_query}

Gather the external research needed for the final report.

You MUST use both permitted research capabilities:

1. Use news_tool to retrieve recent company news.
2. Use web_search_tool to retrieve relevant analyst commentary,
   market commentary, or other external evidence.

After each tool result, review the observation and decide whether
another permitted tool call is needed.

Complete both the news research and web research before finishing
this research phase.

Do not use any tool other than the permitted tools.
"""
        ),
    ]

    observations: list[dict[str, Any]] = []

    max_iterations = 3

    for _ in range(max_iterations):

        response = llm.invoke(messages)

        # Add the assistant response containing the tool calls.
        messages.append(response)

        # If the model decides that no more tools are required,
        # finish the research phase.
        if not response.tool_calls:
            break

        for call in response.tool_calls:

            tool_name = call["name"]

            if tool_name not in TOOL_MAP:
                raise ValueError(
                    f"Agent B attempted to use unauthorized tool: "
                    f"{tool_name}"
                )

            tool_args = call["args"]
            selected_tool = TOOL_MAP[tool_name]

            start_time = time.perf_counter()

            try:
                # Execute the selected tool.
                result = selected_tool.invoke(tool_args)

                duration_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                # Record the tool execution.
                write_trace(
                    tool_name=tool_name,
                    inputs=tool_args,
                    output=result,
                    duration_ms=duration_ms,
                    success=True,
                    agent="agent_b",
                    event_type="tool_call",
                )

                observations.append(
                    {
                        "tool": tool_name,
                        "inputs": tool_args,
                        "output": result,
                    }
                )

                # IMPORTANT:
                # Send the tool result back to Agent B so that
                # it can observe the result and make its next decision.
                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=call["id"],
                    )
                )

            except Exception as exc:

                duration_ms = (
                    time.perf_counter() - start_time
                ) * 1000

                write_trace(
                    tool_name=tool_name,
                    inputs=tool_args,
                    output=str(exc),
                    duration_ms=duration_ms,
                    success=False,
                    agent="agent_b",
                    event_type="tool_call",
                )

                raise

    news_results = [
        observation["output"]
        for observation in observations
        if observation["tool"] == "news_tool"
    ]

    web_results = [
        observation["output"]
        for observation in observations
        if observation["tool"] == "web_search_tool"
    ]

    return news_results, web_results

def create_clarification_request(
    analyst_output: DataAnalystOutput,
    user_query: str,
) -> ClarificationRequest:
    """
    Agent B creates exactly one clarification request
    for Agent A.
    """

    llm = build_llm().with_structured_output(
    ClarificationRequest,
    method="function_calling",
)

    prompt = f"""
You are Agent B, the Research Writer.

User request:
{user_query}

Agent A provided this structured analysis:

{analyst_output.model_dump_json(indent=2)}

Identify ONE important ambiguity or area where additional
quantitative clarification from Agent A would improve the
final financial research report.

Ask exactly ONE concise question.

Do not ask about news or web research.

Ask only about information that belongs to Agent A's
quantitative or sentiment analysis.
"""

    start_time = time.perf_counter()

    try:
        result = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        write_trace(
            tool_name="clarification_request",
            inputs={
                "ticker": analyst_output.ticker,
                "question_context": user_query,
            },
            output=result.model_dump(),
            duration_ms=duration_ms,
            success=True,
            agent="agent_b",
            event_type="clarification_request",
        )

        return result

    except Exception as exc:
        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        write_trace(
            tool_name="clarification_request",
            inputs={
                "ticker": analyst_output.ticker,
                "question_context": user_query,
            },
            output=str(exc),
            duration_ms=duration_ms,
            success=False,
            agent="agent_b",
            event_type="clarification_request",
        )

        raise


def generate_final_report(
    user_query: str,
    analyst_output: DataAnalystOutput,
    clarification_request: ClarificationRequest,
    clarification_response: str,
    news_results: list[dict[str, Any]],
    web_results: list[dict[str, Any]],
) -> ResearchReport:
    """
    Agent B generates the final structured research report
    after incorporating Agent A's clarification.
    """

    llm = build_llm().with_structured_output(
    ResearchReport,
    method="function_calling",
)

    prompt = f"""
Produce the final financial research report.

User request:
{user_query}

========================
AGENT A — DATA ANALYST
========================

{analyst_output.model_dump_json(indent=2)}

========================
CLARIFICATION REQUEST
========================

{clarification_request.question}

========================
AGENT A — CLARIFICATION
========================

{clarification_response}

========================
NEWS RESEARCH
========================

{news_results}

========================
WEB RESEARCH
========================

{web_results}

========================
REPORT REQUIREMENTS
========================

Produce:

1. Financial Health Summary
2. Market Sentiment
3. Top Three Risks with evidence
4. Hedge Strategy Recommendation

Use the quantitative information from Agent A,
the clarification response, and the external research.

Do not invent data.

The final report must contain exactly three risks.
"""

    start_time = time.perf_counter()

    try:
        result = llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        write_trace(
            tool_name="final_report",
            inputs={
                "ticker": analyst_output.ticker,
            },
            output=result.model_dump(),
            duration_ms=duration_ms,
            success=True,
            agent="agent_b",
            event_type="final_report",
        )

        return result

    except Exception as exc:
        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        write_trace(
            tool_name="final_report",
            inputs={
                "ticker": analyst_output.ticker,
            },
            output=str(exc),
            duration_ms=duration_ms,
            success=False,
            agent="agent_b",
            event_type="final_report",
        )

        raise