from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.models.research_models import (
    ClarificationRequest,
    DataAnalystOutput,
    ResearchReport,
)
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
    # from langchain_groq import ChatGroq
    #
    # return ChatGroq(
    #     model="openai/gpt-oss-120b",
    #     temperature=0,
    # )


def run_research_tools(
    ticker: str,
    user_query: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    llm = build_llm().bind_tools(TOOLS)

    prompt = f"""
Ticker:
{ticker}

User request:
{user_query}

Gather the external research needed for the final report.

Use news_tool for recent company news.

Use web_search_tool for relevant analyst commentary,
market commentary, and external evidence.

Choose the queries yourself based on the request.
"""

    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    observations = []
    tool_messages = []

    for call in response.tool_calls:
        tool_name = call["name"]

        if tool_name not in TOOL_MAP:
            raise ValueError(
                f"Agent B attempted to use unauthorized tool: "
                f"{tool_name}"
            )

        tool_args = call["args"]
        selected_tool = TOOL_MAP[tool_name]

        result = selected_tool.invoke(tool_args)

        observations.append(
            {
                "tool": tool_name,
                "inputs": tool_args,
                "output": result,
            }
        )

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=call["id"],
            )
        )

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

    llm = build_llm().with_structured_output(
        ClarificationRequest
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

    return llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )


def generate_final_report(
    user_query: str,
    analyst_output: DataAnalystOutput,
    clarification_request: ClarificationRequest,
    clarification_response: str,
    news_results: list[dict[str, Any]],
    web_results: list[dict[str, Any]],
) -> ResearchReport:

    llm = build_llm().with_structured_output(
        ResearchReport
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

    return llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )