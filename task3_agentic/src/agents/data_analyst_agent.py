import time
from typing import Any, TypedDict
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
# from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from src.llm.provider import build_llm

# Alternative provider — uncomment when testing Groq.
# from langchain_groq import ChatGroq

from src.models.research_models import DataAnalystOutput
from src.observability.trace import write_trace
from src.tools.price_data import get_price_data
from src.tools.sentiment import llm_sentiment
from src.tools.volatility import calculate_volatility


class DataAnalystState(TypedDict):
    ticker: str
    user_query: str
    sentiment_headlines: list[str]
    messages: list[Any]
    observations: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    iteration: int
    final_output: dict[str, Any] | None


@tool
def price_data_tool(
    ticker: str,
    period: str = "1y",
) -> dict[str, Any]:
    """Get price data and technical indicators for a stock."""
    return get_price_data(ticker, period)


@tool
def volatility_tool(
    ticker: str,
    window: int = 30,
) -> dict[str, Any]:
    """Calculate annualized historical volatility."""
    return calculate_volatility(ticker, window)


@tool
def sentiment_tool(
    headlines: list[str],
) -> dict[str, Any]:
    """Analyze financial sentiment from supplied headlines."""
    return llm_sentiment(headlines)


TOOLS = [
    price_data_tool,
    volatility_tool,
    sentiment_tool,
]

TOOL_MAP = {
    tool_.name: tool_
    for tool_ in TOOLS
}

MAX_ITERATIONS = 4


SYSTEM_PROMPT = """
You are Agent A, the Data Analyst in a multi-agent financial
research workflow.

Your responsibility is ONLY quantitative and sentiment analysis.

You may use ONLY these capabilities:

1. price_data_tool
   - price data
   - technical indicators

2. volatility_tool
   - historical annualized volatility

3. sentiment_tool
   - financial sentiment from headlines supplied to you

You MUST NOT perform web searches.
You MUST NOT retrieve news.
You MUST NOT use any other tools.

Your job is to gather the quantitative evidence needed by the
Research Writer agent.

You should:
- inspect the user request
- determine which quantitative evidence is missing
- call the appropriate tools
- observe their results
- decide whether additional analysis is required
- produce a structured DataAnalystOutput

Do not invent financial data.

If a metric cannot be calculated, explicitly state that it
is unavailable rather than guessing.
"""


# def build_llm() -> ChatOpenAI:
#     # ---------------------------------------------------------
#     # ACTIVE PROVIDER — OPENAI
#     # ---------------------------------------------------------
#     return ChatOpenAI(
#         model="gpt-5.4-mini",
#         temperature=0,
#     )

#     # ---------------------------------------------------------
#     # ALTERNATIVE PROVIDER — GROQ
#     # Uncomment this implementation when performing
#     # the final Groq test.
#     # ---------------------------------------------------------
#     # return ChatGroq(
#     #     model="openai/gpt-oss-120b",
#     #     temperature=0,
#     # )


def analyst_node(
    state: DataAnalystState,
) -> dict[str, Any]:

    llm = build_llm().bind_tools(TOOLS)

    messages = list(state["messages"])

    if not messages:
        context = (
            f"Ticker: {state['ticker']}\n"
            f"User request: {state['user_query']}\n"
            f"Available sentiment headlines: "
            f"{state['sentiment_headlines']}\n"
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]

    response = llm.invoke(messages)

    return {
        "messages": messages + [response],
        "iteration": state["iteration"] + 1,
    }


def tool_node(
    state: DataAnalystState,
) -> dict[str, Any]:

    messages = list(state["messages"])
    response = messages[-1]

    observations = list(state["observations"])
    tool_calls = list(state["tool_calls"])

    tool_messages = []

    for call in response.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]

        if tool_name not in TOOL_MAP:
            raise ValueError(
                f"Agent A attempted to use unauthorized tool: "
                f"{tool_name}"
            )

        selected_tool = TOOL_MAP[tool_name]

        start_time = time.perf_counter()

        try:
            result = selected_tool.invoke(tool_args)

            duration_ms = (
                time.perf_counter() - start_time
            ) * 1000

            write_trace(
                tool_name=tool_name,
                inputs=tool_args,
                output=result,
                duration_ms=duration_ms,
                success=True,
                agent="agent_a",
                event_type="tool_call",
            )

            observations.append(
                {
                    "tool": tool_name,
                    "inputs": tool_args,
                    "output": result,
                }
            )

            tool_calls.append(
                {
                    "tool": tool_name,
                    "inputs": tool_args,
                }
            )

            tool_messages.append(
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
                agent="agent_a",
                event_type="tool_call",
            )

            raise

    return {
        "messages": messages + tool_messages,
        "observations": observations,
        "tool_calls": tool_calls,
    }


def route_after_analyst(
    state: DataAnalystState,
) -> str:

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        if state["iteration"] <= MAX_ITERATIONS:
            return "tools"

    return "report"


def report_node(
    state: DataAnalystState,
) -> dict[str, Any]:

    llm = build_llm().with_structured_output(
    DataAnalystOutput,
    method="function_calling",
    )

    observations_text = "\n\n".join(
        f"Tool: {observation['tool']}\n"
        f"Input: {observation['inputs']}\n"
        f"Output: {observation['output']}"
        for observation in state["observations"]
    )

    prompt = f"""
You are Agent A, the Data Analyst.

Create a structured quantitative analysis for ticker:
{state['ticker']}

User request:
{state['user_query']}

Your observations:

{observations_text}

Use ONLY the evidence above.

Return:
- latest price
- available price indicators
- annualized volatility
- market sentiment
- sentiment score
- sentiment confidence
- concise quantitative findings

Do not invent values.

If an indicator is unavailable, use null.
"""

    result = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "final_output": result.model_dump(),
        "messages": state["messages"],
    }


def build_data_analyst_graph():

    graph = StateGraph(DataAnalystState)

    graph.add_node(
        "analyst",
        analyst_node,
    )

    graph.add_node(
        "tools",
        tool_node,
    )

    graph.add_node(
        "report",
        report_node,
    )

    graph.add_edge(
        START,
        "analyst",
    )

    graph.add_conditional_edges(
        "analyst",
        route_after_analyst,
        {
            "tools": "tools",
            "report": "report",
        },
    )

    graph.add_edge(
        "tools",
        "analyst",
    )

    graph.add_edge(
        "report",
        END,
    )

    return graph.compile()


def run_data_analyst(
    ticker: str,
    user_query: str,
    sentiment_headlines: list[str] | None = None,
) -> DataAnalystOutput:

    graph = build_data_analyst_graph()

    initial_state: DataAnalystState = {
        "ticker": ticker.upper().strip(),
        "user_query": user_query,
        "sentiment_headlines": sentiment_headlines or [],
        "messages": [],
        "observations": [],
        "tool_calls": [],
        "iteration": 0,
        "final_output": None,
    }

    result = graph.invoke(initial_state)

    if not result["final_output"]:
        raise RuntimeError(
            "Agent A did not produce a final structured output."
        )

    return DataAnalystOutput.model_validate(
        result["final_output"]
    )


def answer_clarification(
    analyst_output: DataAnalystOutput,
    question: str,
) -> dict[str, Any]:
    """
    Agent A answers one clarification question from Agent B
    using its existing quantitative analysis.
    """

    from src.models.research_models import ClarificationResponse

    llm = build_llm().with_structured_output(
    ClarificationResponse,
    method="function_calling",
)

    prompt = f"""
You are Agent A, the Data Analyst.

You previously produced this structured analysis:

{analyst_output.model_dump_json(indent=2)}

Agent B, the Research Writer, has asked:

{question}

Answer the clarification using ONLY the quantitative
and sentiment information already present in your analysis.

Do not perform web searches.
Do not retrieve news.
Do not invent new financial data.

Provide a concise answer and supporting data.
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

        response = result.model_dump()

        write_trace(
            tool_name="clarification_response",
            inputs={
                "question": question,
            },
            output=response,
            duration_ms=duration_ms,
            success=True,
            agent="agent_a",
            event_type="clarification_response",
        )

        return response

    except Exception as exc:

        duration_ms = (
            time.perf_counter() - start_time
        ) * 1000

        write_trace(
            tool_name="clarification_response",
            inputs={
                "question": question,
            },
            output=str(exc),
            duration_ms=duration_ms,
            success=False,
            agent="agent_a",
            event_type="clarification_response",
        )

        raise


if __name__ == "__main__":

    query = (
        "Analyse the current financial health and market "
        "sentiment of AAPL."
    )

    headlines = [
        "Apple shares rise as investors assess new AI developments.",
        "Apple faces higher costs from memory and component prices.",
    ]

    output = run_data_analyst(
        ticker="AAPL",
        user_query=query,
        sentiment_headlines=headlines,
    )

    print("\n=== TASK 3B — AGENT A DATA ANALYST ===")
    print(
        output.model_dump_json(
            indent=2
        )
    )