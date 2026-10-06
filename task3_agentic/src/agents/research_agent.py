from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import HumanMessage, ToolMessage

from src.models.research_models import ResearchReport
from src.tools.news import get_news
from src.tools.price_data import get_price_data
from src.tools.sentiment import llm_sentiment
from src.tools.volatility import calculate_volatility
from src.tools.web_search import web_search


load_dotenv()


# ============================================================
# 1. AGENT STATE
# ============================================================

class ResearchState(TypedDict):
    ticker: str
    user_query: str
    messages: list[Any]
    observations: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    errors: list[str]
    iteration: int
    final_report: dict[str, Any] | None

# ============================================================
# 2. TOOLS
# ============================================================

@tool
def price_data_tool(ticker: str, period: str = "1y") -> dict:
    """
    Get OHLCV price data and technical indicators for a stock.

    Valid periods:
    1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
    """
    valid_periods = {
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    }

    if period not in valid_periods:
        raise ValueError(
            f"Invalid period '{period}'. "
            f"Valid periods are: {', '.join(sorted(valid_periods))}"
        )

    return get_price_data(ticker, period)


@tool
def news_tool(ticker: str, n: int = 10) -> list[dict]:
    """Get recent news headlines for a stock."""
    return get_news(ticker, n)


@tool
def volatility_tool(ticker: str, window: int = 30) -> dict:
    """Calculate annualized historical volatility for a stock."""
    return calculate_volatility(ticker, window)


@tool
def sentiment_tool(headlines: list[str]) -> dict:
    """Analyze financial news headlines using an LLM."""
    return llm_sentiment(headlines)


@tool
def web_search_tool(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for analyst commentary and market information."""
    return web_search(query, max_results)


RESEARCH_TOOLS = [
    price_data_tool,
    news_tool,
    volatility_tool,
    sentiment_tool,
    web_search_tool,
]


# ============================================================
# 3. LLM
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)

llm_with_tools = llm.bind_tools(RESEARCH_TOOLS)


# ============================================================
# 4. AGENT NODE
# ============================================================

def agent_node(state: ResearchState) -> dict:
    """
    Ask the LLM what information it needs next.
    """

    messages = list(state["messages"])

    if not messages:
        messages.append(
            HumanMessage(
                content=(
                    "You are a financial research agent.\n\n"
                    f"Ticker: {state['ticker']}\n"
                    f"Research request: {state['user_query']}\n\n"

                    "Your job is to autonomously gather enough evidence "
                    "to produce the requested financial research report.\n\n"

                    "Required evidence categories:\n"
                    "1. Recent market/news information.\n"
                    "2. Price and technical indicators.\n"
                    "3. Historical volatility.\n"
                    "4. LLM-based news sentiment.\n"
                    "5. Fundamental or analyst commentary from web search.\n\n"

                    "Decision rules:\n"
                    "- Choose the next tool based on the evidence already collected.\n"
                    "- Before requesting another tool, check which evidence "
                    "categories are still missing.\n"
                    "- Prioritize missing evidence over repeating a tool whose "
                    "information is already available.\n"
                    "- Do not repeat a tool unless the previous result failed, "
                    "was incomplete, or additional information is genuinely needed.\n"
                    "- Once all required evidence categories are available, "
                    "stop requesting tools and provide the final answer.\n"
                    "- Do not invent financial data or facts.\n"
                    "- Use only information returned by the tools for factual claims.\n"
                    "- If a tool fails, adapt and try another valid approach.\n"
                )
            )
        )
    response = llm_with_tools.invoke(messages)

    messages.append(response)

    tool_calls = []

    for tool_call in response.tool_calls:
        tool_calls.append(
            {
                "name": tool_call["name"],
                "args": tool_call["args"],
                "id": tool_call["id"],
            }
        )

    return {
        "messages": messages,
        "tool_calls": tool_calls,
        "iteration": state["iteration"] + 1,
    }

# ============================================================
# 5. TOOL EXECUTION NODE
# ============================================================

def tool_node(state: ResearchState) -> dict:
    """
    Execute tools selected by the LLM and return observations.
    """

    observations = list(state["observations"])
    errors = list(state["errors"])
    messages = list(state["messages"])

    for tool_call in state["tool_calls"]:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        selected_tool = next(
            (
                tool
                for tool in RESEARCH_TOOLS
                if tool.name == tool_name
            ),
            None,
        )

        if selected_tool is None:
            error_message = f"Unknown tool requested: {tool_name}"

            errors.append(error_message)

            messages.append(
                ToolMessage(
                    content=error_message,
                    tool_call_id=tool_call["id"],
                )
            )

            continue

        try:
            result = selected_tool.invoke(tool_args)

            observations.append(
                {
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": result,
                }
            )

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

        except Exception as exc:
            error_message = (
                f"{tool_name} failed: {str(exc)}"
            )

            errors.append(error_message)

            messages.append(
                ToolMessage(
                    content=error_message,
                    tool_call_id=tool_call["id"],
                )
            )

    return {
        "messages": messages,
        "observations": observations,
        "errors": errors,
        "tool_calls": [],
    }

def report_node(state: ResearchState) -> dict:
    """
    Generate the final structured research report
    from a compact representation of the evidence collected by the agent.
    """

    compact_observations = []

    for observation in state["observations"]:
        tool_name = observation["tool"]
        output = observation["output"]

        if tool_name == "price_data_tool":
            compact_output = {
                "ticker": output.get("ticker"),
                "period": output.get("period"),
                "rows": output.get("rows"),
                "latest_price": output.get("latest_price"),
                "latest_ohlcv": output.get("latest_ohlcv"),
                "indicators": output.get("indicators"),
            }

        elif tool_name == "news_tool":
            compact_output = [
                {
                    "title": item.get("title"),
                    "publisher": item.get("publisher"),
                    "published_at": item.get("published_at"),
                }
                for item in output[:8]
            ]

        elif tool_name == "volatility_tool":
            compact_output = {
                "ticker": output.get("ticker"),
                "window": output.get("window"),
                "daily_volatility": output.get("daily_volatility"),
                "annualized_volatility": output.get("annualized_volatility"),
            }

        elif tool_name == "sentiment_tool":
            compact_output = output

        elif tool_name == "web_search_tool":
            compact_output = [
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("snippet"),
                }
                for item in output[:5]
            ]

        else:
            compact_output = output

        compact_observations.append(
            {
                "tool": tool_name,
                "output": compact_output,
            }
        )

    report_prompt = f"""
    You are a senior financial research analyst.

    Prepare the final research report for {state["ticker"]}.

    Original research request:
    {state["user_query"]}

    Compact evidence collected by the research agent:
    {compact_observations}

    Tool errors encountered:
    {state["errors"]}

    Requirements:

    1. Summarize the company's current financial health.
    2. Determine the overall market sentiment.
    3. Identify exactly three risks to the share price over the next 90 days.
    4. Each risk must contain supporting evidence.
    5. Assign each risk a severity: low, medium, or high.
    6. Recommend exactly one data-driven hedge strategy.
    7. Use the sentiment_tool result for market sentiment and sentiment_score.
    8. Do not infer sentiment independently when sentiment_tool evidence is available.
    9. Do not invent facts that are not supported by the collected evidence.
    10. Return a structured ResearchReport.
    """

    structured_llm = llm.with_structured_output(ResearchReport)

    report = structured_llm.invoke(report_prompt)

    return {
        "final_report": report.model_dump()
    }
# ============================================================
# 6. ROUTING
# ============================================================
MAX_ITERATIONS = 8


def has_required_evidence(
    state: ResearchState
) -> bool:
    """
    Check whether the agent has collected enough
    successful evidence to generate the final report.
    """

    successful_tools = {
        observation["tool"]
        for observation in state["observations"]
    }

    required_tools = {
        "price_data_tool",
        "news_tool",
        "volatility_tool",
        "sentiment_tool",
        "web_search_tool",
    }

    return required_tools.issubset(successful_tools)


def route_after_agent(state: ResearchState) -> str:
    """
    Decide whether the graph should execute tools again
    or generate the final report.
    """

    # If the LLM selected tools, execute them first.
    # This check comes before the iteration limit so that
    # a valid tool request is never abandoned simply because
    # the current iteration reached the safety limit.
    if state["tool_calls"]:
        if state["iteration"] <= MAX_ITERATIONS:
            return "tools"

    # Stop once all required evidence has been collected.
    if has_required_evidence(state):
        return "end"

    # Safety fallback.
    return "end"
# ============================================================
# 7. BUILD GRAPH
# ============================================================

def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "agent")

    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "end": "report",
        },
    )

    graph.add_edge("tools", "agent")
    graph.add_edge("report", END)

    return graph.compile()

research_graph = build_research_graph()

if __name__ == "__main__":
    initial_state: ResearchState = {
    "ticker": "AAPL",
    "user_query": (
        "Analyse the current financial health and market sentiment "
        "of AAPL. Identify the top three risks to its share price "
        "over the next 90 days and suggest one data-driven hedge strategy."
    ),
    "messages": [],
    "observations": [],
    "tool_calls": [],
    "errors": [],
    "iteration": 0,
    "final_report": None,
}

    result = research_graph.invoke(initial_state)

    print("=== FINAL STATE ===")
    print(result)

    print("\n=== AGENT SUMMARY ===")
    print(f"Iterations: {result['iteration']}")
    print(f"Observations: {len(result['observations'])}")
    print(f"Errors: {len(result['errors'])}")

    print("\n=== TOOLS USED ===")
    for observation in result["observations"]:
        print(f"- {observation['tool']}")

    if result["errors"]:
        print("\n=== ERRORS ===")
        for error in result["errors"]:
            print(f"- {error}")

    print("\n=== FINAL RESEARCH REPORT ===")
    print(result["final_report"])