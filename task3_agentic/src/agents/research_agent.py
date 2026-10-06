from typing import Any, TypedDict

from langchain_core.tools import tool
from src.tools.price_data import get_price_data
from src.tools.news import get_news
from src.tools.volatility import calculate_volatility
from src.tools.sentiment import llm_sentiment
from src.tools.web_search import web_search

class ResearchState(TypedDict):
    ticker: str
    user_query: str
    observations: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    errors: list[str]
    final_report: dict[str, Any] | None

@tool
def price_data_tool(ticker: str, period: str = "1y") -> dict:
    """Get OHLCV price data and technical indicators for a stock."""
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
def sentiment_tool(headlines: list[dict]) -> dict:
    """Analyze the sentiment of financial news headlines using an LLM."""
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