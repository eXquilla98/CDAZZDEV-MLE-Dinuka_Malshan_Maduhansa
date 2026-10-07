import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CACHE_FILE = Path("memory/research_cache.json")


def _load_cache() -> dict[str, Any]:
    """Load the persistent research cache from disk."""
    if not CACHE_FILE.exists():
        return {}

    with CACHE_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_cache(cache: dict[str, Any]) -> None:
    """Persist the research cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with CACHE_FILE.open("w", encoding="utf-8") as file:
        json.dump(cache, file, indent=2, ensure_ascii=False)


def save_research(
    ticker: str,
    analyst_output: dict[str, Any],
    news_results: list[Any],
    web_results: list[Any],
    final_report: dict[str, Any],
) -> None:
    """
    Save the completed research workflow to the persistent cache.

    Cache entries are grouped by ticker and research date.
    """

    cache = _load_cache()

    ticker = ticker.upper()
    research_date = datetime.now(timezone.utc).date().isoformat()

    if ticker not in cache:
        cache[ticker] = {}

    cache[ticker][research_date] = {
        "ticker": ticker,
        "date": research_date,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "analyst_output": analyst_output,
        "news_results": news_results,
        "web_results": web_results,
        "final_report": final_report,
    }

    _save_cache(cache)


def load_research(
    ticker: str,
    research_date: str | None = None,
) -> dict[str, Any] | None:
    """
    Load cached research for a ticker.

    If no date is provided, the most recent cached date is returned.
    """

    cache = _load_cache()

    ticker = ticker.upper()

    if ticker not in cache:
        return None

    ticker_cache = cache[ticker]

    if not ticker_cache:
        return None

    if research_date:
        return ticker_cache.get(research_date)

    latest_date = max(ticker_cache.keys())

    return ticker_cache[latest_date]


def cache_exists(
    ticker: str,
    research_date: str | None = None,
) -> bool:
    """Return whether cached research exists."""
    return load_research(ticker, research_date) is not None