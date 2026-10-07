from typing import Any
from urllib.parse import quote
import re
import xml.etree.ElementTree as ET

import requests


GOOGLE_NEWS_RSS_URL = (
    "https://news.google.com/rss/search"
    "?q={query}"
    "&hl=en-US"
    "&gl=US"
    "&ceid=US:en"
)


COMPANY_ALIASES = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft"],
    "GOOGL": ["google", "alphabet", "googl"],
    "GOOG": ["google", "alphabet", "goog"],
    "AMZN": ["amazon", "amzn"],
    "NVDA": ["nvidia", "nvda"],
    "META": ["meta", "facebook", "meta platforms"],
    "TSLA": ["tesla", "tsla"],
}


def _normalize_title(title: str) -> str:
    """Normalize a headline for duplicate detection."""
    title = title.lower()
    title = re.sub(r"[^a-z0-9\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _is_relevant_headline(title: str, ticker: str) -> bool:
    """Check whether a headline is relevant to the requested ticker."""
    normalized_title = _normalize_title(title)

    aliases = COMPANY_ALIASES.get(
        ticker,
        [ticker.lower()],
    )

    return any(
        alias in normalized_title
        for alias in aliases
    )


def get_news(
    ticker: str,
    n: int = 10
) -> list[dict[str, Any]]:
    """
    Retrieve recent relevant news headlines for a stock ticker
    using Google News RSS.

    Args:
        ticker: Stock ticker symbol.
        n: Maximum number of headlines.

    Returns:
        Structured list of relevant, deduplicated news items.
    """

    if not ticker:
        raise ValueError("Ticker symbol cannot be empty.")

    if n <= 0:
        raise ValueError(
            "Number of headlines must be greater than zero."
        )

    ticker = ticker.upper().strip()

    query = quote(f"{ticker} stock")
    url = GOOGLE_NEWS_RSS_URL.format(query=query)

    response = requests.get(
        url,
        timeout=10,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "Chrome/120.0 Safari/537.36"
            )
        },
    )

    response.raise_for_status()

    root = ET.fromstring(response.content)

    news_items = []
    seen_titles = set()

    # Fetch more candidates than requested because some
    # results may be irrelevant or duplicates.
    candidate_items = root.findall(".//item")

    for item in candidate_items:
        if len(news_items) >= n:
            break

        title = item.findtext("title")
        link = item.findtext("link")
        published_at = item.findtext("pubDate")
        source = item.findtext("source")

        if not title:
            continue

        title = title.strip()

        if not _is_relevant_headline(title, ticker):
            continue

        normalized_title = _normalize_title(title)

        if normalized_title in seen_titles:
            continue

        seen_titles.add(normalized_title)

        news_items.append(
            {
                "title": title,
                "publisher": (
                    source.strip()
                    if source
                    else None
                ),
                "published_at": published_at,
                "url": link,
            }
        )

    return news_items