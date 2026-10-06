from typing import Any
from urllib.parse import quote
import xml.etree.ElementTree as ET

import requests


GOOGLE_NEWS_RSS_URL = (
    "https://news.google.com/rss/search"
    "?q={query}"
    "&hl=en-US"
    "&gl=US"
    "&ceid=US:en"
)


def get_news(
    ticker: str,
    n: int = 10
) -> list[dict[str, Any]]:
    """
    Retrieve recent news headlines for a stock ticker
    using Google News RSS.

    Args:
        ticker: Stock ticker symbol.
        n: Maximum number of headlines.

    Returns:
        Structured list of news items.
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

    for item in root.findall(".//item")[:n]:
        title = item.findtext("title")
        link = item.findtext("link")
        published_at = item.findtext("pubDate")
        source = item.findtext("source")

        if not title:
            continue

        news_items.append(
            {
                "title": title.strip(),
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