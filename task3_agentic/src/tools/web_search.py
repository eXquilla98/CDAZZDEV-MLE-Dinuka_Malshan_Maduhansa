from typing import Any

from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query.
        max_results: Maximum number of results.

    Returns:
        A list of structured search results.
    """

    if not query or not query.strip():
        raise ValueError("query must not be empty")

    if max_results <= 0:
        raise ValueError("max_results must be greater than 0")

    query = query.strip()

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(
                    query,
                    max_results=max_results,
                )
            )
    except Exception as exc:
        raise RuntimeError(
            f"Web search failed: {exc}"
        ) from exc

    structured_results = []

    for result in results:
        structured_results.append(
            {
                "title": result.get("title"),
                "url": result.get("href"),
                "snippet": result.get("body"),
            }
        )

    return structured_results