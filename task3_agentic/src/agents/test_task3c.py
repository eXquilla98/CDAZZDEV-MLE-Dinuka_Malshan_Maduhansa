from src.agents.followup_agent import answer_followup
from src.memory.research_cache import load_research


def main() -> None:
    ticker = "AAPL"

    print("=== TASK 3C CACHE FOLLOW-UP ===")

    cached = load_research(ticker)

    if cached is None:
        raise RuntimeError(
            f"No cached research found for {ticker}"
        )

    print("[1/3] Cached Research")
    print(f"      Ticker: {cached['ticker']}")
    print(f"      Date: {cached['date']}")
    print("      Persistent cache: available")

    question = (
        "What was the main downside risk identified for AAPL?"
    )

    print("[2/3] Follow-Up Question")
    print(f"      {question}")

    answer = answer_followup(
        ticker=ticker,
        question=question,
    )

    print("[3/3] Follow-Up Answer")
    print(answer)

    print("\n=== TASK 3C COMPLETE ===")
    print("Follow-up answered from persistent cached research.")


if __name__ == "__main__":
    main()
    