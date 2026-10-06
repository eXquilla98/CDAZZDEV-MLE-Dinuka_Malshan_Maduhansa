from src.tools.volatility import calculate_volatility


result = calculate_volatility("AAPL", 30)

print("\n=== VOLATILITY ===")
print(f"Ticker: {result['ticker']}")
print(f"Window: {result['window']} days")
print(
    f"Daily volatility: "
    f"{result['daily_volatility']:.4%}"
)
print(
    f"Annualized volatility: "
    f"{result['annualized_volatility']:.4%}"
)