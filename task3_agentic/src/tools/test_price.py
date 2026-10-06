from src.tools.price_data import get_price_data


result = get_price_data("AAPL", "1y")

print("\n=== PRICE DATA ===")
print(f"Ticker: {result['ticker']}")
print(f"Rows: {result['rows']}")
print(f"Latest price: {result['latest_price']:.2f}")

print("\n=== OHLCV ===")
for key, value in result["latest_ohlcv"].items():
    print(f"{key}: {value}")

print("\n=== TECHNICAL INDICATORS ===")
for key, value in result["indicators"].items():
    print(f"{key}: {value:.4f}")