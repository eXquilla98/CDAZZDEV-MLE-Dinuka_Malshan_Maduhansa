import yfinance as yf


def calculate_volatility(
    ticker: str,
    window: int = 30
) -> dict:
    """
    Calculate annualized historical volatility.

    Args:
        ticker: Stock ticker symbol.
        window: Number of trading days used for the calculation.

    Returns:
        Structured volatility information.
    """

    if not ticker:
        raise ValueError("Ticker symbol cannot be empty.")

    if window <= 1:
        raise ValueError(
            "Volatility window must be greater than one."
        )

    ticker = ticker.upper().strip()

    stock = yf.Ticker(ticker)

    data = stock.history(
        period="1y",
        auto_adjust=False
    )

    if data.empty:
        raise ValueError(
            f"No price data found for ticker: {ticker}"
        )

    if len(data) < window + 1:
        raise ValueError(
            f"Insufficient data for {window}-day volatility."
        )

    returns = data["Close"].pct_change().dropna()

    recent_returns = returns.tail(window)

    daily_volatility = recent_returns.std()

    annualized_volatility = (
        daily_volatility * (252 ** 0.5)
    )

    return {
        "ticker": ticker,
        "window": window,
        "daily_volatility": float(
            daily_volatility
        ),
        "annualized_volatility": float(
            annualized_volatility
        ),
    }