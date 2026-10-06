import numpy as np
import pandas as pd
import yfinance as yf


def fetch_price_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical OHLCV data from Yahoo Finance."""

    if not ticker:
        raise ValueError("Ticker symbol cannot be empty.")

    ticker = ticker.upper().strip()

    stock = yf.Ticker(ticker)
    data = stock.history(
        period=period,
        auto_adjust=False
    )

    if data.empty:
        raise ValueError(f"No price data found for ticker: {ticker}")

    data = data.reset_index()

    data.columns = [
        str(column).lower().replace(" ", "_")
        for column in data.columns
    ]

    return data


def calculate_sma(
    data: pd.DataFrame,
    window: int
) -> pd.Series:
    """Calculate Simple Moving Average."""

    if window <= 0:
        raise ValueError("SMA window must be greater than zero.")

    return data["close"].rolling(window=window).mean()


def calculate_rsi(
    data: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """Calculate RSI using Wilder-style exponential smoothing."""

    if period <= 0:
        raise ValueError("RSI period must be greater than zero.")

    delta = data["close"].diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    relative_strength = average_gain / average_loss

    rsi = 100 - (
        100 / (1 + relative_strength)
    )

    return rsi


def calculate_macd(
    data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """Calculate MACD and signal line."""

    if fast_period >= slow_period:
        raise ValueError(
            "MACD fast period must be smaller than slow period."
        )

    close = data["close"]

    fast_ema = close.ewm(
        span=fast_period,
        adjust=False
    ).mean()

    slow_ema = close.ewm(
        span=slow_period,
        adjust=False
    ).mean()

    macd_line = fast_ema - slow_ema

    signal_line = macd_line.ewm(
        span=signal_period,
        adjust=False
    ).mean()

    histogram = macd_line - signal_line

    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_histogram": histogram,
    })


def calculate_bollinger_bands(
    data: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """Calculate Bollinger Bands."""

    if window <= 0:
        raise ValueError(
            "Bollinger Band window must be greater than zero."
        )

    if num_std <= 0:
        raise ValueError(
            "Number of standard deviations must be greater than zero."
        )

    rolling_mean = data["close"].rolling(
        window=window
    ).mean()

    rolling_std = data["close"].rolling(
        window=window
    ).std()

    upper_band = rolling_mean + (
        rolling_std * num_std
    )

    lower_band = rolling_mean - (
        rolling_std * num_std
    )

    return pd.DataFrame({
        "bollinger_middle": rolling_mean,
        "bollinger_upper": upper_band,
        "bollinger_lower": lower_band,
    })


def get_price_data(
    ticker: str,
    period: str = "1y"
) -> dict:
    """Fetch price data and calculate all required indicators."""

    data = fetch_price_data(ticker, period)

    data["sma_50"] = calculate_sma(data, 50)
    data["sma_200"] = calculate_sma(data, 200)
    data["rsi_14"] = calculate_rsi(data, 14)

    macd = calculate_macd(data)
    data = pd.concat(
        [data, macd],
        axis=1
    )

    bollinger = calculate_bollinger_bands(data)
    data = pd.concat(
        [data, bollinger],
        axis=1
    )

    latest = data.iloc[-1]

    return {
        "ticker": ticker.upper().strip(),
        "period": period,
        "rows": len(data),
        "latest_price": float(latest["close"]),
        "indicators": {
            "sma_50": float(latest["sma_50"]),
            "sma_200": float(latest["sma_200"]),
            "rsi_14": float(latest["rsi_14"]),
            "macd": float(latest["macd"]),
            "macd_signal": float(latest["macd_signal"]),
            "macd_histogram": float(
                latest["macd_histogram"]
            ),
            "bollinger_middle": float(
                latest["bollinger_middle"]
            ),
            "bollinger_upper": float(
                latest["bollinger_upper"]
            ),
            "bollinger_lower": float(
                latest["bollinger_lower"]
            ),
        },
        "latest_ohlcv": {
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "close": float(latest["close"]),
            "volume": int(latest["volume"]),
        },
    }