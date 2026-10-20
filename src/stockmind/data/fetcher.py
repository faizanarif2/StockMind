"""Fetch historical PSX stock prices using psxdata."""

import pandas as pd
import psxdata


REQUIRED_COLUMNS = {
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def fetch_historical_prices(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a PSX ticker.

    Parameters:
        ticker: PSX stock symbol, e.g. "SAZEW".
        start: Optional start date in YYYY-MM-DD format.
        end: Optional end date in YYYY-MM-DD format.

    Returns:
        A copy of the DataFrame returned by psxdata.

    Raises:
        ValueError: If the ticker is empty or required columns are missing.
    """
    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    # Keep psxdata caching enabled.
    df = psxdata.stocks(
        ticker,
        start=start,
        end=end,
        cache=True,
    )

    if df.empty:
        raise ValueError(
            f"No historical price data was returned for {ticker}."
        )

    df = df.copy()
    df.columns = [str(column).strip().lower() for column in df.columns]

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Unexpected response for {ticker}. "
            f"Missing columns: {sorted(missing_columns)}"
        )

    return df