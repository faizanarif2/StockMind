"""Fetch historical PSX stock prices using psxdata."""

import time

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

# PSX occasionally answers with an empty price table instead of an error
# (seen 2026-11-25 for FFL and HCAR while Stage 0 fetched all 30 KMI-30
# companies). A listed company always has prices, so an empty full history
# is a glitch: ask again a couple of times before giving up.
EMPTY_RESPONSE_RETRIES = 2
EMPTY_RESPONSE_PAUSE = 2  # seconds between attempts


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

    # StockMind caches responses itself (see cache.py), so psxdata's
    # separate cache in ~/.psxdata is disabled to avoid two copies.
    # A date-filtered request can be genuinely empty (e.g. a holiday), so
    # only a request for the full history is retried.
    attempts = 1 if (start or end) else 1 + EMPTY_RESPONSE_RETRIES
    for attempt in range(attempts):
        if attempt:
            time.sleep(EMPTY_RESPONSE_PAUSE)
        df = psxdata.stocks(
            ticker,
            start=start,
            end=end,
            cache=False,
        )
        if not df.empty:
            break

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

    # PSX returns newest rows first; downstream code expects oldest first.
    return df.sort_values("date").reset_index(drop=True)