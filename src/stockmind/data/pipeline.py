"""Coordinate fetching, merging, cleaning, and local storage."""

import pandas as pd

from stockmind.data.cleaning import clean_historical_data
from stockmind.data.fetcher import fetch_historical_prices
from stockmind.data.storage import (
    load_raw_data,
    save_processed_data,
    save_quarantine_data,
    save_raw_data,
)


OVERLAP_DAYS = 7


def update_ticker(ticker: str) -> dict:
    """
    Update locally stored historical data for one ticker.

    Returns a summary of the update and output file paths.
    """
    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    existing = load_raw_data(ticker)

    # First run: retrieve the full history.
    if existing.empty:
        fetched = fetch_historical_prices(ticker)
    else:
        existing["date"] = pd.to_datetime(
            existing["date"], errors="coerce"
        )

        last_date = existing["date"].max()

        if pd.isna(last_date):
            # Cannot safely determine an incremental start date.
            fetched = fetch_historical_prices(ticker)
        else:
            start_date = (
                last_date - pd.Timedelta(days=OVERLAP_DAYS)
            ).strftime("%Y-%m-%d")

            fetched = fetch_historical_prices(
                ticker,
                start=start_date,
            )

    # Combine old and newly fetched records.
    if existing.empty:
        merged = fetched.copy()
    else:
        merged = pd.concat(
            [existing, fetched],
            ignore_index=True,
            sort=False,
        )

    merged.columns = [
        str(column).strip().lower() for column in merged.columns
    ]
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")

    # If a date appears in both datasets, prefer the newly fetched version.
    merged["_fetch_order"] = range(len(merged))
    merged = merged.sort_values(
        ["date", "_fetch_order"],
        na_position="last",
    )
    merged = merged.drop_duplicates(subset=["date"], keep="last")
    merged = merged.drop(columns=["_fetch_order"])
    merged = merged.sort_values("date", na_position="last")
    merged = merged.reset_index(drop=True)

    # Save source history before applying any cleaning.
    raw_path = save_raw_data(ticker, merged)

    # Revalidate the local history so corrections to recent records
    # can affect the cleaned output.
    clean_df, quarantine_df = clean_historical_data(merged)

    processed_path = save_processed_data(ticker, clean_df)
    quarantine_path = save_quarantine_data(ticker, quarantine_df)

    return {
        "ticker": ticker,
        "raw_records": len(merged),
        "clean_records": len(clean_df),
        "quarantined_records": len(quarantine_df),
        "raw_path": str(raw_path),
        "processed_path": str(processed_path),
        "quarantine_path": str(quarantine_path),
        "latest_date": (
            merged["date"].max().strftime("%Y-%m-%d")
            if merged["date"].notna().any()
            else None
        ),
    }