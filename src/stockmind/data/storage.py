"""Local CSV storage for StockMind historical price data."""

from pathlib import Path

import pandas as pd


# Resolve the project root from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
QUARANTINE_DIR = DATA_DIR / "quarantine"


def ensure_data_directories() -> None:
    """Create the required local data directories."""
    for directory in (RAW_DIR, PROCESSED_DIR, QUARANTINE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _validate_ticker(ticker: str) -> str:
    """Normalize a ticker before using it in a filename."""
    ticker = ticker.strip().upper()

    if not ticker or any(
        character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"
        for character in ticker
    ):
        raise ValueError(f"Invalid ticker for a filename: {ticker!r}")

    return ticker


def _ticker_path(directory: Path, ticker: str) -> Path:
    """Build a CSV path for one ticker."""
    ticker = _validate_ticker(ticker)
    return directory / f"{ticker}.csv"


def load_raw_data(ticker: str) -> pd.DataFrame:
    """Load previously saved raw data, or return an empty DataFrame."""
    ensure_data_directories()
    path = _ticker_path(RAW_DIR, ticker)

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path, parse_dates=["date"])


def load_processed_data(ticker: str) -> pd.DataFrame:
    """Load previously saved cleaned data, or return an empty DataFrame."""
    ensure_data_directories()
    path = _ticker_path(PROCESSED_DIR, ticker)

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path, parse_dates=["date"])


def save_raw_data(ticker: str, df: pd.DataFrame) -> Path:
    """Save raw source records without cleaning them."""
    ensure_data_directories()
    path = _ticker_path(RAW_DIR, ticker)

    df.to_csv(path, index=False)

    return path


def save_processed_data(ticker: str, df: pd.DataFrame) -> Path:
    """Save the cleaned dataset for a ticker."""
    ensure_data_directories()
    path = _ticker_path(PROCESSED_DIR, ticker)

    df.to_csv(path, index=False)

    return path


def save_quarantine_data(ticker: str, df: pd.DataFrame) -> Path:
    """Save suspicious or invalid records for later investigation."""
    ensure_data_directories()
    path = _ticker_path(QUARANTINE_DIR, ticker)

    df.to_csv(path, index=False)

    return path