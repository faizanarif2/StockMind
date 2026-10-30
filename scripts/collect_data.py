"""Run the Data Collection Agent from the command line.

Usage:
    python scripts/collect_data.py              # all supported tickers
    python scripts/collect_data.py OGDC MEBL    # selected tickers
    python scripts/collect_data.py --force      # ignore the cache, fetch live
"""

import argparse
import sys
import time
from pathlib import Path


# Make the src/ directory importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stockmind.agents.data_collection import DataCollectionAgent
from stockmind.config import SUPPORTED_TICKERS


def _span(df, column: str) -> str:
    """'<rows> rows, <oldest> to <newest>' for a dated table."""
    if df.empty:
        return "none"
    return (f"{len(df)} rows, {df[column].min():%Y-%m-%d} "
            f"to {df[column].max():%Y-%m-%d}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="*", default=list(SUPPORTED_TICKERS))
    parser.add_argument("--force", action="store_true",
                        help="ignore cached responses and fetch live")
    args = parser.parse_args()

    agent = DataCollectionAgent()

    for ticker in args.tickers:
        started = time.perf_counter()
        data = agent.collect(ticker, force_refresh=args.force)
        seconds = time.perf_counter() - started

        equity = data.profile.get("equity", {})
        years = sorted(data.balance_sheet.get("fiscal_year", []).unique()) \
            if not data.balance_sheet.empty else []
        print(f"\n{data.ticker} - {data.profile.get('name', 'unknown')} "
              f"({seconds:.2f}s)")
        print(f"  prices           : {_span(data.prices, 'date')}"
              f" ({len(data.price_issues)} rows rejected)")
        print(f"  shares           : {equity.get('shares_outstanding')}")
        print(f"  PSX financials   : {len(data.financials)} values")
        print(f"  income statement : {len(data.income_statement)} values")
        print(f"  balance sheet    : {len(data.balance_sheet)} values, {years}")
        print(f"  dividend history : {_span(data.dividend_history, 'ex_date')}")
        print(f"  PSX payouts      : {_span(data.payouts, 'announced_at')}")
        peers_with_data = sorted(set(data.peers.balance_sheet.get("ticker", [])))
        print(f"  sector peers     : {data.peers.sector} -> "
              f"{', '.join(data.peers.tickers) or 'none'} "
              f"(statements for {len(peers_with_data)})")
        for warning in data.warnings:
            print(f"  WARNING          : {warning}")


if __name__ == "__main__":
    main()
