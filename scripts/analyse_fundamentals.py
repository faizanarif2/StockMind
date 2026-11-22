"""Run the Fundamental Analysis Agent from the command line.

Usage:
    python scripts/analyse_fundamentals.py              # all supported tickers
    python scripts/analyse_fundamentals.py OGDC MEBL    # selected tickers
    python scripts/analyse_fundamentals.py --force      # ignore the cache, fetch live
"""

import argparse
import sys
from pathlib import Path


# Make the src/ directory importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stockmind.agents.data_collection import DataCollectionAgent, DataUnavailableError
from stockmind.agents.fundamental_analysis import FundamentalAnalysisAgent
from stockmind.agents.stock_selection import StockSelectionAgent
from stockmind.data.tickers import UnsupportedTickerError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="*")
    parser.add_argument("--force", action="store_true",
                        help="ignore cached responses and fetch live")
    args = parser.parse_args()

    selection = StockSelectionAgent().current(force_refresh=args.force)
    tickers = args.tickers or selection.selected
    collector = DataCollectionAgent(universe=selection.universe)
    if not args.tickers:
        print(selection.summary)
        for warning in selection.warnings:
            print(f"WARNING: {warning}")
    analyst = FundamentalAnalysisAgent()

    for ticker in tickers:
        result = analyst.analyse(collector.collect(ticker, force_refresh=args.force))

        print(f"\n{result.ticker}: {result.signal} "
              f"({result.confidence:.0f}% confidence, "
              f"health {result.health_score:.0f}/100)")
        print(f"  {result.summary}")
        for sentence in result.explanation:
            print(f"  - {sentence}")
        for warning in result.warnings:
            print(f"  WARNING: {warning}")


if __name__ == "__main__":
    try:
        main()
    except (DataUnavailableError, UnsupportedTickerError) as exc:
        raise SystemExit(f"Error: {exc}")
