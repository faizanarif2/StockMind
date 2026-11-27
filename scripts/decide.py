"""Run the Decision Making Agent from the command line.

Usage:
    python scripts/decide.py              # the Stage 0 selection
    python scripts/decide.py OGDC MEBL    # selected tickers
    python scripts/decide.py --force      # ignore the cache, fetch live

Runs Agents 1-4 for each company and combines Agents 2-4 into one final
Buy / Hold / Sell decision. Sentiment needs FinBERT: the first run
downloads it (~440 MB, once).
"""

import argparse
import logging
import sys
from pathlib import Path


# Make the src/ directory importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stockmind.agents.data_collection import DataCollectionAgent, DataUnavailableError
from stockmind.agents.decision_making import DecisionMakingAgent
from stockmind.agents.stock_selection import StockSelectionAgent
from stockmind.data.tickers import UnsupportedTickerError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="*")
    parser.add_argument("--force", action="store_true",
                        help="ignore cached responses and fetch live")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    selection = StockSelectionAgent().current(force_refresh=args.force)
    tickers = args.tickers or selection.selected
    collector = DataCollectionAgent(universe=selection.universe)
    if not args.tickers:
        print(selection.summary)
        for warning in selection.warnings:
            print(f"WARNING: {warning}")
    decider = DecisionMakingAgent()

    for ticker in tickers:
        decision = decider.decide(collector.collect(ticker, force_refresh=args.force))

        print(f"\n{decision.ticker}: {decision.signal} "
              f"({decision.confidence:.0f}% confidence, "
              f"decision score {decision.decision_score:.0f}/100, "
              f"agents {decision.agreement})")
        print(f"  {decision.summary}")
        for sentence in decision.explanation:
            print(f"  - {sentence}")
        for warning in decision.warnings:
            print(f"  WARNING: {warning}")


if __name__ == "__main__":
    try:
        main()
    except (DataUnavailableError, UnsupportedTickerError) as exc:
        raise SystemExit(f"Error: {exc}")
