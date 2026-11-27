"""Run the Sentiment Analysis Agent from the command line.

Usage:
    python scripts/analyse_sentiment.py              # the Stage 0 selection
    python scripts/analyse_sentiment.py OGDC MEBL    # selected tickers
    python scripts/analyse_sentiment.py --force      # ignore the cache, fetch live
    python scripts/analyse_sentiment.py --headlines 10   # show more headlines

The first run downloads FinBERT (~440 MB, once) and scores every headline;
later runs reuse the remembered scores and take about a second per company.
"""

import argparse
import logging
import sys
from pathlib import Path


# Make the src/ directory importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stockmind.agents.data_collection import DataCollectionAgent, DataUnavailableError
from stockmind.agents.sentiment_analysis import SentimentAnalysisAgent
from stockmind.agents.stock_selection import StockSelectionAgent
from stockmind.data.tickers import UnsupportedTickerError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tickers", nargs="*")
    parser.add_argument("--force", action="store_true",
                        help="ignore cached responses and fetch live")
    parser.add_argument("--headlines", type=int, default=5,
                        help="how many recent company headlines to show")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    selection = StockSelectionAgent().current(force_refresh=args.force)
    tickers = args.tickers or selection.selected
    collector = DataCollectionAgent(universe=selection.universe)
    if not args.tickers:
        print(selection.summary)
        for warning in selection.warnings:
            print(f"WARNING: {warning}")
    analyst = SentimentAnalysisAgent()

    for ticker in tickers:
        result = analyst.analyse(collector.collect(ticker, force_refresh=args.force))

        print(f"\n{result.ticker}: {result.signal} "
              f"({result.confidence:.0f}% confidence, "
              f"sentiment score {result.sentiment_score:.0f}/100, {result.sentiment})")
        print(f"  {result.summary}")
        for sentence in result.explanation:
            print(f"  - {sentence}")

        company = result.articles[result.articles["relevance"] == "company"]
        if args.headlines and not company.empty:
            print("  Latest company news:")
            for row in company.head(args.headlines).itertuples():
                print(f"    {row.published_at:%Y-%m-%d}  {row.sentiment_label:<8} "
                      f"{row.sentiment:+.2f}  {row.title[:80]}")
        for warning in result.warnings:
            print(f"  WARNING: {warning}")


if __name__ == "__main__":
    try:
        main()
    except (DataUnavailableError, UnsupportedTickerError) as exc:
        raise SystemExit(f"Error: {exc}")
