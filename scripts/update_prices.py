"""Command-line entry point for updating PSX stock prices."""

import argparse
import sys
from pathlib import Path


# Make the src/ directory importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stockmind.data.pipeline import update_ticker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update historical PSX stock-price data."
    )
    parser.add_argument(
        "ticker",
        help="PSX ticker symbol, e.g. SAZEW or OGDC",
    )

    args = parser.parse_args()

    summary = update_ticker(args.ticker)

    print("\nStockMind update completed")
    print("-" * 32)

    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()