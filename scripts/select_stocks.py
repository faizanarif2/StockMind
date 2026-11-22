"""Show the Stock Selection stage's ranking from the command line.

Usage:
    python scripts/select_stocks.py            # rank the KMI-30, show the picks
    python scripts/select_stocks.py --all      # also show excluded companies
    python scripts/select_stocks.py --force    # ignore the cache, fetch live

scripts/collect_data.py runs Agent 1 on this same selection.
"""

import argparse
import sys
import time
from pathlib import Path


# Make the src/ directory importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stockmind.agents.data_collection import DataUnavailableError
from stockmind.agents.stock_selection import StockSelectionAgent


def _num(value, pattern: str) -> str:
    return "-" if value is None or value != value else pattern.format(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true",
                        help="list every constituent, including excluded ones")
    parser.add_argument("--force", action="store_true",
                        help="ignore cached responses and fetch live")
    args = parser.parse_args()

    started = time.perf_counter()
    selection = StockSelectionAgent().current(force_refresh=args.force)
    print(f"{selection.summary}  ({time.perf_counter() - started:.1f}s)\n")

    header = (f"  {'':2}{'#':>2} {'symbol':<8}{'score':>6}{'12m ret':>9}"
              f"{'risk-adj':>9}{'max DD':>8}{'E/P':>7}{'div':>6}"
              f"{'PKR m/day':>11}")
    for category, group in selection.ranking.groupby("category", sort=True):
        print(f"{category}")
        print(header)
        for _, row in group.iterrows():
            if not row["eligible"]:
                if args.all:
                    print(f"     {row['symbol']:<8} excluded: {row['excluded_because']}")
                continue
            mark = "->" if row["selected"] else ""
            print(f"  {mark:2}{row['category_rank']:>2} {row['symbol']:<8}"
                  f"{row['score']:>6.0f}"
                  f"{_num(row['return_12m'], '{:+.1f}%'):>9}"
                  f"{_num(row['risk_adjusted_return'], '{:.2f}'):>9}"
                  f"{_num(row['max_drawdown'], '{:.0f}%'):>8}"
                  f"{_num(row['earnings_yield'], '{:.1f}%'):>7}"
                  f"{_num(row['dividend_yield'], '{:.1f}%'):>6}"
                  f"{_num(row['liquidity'], '{:,.0f}'):>11}")
        print()

    print("Why each company was selected:")
    for ticker, reason in selection.reasons.items():
        print(f"  - {reason}")
    excluded = selection.ranking[~selection.ranking["eligible"]]
    if not excluded.empty:
        print(f"\n{len(excluded)} excluded (use --all to list them).")
    for warning in selection.warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    try:
        main()
    except DataUnavailableError as exc:
        raise SystemExit(f"Error: {exc}")
