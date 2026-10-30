"""Show the current KMI-30 constituents ranked by index weight.

Use this to check whether SUPPORTED_TICKERS in config.py is still the
top five after PSX rebalances the index.

Usage:
    python scripts/kmi30_weights.py
"""

import psxdata


def main() -> None:
    index = psxdata.indices("KMI30", cache=False)
    top = index.sort_values("idx_weight", ascending=False).head(10)
    print(top[["symbol", "name", "idx_weight", "market_cap_m"]].to_string(index=False))


if __name__ == "__main__":
    main()
