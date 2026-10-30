"""Project-wide settings for StockMind.

Keeping these values in one place makes them easy to find, explain,
and change without touching the agent code.
"""

import os
from datetime import timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Supported companies
# ---------------------------------------------------------------------------
# StockMind is specialised for the five largest constituents of the KMI-30
# index, ranked by index weight on the PSX Data Portal (checked 2026-10-30):
#
#   FFC    12.86%  Fauji Fertilizer Company Limited
#   ENGROH 10.61%  Engro Holdings Limited
#   MEBL   10.22%  Meezan Bank Limited
#   OGDC    8.50%  Oil & Gas Development Company Limited
#   HUBC    8.07%  The Hub Power Company Limited
#
# Index weights change when PSX rebalances the index. Run
# `python scripts/kmi30_weights.py` to see the current ranking.
#
# Scaling up: adding a symbol here is all that is needed. Nothing else in the
# code assumes five companies - sector peers are found automatically and
# every source is fetched and cached per symbol.
SUPPORTED_TICKERS: tuple[str, ...] = ("FFC", "ENGROH", "MEBL", "OGDC", "HUBC")


# ---------------------------------------------------------------------------
# Sector peers
# ---------------------------------------------------------------------------
# For industry comparisons, Agent 1 also fetches the financial statements of
# the largest other companies in the same PSX sector (ranked by market cap).
PEERS_PER_COMPANY = 4

# Upper limit on simultaneous requests to one website, to stay polite.
MAX_PARALLEL_REQUESTS = 4


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------
# Fetched data is cached outside the project, in the user's home folder.
# Set the STOCKMIND_CACHE_DIR environment variable to use another folder.
CACHE_DIR = Path(
    os.environ.get("STOCKMIND_CACHE_DIR", Path.home() / ".stockmind" / "cache")
)

# How long a cached response is reused before it is fetched again.
# - Prices and the market snapshot change during the trading day.
# - Dividends are announced occasionally.
# - Balance sheets only change when a new annual report is published.
CACHE_MAX_AGE: dict[str, timedelta] = {
    "prices": timedelta(hours=1),
    "company_page": timedelta(hours=1),
    "payouts": timedelta(days=1),
    "dividend_history": timedelta(days=1),
    "income_statement": timedelta(days=7),
    "balance_sheet": timedelta(days=7),
    "market_listing": timedelta(days=1),
}
