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


# ---------------------------------------------------------------------------
# Fundamental analysis (Agent 3)
# ---------------------------------------------------------------------------
# Each scored ratio: (weight, bad, good).
#
# Absolute score: a ratio at `bad` scores -1, at `good` +1, linearly in
# between and capped at the ends. The direction follows bad -> good, so for
# P/E and debt-to-equity lower is better.
#
# Industry score: the same bad-to-good span sets how big a gap to the sector
# median matters. Being better than the median by half the span scores +1.
# E.g. ROE span = 20 points, so ROE 10 points above the median scores +1.
#
# The ratio's score is the average of the two (absolute only if there are
# too few peers). P/E, EPS growth, ROE, and revenue growth are the ratios
# named in the proposal, so they carry most of the weight. A ratio that is
# missing is skipped and the weights of the others are rescaled.
FUNDAMENTAL_METRICS: dict[str, tuple[float, float, float]] = {
    "pe_ratio":       (0.20, 15.0, 5.0),    # x
    "roe":            (0.20, 5.0, 25.0),    # %
    "eps_growth":     (0.15, -10.0, 20.0),  # % vs previous 12 months
    "revenue_growth": (0.15, -5.0, 15.0),   # % vs previous 12 months
    "net_margin":     (0.10, 2.0, 20.0),    # %
    "debt_to_equity": (0.10, 2.0, 0.3),     # x, not used for banks
    "price_to_book":  (0.05, 3.0, 1.0),     # x
    "dividend_yield": (0.05, 1.0, 9.0),     # %, last 12 months
}

# Overall score at or above +threshold -> BUY, at or below -threshold -> SELL,
# otherwise HOLD. Each ratio also "votes" with the same rule.
FUNDAMENTAL_SIGNAL_THRESHOLD = 0.2

# A BUY or SELL score this far from zero counts as a full-strength signal
# when calculating confidence.
FUNDAMENTAL_FULL_STRENGTH = 0.6

# Industry medians need at least this many peers with usable figures.
MIN_PEERS_FOR_COMPARISON = 2

# Peers whose latest statements end this long before the company's own are
# left out of the comparison (e.g. a peer that stopped filing).
PEER_MAX_STATEMENT_LAG = timedelta(days=548)  # about 18 months
