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
#
# This is Agent 1's default list. The Stock Selection stage (below) replaces
# it with companies chosen from the whole index: run
# `python scripts/select_stocks.py`.


# ---------------------------------------------------------------------------
# Stock selection (Stage 0, runs before Agent 1)
# ---------------------------------------------------------------------------
# The selection stage starts from every current constituent of this PSX
# index, fetched live, so index rebalances are picked up automatically.
SELECTION_INDEX = "KMI30"

# Categories. A company's sector is read live from PSX's own listing, and
# PSX sectors are grouped into broad categories that follow the GICS sector
# structure (the international standard used by MSCI and S&P), so that
# small PSX sectors are not categories of one.
#
# The table covers every sector on the exchange (37, checked 2026-11-20),
# not individual companies, so any company that joins the index is
# classified without code changes. If PSX ever adds a new sector, it
# becomes its own category and a warning is shown until it is added here.
SECTOR_CATEGORIES: dict[str, str] = {
    # Energy
    "OIL & GAS EXPLORATION COMPANIES": "Energy",
    "OIL & GAS MARKETING COMPANIES": "Energy",
    "REFINERY": "Energy",
    # Utilities
    "POWER GENERATION & DISTRIBUTION": "Utilities",
    # Materials
    "CEMENT": "Materials",
    "CHEMICAL": "Materials",
    "FERTILIZER": "Materials",
    "GLASS & CERAMICS": "Materials",
    "PAPER, BOARD & PACKAGING": "Materials",
    "SYNTHETIC & RAYON": "Materials",
    # Industrials
    "CABLE & ELECTRICAL GOODS": "Industrials",
    "ENGINEERING": "Industrials",
    "TRANSPORT": "Industrials",
    # Consumer discretionary
    "AUTOMOBILE ASSEMBLER": "Consumer Discretionary",
    "AUTOMOBILE PARTS & ACCESSORIES": "Consumer Discretionary",
    "TEXTILE COMPOSITE": "Consumer Discretionary",
    "TEXTILE SPINNING": "Consumer Discretionary",
    "TEXTILE WEAVING": "Consumer Discretionary",
    "APPAREL": "Consumer Discretionary",
    "JUTE": "Consumer Discretionary",
    "WOOLLEN": "Consumer Discretionary",
    "LEATHER & TANNERIES": "Consumer Discretionary",
    # Consumer staples
    "FOOD & PERSONAL CARE PRODUCTS": "Consumer Staples",
    "SUGAR & ALLIED INDUSTRIES": "Consumer Staples",
    "VANASPATI & ALLIED INDUSTRIES": "Consumer Staples",
    "TOBACCO": "Consumer Staples",
    # Health care
    "PHARMACEUTICALS": "Health Care",
    # Financials
    "COMMERCIAL BANKS": "Financials",
    "INV. BANKS / INV. COS. / SECURITIES COS.": "Financials",
    "INSURANCE": "Financials",
    "LEASING COMPANIES": "Financials",
    "MODARABAS": "Financials",
    "CLOSE - END MUTUAL FUND": "Financials",
    # Technology (PSX does not separate IT from telecoms)
    "TECHNOLOGY & COMMUNICATION": "Technology & Communication",
    # Real estate
    "PROPERTY": "Real Estate",
    "REAL ESTATE INVESTMENT TRUST": "Real Estate",
    # PSX's catch-all sector
    "MISCELLANEOUS": "Miscellaneous",
}

# How companies are ranked within their category. Each measure is turned
# into a percentile rank inside the category (0-1, higher is better) and
# the selection score is the weighted sum. The weights add up to 1.
SELECTION_WEIGHTS: dict[str, float] = {
    "return_12m": 0.25,            # price performance, bonus-adjusted
    "risk_adjusted_return": 0.20,  # 12-month return / annualised volatility
    "max_drawdown": 0.10,          # worst fall from a peak (smaller is better)
    "earnings_yield": 0.20,        # 1 / P/E: cheapness relative to profit
    "dividend_yield": 0.10,        # cash returned to shareholders
    "liquidity": 0.15,             # average daily traded value
}

# Performance is measured over this period.
SELECTION_LOOKBACK = timedelta(days=365)

# Eligibility: a company is only ranked if it passes all of these.
SELECTION_MIN_TRADING_DAYS = 200        # within the lookback period
SELECTION_MAX_PRICE_AGE = timedelta(days=10)  # older = suspended / not trading
SELECTION_LIQUIDITY_DAYS = 63           # about 3 months of trading days
SELECTION_MIN_TRADED_VALUE_M = 20.0     # PKR millions per day, on average

# One company is selected per category, plus one more for every further
# CATEGORY_SIZE_PER_PICK eligible companies: a category of 1-5 gets 1,
# 6-10 gets 2, and so on. Large categories are not under-represented.
CATEGORY_SIZE_PER_PICK = 5


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
    "index_constituents": timedelta(days=1),
    "dividend_history": timedelta(days=1),
    "income_statement": timedelta(days=7),
    "balance_sheet": timedelta(days=7),
    "market_listing": timedelta(days=1),
}
