"""Project-wide settings for StockMind.

Keeping these values in one place makes them easy to find, explain,
and change without touching the agent code.
"""

import os
from datetime import time, timedelta
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

# A new selection is not saved if more companies than this could not get
# fresh prices: the outage would distort the ranking, so the last one is kept.
SELECTION_MAX_FAILED_PRICES = 3

# One company is selected per category, plus one more for every further
# CATEGORY_SIZE_PER_PICK eligible companies: a category of 1-5 gets 1,
# 6-10 gets 2, and so on. Large categories are not under-represented.
CATEGORY_SIZE_PER_PICK = 5


# ---------------------------------------------------------------------------
# Market hours (market-aware refreshing)
# ---------------------------------------------------------------------------
# All market times are Pakistan time, whatever the computer's own time zone.
MARKET_TIMEZONE = "Asia/Karachi"

# Per weekday (Monday = 0): a broad window in which PSX's regular market MAY
# be trading. It is deliberately wider than the official hours (about
# 09:15-15:30; Friday 09:15-12:00 and 14:30-16:30; post-close until about
# 16:50) so that pre-open, post-close and Ramadan timings all fit inside it.
#   - Outside the window nothing market-driven is fetched (nights, weekends).
#   - Inside it, PSX's own live market "State" decides whether trading is
#     happening, so holidays (including moon-sighted ones) and unscheduled
#     closures need no calendar.
#   - A session's data is final once its window has ended.
TRADING_WINDOWS: dict[int, tuple[time, time]] = {
    0: (time(9, 0), time(16, 15)),
    1: (time(9, 0), time(16, 15)),
    2: (time(9, 0), time(16, 15)),
    3: (time(9, 0), time(16, 15)),
    4: (time(9, 0), time(17, 15)),
}

# How often data that changes during a session is refreshed while trading.
SESSION_REFRESH = timedelta(minutes=15)

# Sources whose data changes during a trading session. They are refreshed
# by the market clock; CACHE_MAX_AGE is only an upper limit for them.
MARKET_AWARE_SOURCES = ("prices", "company_page", "market_listing")

# PSX's live market state (homepage "State" of this board) is re-read at
# most this often. State words that mean "not trading"; any other word is
# treated as trading, so unknown states never cause stale data.
MARKET_STATE_MAX_AGE = timedelta(minutes=5)
MARKET_STATE_BOARD = "REG"
MARKET_CLOSED_STATES = ("CLOSED", "SUSPENDED", "HALTED", "HALT")

# Market Watch (today's open/high/low/current/volume for every company in
# one request) updates today's price bar between full downloads.
# Kept OFF until its behaviour during a live session has been verified with
# `python scripts/check_market_session.py`. While off, prices are refreshed
# by downloading the full history every SESSION_REFRESH during a session.
LIVE_PRICES_ENABLED = False
# A Market Watch read is rejected (and the full download used instead) when
# PSX's "As of" time lags the clock by more than this during trading, or
# when Market Watch's total volume differs from PSX's regular-market volume
# by more than this share.
LIVE_PRICES_MAX_LAG = timedelta(minutes=20)
LIVE_PRICES_VOLUME_TOLERANCE = 0.02


# ---------------------------------------------------------------------------
# Permanent history store
# ---------------------------------------------------------------------------
# Unlike the cache, this folder keeps every record ever collected, including
# records that have since dropped out of a source's window (PSX shows 10
# years of prices and 2 years of payouts). It CANNOT be rebuilt by
# downloading again, so back it up (see README / scripts/backup_history.py).
# Set the STOCKMIND_DATA_DIR environment variable to use another folder.
HISTORY_DIR = Path(
    os.environ.get("STOCKMIND_DATA_DIR", Path.home() / ".stockmind" / "data")
)

# Previous versions of each file kept in _backups/ before it is rewritten.
HISTORY_BACKUPS_KEPT = 3

# An update that would change more than this share of the stored records it
# overlaps (and at least HISTORY_MIN_SUSPICIOUS_CHANGES of them) is refused
# as suspicious, e.g. a unit change or a broken page. The stored data is
# kept and a warning explains why.
HISTORY_MAX_CORRECTION_SHARE = 0.05
HISTORY_MIN_SUSPICIOUS_CHANGES = 5


# ---------------------------------------------------------------------------
# Sector peers
# ---------------------------------------------------------------------------
# For industry comparisons, Agent 1 also fetches the financial statements of
# the largest other companies in the same PSX sector (ranked by market cap).
PEERS_PER_COMPANY = 4

# Upper limit on simultaneous requests to one website, to stay polite.
MAX_PARALLEL_REQUESTS = 4


# ---------------------------------------------------------------------------
# Technical analysis (Agent 2)
# ---------------------------------------------------------------------------
# Indicator window lengths in trading days (the standard textbook values).
TECHNICAL_WINDOWS: dict[str, int] = {
    "sma_short": 50,    # medium-term trend
    "sma_long": 200,    # long-term trend
    "ema": 20,          # short-term trend
    "rsi": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger": 20,
}
BOLLINGER_STD = 2.0

# Each scored signal: (weight, scale).
#
# Trend signals score the % gap between two lines; a gap of `scale` %
# (or more) scores +1 when bullish and -1 when bearish, linearly in between.
#   long_term_trend  50-day SMA vs 200-day SMA ("golden" / "death" cross)
#   price_vs_sma     close vs 50-day SMA
#   price_vs_ema     close vs 20-day EMA
#   macd             MACD histogram as a % of the price (momentum)
#
# Oscillators score against fixed levels (full buy, buy starts,
# sell starts, full sell): inside the middle range they score 0.
#   rsi        RSI <= 30 oversold (+1) ... RSI >= 70 overbought (-1)
#   bollinger  %B <= 0 at/below lower band (+1) ... %B >= 1 at/above upper (-1)
#
# Trend signals follow the market; oscillators expect a reversal. When they
# disagree, agreement - and therefore confidence - goes down.
TECHNICAL_SIGNALS: dict[str, tuple[float, object]] = {
    "long_term_trend": (0.20, 5.0),
    "price_vs_sma":    (0.15, 5.0),
    "price_vs_ema":    (0.15, 3.0),
    "macd":            (0.20, 1.0),
    "rsi":             (0.15, (30.0, 40.0, 60.0, 70.0)),
    "bollinger":       (0.15, (0.0, 0.2, 0.8, 1.0)),
}

# Same signal and confidence rules as Agent 3: score >= +threshold -> BUY,
# <= -threshold -> SELL, otherwise HOLD.
TECHNICAL_SIGNAL_THRESHOLD = 0.2
TECHNICAL_FULL_STRENGTH = 0.6

# Warn when the latest price is older than this (e.g. a stale cache).
TECHNICAL_MAX_DATA_AGE = timedelta(days=7)


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------
# Fetched data is cached outside the project, in the user's home folder.
# Set the STOCKMIND_CACHE_DIR environment variable to use another folder.
CACHE_DIR = Path(
    os.environ.get("STOCKMIND_CACHE_DIR", Path.home() / ".stockmind" / "cache")
)

# How long a cached response is reused before it is fetched again.
# - Prices and the market snapshot change during the trading day. They are
#   refreshed by the market clock (see MARKET_AWARE_SOURCES); the values
#   here are only an upper limit in case the clock is ever wrong.
# - Dividends are announced occasionally.
# - Balance sheets only change when a new annual report is published.
CACHE_MAX_AGE: dict[str, timedelta] = {
    "prices": timedelta(days=1),
    "company_page": timedelta(days=1),
    "payouts": timedelta(days=1),
    "index_constituents": timedelta(days=1),
    # Stage 0's result: recalculated when a session ends (see current()).
    "selection": timedelta(days=7),
    "dividend_history": timedelta(days=1),
    "income_statement": timedelta(days=7),
    "balance_sheet": timedelta(days=7),
    "market_listing": timedelta(days=1),
}
