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
