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
# News (collected by Agent 1 for the Sentiment Agent, Agent 4)
# ---------------------------------------------------------------------------
# Sources and how items are tagged are explained in data/news.py.
#
# PSX company announcements: the first time a company is collected, this
# many years of its announcements are loaded; after that only the latest
# page is checked (CACHE_MAX_AGE["announcements"]).
NEWS_HISTORY_YEARS = 5

# PSX files each announcement under the symbol the company had at the time,
# so a renamed or restructured company's earlier news sits under its old
# symbols. These are added to its announcement history (tagged with the old
# symbol in `source`). Engro Holdings (ENGROH) is Dawood Hercules (DAWH)
# renamed in January 2025, when Engro Corporation (ENGRO) became its wholly
# owned subsidiary: ENGRO shareholders received ENGROH shares and ENGRO was
# delisted. DAWH is the same legal company; ENGRO is the business it now owns.
NEWS_PREDECESSOR_SYMBOLS: dict[str, tuple[str, ...]] = {
    "ENGROH": ("DAWH", "ENGRO"),
}

# Business-news RSS feeds: name -> (source shown to the user, feed URL).
# All of them allow automated access in their robots.txt (checked
# 2026-11-25). They are shared by every company and fetched once per
# refresh (CACHE_MAX_AGE["news_feeds"]).
NEWS_FEEDS: dict[str, tuple[str, str]] = {
    "brecorder-markets": ("Business Recorder", "https://www.brecorder.com/feeds/markets"),
    "brecorder-business": ("Business Recorder", "https://www.brecorder.com/feeds/business-finance"),
    "brecorder-latest": ("Business Recorder", "https://www.brecorder.com/feeds/latest-news"),
    "dawn-business": ("Dawn", "https://www.dawn.com/feeds/business"),
    "tribune-business": ("Express Tribune", "https://tribune.com.pk/feed/business"),
    "propakistani-business": ("ProPakistani", "https://propakistani.pk/category/business/feed/"),
}

# Publisher summaries are cut to this many characters: enough context for
# sentiment analysis, without keeping article text.
NEWS_SUMMARY_CHARS = 400

# Company names are derived from PSX's official name automatically (see
# news.company_names). These add short names the press uses that cannot be
# derived, e.g. "OGDCL" for OGDC.
NEWS_EXTRA_NAMES: dict[str, tuple[str, ...]] = {
    "OGDC": ("OGDCL",),
    "HUBC": ("HUBCO",),
    "MARI": ("Mari Petroleum",),
    "ENGROH": ("Engro Corporation", "Engro Corp", "Engro"),  # holding company of the Engro group
    "PSO": ("Pakistan State Oil",),
    "SNGP": ("SNGPL", "Sui Northern"),
    "SSGC": ("SSGCL", "Sui Southern"),
}

# A headline is "sector" news for a company when it uses one of its PSX
# sector's terms. Kept deliberately specific (e.g. "banks", not "bank",
# which would match "World Bank"). Sectors not listed get company and
# market news only.
NEWS_SECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "COMMERCIAL BANKS": ("banks", "banking sector", "banking industry",
                         "Islamic banking", "bank deposits", "bank advances"),
    "FERTILIZER": ("fertiliser", "fertilizer", "fertilisers", "fertilizers",
                   "urea", "DAP"),
    "OIL & GAS EXPLORATION COMPANIES": ("E&P", "exploration and production",
                                        "oil and gas exploration", "gas discovery",
                                        "oil discovery", "upstream"),
    "OIL & GAS MARKETING COMPANIES": ("OMC", "OMCs", "oil marketing", "petroleum sales",
                                      "fuel sales", "petrol prices", "diesel prices"),
    "REFINERY": ("refinery", "refineries", "refining"),
    "POWER GENERATION & DISTRIBUTION": ("IPP", "IPPs", "power sector", "power producers",
                                        "circular debt", "electricity tariff",
                                        "capacity payments"),
    "CEMENT": ("cement",),
    "TECHNOLOGY & COMMUNICATION": ("IT exports", "telecom", "software exports",
                                   "tech sector"),
    "AUTOMOBILE ASSEMBLER": ("car sales", "auto sales", "auto sector", "automobile"),
    "PHARMACEUTICALS": ("pharma", "pharmaceutical", "drug prices"),
    "CHEMICAL": ("chemical", "chemicals", "petrochemical", "PVC", "soda ash"),
    "ENGINEERING": ("steel",),
    "TEXTILE COMPOSITE": ("textile", "textiles", "apparel exports"),
    "GLASS & CERAMICS": ("glass", "tiles", "ceramics"),
    "INSURANCE": ("insurance", "takaful"),
    "CABLE & ELECTRICAL GOODS": ("home appliances", "electrical appliances",
                                 "electrical goods", "electronics industry",
                                 "air conditioners", "refrigerators", "transformers"),
    "FOOD & PERSONAL CARE PRODUCTS": ("FMCG", "consumer goods", "food industry",
                                      "packaged food", "edible oil", "cooking oil",
                                      "ghee", "personal care"),
    "INV. BANKS / INV. COS. / SECURITIES COS.": ("brokerage", "brokerages",
                                                 "brokerage houses", "mutual funds",
                                                 "asset management", "investment banks",
                                                 "securities firms"),
}

# A headline is "market" news when it is about the Pakistani stock market as
# a whole. Only PSX-specific terms: generic words such as "stocks" also match
# "Asian stocks rally" or "US stocks surge" in the same feeds.
NEWS_MARKET_KEYWORDS: tuple[str, ...] = (
    "KSE-100", "KSE 100", "KSE100", "KMI-30", "PSX", "Pakistan Stock Exchange",
    "bourse",
)

# Press archive: Express Tribune's sitemap lists every story it has
# published, newest first, 25,000 per file (about 5 months), each with its
# web address (whose last part is the headline) and the date it was last
# modified. robots.txt allows it (checked 2026-11-25). It gives a year of
# press headlines on the very first run, so the sector and market signals
# do not have to wait for the RSS feeds above to build up. The first page
# (CACHE_MAX_AGE["press_latest"]) keeps it up to date.
NEWS_PRESS_ARCHIVE: tuple[str, str] = (
    "Express Tribune", "https://tribune.com.pk/sitemap/posts-{page}.xml")
NEWS_PRESS_HISTORY_DAYS = 365

# Archive headlines come from web addresses ("cement-sales-grow-97-to-258m-
# tonnes"), so capitals and punctuation are lost, and the sitemap covers
# every section of the paper, not just business. Two rules keep matching
# accurate for them:
# - A sector keyword only counts together with one of these business terms,
#   so "Bethell looks to cement place" (cricket) is not cement news.
NEWS_BUSINESS_TERMS: tuple[str, ...] = (
    "price", "prices", "sales", "exports", "imports", "production", "output",
    "demand", "supply", "profit", "profits", "earnings", "revenue", "tariff",
    "tariffs", "tax", "taxes", "duty", "budget", "investment", "investors",
    "billion", "million", "bn", "growth", "industry", "sector", "companies",
    "capacity", "despatches", "dispatches", "imf", "sbp", "gdp", "inflation",
    "shares", "stocks", "market", "subsidy", "deficit", "debt", "loans",
    "financing", "plant", "plants", "rs",
)
# - A ticker is matched in any case ("ogdc plans partnerships"), except
#   tickers that are also ordinary words; those still need capitals, so
#   "luck" in a headline is not Lucky Cement.
NEWS_WORD_TICKERS: tuple[str, ...] = (
    "LUCK", "MARI", "PACE", "LOADS", "POWER", "GAL", "TELE", "AIRLINK",
)


# ---------------------------------------------------------------------------
# Sentiment analysis (Agent 4)
# ---------------------------------------------------------------------------
# FinBERT (Araci, 2019): BERT trained further on financial text and
# fine-tuned on Financial PhraseBank to label a sentence positive, negative
# or neutral. Pinned to one revision so scores never change silently; the
# model (~440 MB) is downloaded once into the Hugging Face cache
# (~/.cache/huggingface), outside the project.
SENTIMENT_MODEL = "ProsusAI/finbert"
SENTIMENT_MODEL_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
SENTIMENT_BATCH_SIZE = 32
SENTIMENT_MAX_TOKENS = 64   # headlines are short; longer text is cut

# Each scored signal: (weight, relevance, window in days, half-life in days).
# A signal averages the FinBERT sentiment (P(positive) - P(negative)) of the
# informative news items with that relevance published in the window. Newer
# items count more: an item's weight halves every half-life (the proposal's
# recency weighting; short-term news moves prices most, Mohan et al. 2019).
#   company_recent     the company's own news: PSX announcements and
#                      headlines naming it
#   company_long_term  the tone of the company's news over the last year
#   sector             headlines about its industry, e.g. urea prices
#   market             headlines about the PSX as a whole
SENTIMENT_SIGNALS: dict[str, tuple[float, str, int, float]] = {
    "company_recent":    (0.45, "company", 60, 14),
    "company_long_term": (0.15, "company", 365, 120),
    "sector":            (0.20, "sector", 30, 7),
    "market":            (0.20, "market", 14, 3),
}

# An average sentiment of +/- this much scores +/-1. Most financial
# headlines are neutral, so averages rarely come near 1.
SENTIMENT_SCALE = 0.5

# A signal needs this many recency-weighted items to count fully; with
# fewer it counts partly (one fresh headline counts 1/4), which lowers the
# confidence rather than letting one headline decide.
SENTIMENT_FULL_EVIDENCE = 4.0

# Same signal and confidence rules as Agents 2 and 3: score >= +threshold
# -> BUY, <= -threshold -> SELL, otherwise HOLD.
SENTIMENT_SIGNAL_THRESHOLD = 0.2
SENTIMENT_FULL_STRENGTH = 0.6


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
# - Prices and the market snapshot change during the trading day. They are
#   refreshed by the market clock (see MARKET_AWARE_SOURCES); the values
#   here are only an upper limit in case the clock is ever wrong.
# - Dividends are announced occasionally.
# - Balance sheets only change when a new annual report is published.
CACHE_MAX_AGE: dict[str, timedelta] = {
    "prices": timedelta(days=1),
    "company_page": timedelta(days=1),
    "payouts": timedelta(days=1),
    # News (data/news.py). Companies file announcements during the day and
    # they can move the price at once, so they are checked hourly. The press
    # mostly reports the next day, so every 2 hours is enough for the feeds.
    # The announcement history is reloaded monthly to fill any gap.
    "announcements": timedelta(hours=1),
    "announcement_history": timedelta(days=30),
    "news_feeds": timedelta(hours=2),
    # The press archive's first page (about 800 KB compressed) twice a day;
    # the whole year (about 2.5 MB) monthly, for companies added later.
    "press_latest": timedelta(hours=12),
    "press_archive": timedelta(days=30),
    "index_constituents": timedelta(days=1),
    # Stage 0's result: recalculated when a session ends (see current()).
    "selection": timedelta(days=7),
    "dividend_history": timedelta(days=1),
    "income_statement": timedelta(days=7),
    "balance_sheet": timedelta(days=7),
    "market_listing": timedelta(days=1),
}
