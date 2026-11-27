"""Agent 1 - Data Collection Agent.

Gives the other agents everything they need about one company, fetched on
demand from the live sources:

    Source                 Dataset
    ---------------------  -----------------------------------------------
    PSX historical prices  prices: ~10 years of daily OHLCV (validated)
    PSX company page       profile: name, sector, shares, face value,
                           free float, market cap, price, P/E, 52-week range
                           financials: 4 years + 4 quarters of sales/income,
                           profit after tax, EPS
                           ratios: margins, EPS growth, PEG
    PSX payouts            payouts: official dividend / bonus / right
                           announcements (~2 years)
    stockanalysis.com      income_statement: 5 fiscal years + TTM of
                           consolidated revenue, net income, EPS, margins
                           balance_sheet: 5 fiscal years (equity, assets,
                           liabilities, debt, book value per share)
                           dividend_history: cash dividends per share with
                           ex-dates (~5+ years)

    PSX market listing     peers: the largest other companies in the same
    + stockanalysis.com    PSX sector (config.PEERS_PER_COMPANY), with their
                           market cap, P/E, dividend yield, income statement
                           and balance sheet - for industry-average comparisons

    PSX announcements      news: for the Sentiment Agent (Agent 4). The
    + business-news RSS    company's official announcements (5 years) and
    + press archive        headlines from Business Recorder, Dawn, Express
                           Tribune and ProPakistani that concern the
                           company, its sector or the whole market (a year
                           of Express Tribune headlines from the first run).
                           One row per item, newest first; see data/news.py.

Use income_statement + balance_sheet together for ratios such as ROE:
both are consolidated. PSX `financials` can be standalone figures.

Which companies
---------------
By default the agent accepts config.SUPPORTED_TICKERS. The Stock Selection
stage (agents/stock_selection.py) chooses companies from the whole KMI-30
instead and passes them in:

    selection = StockSelectionAgent().select()
    agent = DataCollectionAgent(universe=selection.universe)
    data = agent.collect_many(selection.selected)

Everything is fetched and cached per symbol, and peers are found from the
PSX sector listing, so nothing assumes a fixed number of companies.

How it stays fast
-----------------
1. All sources (and the peers) are requested in parallel, so a cold request takes
   about as long as the slowest source instead of the sum of all of them.
2. Responses go through an automatic cache (see data/cache.py) kept in the
   user's home folder. Repeat requests within the cache's max age are
   answered in milliseconds; older entries are fetched again.
3. If a source is down, the last fetched copy is used and a warning is
   attached, so the rest of the pipeline can still run. If nothing is
   cached either, the permanent history store is used.

Market-aware refreshing and permanent history
---------------------------------------------
4. Prices, the company page and the market listing are refreshed by the
   market clock (data/market_clock.py): every config.SESSION_REFRESH while
   PSX is trading, once more when a session ends, and not at all at night,
   at weekends or on holidays.
5. Every download is merged into the permanent history store
   (data/history_store.py), so records that drop out of a source's window
   (PSX keeps 10 years of prices, 2 years of payouts) are kept. The values
   handed to the other agents have exactly the same columns as before;
   they simply reach further back over time.
6. When PSX shows a reporting quarter we have not seen before, the
   financial statements are fetched again at once instead of waiting for
   their weekly refresh.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import requests
from psxdata.exceptions import PSXDataError

from stockmind import config
from stockmind.data import company, market, news, stockanalysis
from stockmind.data.cache import ResponseCache
from stockmind.data.history_store import HistoryStore
from stockmind.data.market_clock import MarketClock
from stockmind.data.cleaning import clean_historical_data
from stockmind.data.fetcher import fetch_historical_prices
from stockmind.data.tickers import resolve_ticker


logger = logging.getLogger(__name__)

# Errors that mean "the source is unavailable or returned something
# unexpected". The agent can recover from these; anything else is a bug.
SOURCE_ERRORS = (PSXDataError, requests.RequestException, ValueError)


class DataUnavailableError(RuntimeError):
    """Raised when price data, which every analysis needs, cannot be fetched."""


@dataclass
class SectorPeers:
    """The largest other companies in the same PSX sector, for comparisons."""

    sector: str | None
    tickers: list[str]
    market: pd.DataFrame            # symbol, name, market_cap_m, price, pe_ratio, ...
    income_statement: pd.DataFrame  # same layout as CompanyData, plus "ticker"
    balance_sheet: pd.DataFrame     # same layout as CompanyData, plus "ticker"

    @classmethod
    def empty(cls, sector: str | None = None) -> "SectorPeers":
        return cls(sector, [], pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


@dataclass
class CompanyData:
    """Everything Agent 1 hands to the analysis agents for one ticker."""

    ticker: str
    prices: pd.DataFrame
    price_issues: pd.DataFrame
    profile: dict
    financials: pd.DataFrame
    ratios: pd.DataFrame
    income_statement: pd.DataFrame
    balance_sheet: pd.DataFrame
    payouts: pd.DataFrame
    dividend_history: pd.DataFrame
    peers: SectorPeers
    # Announcements and headlines for Agent 4, newest first; columns in
    # news.NEWS_COLUMNS. Has a default so code that builds CompanyData
    # without news (e.g. other agents' tests) keeps working.
    news: pd.DataFrame = field(default_factory=news.empty_news)
    fetched_at: dict[str, datetime] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def latest_price_date(self) -> pd.Timestamp | None:
        return self.prices["date"].max() if not self.prices.empty else None


def _fetch_prices(symbol: str) -> dict:
    """Download price history and split it into clean rows and problem rows."""
    clean, issues = clean_historical_data(fetch_historical_prices(symbol))
    return {"prices": clean, "price_issues": issues}


# Each source: name -> (fetch function, value to use if it is unavailable).
SOURCES = {
    "prices": (_fetch_prices, None),  # required, no fallback
    "company_page": (
        company.fetch_company,
        {"profile": {}, "financials": pd.DataFrame(), "ratios": pd.DataFrame()},
    ),
    "payouts": (company.fetch_payouts, pd.DataFrame()),
    "income_statement": (stockanalysis.fetch_income_statement, pd.DataFrame()),
    "balance_sheet": (stockanalysis.fetch_balance_sheet, pd.DataFrame()),
    "dividend_history": (stockanalysis.fetch_dividend_history, pd.DataFrame()),
}

# Statements fetched for each sector peer. They use the same cache entries
# as SOURCES, so a company that is both supported and a peer is fetched once.
PEER_SOURCES = ("income_statement", "balance_sheet")

# Cache owner for market-wide data that is shared by every company.
MARKET = "PSX"

# Cache owner for the business-news feeds, also shared by every company.
MEDIA = "MEDIA"


class DataCollectionAgent:
    """Fetch, validate, and cache PSX data for the supported companies."""

    name = "Data Collection Agent"

    def __init__(
        self,
        cache: ResponseCache | None = None,
        universe: list[str] | tuple[str, ...] | None = None,
        history: HistoryStore | None = None,
        clock: MarketClock | None = None,
    ) -> None:
        """
        Args:
            cache: Response cache to use (default: config.CACHE_DIR).
            universe: The tickers this agent accepts. Defaults to
                config.SUPPORTED_TICKERS; the Stock Selection stage passes
                the whole index here.
            history: Permanent history store (default: config.HISTORY_DIR).
            clock: Market clock deciding when market data is refreshed.
        """
        self._cache = cache or ResponseCache(config.CACHE_DIR)
        self.universe: tuple[str, ...] = (
            tuple(universe) if universe else config.SUPPORTED_TICKERS
        )
        self._history = history or HistoryStore(config.HISTORY_DIR)
        self._clock = clock or MarketClock()

    def collect(self, ticker: str, force_refresh: bool = False) -> CompanyData:
        """
        Return all data for one ticker.

        Args:
            ticker: A PSX symbol in the agent's universe, e.g. "OGDC"
                (case-insensitive).
            force_refresh: Ignore cached responses and fetch everything live.

        Raises:
            UnsupportedTickerError: The ticker is not in the agent's universe.
            DataUnavailableError: Price data could not be fetched.
        """
        ticker = resolve_ticker(ticker, self.universe)

        # The company's own sources, its sector peers and its news are
        # fetched together.
        with ThreadPoolExecutor(max_workers=len(SOURCES) + 2) as pool:
            futures = {
                name: pool.submit(self.load_source, ticker, name, force_refresh)
                for name in SOURCES
            }
            peers_future = pool.submit(self._collect_peers, ticker, force_refresh)
            news_future = pool.submit(self._collect_news, ticker,
                                      futures["company_page"], force_refresh)
            results = {name: future.result() for name, future in futures.items()}
            peers, peer_warnings = peers_future.result()
            company_news, news_warnings, news_fetched_at = news_future.result()

        # A reporting quarter we have not seen before has appeared on PSX:
        # fetch the statements now rather than at their weekly refresh.
        new_periods = results["company_page"]["history"].get("new_periods", {})
        if new_periods.get("financials") and not force_refresh:
            for name in ("income_statement", "balance_sheet"):
                results[name] = self.load_source(ticker, name, force_refresh=True)

        warnings = [r["warning"] for r in results.values() if r["warning"]]
        warnings += peer_warnings + news_warnings
        fetched_at = {
            name: r["fetched_at"] for name, r in results.items() if r["fetched_at"]
        }
        fetched_at.update(news_fetched_at)

        if results["prices"]["value"] is None:
            raise DataUnavailableError(
                f"No price data available for {ticker}. " + " ".join(warnings)
            )

        prices = results["prices"]["value"]
        latest = prices["prices"]["date"].max() if not prices["prices"].empty else None
        if latest is not None and not self._clock.day_is_final(pd.Timestamp(latest).date()):
            warnings.append(
                f"{ticker}: the latest price ({pd.Timestamp(latest):%Y-%m-%d}) is "
                "from a session still in progress; it will change until the close."
            )
        page = results["company_page"]["value"]
        face_value = page["profile"].get("equity", {}).get("face_value")
        payouts = results["payouts"]["value"]
        if not payouts.empty:
            payouts = company.add_cash_per_share(payouts, face_value)

        return CompanyData(
            ticker=ticker,
            prices=prices["prices"],
            price_issues=prices["price_issues"],
            profile=page["profile"],
            financials=page["financials"],
            ratios=page["ratios"],
            income_statement=results["income_statement"]["value"],
            balance_sheet=results["balance_sheet"]["value"],
            payouts=payouts,
            dividend_history=results["dividend_history"]["value"],
            peers=peers,
            news=company_news,
            fetched_at=fetched_at,
            warnings=warnings,
        )

    def collect_many(
        self,
        tickers: list[str] | tuple[str, ...] | None = None,
        force_refresh: bool = False,
    ) -> dict[str, CompanyData]:
        """Collect several tickers (default: the whole universe)."""
        return {
            data.ticker: data
            for data in (self.collect(t, force_refresh)
                         for t in (tickers or self.universe))
        }

    # ------------------------------------------------------------------

    def _collect_peers(
        self, ticker: str, force_refresh: bool
    ) -> tuple[SectorPeers, list[str]]:
        """Find the sector's largest other companies and fetch their statements."""
        listing = self.load_source(
            MARKET, "market_listing", force_refresh,
            fetch=lambda _: market.fetch_market_listing(),
            fallback=pd.DataFrame(),
        )
        warnings = [listing["warning"]] if listing["warning"] else []
        table = listing["value"]

        if table.empty:
            return SectorPeers.empty(), warnings

        sector_rows = table.loc[table["symbol"] == ticker, "sector"]
        sector = sector_rows.iloc[0] if not sector_rows.empty else None
        symbols = market.find_sector_peers(ticker, table, config.PEERS_PER_COMPANY)
        if not symbols:
            return SectorPeers.empty(sector), warnings

        jobs = [(peer, source) for peer in symbols for source in PEER_SOURCES]
        with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_REQUESTS) as pool:
            loaded = list(pool.map(
                lambda job: self.load_source(job[0], job[1], force_refresh), jobs
            ))

        statements = {source: [] for source in PEER_SOURCES}
        for (peer, source), result in zip(jobs, loaded):
            if result["warning"]:
                warnings.append(result["warning"])
            if not result["value"].empty:
                statements[source].append(result["value"].assign(ticker=peer))

        def combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        peer_rows = table.set_index("symbol").loc[symbols].reset_index()
        return SectorPeers(
            sector=sector,
            tickers=symbols,
            market=peer_rows,
            income_statement=combine(statements["income_statement"]),
            balance_sheet=combine(statements["balance_sheet"]),
        ), warnings

    def _collect_news(
        self, ticker: str, company_page, force_refresh: bool
    ) -> tuple[pd.DataFrame, list[str], dict[str, datetime]]:
        """
        Announcements and matching headlines for one company.

        1. Announcement history (config.NEWS_HISTORY_YEARS): downloaded the
           first time, then monthly; merged into the history store.
        2. Latest announcements: hourly; merged into the history store, which
           returns everything kept so far.
        3. Business-news feeds: shared by all companies, every 2 hours.
           Headlines about this company, its sector or the market are kept
           in the history store, because feeds only show the latest ~30.
        4. Press archive (config.NEWS_PRESS_ARCHIVE): shared by all
           companies; a year of headlines monthly, the newest page twice a
           day. Matching headlines are kept like the feeds' ones.

        `company_page` is the future of the company-page download: matching
        headlines needs the company's name and sector, but the downloads
        above do not, so they run while the company page is still loading.
        """
        warnings: list[str] = []
        fetched_at: dict[str, datetime] = {}

        def load(owner: str, source: str, fetch, fallback):
            result = self.load_source(owner, source, force_refresh,
                                      fetch=fetch, fallback=fallback)
            if result["warning"]:
                warnings.append(result["warning"])
            if result["fetched_at"]:
                fetched_at[source] = result["fetched_at"]
            return result["value"]

        # Not a SOURCE_DATASETS source, so it is merged into the store here,
        # before the latest page is loaded and merged on top of it.
        history = load(ticker, "announcement_history",
                       lambda t: news.fetch_announcement_history(t, config.NEWS_HISTORY_YEARS),
                       news.empty_news())
        if not history.empty:
            history, info = self._history.retain(ticker, "announcements", history,
                                                 self._clock.day_is_final)
            warnings += info["warnings"]
        latest = load(ticker, "announcements", news.fetch_announcements,
                      news.empty_news())

        feeds = load(MEDIA, "news_feeds", lambda _: news.fetch_feeds(),
                     {"items": news.empty_news(), "failed": []})
        if feeds["failed"]:
            warnings.append("News feeds unavailable this time: "
                            + ", ".join(feeds["failed"]) + ".")
        archive = load(MEDIA, "press_archive",
                       lambda _: news.fetch_press_archive(config.NEWS_PRESS_HISTORY_DAYS),
                       news.empty_news())
        archive_latest = load(MEDIA, "press_latest",
                              lambda _: news.fetch_press_latest(),
                              news.empty_news())

        profile = company_page.result()["value"]["profile"]
        if not profile.get("name"):
            warnings.append(f"{ticker}: company name unknown; headlines are "
                            "matched by ticker only.")
        def match(items):
            return news.match_articles(items, ticker, profile.get("name"),
                                       profile.get("sector"))

        headlines = news.combine_news([feeds["items"], archive_latest, archive])
        articles, info = self._history.retain(ticker, "articles", match(headlines),
                                              self._clock.day_is_final)
        warnings += info["warnings"]
        # Stored headlines are matched again with today's rules, so a change
        # to the keywords in config.py also applies to older headlines.
        articles = match(articles)

        # Announcements kept from earlier runs are classified again too.
        combined = news.reclassify_announcements(
            news.combine_news([latest, history, articles]))
        return combined, warnings, fetched_at

    def load_source(
        self,
        owner: str,
        source: str,
        force_refresh: bool,
        fetch=None,
        fallback=None,
    ) -> dict:
        """
        Load one source for one owner (a ticker, or MARKET) through the cache.

        Never raises for source errors: returns the fallback with a warning
        (or, if the history store has the data, the stored history).
        Returns {"value", "fetched_at", "warning", "history"}. Also used by
        the Stock Selection stage, so both share the same cache entries.

        Steps: the cache decides whether to download (market-aware for
        config.MARKET_AWARE_SOURCES); the result is merged into the history
        store, which returns the full retained history in the same shape.
        """
        if fetch is None:
            fetch, fallback = SOURCES[source]
        try:
            result = self._cache.get_or_fetch(
                key=f"{owner}:{source}",
                max_age=config.CACHE_MAX_AGE[source],
                fetch=lambda: fetch(owner),
                fallback_errors=SOURCE_ERRORS,
                force_refresh=force_refresh,
                is_fresh=self._freshness_rule(source),
            )
        except SOURCE_ERRORS as exc:
            message = f"{owner}:{source} unavailable ({type(exc).__name__}: {exc})."
            stored = self._history.stored_value(owner, source)
            if stored is not None:
                message += " Using the permanent history store instead."
            logger.warning(message)
            return {"value": stored if stored is not None else fallback,
                    "fetched_at": None, "warning": message, "history": {}}

        warnings = [result.warning] if result.warning else []
        value, history = self._history.retain(
            owner, source, result.value, self._clock.day_is_final)
        warnings += history["warnings"]

        message = " ".join(warnings) or None
        if message:
            logger.warning(message)
        return {"value": value, "fetched_at": result.fetched_at,
                "warning": message, "history": history}

    def _freshness_rule(self, source: str):
        """The market clock's rule for market-driven sources, else None."""
        if source not in config.MARKET_AWARE_SOURCES:
            return None
        return lambda fetched_at: self._clock.is_fresh(fetched_at)
