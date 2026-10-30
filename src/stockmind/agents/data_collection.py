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

Use income_statement + balance_sheet together for ratios such as ROE:
both are consolidated. PSX `financials` can be standalone figures.

Scaling
-------
Everything is fetched and cached per symbol, and peers are found from the
PSX sector listing, so supporting more companies only means adding them to
config.SUPPORTED_TICKERS. `collect_many()` collects several at once.

How it stays fast
-----------------
1. All sources (and the peers) are requested in parallel, so a cold request takes
   about as long as the slowest source instead of the sum of all of them.
2. Responses go through an automatic cache (see data/cache.py) kept in the
   user's home folder. Repeat requests within the cache's max age are
   answered in milliseconds; older entries are fetched again.
3. If a source is down, the last fetched copy is used and a warning is
   attached, so the rest of the pipeline can still run.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import requests
from psxdata.exceptions import PSXDataError

from stockmind import config
from stockmind.data import company, market, stockanalysis
from stockmind.data.cache import ResponseCache
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


class DataCollectionAgent:
    """Fetch, validate, and cache PSX data for the supported companies."""

    name = "Data Collection Agent"

    def __init__(self, cache: ResponseCache | None = None) -> None:
        self._cache = cache or ResponseCache(config.CACHE_DIR)

    def collect(self, ticker: str, force_refresh: bool = False) -> CompanyData:
        """
        Return all data for one ticker.

        Args:
            ticker: A supported PSX symbol, e.g. "OGDC" (case-insensitive).
            force_refresh: Ignore cached responses and fetch everything live.

        Raises:
            UnsupportedTickerError: The ticker is not in config.SUPPORTED_TICKERS.
            DataUnavailableError: Price data could not be fetched.
        """
        ticker = resolve_ticker(ticker)

        # The company's own sources and its sector peers are fetched together.
        with ThreadPoolExecutor(max_workers=len(SOURCES) + 1) as pool:
            futures = {
                name: pool.submit(self._load, ticker, name, force_refresh)
                for name in SOURCES
            }
            peers_future = pool.submit(self._collect_peers, ticker, force_refresh)
            results = {name: future.result() for name, future in futures.items()}
            peers, peer_warnings = peers_future.result()

        warnings = [r["warning"] for r in results.values() if r["warning"]]
        warnings += peer_warnings
        fetched_at = {
            name: r["fetched_at"] for name, r in results.items() if r["fetched_at"]
        }

        if results["prices"]["value"] is None:
            raise DataUnavailableError(
                f"No price data available for {ticker}. " + " ".join(warnings)
            )

        prices = results["prices"]["value"]
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
            fetched_at=fetched_at,
            warnings=warnings,
        )

    def collect_many(
        self,
        tickers: list[str] | tuple[str, ...] = config.SUPPORTED_TICKERS,
        force_refresh: bool = False,
    ) -> dict[str, CompanyData]:
        """Collect several tickers, e.g. for dashboards or evaluation runs."""
        return {
            data.ticker: data
            for data in (self.collect(t, force_refresh) for t in tickers)
        }

    # ------------------------------------------------------------------

    def _collect_peers(
        self, ticker: str, force_refresh: bool
    ) -> tuple[SectorPeers, list[str]]:
        """Find the sector's largest other companies and fetch their statements."""
        listing = self._load(
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
                lambda job: self._load(job[0], job[1], force_refresh), jobs
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

    def _load(
        self,
        owner: str,
        source: str,
        force_refresh: bool,
        fetch=None,
        fallback=None,
    ) -> dict:
        """
        Load one source for one owner (a ticker, or MARKET) through the cache.

        Never raises for source errors: returns the fallback with a warning.
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
            )
        except SOURCE_ERRORS as exc:
            message = f"{owner}:{source} unavailable ({type(exc).__name__}: {exc})."
            logger.warning(message)
            return {"value": fallback, "fetched_at": None, "warning": message}

        if result.warning:
            logger.warning(result.warning)
        return {"value": result.value, "fetched_at": result.fetched_at,
                "warning": result.warning}
