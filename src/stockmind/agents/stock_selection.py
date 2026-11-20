"""Stage 0 - Stock Selection Agent.

Decides WHICH companies the pipeline analyses. It runs before Agent 1 and
picks the most promising companies in each category of the KMI-30, using
transparent, market-based criteria. Whether to buy them is still decided
later by Agents 2 and 3 and the Decision Agent, so picking "the best of a
weak category" is fine: the later agents can still say HOLD or SELL.

    KMI-30 (live) -> categories -> eligibility -> rank in category -> top picks
                                                                       |
                                     Agent 1 collects data for them <--+

How companies are chosen
------------------------
1. Universe. Every current KMI-30 constituent, fetched live from PSX, so
   index rebalances need no code changes (see data/universe.py).

2. Categories. Each company's PSX sector, read live from PSX, is mapped to
   a broad category (config.SECTOR_CATEGORIES, GICS-style). The table
   covers all PSX sectors, so new constituents are classified automatically.

3. Eligibility. A company is only ranked if it has:
     - prices covering the full lookback (config.SELECTION_LOOKBACK) and at
       least config.SELECTION_MIN_TRADING_DAYS trading days in it
       (Agent 2's 200-day SMA needs this much history);
     - traded recently (latest price within config.SELECTION_MAX_PRICE_AGE),
       so suspended companies are left out;
     - an average daily traded value of at least
       config.SELECTION_MIN_TRADED_VALUE_M (PKR millions) over the last
       config.SELECTION_LIQUIDITY_DAYS trading days.

4. Measures, all "higher is better":
     return_12m            % price change over the lookback, with prices
                           before a bonus issue adjusted (a bonus is not a loss)
     risk_adjusted_return  return_12m / annualised volatility of daily returns
     max_drawdown          worst fall from a previous peak, % (e.g. -18 is
                           better than -35)
     earnings_yield        100 / P/E, % (PSX P/E). No P/E (a loss, or not
                           reported) ranks last.
     dividend_yield        PSX trailing dividend yield, %. Missing ranks last.
     liquidity             average daily traded value, PKR millions

5. Ranking. Within its category, each measure becomes a percentile rank
   (0-1, the best company gets 1). Ranks are used instead of raw values so
   that measures in different units can be added up, one extreme value
   cannot dominate, and sectors with different norms (e.g. banks' low P/E)
   are only compared with themselves.

       selection score = 100 x sum(weight x rank) / sum(weights)

   with the weights in config.SELECTION_WEIGHTS.

6. Picks. The top ceil(eligible / config.CATEGORY_SIZE_PER_PICK) companies
   of each category are selected (1 for a category of 1-5, 2 for 6-10 ...).

Only market data is used (prices, P/E, dividend yield). Profitability and
balance-sheet strength are left to Agent 3, so the selection does not
pre-judge what the Fundamental Agent will conclude.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from stockmind import config
from stockmind.agents.data_collection import (
    MARKET,
    DataCollectionAgent,
    DataUnavailableError,
)
from stockmind.analysis.price_adjustments import adjust_for_bonus
from stockmind.data import market, universe
from stockmind.data.cache import ResponseCache


TRADING_DAYS_PER_YEAR = 252

MEASURE_LABELS: dict[str, str] = {
    "return_12m": "12-month return",
    "risk_adjusted_return": "risk-adjusted return",
    "max_drawdown": "maximum drawdown",
    "earnings_yield": "earnings yield",
    "dividend_yield": "dividend yield",
    "liquidity": "liquidity",
}

PRICE_METRICS = (
    "first_date", "last_date", "trading_days", "return_12m", "volatility",
    "risk_adjusted_return", "max_drawdown", "liquidity",
)

RANKING_COLUMNS = [
    "category", "category_rank", "symbol", "name", "sector", "idx_weight",
    "selected", "eligible", "excluded_because", "score",
    "return_12m", "volatility", "risk_adjusted_return", "max_drawdown",
    "earnings_yield", "pe_ratio", "dividend_yield", "liquidity",
    "trading_days", "last_date",
] + [f"rank_{measure}" for measure in MEASURE_LABELS]


@dataclass
class StockSelection:
    """The Stock Selection Agent's result."""

    index: str                         # e.g. "KMI30"
    selected: list[str]                # tickers to pass to Agent 1
    by_category: dict[str, list[str]]  # category -> selected, best first
    reasons: dict[str, str]            # ticker -> why it was selected
    ranking: pd.DataFrame              # every constituent, see RANKING_COLUMNS
    universe: list[str]                # every constituent (Agent 1's universe)
    as_of: pd.Timestamp | None         # latest price date used
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        groups = "; ".join(f"{category}: {', '.join(tickers)}"
                           for category, tickers in self.by_category.items())
        as_of = f" (prices to {self.as_of:%Y-%m-%d})" if self.as_of is not None else ""
        return (f"Selected {len(self.selected)} of {len(self.universe)} "
                f"{self.index} companies across {len(self.by_category)} "
                f"categories{as_of}. {groups}.")


# ---------------------------------------------------------------------------
# Measures (pure functions, testable offline)
# ---------------------------------------------------------------------------

def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def price_metrics(
    prices: pd.DataFrame, payouts: pd.DataFrame | None = None
) -> tuple[dict, list[str]]:
    """
    Performance, risk, and liquidity measures from daily prices.

    Returns (metrics, warnings). Metrics that need a full lookback period of
    history are None when the history is shorter.
    """
    metrics = dict.fromkeys(PRICE_METRICS)
    metrics["trading_days"] = 0
    if prices is None or prices.empty:
        return metrics, []

    prices = prices.sort_values("date").reset_index(drop=True)
    warnings: list[str] = []
    adjusted = prices
    if payouts is not None and not payouts.empty:
        adjusted, _, warnings = adjust_for_bonus(prices, payouts)

    last_date = adjusted["date"].iloc[-1]
    start = last_date - config.SELECTION_LOOKBACK
    window = adjusted[adjusted["date"] > start]
    # Traded value uses the prices and volumes as they actually traded.
    recent = prices.tail(config.SELECTION_LIQUIDITY_DAYS)

    metrics.update(
        first_date=prices["date"].iloc[0],
        last_date=last_date,
        trading_days=len(window),
        liquidity=_number((recent["close"] * recent["volume"]).mean() / 1e6),
    )

    before = adjusted[adjusted["date"] <= start]
    if before.empty:
        return metrics, warnings  # less than a full lookback of history

    # The last close on or before the start of the period is the base.
    path = pd.concat([before.tail(1), window])["close"].astype(float)
    return_pct = (path.iloc[-1] / path.iloc[0] - 1) * 100
    daily = path.pct_change().dropna()
    volatility = (daily.std() * math.sqrt(TRADING_DAYS_PER_YEAR) * 100
                  if len(daily) > 1 else None)
    drawdown = ((path / path.cummax()) - 1).min() * 100

    metrics.update(
        return_12m=round(return_pct, 2),
        volatility=_number(round(volatility, 2)) if volatility is not None else None,
        risk_adjusted_return=(round(return_pct / volatility, 3)
                              if volatility else None),
        max_drawdown=round(drawdown, 2),
    )
    return metrics, warnings


def exclusion_reason(metrics: dict, now: datetime) -> str | None:
    """Why a company cannot be ranked, or None if it is eligible."""
    if metrics["last_date"] is None:
        return "no price data available"
    age = pd.Timestamp(now) - pd.Timestamp(metrics["last_date"])
    if age > config.SELECTION_MAX_PRICE_AGE:
        return f"no trading since {metrics['last_date']:%Y-%m-%d}"
    if metrics["return_12m"] is None:
        return (f"price history starts {metrics['first_date']:%Y-%m-%d}, "
                "less than the full lookback period")
    if metrics["trading_days"] < config.SELECTION_MIN_TRADING_DAYS:
        return (f"only {metrics['trading_days']} trading days in the lookback "
                f"period (minimum {config.SELECTION_MIN_TRADING_DAYS})")
    liquidity = metrics["liquidity"]
    if liquidity is None or liquidity < config.SELECTION_MIN_TRADED_VALUE_M:
        shown = "unknown" if liquidity is None else f"PKR {liquidity:,.1f}m"
        return (f"average traded value {shown} a day is below the "
                f"minimum of PKR {config.SELECTION_MIN_TRADED_VALUE_M:g}m")
    return None


def rank_within_categories(
    table: pd.DataFrame, weights: dict[str, float] | None = None
) -> pd.DataFrame:
    """
    Add percentile ranks, the selection score, the rank within the category
    and the selected flag to a table with one row per company.

    Needs the columns: symbol, category, eligible, idx_weight and one column
    per measure in `weights`. Ineligible companies get no rank or score.
    """
    weights = weights or config.SELECTION_WEIGHTS
    table = table.copy()
    eligible = table["eligible"].astype(bool)
    total_weight = sum(weights.values())

    score = pd.Series(0.0, index=table.index)
    for measure, weight in weights.items():
        values = pd.to_numeric(table[measure], errors="coerce")
        ranks = (values[eligible]
                 .groupby(table.loc[eligible, "category"])
                 .rank(pct=True, method="average")
                 .fillna(0.0))  # a missing measure ranks last
        table[f"rank_{measure}"] = ranks.round(3)
        score = score.add(ranks * weight, fill_value=0.0)
    table["score"] = (100 * score / total_weight).round(1).where(eligible)

    table = table.sort_values(
        ["category", "eligible", "score", "idx_weight"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    eligible = table["eligible"].astype(bool)
    table["category_rank"] = (
        table[eligible].groupby("category").cumcount() + 1
    ).astype("Int64")

    counts = table[eligible].groupby("category").size()
    picks = (counts / config.CATEGORY_SIZE_PER_PICK).apply(math.ceil)
    table["selected"] = eligible & (
        table["category_rank"] <= table["category"].map(picks)
    ).fillna(False).astype(bool)
    return table


def _fmt(value, pattern: str) -> str:
    number = _number(value)
    return "n/a" if number is None else pattern.format(number)


def selection_reason(row: pd.Series, eligible_in_category: int) -> str:
    """One paragraph explaining a selected company's rank."""
    position = (
        f"is the only eligible company in {row['category']}"
        if eligible_in_category == 1 else
        f"ranks {row['category_rank']} of {eligible_in_category} eligible "
        f"companies in {row['category']}"
    )
    text = (
        f"{row['symbol']} {position} (selection score "
        f"{row['score']:.0f}/100). 12-month return "
        f"{_fmt(row['return_12m'], '{:+.1f}%')}, risk-adjusted return "
        f"{_fmt(row['risk_adjusted_return'], '{:.2f}')}, maximum drawdown "
        f"{_fmt(row['max_drawdown'], '{:.1f}%')}, earnings yield "
        f"{_fmt(row['earnings_yield'], '{:.1f}%')} (P/E "
        f"{_fmt(row['pe_ratio'], '{:.1f}')}), dividend yield "
        f"{_fmt(row['dividend_yield'], '{:.1f}%')}, average traded value "
        f"{_fmt(row['liquidity'], 'PKR {:,.0f}m')} a day."
    )
    if eligible_in_category == 1:
        return text
    best =[label for measure, label in MEASURE_LABELS.items()
            if row.get(f"rank_{measure}") == 1.0]
    if best:
        text += " Best in its category on " + ", ".join(best) + "."
    return text


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class StockSelectionAgent:
    """Choose the most promising companies in each category of an index."""

    name = "Stock Selection Agent"

    def __init__(
        self,
        cache: ResponseCache | None = None,
        index: str = config.SELECTION_INDEX,
    ) -> None:
        self.index = index
        # Agent 1's loader gives the same caching and stale-if-error
        # behaviour, and the same cache entries: prices fetched here are
        # reused when Agent 1 collects the selected companies.
        self._loader = DataCollectionAgent(cache=cache)

    def select(
        self, force_refresh: bool = False, now: datetime | None = None
    ) -> StockSelection:
        """
        Rank every index constituent within its category and select the best.

        Raises:
            DataUnavailableError: The index constituents or the PSX market
                listing could not be fetched and nothing is cached.
        """
        now = now or datetime.now()
        constituents, warnings = self._constituents(force_refresh)
        symbols = constituents["symbol"].tolist()

        jobs = [(symbol, source) for symbol in symbols
                for source in ("prices", "payouts")]
        with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL_REQUESTS) as pool:
            loaded = dict(zip(jobs, pool.map(
                lambda job: self._loader.load_source(job[0], job[1], force_refresh),
                jobs,
            )))

        rows = []
        for company in constituents.to_dict("records"):
            symbol = company["symbol"]
            prices_result = loaded[(symbol, "prices")]
            payouts_result = loaded[(symbol, "payouts")]
            warnings += [r["warning"] for r in (prices_result, payouts_result)
                         if r["warning"]]

            prices = (prices_result["value"]["prices"]
                      if prices_result["value"] is not None else pd.DataFrame())
            metrics, price_warnings = price_metrics(prices, payouts_result["value"])
            warnings += [f"{symbol}: {warning}" for warning in price_warnings]

            pe_ratio = _number(company["pe_ratio"])
            reason = exclusion_reason(metrics, now)
            rows.append({
                **company,
                **metrics,
                "earnings_yield": (round(100 / pe_ratio, 2)
                                   if pe_ratio and pe_ratio > 0 else None),
                "eligible": reason is None,
                "excluded_because": reason,
            })

        ranking = rank_within_categories(pd.DataFrame(rows))[RANKING_COLUMNS]
        return self._result(ranking, symbols, warnings)

    # ------------------------------------------------------------------

    def _constituents(self, force_refresh: bool) -> tuple[pd.DataFrame, list[str]]:
        index_result = self._loader.load_source(
            self.index, "index_constituents", force_refresh,
            fetch=universe.fetch_index_table, fallback=pd.DataFrame(),
        )
        listing_result = self._loader.load_source(
            MARKET, "market_listing", force_refresh,
            fetch=lambda _: market.fetch_market_listing(),
            fallback=pd.DataFrame(),
        )
        warnings = [r["warning"] for r in (index_result, listing_result)
                    if r["warning"]]

        if index_result["value"].empty:
            raise DataUnavailableError(
                f"The {self.index} constituents could not be fetched. "
                + " ".join(warnings))
        if listing_result["value"].empty:
            raise DataUnavailableError(
                "The PSX market listing (sectors, P/E, dividend yield) could "
                "not be fetched. " + " ".join(warnings))

        constituents, category_warnings = universe.build_constituents(
            index_result["value"], listing_result["value"])
        return constituents, warnings + category_warnings

    def _result(
        self, ranking: pd.DataFrame, symbols: list[str], warnings: list[str]
    ) -> StockSelection:
        chosen = ranking[ranking["selected"]]
        eligible_counts = ranking[ranking["eligible"]].groupby("category").size()

        by_category = {
            category: group["symbol"].tolist()
            for category, group in chosen.groupby("category", sort=True)
        }
        reasons = {
            row["symbol"]: selection_reason(row, int(eligible_counts[row["category"]]))
            for _, row in chosen.iterrows()
        }
        for category in sorted(set(ranking["category"]) - set(by_category)):
            warnings.append(f"No company in {category} is eligible, "
                            "so nothing is selected from it.")

        last_dates = ranking.loc[ranking["eligible"], "last_date"].dropna()
        return StockSelection(
            index=self.index,
            selected=chosen["symbol"].tolist(),
            by_category=by_category,
            reasons=reasons,
            ranking=ranking,
            universe=symbols,
            as_of=last_dates.max() if not last_dates.empty else None,
            warnings=warnings,
        )
