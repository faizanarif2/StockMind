"""Tests for the Stock Selection stage.

Prices are synthetic and every source is replaced with a fake, so these
tests run offline.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import pytest
from psxdata.exceptions import PSXConnectionError

from stockmind import config
from stockmind.agents import data_collection
from stockmind.agents.data_collection import DataCollectionAgent
from stockmind.agents.stock_selection import (
    StockSelectionAgent,
    exclusion_reason,
    price_metrics,
    rank_within_categories,
)
from stockmind.data import market, universe
from stockmind.data.cache import ResponseCache


END = pd.Timestamp("2026-11-20")
NOW = datetime(2026, 11, 20, 16, 0)


def make_prices(closes=None, days=300, volume=1_000_000, end=END) -> pd.DataFrame:
    """Daily prices on business days ending at `end` (flat at 100 by default)."""
    dates = pd.bdate_range(end=end, periods=days)
    closes = [100.0] * days if closes is None else closes
    return pd.DataFrame({
        "date": dates, "open": closes, "high": closes, "low": closes,
        "close": closes, "volume": [volume] * days,
    })


def rising(start: float, end: float, days: int = 300) -> pd.DataFrame:
    step = (end - start) / (days - 1)
    return make_prices([start + step * i for i in range(days)], days)


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def test_return_and_drawdown():
    closes = [100.0] * 300
    closes[-50], closes[-30], closes[-1] = 120.0, 90.0, 110.0
    metrics, _ = price_metrics(make_prices(closes))

    assert metrics["return_12m"] == pytest.approx(10.0)
    assert metrics["max_drawdown"] == pytest.approx(-25.0)  # 120 -> 90
    assert metrics["volatility"] > 0
    assert metrics["risk_adjusted_return"] == pytest.approx(
        10.0 / metrics["volatility"], abs=1e-3)


def test_liquidity_is_average_daily_traded_value_in_millions():
    metrics, _ = price_metrics(make_prices(volume=500_000))
    assert metrics["liquidity"] == pytest.approx(50.0)  # 100 x 500k = PKR 50m


def test_bonus_issue_is_not_counted_as_a_loss():
    # 10% bonus: the price drops from 110 to 100 on the ex-date.
    prices = make_prices([110.0] * 280 + [100.0] * 20)
    ex_date = prices["date"].iloc[-20]
    payouts = pd.DataFrame({
        "details": ["10% (B)"], "bonus_pct": [10.0], "right_pct": [None],
        "book_closure_start": [prices["date"].iloc[-19]],
    })

    unadjusted, _ = price_metrics(prices)
    adjusted, _ = price_metrics(prices, payouts)

    assert ex_date < prices["date"].iloc[-19]
    assert unadjusted["return_12m"] == pytest.approx(-9.09, abs=0.01)
    assert adjusted["return_12m"] == pytest.approx(0.0)
    assert adjusted["max_drawdown"] == pytest.approx(0.0)


def test_eligible_company_has_no_exclusion_reason():
    metrics, _ = price_metrics(make_prices())
    assert exclusion_reason(metrics, NOW) is None


def test_short_history_is_excluded():
    metrics, _ = price_metrics(make_prices(days=150))
    assert metrics["return_12m"] is None
    assert "less than the full lookback" in exclusion_reason(metrics, NOW)


def test_suspended_company_is_excluded():
    metrics, _ = price_metrics(make_prices())
    reason = exclusion_reason(metrics, NOW + timedelta(days=30))
    assert reason == "no trading since 2026-11-20"


def test_illiquid_company_is_excluded():
    metrics, _ = price_metrics(make_prices(volume=10_000))  # PKR 1m a day
    assert "below the minimum" in exclusion_reason(metrics, NOW)


def test_no_prices_is_excluded():
    metrics, _ = price_metrics(pd.DataFrame())
    assert exclusion_reason(metrics, NOW) == "no price data available"


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def company(symbol, category, value, eligible=True, weight=1.0, **overrides):
    """A row in which every measure equals `value`, unless overridden."""
    row = {"symbol": symbol, "category": category, "eligible": eligible,
           "idx_weight": weight}
    row.update({measure: value for measure in config.SELECTION_WEIGHTS})
    row.update(overrides)
    return row


def test_best_company_in_each_category_is_selected():
    table = rank_within_categories(pd.DataFrame([
        company("WEAK", "Energy", 1.0),
        company("STRONG", "Energy", 5.0),
        company("ALONE", "Financials", -3.0),  # the only one: selected anyway
    ])).set_index("symbol")

    assert table.loc["STRONG", "selected"]
    assert not table.loc["WEAK", "selected"]
    assert table.loc["STRONG", "score"] == 100.0
    assert table.loc["WEAK", "score"] == 50.0
    assert table.loc["ALONE", "selected"]
    assert table.loc["ALONE", "category_rank"] == 1


def test_companies_are_only_compared_within_their_category():
    table = rank_within_categories(pd.DataFrame([
        company("BANK1", "Financials", 1.0, pe_ratio=5),
        company("BANK2", "Financials", 2.0),
        company("TECH1", "Technology", 50.0),
        company("TECH2", "Technology", 60.0),
    ])).set_index("symbol")

    # BANK2 is far below the tech companies in absolute terms, but it is
    # the best bank.
    assert table.loc["BANK2", "selected"]
    assert table.loc["TECH2", "selected"]
    assert table.loc["BANK2", "score"] == table.loc["TECH2", "score"]


def test_a_missing_measure_ranks_last():
    table = rank_within_categories(pd.DataFrame([
        company("NOPE", "Energy", 1.0, earnings_yield=None),
        company("PE", "Energy", 1.0, earnings_yield=5.0),
    ])).set_index("symbol")

    assert table.loc["NOPE", "rank_earnings_yield"] == 0.0
    assert table.loc["PE", "rank_earnings_yield"] == 1.0
    assert table.loc["PE", "selected"]


def test_ineligible_companies_are_not_ranked():
    table = rank_within_categories(pd.DataFrame([
        company("GOOD", "Energy", 1.0),
        company("BAD", "Energy", 99.0, eligible=False),
    ])).set_index("symbol")

    assert pd.isna(table.loc["BAD", "score"])
    assert pd.isna(table.loc["BAD", "category_rank"])
    assert not table.loc["BAD", "selected"]
    assert table.loc["GOOD", "selected"]


def test_large_categories_get_more_picks():
    rows = [company(f"E{i}", "Energy", float(i)) for i in range(6)]
    rows += [company(f"M{i}", "Materials", float(i)) for i in range(5)]
    table = rank_within_categories(pd.DataFrame(rows))

    selected = table[table["selected"]]
    assert sorted(selected["symbol"]) == ["E4", "E5", "M4"]


# ---------------------------------------------------------------------------
# Agent, end to end with fake sources
# ---------------------------------------------------------------------------

INDEX = pd.DataFrame({
    "symbol": ["OGDC", "PPLXD", "MEBL", "NEWCO"],
    "name": ["OGDC", "PPL", "Meezan", "New Co"],
    "idx_weight": [8.1, 6.0, 9.9, 0.5],
})
LISTING = pd.DataFrame({
    "symbol": ["OGDC", "PPL", "MEBL", "NEWCO"],
    "name": ["OGDC", "PPL", "Meezan", "New Co"],
    "sector": ["OIL & GAS EXPLORATION COMPANIES", "OIL & GAS EXPLORATION COMPANIES",
               "COMMERCIAL BANKS", "SPACE MINING"],
    "market_cap_m": [1.0, 1.0, 1.0, 1.0],
    "price": [120.0, 105.0, 100.0, 100.0],
    "pe_ratio": [5.0, 6.0, 10.0, float("nan")],
    "dividend_yield": [4.5, 5.3, 5.5, float("nan")],
})
PRICES = {
    "OGDC": rising(100, 120),
    "PPL": rising(100, 105),
    "MEBL": make_prices(days=100),   # too short: excluded
    "NEWCO": rising(100, 101),
}


@pytest.fixture
def fake_sources(monkeypatch):
    calls = {"prices": 0, "index": 0}

    def fake_prices(symbol):
        calls["prices"] += 1
        return {"prices": PRICES[symbol], "price_issues": PRICES[symbol].iloc[0:0]}

    def fake_index(index):
        calls["index"] += 1
        return INDEX

    monkeypatch.setitem(data_collection.SOURCES, "prices",
                        (fake_prices, None))
    monkeypatch.setitem(data_collection.SOURCES, "payouts",
                        (lambda symbol: pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(universe, "fetch_index_table", fake_index)
    monkeypatch.setattr(market, "fetch_market_listing", lambda: LISTING)
    return calls


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path)


def test_agent_selects_from_the_live_index(fake_sources, cache):
    selection = StockSelectionAgent(cache=cache).select(now=NOW)

    assert selection.universe == ["MEBL", "OGDC", "PPL", "NEWCO"]  # by weight
    assert selection.selected == ["OGDC", "NEWCO"]
    assert selection.by_category == {"Energy": ["OGDC"], "Space Mining": ["NEWCO"]}
    assert selection.as_of == END
    assert selection.reasons["OGDC"].startswith(
        "OGDC ranks 1 of 2 eligible companies in Energy")
    assert "only eligible company in Space Mining" in selection.reasons["NEWCO"]


def test_agent_explains_exclusions_and_new_sectors(fake_sources, cache):
    selection = StockSelectionAgent(cache=cache).select(now=NOW)
    ranking = selection.ranking.set_index("symbol")

    assert not ranking.loc["MEBL", "eligible"]
    assert "less than the full lookback" in ranking.loc["MEBL", "excluded_because"]
    assert any("NEWCO is in PSX sector 'SPACE MINING'" in w for w in selection.warnings)
    assert any("No company in Financials is eligible" in w for w in selection.warnings)


def test_selection_and_agent_1_share_the_cache(fake_sources, cache):
    StockSelectionAgent(cache=cache).select(now=NOW)
    StockSelectionAgent(cache=cache).select(now=NOW)
    assert fake_sources == {"prices": 4, "index": 1}  # second run: all cached

    agent = DataCollectionAgent(cache=cache, universe=["OGDC", "PPL"])
    data = agent.load_source("OGDC", "prices", False)
    assert len(data["value"]["prices"]) == 300
    assert fake_sources["prices"] == 4  # Agent 1 reused the selection's prices


def test_missing_index_raises(fake_sources, cache, monkeypatch):
    def offline(index):
        raise ValueError("PSX returned no constituents")
    monkeypatch.setattr(universe, "fetch_index_table", offline)

    with pytest.raises(data_collection.DataUnavailableError):
        StockSelectionAgent(cache=cache).select(now=NOW)


# ---------------------------------------------------------------------------
# One selection per session: StockSelectionAgent.current()
# ---------------------------------------------------------------------------

class FakeClock:
    """`selection_fresh`: no session has ended since the selection was made.
    `data_fresh`: market data in the cache is still current."""

    def __init__(self):
        self.selection_fresh = True
        self.data_fresh = True

    def is_fresh(self, fetched_at, interval=config.SESSION_REFRESH):
        return self.selection_fresh if interval is None else self.data_fresh

    def day_is_final(self, day, moment=None):
        return True

    def in_window(self, moment=None):
        return False


@pytest.fixture
def stage0(fake_sources, cache):
    """A selection agent that counts how often the ranking is recalculated."""
    agent = StockSelectionAgent(cache=cache, clock=FakeClock())
    agent.calculations = 0
    rank = agent.select

    def counted(*args, **kwargs):
        agent.calculations += 1
        return rank(*args, **kwargs)
    agent.select = counted
    return agent


def _psx_down(monkeypatch):
    def offline(*args):
        raise PSXConnectionError("PSX unreachable")
    monkeypatch.setitem(data_collection.SOURCES, "prices", (offline, None))
    monkeypatch.setitem(data_collection.SOURCES, "payouts", (offline, pd.DataFrame()))
    monkeypatch.setattr(universe, "fetch_index_table", offline)
    monkeypatch.setattr(market, "fetch_market_listing", offline)


def test_selection_itself_is_kept_during_a_session(stage0):
    first = stage0.current(now=NOW)
    second = stage0.current(now=NOW)

    assert second.selected == first.selected == ["OGDC", "NEWCO"]
    assert stage0.calculations == 1


def test_selection_is_recalculated_after_a_session_ends(stage0):
    stage0.current(now=NOW)
    stage0._clock.selection_fresh = False

    stage0.current(now=NOW)

    assert stage0.calculations == 2


def test_force_recalculates_the_selection(stage0):
    stage0.current(now=NOW)
    stage0.current(force_refresh=True, now=NOW)

    assert stage0.calculations == 2


def test_last_selection_is_reused_while_psx_is_down(stage0, monkeypatch):
    first = stage0.current(now=NOW)
    stage0._clock.selection_fresh = stage0._clock.data_fresh = False
    _psx_down(monkeypatch)

    reused = stage0.current(now=NOW)

    assert reused.selected == first.selected
    assert "KMI30:selection: source unavailable" in reused.warnings[0]
    assert stage0.calculations == 1  # stopped before the per-company requests


def test_outage_cannot_overwrite_the_last_good_selection(stage0, monkeypatch):
    first = stage0.current(now=NOW)
    stage0._clock.selection_fresh = False
    monkeypatch.setitem(config.CACHE_MAX_AGE, "prices", timedelta(0))

    def offline(symbol):
        raise PSXConnectionError("PSX unreachable")
    monkeypatch.setitem(data_collection.SOURCES, "prices", (offline, None))

    reused = stage0.current(now=NOW)

    assert reused.selected == first.selected
    assert "4 PSX data sources were unavailable" in reused.warnings[0]


def test_one_failed_company_does_not_block_a_new_selection(stage0, monkeypatch):
    stage0.current(now=NOW)
    stage0._clock.selection_fresh = False
    monkeypatch.setitem(config.CACHE_MAX_AGE, "prices", timedelta(0))
    working = data_collection.SOURCES["prices"][0]

    def one_down(symbol):
        if symbol == "PPL":
            raise PSXConnectionError("PSX unreachable")
        return working(symbol)
    monkeypatch.setitem(data_collection.SOURCES, "prices", (one_down, None))

    selection = stage0.current(now=NOW)

    assert stage0.calculations == 2
    assert not any("KMI30:selection" in warning for warning in selection.warnings)


def test_no_previous_selection_is_a_clear_error(stage0, monkeypatch):
    _psx_down(monkeypatch)

    with pytest.raises(data_collection.DataUnavailableError,
                       match="no previous selection to reuse"):
        stage0.current(now=NOW)


def test_an_empty_selection_is_never_saved(stage0, monkeypatch):
    for symbol in PRICES:
        monkeypatch.setitem(PRICES, symbol, make_prices(days=100))
    with pytest.raises(data_collection.DataUnavailableError,
                       match="no company passed the eligibility checks"):
        stage0.current(now=NOW)



def test_routine_library_warnings_are_hidden():
    assert not logging.getLogger("psxdata.utils").isEnabledFor(logging.WARNING)
    assert logging.getLogger("psxdata.utils").isEnabledFor(logging.ERROR)
