"""Tests for the Data Collection Agent.

Every source is replaced with a fake function and the cache lives in a
temporary folder, so these tests run offline and never touch ~/.stockmind.
"""

from datetime import timedelta

import pandas as pd
import pytest
from psxdata.exceptions import PSXConnectionError

from stockmind import config
from stockmind.agents import data_collection
from stockmind.agents.data_collection import (
    DataCollectionAgent,
    DataUnavailableError,
)
from stockmind.data import market
from stockmind.data.cache import ResponseCache
from stockmind.data.tickers import UnsupportedTickerError, resolve_ticker


PRICES = pd.DataFrame(
    {"date": pd.to_datetime(["2026-10-29", "2026-10-30"]),
     "open": [10.0, 11.0], "high": [11.0, 12.0], "low": [9.0, 10.0],
     "close": [10.5, 11.5], "volume": [100, 200]}
)
LISTING = pd.DataFrame({
    "symbol": ["OGDC", "MARI", "PPL", "MEBL"],
    "name": ["OGDC", "Mari", "PPL", "Meezan"],
    "sector": ["E&P", "E&P", "E&P", "BANKS"],
    "market_cap_m": [1_300_000.0, 160_000.0, 150_000.0, 955_000.0],
    "price": [306.0, 631.0, 214.0, 528.0],
    "pe_ratio": [5.4, 8.1, 6.1, 9.9],
    "dividend_yield": [3.8, 4.0, 4.2, 5.5],
})
PAYOUTS = pd.DataFrame(
    {"announced_at": pd.to_datetime(["2026-09-08"]), "details": ["60% (F) (D)"],
     "cash_dividend_pct": [60.0]}
)


@pytest.fixture
def calls(monkeypatch):
    """Replace every source with a fake that counts how often it is called."""
    calls = {name: 0 for name in data_collection.SOURCES}

    def fake(name, value):
        def fetch(ticker):
            calls[name] += 1
            return value
        return fetch

    fakes = {
        "prices": {"prices": PRICES, "price_issues": PRICES.iloc[0:0]},
        "company_page": {"profile": {"name": "Test Co",
                                     "equity": {"face_value": 10.0}},
                         "financials": pd.DataFrame(),
                         "ratios": pd.DataFrame()},
        "payouts": PAYOUTS,
        "income_statement": pd.DataFrame({"metric": ["net_income"]}),
        "balance_sheet": pd.DataFrame({"metric": ["shareholders_equity"]}),
        "dividend_history": pd.DataFrame({"cash_per_share": [6.0]}),
    }
    for name, (_, fallback) in data_collection.SOURCES.items():
        monkeypatch.setitem(data_collection.SOURCES, name,
                            (fake(name, fakes[name]), fallback))

    # By default OGDC is alone in its sector, so no peers are fetched.
    monkeypatch.setattr(market, "fetch_market_listing", lambda: LISTING.iloc[:1])
    return calls


@pytest.fixture
def agent(tmp_path):
    return DataCollectionAgent(cache=ResponseCache(tmp_path))


def _break_source(monkeypatch, name):
    """Make one source fail as if PSX were unreachable."""
    def offline(ticker):
        raise PSXConnectionError("PSX unreachable")
    fallback = data_collection.SOURCES[name][1]
    monkeypatch.setitem(data_collection.SOURCES, name, (offline, fallback))


def test_collect_returns_every_dataset(calls, agent):
    data = agent.collect("ogdc")

    assert data.ticker == "OGDC"
    assert len(data.prices) == 2
    assert data.profile["name"] == "Test Co"
    assert not data.income_statement.empty
    assert not data.balance_sheet.empty
    assert data.dividend_history["cash_per_share"].item() == 6.0
    # 60% of a Rs 10 face value
    assert data.payouts["cash_dividend_per_share"].item() == 6.0
    assert set(data.fetched_at) == set(data_collection.SOURCES)
    assert data.warnings == []


def test_second_request_is_served_from_cache(calls, agent):
    agent.collect("OGDC")
    agent.collect("OGDC")

    assert all(count == 1 for count in calls.values())


def test_force_refresh_fetches_again(calls, agent):
    agent.collect("OGDC")
    agent.collect("OGDC", force_refresh=True)

    assert all(count == 2 for count in calls.values())


def test_expired_entries_are_fetched_again(calls, agent, monkeypatch):
    monkeypatch.setitem(config.CACHE_MAX_AGE, "prices", timedelta(0))

    agent.collect("OGDC")
    agent.collect("OGDC")

    assert calls["prices"] == 2
    assert calls["balance_sheet"] == 1


def test_source_failure_uses_previous_response(calls, agent, monkeypatch):
    agent.collect("OGDC")
    monkeypatch.setitem(config.CACHE_MAX_AGE, "prices", timedelta(0))
    _break_source(monkeypatch, "prices")

    data = agent.collect("OGDC")

    assert len(data.prices) == 2
    assert any("OGDC:prices: source unavailable" in w for w in data.warnings)


def test_optional_source_failure_returns_empty_with_warning(calls, agent, monkeypatch):
    _break_source(monkeypatch, "balance_sheet")

    data = agent.collect("OGDC")

    assert data.balance_sheet.empty
    assert len(data.prices) == 2
    assert any("balance_sheet unavailable" in w for w in data.warnings)


def test_missing_prices_raise(calls, agent, monkeypatch):
    _break_source(monkeypatch, "prices")

    with pytest.raises(DataUnavailableError):
        agent.collect("OGDC")


def test_sector_peers_are_collected(calls, agent, monkeypatch):
    monkeypatch.setattr(market, "fetch_market_listing", lambda: LISTING)

    data = agent.collect("OGDC")

    assert data.peers.sector == "E&P"
    assert data.peers.tickers == ["MARI", "PPL"]  # largest first, no banks
    assert list(data.peers.market["symbol"]) == ["MARI", "PPL"]
    assert set(data.peers.balance_sheet["ticker"]) == {"MARI", "PPL"}
    assert set(data.peers.income_statement["ticker"]) == {"MARI", "PPL"}
    # OGDC + 2 peers each need an income statement.
    assert calls["income_statement"] == 3


def test_peer_statements_share_the_cache(calls, agent, monkeypatch):
    monkeypatch.setattr(market, "fetch_market_listing", lambda: LISTING)

    agent.collect("OGDC")
    agent.collect("OGDC")

    assert calls["income_statement"] == 3  # nothing fetched the second time


def test_missing_market_listing_gives_no_peers_but_still_works(calls, agent, monkeypatch):
    def offline():
        raise PSXConnectionError("PSX unreachable")
    monkeypatch.setattr(market, "fetch_market_listing", offline)

    data = agent.collect("OGDC")

    assert data.peers.tickers == []
    assert len(data.prices) == 2
    assert any("market_listing unavailable" in w for w in data.warnings)


def test_collect_many(calls, agent):
    results = agent.collect_many(["OGDC", "mebl"])
    assert list(results) == ["OGDC", "MEBL"]


def test_unsupported_ticker_is_rejected_before_any_fetch(calls, agent):
    with pytest.raises(UnsupportedTickerError):
        agent.collect("AAPL")
    assert all(count == 0 for count in calls.values())


@pytest.mark.parametrize(
    "raw, expected",
    [("ogdc", "OGDC"), (" MEBL ", "MEBL"), ("HUBCXD", "HUBC"), ("FFCXB", "FFC")],
)
def test_resolve_ticker(raw, expected):
    assert resolve_ticker(raw) == expected
