"""Tests for the PSX price fetcher (offline)."""

import pandas as pd
import pytest

from stockmind.data import fetcher

PRICES = pd.DataFrame({
    "date": pd.to_datetime(["2026-11-24", "2026-11-23"]),
    "open": [10.0, 9.0], "high": [11.0, 10.0], "low": [9.5, 8.5],
    "close": [10.5, 9.5], "volume": [100, 200],
})


@pytest.fixture
def no_pause(monkeypatch):
    monkeypatch.setattr(fetcher.time, "sleep", lambda seconds: None)


def answers(monkeypatch, *responses):
    """Make psxdata.stocks return these tables in turn; record the calls."""
    calls = []

    def stocks(ticker, start=None, end=None, cache=True):
        calls.append(ticker)
        return responses[min(len(calls), len(responses)) - 1]

    monkeypatch.setattr(fetcher.psxdata, "stocks", stocks)
    return calls


def test_prices_are_returned_oldest_first(monkeypatch, no_pause):
    answers(monkeypatch, PRICES)
    prices = fetcher.fetch_historical_prices("ffc")
    assert prices["date"].is_monotonic_increasing


def test_an_empty_answer_is_retried(monkeypatch, no_pause):
    calls = answers(monkeypatch, PRICES.iloc[0:0], PRICES)
    assert len(fetcher.fetch_historical_prices("FFL")) == 2
    assert calls == ["FFL", "FFL"]


def test_gives_up_after_repeated_empty_answers(monkeypatch, no_pause):
    calls = answers(monkeypatch, PRICES.iloc[0:0])
    with pytest.raises(ValueError, match="No historical price data"):
        fetcher.fetch_historical_prices("HCAR")
    assert len(calls) == 1 + fetcher.EMPTY_RESPONSE_RETRIES


def test_a_date_filtered_empty_answer_is_not_retried(monkeypatch, no_pause):
    calls = answers(monkeypatch, PRICES.iloc[0:0])
    with pytest.raises(ValueError):
        fetcher.fetch_historical_prices("FFC", start="2026-11-22", end="2026-11-22")
    assert len(calls) == 1
