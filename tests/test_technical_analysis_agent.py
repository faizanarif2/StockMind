"""Tests for the Technical Analysis Agent, using synthetic prices (offline)."""

import numpy as np
import pandas as pd
import pytest

from stockmind.agents.data_collection import CompanyData, SectorPeers
from stockmind.agents.technical_analysis import (
    BUY,
    HOLD,
    SELL,
    SIGNAL_COLUMNS,
    TechnicalAnalysisAgent,
    oscillator_score,
    signal_confidence,
    trend_score,
)


def make_prices(daily_return: float, days: int = 300, noise: float = 0.01,
                end: str | None = None) -> pd.DataFrame:
    """Prices growing by `daily_return` a day with reproducible noise."""
    rng = np.random.default_rng(seed=7)
    close = 100 * np.cumprod(1 + daily_return + rng.normal(0, noise, days))
    dates = pd.bdate_range(end=end or pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame({"date": dates, "open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close, "volume": 1000})


def company(prices: pd.DataFrame, payouts: pd.DataFrame | None = None) -> CompanyData:
    empty = pd.DataFrame()
    return CompanyData(
        ticker="TEST", prices=prices, price_issues=empty, profile={},
        financials=empty, ratios=empty, income_statement=empty,
        balance_sheet=empty, payouts=empty if payouts is None else payouts,
        dividend_history=empty, peers=SectorPeers.empty(),
    )


@pytest.fixture
def agent():
    return TechnicalAnalysisAgent()


# --- scoring rules ---------------------------------------------------------

def test_trend_score_scales_and_caps():
    assert trend_score(2.5, 5.0) == 0.5
    assert trend_score(-10.0, 5.0) == -1.0


@pytest.mark.parametrize("rsi, expected", [
    (20, 1.0), (30, 1.0), (35, 0.5), (50, 0.0), (65, -0.5), (70, -1.0), (90, -1.0),
])
def test_oscillator_score_matches_rsi_meaning(rsi, expected):
    # Oversold (low RSI) is bullish, overbought (high RSI) bearish.
    assert oscillator_score(rsi, (30, 40, 60, 70)) == pytest.approx(expected)


def test_confidence_formula():
    # Clear BUY, all data, everything agrees -> 100%.
    assert signal_confidence(0.6, coverage=1.0, agreement=1.0) == (BUY, 100.0)
    # Score exactly 0 is the clearest HOLD.
    assert signal_confidence(0.0, coverage=1.0, agreement=1.0) == (HOLD, 100.0)
    # Half the data missing halves confidence.
    assert signal_confidence(-0.6, coverage=0.5, agreement=1.0) == (SELL, 50.0)


# --- whole agent -----------------------------------------------------------

def test_steady_uptrend_is_bullish(agent):
    result = agent.analyse(company(make_prices(0.004)))

    assert result.signal == BUY
    assert result.trend == "uptrend"
    votes = result.signals.set_index("signal")["vote"]
    assert votes["long_term_trend"] == BUY
    assert votes["price_vs_sma"] == BUY
    assert 50 < result.technical_score <= 100


def test_steady_downtrend_trend_signals_are_bearish(agent):
    result = agent.analyse(company(make_prices(-0.004)))

    votes = result.signals.set_index("signal")["vote"]
    assert result.trend == "downtrend"
    assert votes[["long_term_trend", "price_vs_sma", "price_vs_ema"]].eq(SELL).all()
    assert result.score < 0
    assert result.technical_score < 50


def test_oversold_downtrend_gives_low_confidence(agent):
    # After an 80% fall the stock is oversold: RSI and Bollinger expect a
    # rebound while the trend says sell. The disagreement must lower the
    # confidence rather than give a confident SELL.
    result = agent.analyse(company(make_prices(-0.004)))

    votes = result.signals.set_index("signal")["vote"]
    assert votes["rsi"] == BUY
    assert result.signal == HOLD
    assert result.confidence < 30


def test_result_has_the_same_shape_as_agent_3(agent):
    result = agent.analyse(company(make_prices(0.001)))

    assert list(result.signals.columns) == SIGNAL_COLUMNS
    assert len(result.explanation) == 6
    assert 0 <= result.confidence <= 100
    assert -1 <= result.score <= 1
    assert result.technical_score == pytest.approx(50 * (result.score + 1), abs=0.1)
    assert result.summary.startswith("TEST's price action looks")
    assert len(result.indicators) == 300  # full series for charts


def test_short_history_skips_long_term_trend(agent):
    result = agent.analyse(company(make_prices(0.004, days=60)))

    trend = result.signals.set_index("signal").loc["long_term_trend"]
    assert pd.isna(trend["score"])
    assert "could not be calculated" in trend["comment"]
    assert result.trend == "unknown"
    assert any("200-day" in w for w in result.warnings)


def test_no_prices_gives_hold_with_zero_confidence(agent):
    result = agent.analyse(company(pd.DataFrame()))

    assert (result.signal, result.confidence) == (HOLD, 0.0)
    assert result.warnings


def test_stale_prices_are_warned_about(agent):
    result = agent.analyse(company(make_prices(0.001, end="2024-01-31")))
    assert any("out of date" in w for w in result.warnings)


def test_bonus_issue_is_not_read_as_a_crash(agent):
    # Flat price of 110, then a 10% bonus: price drops to 100 on the ex-date.
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=260)
    close = np.where(np.arange(260) < 250, 110.0, 100.0)
    prices = pd.DataFrame({"date": dates, "open": close, "high": close,
                           "low": close, "close": close, "volume": 1000})
    # Ex-date is day 250 (first day at 100), so book closure starts the
    # next trading day.
    payouts = pd.DataFrame({
        "details": ["10% (B)"], "bonus_pct": [10.0], "right_pct": [None],
        "book_closure_start": [dates[251]],
    })

    unadjusted = agent.analyse(company(prices))
    adjusted = agent.analyse(company(prices, payouts))

    assert unadjusted.signals.set_index("signal").loc["price_vs_sma", "vote"] == SELL
    assert adjusted.signals.set_index("signal").loc["price_vs_sma", "vote"] == HOLD
    assert adjusted.adjustments
