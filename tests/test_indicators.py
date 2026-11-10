"""Check our indicator calculations against the `ta` library.

This is the evidence for proposal criterion 6.2: indicator values must match
verified calculations.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from stockmind.analysis import indicators as ind

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import ta


@pytest.fixture
def close():
    """A reproducible random walk of 400 daily closes."""
    rng = np.random.default_rng(seed=42)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 400))))


def assert_matches(ours: pd.Series, reference: pd.Series):
    assert (ours.isna() == reference.isna()).all(), "warm-up periods differ"
    both = ours.notna()
    np.testing.assert_allclose(ours[both], reference[both], rtol=1e-9, atol=1e-9)


def test_sma(close):
    assert_matches(ind.sma(close, 50), ta.trend.SMAIndicator(close, 50).sma_indicator())


def test_ema(close):
    assert_matches(ind.ema(close, 20), ta.trend.EMAIndicator(close, 20).ema_indicator())


def test_rsi(close):
    assert_matches(ind.rsi(close, 14), ta.momentum.RSIIndicator(close, 14).rsi())


def test_macd(close):
    ours, reference = ind.macd(close), ta.trend.MACD(close)
    assert_matches(ours["macd"], reference.macd())
    assert_matches(ours["macd_signal"], reference.macd_signal())
    assert_matches(ours["macd_hist"], reference.macd_diff())


def test_bollinger_bands(close):
    ours, reference = ind.bollinger_bands(close), ta.volatility.BollingerBands(close)
    assert_matches(ours["bb_middle"], reference.bollinger_mavg())
    assert_matches(ours["bb_upper"], reference.bollinger_hband())
    assert_matches(ours["bb_lower"], reference.bollinger_lband())
    assert_matches(ours["bb_percent_b"], reference.bollinger_pband())


def test_rsi_edge_cases():
    rising = pd.Series(np.arange(1.0, 31.0))
    flat = pd.Series([50.0] * 30)

    assert ind.rsi(rising).iloc[-1] == 100.0   # only gains
    assert ind.rsi(flat).iloc[-1] == 50.0      # no movement
    assert ind.rsi(rising).iloc[:13].isna().all()  # warm-up is not guessed


def test_compute_indicators_columns_and_warm_up(close):
    prices = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=len(close)),
                           "close": close})
    table = ind.compute_indicators(prices)

    assert list(table.columns) == [
        "date", "close", "sma_short", "sma_long", "ema", "rsi", "macd",
        "macd_signal", "macd_hist", "bb_middle", "bb_upper", "bb_lower",
        "bb_percent_b",
    ]
    assert table["sma_long"].iloc[:199].isna().all()
    assert table["sma_long"].iloc[199:].notna().all()
