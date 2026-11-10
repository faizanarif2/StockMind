"""Technical indicators calculated from daily closing prices.

All functions take a pandas Series of closes (oldest first) and return
Series aligned with it. Values are NaN until enough history exists
(e.g. the first 199 days of a 200-day average), never guessed.

The formulas are the standard textbook definitions, the same ones used by
TA-Lib and the `ta` library; tests/test_indicators.py checks our values
against `ta`.

    SMA   simple moving average: mean of the last n closes
    EMA   exponential moving average: recent closes weigh more,
          alpha = 2 / (n + 1)
    RSI   relative strength index (Wilder): 100 - 100 / (1 + avg gain / avg loss),
          gains and losses smoothed with alpha = 1 / n
    MACD  EMA(12) - EMA(26); signal line = EMA(9) of MACD;
          histogram = MACD - signal
    Bollinger Bands  SMA(20) +/- 2 standard deviations;
          %B = where the close sits between the bands (0 = lower, 1 = upper)
"""

import numpy as np
import pandas as pd

from stockmind import config


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, min_periods=window, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    # The first day has no previous close; count it as no change, as TA-Lib
    # and `ta` do, so the smoothing starts from the same point.
    change = close.diff().fillna(0.0)
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    def wilder(series: pd.Series) -> pd.Series:
        return series.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    average_gain, average_loss = wilder(gain), wilder(loss)
    values = 100 - 100 / (1 + average_gain / average_loss)
    # Only gains: RSI is 100 by definition. No movement at all: neutral 50.
    values = values.where(average_loss != 0, 100.0)
    values = values.where((average_gain != 0) | (average_loss != 0), 50.0)
    return values.where(average_gain.notna())


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Columns: macd, macd_signal, macd_hist."""
    line = ema(close, fast) - ema(close, slow)
    signal_line = ema(line, signal)
    return pd.DataFrame(
        {"macd": line, "macd_signal": signal_line, "macd_hist": line - signal_line}
    )


def bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Columns: bb_middle, bb_upper, bb_lower, bb_percent_b."""
    middle = sma(close, window)
    # Population standard deviation (ddof=0), as in Bollinger's definition.
    spread = num_std * close.rolling(window, min_periods=window).std(ddof=0)
    upper, lower = middle + spread, middle - spread
    width = (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "bb_middle": middle,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_percent_b": (close - lower) / width,
    })


def compute_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Every indicator Agent 2 uses, one row per trading day.

    Columns: date, close, sma_short, sma_long, ema, rsi, macd, macd_signal,
    macd_hist, bb_middle, bb_upper, bb_lower, bb_percent_b.
    Window lengths come from config.TECHNICAL_WINDOWS.
    """
    windows = config.TECHNICAL_WINDOWS
    close = prices["close"].astype(float).reset_index(drop=True)

    table = pd.DataFrame({
        "date": prices["date"].reset_index(drop=True),
        "close": close,
        "sma_short": sma(close, windows["sma_short"]),
        "sma_long": sma(close, windows["sma_long"]),
        "ema": ema(close, windows["ema"]),
        "rsi": rsi(close, windows["rsi"]),
    })
    return pd.concat([
        table,
        macd(close, windows["macd_fast"], windows["macd_slow"], windows["macd_signal"]),
        bollinger_bands(close, windows["bollinger"], config.BOLLINGER_STD),
    ], axis=1)
