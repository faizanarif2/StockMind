"""Agent 2 - Technical Analysis Agent.

Studies how a company's share price has been moving, using the indicators
named in the proposal (RSI, MACD, SMA, EMA, Bollinger Bands), and returns a
Buy / Hold / Sell signal with a confidence and a plain-English explanation
for Agent 5. It answers "what is the price doing?"; Agent 3 answers "how
healthy is the company?".

It only calculates from Agent 1's prices - it never fetches anything - so it
runs in milliseconds and can be tested offline. Its result has the same
shape and the same signal / confidence rules as Agent 3's, so the Decision
Agent can combine them directly.

How the signal is made
----------------------
1. Prices. Agent 1's validated daily prices, adjusted for bonus issues
   (see analysis/price_adjustments.py) so a bonus is not mistaken for a
   price crash.

2. Indicators, calculated for every day (analysis/indicators.py): SMA 50,
   SMA 200, EMA 20, RSI 14, MACD (12, 26, 9), Bollinger Bands (20, 2).

3. Signal scores for the latest day, from -1 (bearish) to +1 (bullish),
   using the scales in config.TECHNICAL_SIGNALS:
     long_term_trend  SMA 50 vs SMA 200              (trend)
     price_vs_sma     close vs SMA 50                (trend)
     price_vs_ema     close vs EMA 20                (trend)
     macd             MACD histogram as % of price   (momentum)
     rsi              RSI <= 30 oversold ... >= 70 overbought  (reversal)
     bollinger        %B <= 0 lower band ... >= 1 upper band   (reversal)
   A signal that cannot be calculated (e.g. under 200 days of history for
   SMA 200) is skipped and the weights of the others are rescaled.

4. Overall score = weighted average of the signal scores.
       score >= +T  -> BUY      score <= -T  -> SELL     otherwise HOLD
   with T = config.TECHNICAL_SIGNAL_THRESHOLD. Each signal also "votes"
   BUY / HOLD / SELL with the same rule.

5. Confidence (0-100), the same formula as Agent 3:
       confidence = 100 x coverage x (agreement + strength) / 2
   coverage   share of the weight whose signal could be calculated
   agreement  share of that weight voting the same way as the overall signal
   strength   for BUY / SELL: |score| / config.TECHNICAL_FULL_STRENGTH,
              capped at 1; for HOLD: 1 - |score| / T

   Technical score (0-100) = 50 x (score + 1), for display.
"""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from stockmind import config
from stockmind.agents.data_collection import CompanyData
from stockmind.analysis.indicators import compute_indicators
from stockmind.analysis.price_adjustments import adjust_for_bonus


BUY, HOLD, SELL = "BUY", "HOLD", "SELL"

SIGNAL_LABELS: dict[str, str] = {
    "long_term_trend": "Long-term trend (SMA 50 vs SMA 200)",
    "price_vs_sma": "Price vs 50-day SMA",
    "price_vs_ema": "Price vs 20-day EMA",
    "macd": "MACD momentum",
    "rsi": "RSI (14)",
    "bollinger": "Bollinger Bands",
}

SIGNAL_COLUMNS = [
    "signal", "label", "value", "reference", "unit", "score", "weight",
    "vote", "comment",
]

DIRECTION = {BUY: "bullish", SELL: "bearish", HOLD: "neutral"}


@dataclass
class TechnicalAnalysis:
    """Agent 2's result for one ticker."""

    ticker: str
    signal: str                 # "BUY", "HOLD" or "SELL"
    confidence: float           # 0-100
    score: float                # -1 (bearish) to +1 (bullish)
    technical_score: float      # 0-100
    period: str | None          # prices used, e.g. "daily prices to 2026-11-10"
    trend: str                  # "uptrend", "downtrend" or "sideways"
    signals: pd.DataFrame       # one row per signal, see SIGNAL_COLUMNS
    indicators: pd.DataFrame    # every indicator for every day (for charts)
    adjustments: list[str]      # bonus-share adjustments applied to prices
    summary: str                # plain-English overview
    warnings: list[str] = field(default_factory=list)

    @property
    def explanation(self) -> list[str]:
        """One sentence per signal, for the report."""
        return [comment for comment in self.signals["comment"] if comment]


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------

def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _vote(score: float) -> str:
    threshold = config.TECHNICAL_SIGNAL_THRESHOLD
    if score >= threshold:
        return BUY
    if score <= -threshold:
        return SELL
    return HOLD


def gap_pct(value: float, reference: float) -> float:
    """How far `value` is above (+) or below (-) `reference`, in %."""
    return (value / reference - 1) * 100


def trend_score(gap: float, full_scale: float) -> float:
    """+1 at a gap of +full_scale % or more, -1 at -full_scale %."""
    return _clip(gap / full_scale)


def oscillator_score(value: float, levels: tuple[float, float, float, float]) -> float:
    """
    Score a reversal indicator such as RSI against four levels:
    (full buy, buy starts, sell starts, full sell), e.g. (30, 40, 60, 70).

    At or below `full buy` -> +1 (oversold), between `buy starts` and
    `sell starts` -> 0, at or above `full sell` -> -1 (overbought),
    linear in between.
    """
    full_buy, buy_starts, sell_starts, full_sell = levels
    if value <= buy_starts:
        return _clip((buy_starts - value) / (buy_starts - full_buy))
    if value >= sell_starts:
        return -_clip((value - sell_starts) / (full_sell - sell_starts))
    return 0.0


def signal_confidence(
    score: float, coverage: float, agreement: float
) -> tuple[str, float]:
    """The overall signal and its confidence (see module docstring)."""
    signal = _vote(score)
    threshold = config.TECHNICAL_SIGNAL_THRESHOLD
    if signal == HOLD:
        strength = 1 - abs(score) / threshold
    else:
        strength = min(1.0, abs(score) / config.TECHNICAL_FULL_STRENGTH)
    return signal, round(100 * coverage * (agreement + strength) / 2, 1)


def _pkr(value: float) -> str:
    return f"PKR {value:,.2f}"


def _position(gap: float) -> str:
    return f"{abs(gap):.1f}% {'above' if gap >= 0 else 'below'}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TechnicalAnalysisAgent:
    """Read the price trend, momentum, and overbought/oversold levels."""

    name = "Technical Analysis Agent"

    def analyse(self, data: CompanyData) -> TechnicalAnalysis:
        """
        Analyse one company from Agent 1's output.

        Never raises for missing data: signals that cannot be calculated are
        skipped, which lowers the confidence, and a warning explains why.
        """
        warnings: list[str] = []
        if data.prices.empty or "close" not in data.prices.columns:
            return self._no_data(data.ticker, [
                f"{data.ticker}: no price data available; "
                "technical analysis is not possible."
            ])
        prices = data.prices.sort_values("date").reset_index(drop=True)

        prices, adjustments, adjust_warnings = adjust_for_bonus(prices, data.payouts)
        warnings += adjust_warnings

        indicators = compute_indicators(prices)
        latest = indicators.iloc[-1]
        as_of = pd.Timestamp(latest["date"])
        if datetime.now() - as_of > config.TECHNICAL_MAX_DATA_AGE:
            warnings.append(
                f"{data.ticker}: latest price is from {as_of:%Y-%m-%d}; "
                "the signal may be out of date."
            )
        if len(prices) < config.TECHNICAL_WINDOWS["sma_long"]:
            warnings.append(
                f"{data.ticker}: only {len(prices)} trading days of prices; "
                f"the {config.TECHNICAL_WINDOWS['sma_long']}-day trend "
                "cannot be calculated."
            )

        signals = self._score_signals(latest)
        score, coverage, agreement = self._combine(signals)
        signal, confidence = signal_confidence(score, coverage, agreement)
        if coverage == 0:
            signal, confidence = HOLD, 0.0

        result = TechnicalAnalysis(
            ticker=data.ticker,
            signal=signal,
            confidence=confidence,
            score=round(score, 3),
            technical_score=round(50 * (score + 1), 1),
            period=f"daily prices to {as_of:%Y-%m-%d}",
            trend=self._trend(signals),
            signals=signals,
            indicators=indicators,
            adjustments=adjustments,
            summary="",
            warnings=warnings,
        )
        result.summary = self._summarise(result, coverage)
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _score_signals(latest: pd.Series) -> pd.DataFrame:
        """Score each signal on the latest day and describe it in words."""
        close = latest["close"]
        rows = []

        def add(name, value, reference, unit, score, comment):
            weight, _ = config.TECHNICAL_SIGNALS[name]
            rows.append({
                "signal": name, "label": SIGNAL_LABELS[name], "value": value,
                "reference": reference, "unit": unit,
                "score": None if score is None else round(score, 3),
                "weight": weight,
                "vote": None if score is None else _vote(score),
                "comment": comment,
            })

        def missing(name, why):
            add(name, None, None, None, None,
                f"{SIGNAL_LABELS[name]} could not be calculated ({why}).")

        def judged(score):
            return DIRECTION[_vote(score)]

        # Trend signals: % gap between two lines.
        trend_inputs = {
            "long_term_trend": (latest["sma_short"], latest["sma_long"]),
            "price_vs_sma": (close, latest["sma_short"]),
            "price_vs_ema": (close, latest["ema"]),
        }
        for name, (value, reference) in trend_inputs.items():
            if pd.isna(value) or pd.isna(reference):
                missing(name, "not enough price history")
                continue
            gap = gap_pct(value, reference)
            score = trend_score(gap, config.TECHNICAL_SIGNALS[name][1])
            if name == "long_term_trend":
                meaning = {BUY: "long-term uptrend", SELL: "long-term downtrend",
                           HOLD: "no clear long-term trend"}[_vote(score)]
                comment = (f"The 50-day SMA ({_pkr(value)}) is {_position(gap)} "
                           f"the 200-day SMA ({_pkr(reference)}): {meaning} "
                           f"({judged(score)}).")
            else:
                line = "50-day SMA" if name == "price_vs_sma" else "20-day EMA"
                comment = (f"Price {_pkr(value)} is {_position(gap)} its {line} "
                           f"of {_pkr(reference)} ({judged(score)}).")
            add(name, round(value, 2), round(reference, 2), "PKR", score, comment)

        # MACD: histogram relative to price.
        if pd.isna(latest["macd_hist"]):
            missing("macd", "not enough price history")
        else:
            hist_pct = latest["macd_hist"] / close * 100
            score = trend_score(hist_pct, config.TECHNICAL_SIGNALS["macd"][1])
            side = "above" if latest["macd_hist"] >= 0 else "below"
            add("macd", round(latest["macd"], 3), round(latest["macd_signal"], 3),
                "", score,
                f"MACD ({latest['macd']:.2f}) is {side} its signal line "
                f"({latest['macd_signal']:.2f}); the gap is {hist_pct:+.2f}% "
                f"of the price ({judged(score)} momentum).")

        # RSI: overbought / oversold.
        if pd.isna(latest["rsi"]):
            missing("rsi", "not enough price history")
        else:
            full_buy, buy_starts, sell_starts, full_sell = config.TECHNICAL_SIGNALS["rsi"][1]
            value = latest["rsi"]
            zone = (f"at or below {full_buy:g}: oversold" if value <= full_buy else
                    f"close to the oversold level of {full_buy:g}" if value <= buy_starts else
                    f"at or above {full_sell:g}: overbought" if value >= full_sell else
                    f"close to the overbought level of {full_sell:g}" if value >= sell_starts else
                    f"in the neutral {buy_starts:g}-{sell_starts:g} range")
            score = oscillator_score(value, (full_buy, buy_starts, sell_starts, full_sell))
            add("rsi", round(value, 1), None, "", score,
                f"RSI (14) is {value:.1f}, {zone} ({judged(score)}).")

        # Bollinger Bands: where the price sits between the bands.
        if pd.isna(latest["bb_percent_b"]):
            missing("bollinger", "not enough price history")
        else:
            levels = config.TECHNICAL_SIGNALS["bollinger"][1]
            value = latest["bb_percent_b"]
            place = ("below the lower band" if value < 0 else
                     "above the upper band" if value > 1 else
                     "near the lower band" if value <= levels[1] else
                     "near the upper band" if value >= levels[2] else
                     "in the middle of the bands")
            score = oscillator_score(value, levels)
            add("bollinger", round(value, 3), None, "%B", score,
                f"Price is at {value * 100:.0f}% of its Bollinger Band range "
                f"({_pkr(latest['bb_lower'])} to {_pkr(latest['bb_upper'])}): "
                f"{place} ({judged(score)}).")

        return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)

    @staticmethod
    def _combine(signals: pd.DataFrame) -> tuple[float, float, float]:
        """Weighted score, coverage, and agreement (see module docstring)."""
        total_weight = signals["weight"].sum()
        scored = signals[signals["score"].notna()]
        scored_weight = scored["weight"].sum()
        if scored_weight == 0:
            return 0.0, 0.0, 0.0

        scores = scored["score"].astype(float)
        score = float((scores * scored["weight"]).sum() / scored_weight)
        signal = _vote(score)
        agreement = float(scored.loc[scored["vote"] == signal, "weight"].sum()
                          / scored_weight)
        return score, float(scored_weight / total_weight), agreement

    @staticmethod
    def _trend(signals: pd.DataFrame) -> str:
        """
        Uptrend: the long-term trend votes BUY and the price is not below
        its 50-day SMA. Downtrend: the mirror image. Otherwise sideways.
        """
        votes = signals.set_index("signal")["vote"]
        long_term, price = votes.get("long_term_trend"), votes.get("price_vs_sma")
        if long_term is None or pd.isna(long_term):
            return "unknown"
        if long_term == BUY and price != SELL:
            return "uptrend"
        if long_term == SELL and price != BUY:
            return "downtrend"
        return "sideways"

    @staticmethod
    def _no_data(ticker: str, warnings: list[str]) -> TechnicalAnalysis:
        result = TechnicalAnalysis(
            ticker=ticker, signal=HOLD, confidence=0.0, score=0.0,
            technical_score=50.0, period=None, trend="unknown",
            signals=pd.DataFrame(columns=SIGNAL_COLUMNS),
            indicators=pd.DataFrame(), adjustments=[], summary="",
            warnings=warnings,
        )
        result.summary = TechnicalAnalysisAgent._summarise(result, coverage=0)
        return result

    @staticmethod
    def _summarise(result: TechnicalAnalysis, coverage: float) -> str:
        if coverage == 0:
            return (f"{result.ticker}: not enough price data for a technical "
                    "assessment. Signal HOLD with 0% confidence.")

        mood = {BUY: "bullish", SELL: "bearish", HOLD: "mixed"}[result.signal]
        scored = result.signals[result.signals["score"].notna()]
        bullish = scored.loc[scored["vote"] == BUY, "label"].tolist()
        bearish = scored.loc[scored["vote"] == SELL, "label"].tolist()

        parts = [
            f"{result.ticker}'s price action looks {mood} "
            f"(technical score {result.technical_score:.0f}/100, "
            f"{result.trend}, {result.period})."
        ]
        if bullish:
            parts.append("Bullish: " + ", ".join(bullish) + ".")
        if bearish:
            parts.append("Bearish: " + ", ".join(bearish) + ".")
        if result.adjustments:
            parts.append("Prices adjusted for bonus shares.")
        parts.append(f"Technical signal: {result.signal} with "
                     f"{result.confidence:.0f}% confidence.")
        return " ".join(parts)
