"""Signal and confidence rules shared by the analysis agents (2, 3 and 4).

Each analysis agent turns its evidence into a few scored signals, each from
-1 (bearish) to +1 (bullish) with a weight, and combines them with the same
rules, so the Decision Agent (Agent 5) can compare and combine the results:

    score       weighted average of the signal scores
    signal      score >= +T -> BUY, score <= -T -> SELL, otherwise HOLD
                (each signal also "votes" with the same rule)
    confidence  100 x coverage x (agreement + strength) / 2
      coverage   share of the weight backed by evidence
      agreement  share of that weight voting the same way as the signal
      strength   BUY / SELL: |score| / full_strength, capped at 1
                 HOLD: 1 - |score| / T (a score of exactly 0 is the clearest HOLD)

A signal's evidence is 1 when it could be calculated and 0 when it could not
(Agents 2 and 3: an indicator or a ratio is either available or missing).
News comes in varying amounts, so Agent 4 also uses values in between: a
signal based on only one or two headlines counts partly.

Agents 2 and 3 were written in parallel on their own branches and still keep
their own copy of these rules; they can switch to this module once all three
are merged.
"""

import pandas as pd

BUY, HOLD, SELL = "BUY", "HOLD", "SELL"


def clip(value: float) -> float:
    """Limit a score to the range -1 to +1."""
    return max(-1.0, min(1.0, value))


def vote(score: float, threshold: float) -> str:
    """BUY, HOLD or SELL for a score (see the module docstring)."""
    if score >= threshold:
        return BUY
    if score <= -threshold:
        return SELL
    return HOLD


def signal_confidence(
    score: float, coverage: float, agreement: float,
    threshold: float, full_strength: float,
) -> tuple[str, float]:
    """The overall signal and its confidence (0-100)."""
    signal = vote(score, threshold)
    if signal == HOLD:
        strength = 1 - abs(score) / threshold
    else:
        strength = min(1.0, abs(score) / full_strength)
    return signal, round(100 * coverage * (agreement + strength) / 2, 1)


def combine(signals: pd.DataFrame, threshold: float) -> tuple[float, float, float]:
    """
    Weighted score, coverage and agreement of a table of signals.

    Uses the columns `score`, `weight`, `vote` and, if present, `evidence`
    (0-1; 1 when missing). A signal without a score counts as no evidence.
    """
    if signals.empty:
        return 0.0, 0.0, 0.0
    total_weight = float(signals["weight"].sum())
    evidence = (signals["evidence"].fillna(0).astype(float)
                if "evidence" in signals else pd.Series(1.0, index=signals.index))
    used = signals["weight"].astype(float) * evidence.where(signals["score"].notna(), 0.0)
    used_weight = float(used.sum())
    if used_weight == 0 or total_weight == 0:
        return 0.0, 0.0, 0.0

    scores = signals["score"].astype(float).fillna(0.0)
    score = float((scores * used).sum() / used_weight)
    agreement = float(used[signals["vote"] == vote(score, threshold)].sum() / used_weight)
    return score, used_weight / total_weight, agreement
