"""Tests for the shared signal and confidence rules."""

import pandas as pd
import pytest

from stockmind.analysis.scoring import BUY, HOLD, SELL, combine, signal_confidence, vote


def signals(*rows):
    """(score, weight, evidence) rows; votes use a threshold of 0.2."""
    table = pd.DataFrame(rows, columns=["score", "weight", "evidence"])
    table["vote"] = [None if pd.isna(s) else vote(s, 0.2) for s in table["score"]]
    return table


@pytest.mark.parametrize("score, expected", [
    (0.2, BUY), (0.19, HOLD), (0.0, HOLD), (-0.2, SELL), (-1.0, SELL),
])
def test_vote(score, expected):
    assert vote(score, 0.2) == expected


def test_confidence_formula():
    # BUY at 0.3: strength 0.3 / 0.6 = 0.5 -> 100 x 1 x (1 + 0.5) / 2 = 75.
    assert signal_confidence(0.3, 1.0, 1.0, 0.2, 0.6) == (BUY, 75.0)
    # HOLD at 0.0 is the clearest HOLD; half the weight missing halves it.
    assert signal_confidence(0.0, 0.5, 1.0, 0.2, 0.6) == (HOLD, 50.0)


def test_combine_without_evidence_column_matches_agents_2_and_3():
    table = signals((0.5, 0.6, 1.0), (-0.5, 0.2, 1.0), (None, 0.2, 1.0)).drop(columns="evidence")
    score, coverage, agreement = combine(table, 0.2)
    assert score == pytest.approx(0.25)       # (0.3 - 0.1) / 0.8
    assert coverage == pytest.approx(0.8)
    assert agreement == pytest.approx(0.75)   # BUY weight 0.6 of 0.8


def test_partial_evidence_counts_partly():
    score, coverage, agreement = combine(signals((1.0, 0.5, 0.25), (0.0, 0.5, 1.0)), 0.2)
    assert score == pytest.approx(0.2)        # 0.125 / 0.625
    assert coverage == pytest.approx(0.625)
    assert agreement == pytest.approx(0.2)


def test_combine_with_nothing_usable():
    assert combine(signals((None, 1.0, 1.0)), 0.2) == (0.0, 0.0, 0.0)
    assert combine(signals(), 0.2) == (0.0, 0.0, 0.0)
