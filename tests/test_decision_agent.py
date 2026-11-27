"""Tests for the Decision Making Agent (Agent 5), offline.

Agents 2-4 are replaced by hand-made results, so every expected decision
can be worked out by hand. Default weights: technical 0.35, fundamental
0.40, sentiment 0.25.
"""

import itertools

import pandas as pd
import pytest

from stockmind import config
from stockmind.agents.decision_making import (
    CONTRIBUTION_COLUMNS,
    DecisionMakingAgent,
    normalise_weights,
)
from stockmind.agents.fundamental_analysis import METRIC_COLUMNS, FundamentalAnalysis
from stockmind.agents.sentiment_analysis import SIGNAL_COLUMNS as SENTIMENT_COLUMNS
from stockmind.agents.sentiment_analysis import SentimentAnalysis
from stockmind.agents.technical_analysis import SIGNAL_COLUMNS as TECHNICAL_COLUMNS
from stockmind.agents.technical_analysis import TechnicalAnalysis
from stockmind.analysis.scoring import BUY, HOLD, SELL


def _table(columns, signal, score, **extra):
    """One detail row voting with the agent, plus one weaker row."""
    rows = [{"score": score, "weight": 0.6, "vote": signal,
             "comment": f"main reason ({signal})", **extra},
            {"score": score / 4, "weight": 0.4, "vote": HOLD,
             "comment": "minor reason", **extra}]
    return pd.DataFrame(rows).reindex(columns=columns)


def technical(signal, confidence, score, ticker="TEST", warnings=()):
    return TechnicalAnalysis(
        ticker=ticker, signal=signal, confidence=confidence, score=score,
        technical_score=50 * (score + 1), period="daily prices to 2026-11-27",
        trend="sideways", signals=_table(TECHNICAL_COLUMNS, signal, score),
        indicators=pd.DataFrame(), adjustments=[], summary="",
        warnings=list(warnings))


def fundamental(signal, confidence, score, ticker="TEST", warnings=()):
    return FundamentalAnalysis(
        ticker=ticker, signal=signal, confidence=confidence, score=score,
        health_score=50 * (score + 1), period="TTM to 2026-09-30",
        sector="TEST SECTOR", peers=[], metrics=_table(METRIC_COLUMNS, signal, score),
        peer_ratios=pd.DataFrame(), summary="", warnings=list(warnings))


def sentiment(signal, confidence, score, ticker="TEST", warnings=()):
    return SentimentAnalysis(
        ticker=ticker, signal=signal, confidence=confidence, score=score,
        sentiment_score=50 * (score + 1), sentiment="neutral",
        period="news to 2026-11-27",
        signals=_table(SENTIMENT_COLUMNS, signal, score, evidence=1.0),
        articles=pd.DataFrame(), history=pd.DataFrame(), summary="",
        warnings=list(warnings))


@pytest.fixture
def agent():
    return DecisionMakingAgent()


def by_agent(decision):
    return decision.contributions.set_index("agent")


# --- weights -----------------------------------------------------------------

def test_default_weights_add_up_to_one(agent):
    assert sum(agent.weights.values()) == pytest.approx(1.0)
    assert agent.weights == pytest.approx(config.DECISION_AGENT_WEIGHTS)


def test_weights_are_rescaled_to_one():
    assert normalise_weights({"technical": 1, "fundamental": 1, "sentiment": 2}) == {
        "technical": 0.25, "fundamental": 0.25, "sentiment": 0.5}


@pytest.mark.parametrize("weights", [
    {"technical": 1, "fundamental": 1},                              # one missing
    {"technical": 1, "fundamental": 1, "sentiment": 1, "risk": 1},   # unknown
    {"technical": 1, "fundamental": -1, "sentiment": 1},             # negative
    {"technical": 0, "fundamental": 0, "sentiment": 0},              # nothing
    {"technical": float("nan"), "fundamental": 1, "sentiment": 1},
])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        normalise_weights(weights)


# --- combining ---------------------------------------------------------------

def test_agents_that_agree(agent):
    decision = agent.combine(technical(BUY, 65, 0.36), fundamental(BUY, 50, 0.24),
                             sentiment(BUY, 29, 0.34))

    # Effective weights 0.2275, 0.20, 0.0725 (sum 0.50).
    assert decision.signal == BUY
    assert decision.score == pytest.approx(0.309, abs=0.001)
    # All support: 0.35 x 65 + 0.40 x 50 + 0.25 x 29 = 50.
    assert decision.confidence == pytest.approx(50.0)
    assert decision.agreement == "agree"
    assert decision.decision_score == pytest.approx(50 * (decision.score + 1), abs=0.1)
    assert by_agent(decision)["relation"].eq("supports").all()
    assert by_agent(decision)["share"].sum() == pytest.approx(1.0, abs=0.01)
    assert "All 3 agents" in decision.summary


def test_conflicting_agents_give_a_hold_with_low_confidence(agent):
    decision = agent.combine(technical(BUY, 65, 0.40), fundamental(SELL, 60, -0.35),
                             sentiment(HOLD, 20, 0.05))

    # BUY and SELL cancel out: score +0.018 -> HOLD. Sentiment supports it
    # (0.25 x 20 = 5); BUY and SELL are each one step from HOLD, so they
    # count half: 0.5 x (0.35 x 65 + 0.40 x 60) = 23.4. Total 28.4%.
    assert decision.signal == HOLD
    assert decision.confidence == pytest.approx(28.4)
    assert decision.agreement == "conflict"
    relations = by_agent(decision)["relation"]
    assert relations.to_dict() == {"technical": "partly supports",
                                   "fundamental": "partly supports",
                                   "sentiment": "supports"}
    assert any("conflict" in w and "Technical Analysis says BUY" in w
               and "Fundamental Analysis says SELL" in w for w in decision.warnings)
    assert "The agents conflict." in decision.summary


def test_an_opposing_agent_lowers_confidence(agent):
    agreeing = agent.combine(technical(BUY, 65, 0.36), fundamental(BUY, 50, 0.24),
                             sentiment(HOLD, 30, 0.0))
    opposed = agent.combine(technical(BUY, 65, 0.36), fundamental(BUY, 50, 0.24),
                            sentiment(SELL, 30, -0.30))

    assert agreeing.signal == opposed.signal == BUY
    # Technical + Fundamental support: 22.75 + 20 = 42.75. A HOLD is one
    # step away and adds half of 0.25 x 30; a SELL subtracts all of it.
    assert agreeing.confidence == pytest.approx(42.75 + 3.75, abs=0.05)
    assert opposed.confidence == pytest.approx(42.75 - 7.5, abs=0.05)
    assert by_agent(agreeing).at["sentiment", "relation"] == "partly supports"
    assert by_agent(opposed).at["sentiment", "relation"] == "opposes"
    assert opposed.agreement == "conflict"


def test_confidence_can_never_go_below_zero(agent):
    # The score leans BUY (a very bullish but unsure technical view), but
    # more confidence-weighted evidence points to SELL.
    decision = agent.combine(technical(BUY, 30, 1.0), fundamental(SELL, 40, -0.2),
                             sentiment(HOLD, 0, 0.0))

    assert decision.signal == BUY
    assert decision.confidence == 0.0


def test_all_agents_saying_hold_supports_a_hold(agent):
    decision = agent.combine(technical(HOLD, 80, 0.0), fundamental(HOLD, 70, 0.05),
                             sentiment(HOLD, 60, -0.05))

    assert decision.signal == HOLD
    assert decision.confidence == pytest.approx(0.35 * 80 + 0.40 * 70 + 0.25 * 60)
    assert decision.agreement == "agree"


def test_hold_between_two_buys_is_partly_supported(agent):
    # GHNI on 2026-11-27, before Agent 4 stopped using market news: two BUYs
    # and a SELL average to +0.195, just inside the HOLD zone. No agent says
    # HOLD, but all three are one step from it.
    decision = agent.combine(technical(BUY, 36, 0.22), fundamental(BUY, 58, 0.41),
                             sentiment(SELL, 26, -0.62))

    assert decision.signal == HOLD
    assert decision.score == pytest.approx(0.195, abs=0.001)
    assert decision.confidence == pytest.approx(0.5 * (12.6 + 23.2 + 6.5), abs=0.1)
    assert by_agent(decision)["relation"].eq("partly supports").all()
    assert "Partly supports: Technical Analysis, Fundamental Analysis, Sentiment Analysis." \
        in decision.summary


def test_no_cliff_at_the_buy_threshold(agent):
    # Score = 0.21 x technical score / 0.45: +0.196 and +0.205, just below
    # and just above +0.2. HOLD and BUY, both partly confident.
    below = agent.combine(technical(BUY, 60, 0.42), fundamental(HOLD, 60, 0.0))
    above = agent.combine(technical(BUY, 60, 0.44), fundamental(HOLD, 60, 0.0))

    assert (below.signal, above.signal) == (HOLD, BUY)
    # HOLD: 24 (Fundamental) + 0.5 x 21 (Technical); BUY: 21 + 0.5 x 24.
    assert below.confidence == pytest.approx(34.5)
    assert above.confidence == pytest.approx(33.0)


def test_hold_agent_partly_supports_a_buy(agent):
    decision = agent.combine(technical(BUY, 60, 0.5), fundamental(HOLD, 60, 0.0),
                             sentiment(BUY, 40, 0.4))

    assert decision.signal == BUY
    # 0.35 x 60 + 0.25 x 40 support; 0.40 x 60 counts half.
    assert decision.confidence == pytest.approx(21 + 10 + 12)
    assert decision.agreement == "mixed"


def test_partial_support_must_stay_between_zero_and_one(monkeypatch):
    monkeypatch.setattr(config, "DECISION_PARTIAL_SUPPORT", 1.5)
    with pytest.raises(ValueError):
        DecisionMakingAgent()


def test_confidence_is_never_above_the_agents_weighted_confidence(agent):
    # Conservative: for every combination, including missing agents, the
    # decision is never more confident than the weighted confidence of the
    # agents that are there (missing ones count as 0), and never above the
    # most confident agent.
    signals = [(BUY, 0.5), (HOLD, 0.0), (SELL, -0.5)]
    builders = (technical, fundamental, sentiment)
    weights = (0.35, 0.40, 0.25)
    options = [(signal, confidence, score) for signal, score in signals
               for confidence in (0, 30, 90)] + [None]
    for chosen in itertools.product(options, repeat=3):
        results = [None if c is None else build(*c) for build, c in zip(builders, chosen)]
        confidences = [0 if c is None else c[1] for c in chosen]
        decision = agent.combine(*results, ticker="TEST")
        ceiling = sum(w * c for w, c in zip(weights, confidences))
        assert 0 <= decision.confidence <= ceiling + 0.05
        assert decision.confidence <= max(confidences)


# --- missing agents and failures ------------------------------------------------

def test_missing_agent_does_not_pull_the_score_but_caps_confidence(agent):
    decision = agent.combine(technical(BUY, 60, 0.4), fundamental(BUY, 60, 0.4), None)

    # Score: rescaled over the agents that are there, so it stays +0.40.
    assert decision.score == pytest.approx(0.4)
    assert decision.signal == BUY
    # Confidence: NOT rescaled; the missing 25% counts as no support:
    # 0.35 x 60 + 0.40 x 60 = 45, not 60.
    assert decision.confidence == pytest.approx(45.0)
    row = by_agent(decision).loc["sentiment"]
    assert (row["status"], row["relation"], row["effective_weight"]) == (
        "missing", "not used", 0.0)
    assert any("Sentiment Analysis" in w and "at most 75%" in w for w in decision.warnings)
    assert "No usable result from Sentiment Analysis." in decision.summary


def test_zero_confidence_agent_is_treated_like_a_missing_one(agent):
    missing = agent.combine(technical(BUY, 60, 0.4), fundamental(BUY, 60, 0.4), None)
    empty = agent.combine(technical(BUY, 60, 0.4), fundamental(BUY, 60, 0.4),
                          sentiment(HOLD, 0.0, 0.0))

    assert (empty.signal, empty.score, empty.confidence) == (
        missing.signal, missing.score, missing.confidence)
    assert by_agent(empty).at["sentiment", "status"] == "no evidence"
    assert empty.agreement == "agree"


def test_conflict_with_a_missing_agent(agent):
    # SYS on 2026-11-27: technicals SELL, fundamentals BUY, no sentiment.
    decision = agent.combine(technical(SELL, 54, -0.34), fundamental(BUY, 55, 0.24), None)

    assert decision.signal == HOLD
    assert decision.agreement == "conflict"
    # Both are one step from HOLD; the missing 25% adds nothing.
    assert decision.confidence == pytest.approx(0.5 * (0.35 * 54 + 0.40 * 55), abs=0.1)
    assert decision.confidence <= 75
    assert any("at most 75%" in w for w in decision.warnings)
    assert any("conflict" in w for w in decision.warnings)


def test_only_one_agent_left(agent):
    decision = agent.combine(None, fundamental(SELL, 70, -0.5), None)

    assert decision.signal == SELL
    assert decision.score == pytest.approx(-0.5)
    assert decision.confidence == pytest.approx(0.40 * 70)
    assert any("Technical Analysis and Sentiment Analysis" in w and "at most 40%" in w
               for w in decision.warnings)


def test_no_evidence_at_all_gives_hold_with_zero_confidence(agent):
    decision = agent.combine(technical(HOLD, 0.0, 0.0), None, None)

    assert (decision.signal, decision.confidence, decision.score) == (HOLD, 0.0, 0.0)
    assert decision.decision_score == 50.0
    assert decision.agreement == "none"
    assert "no basis for a decision" in decision.summary


def test_all_missing_needs_a_ticker(agent):
    with pytest.raises(ValueError):
        agent.combine(None, None, None)
    assert agent.combine(None, None, None, ticker="TEST").signal == HOLD


def test_results_for_different_tickers_are_rejected(agent):
    with pytest.raises(ValueError, match="one ticker"):
        agent.combine(technical(BUY, 60, 0.4, ticker="OGDC"),
                      fundamental(BUY, 60, 0.4, ticker="MEBL"))


class Fixed:
    """Stands in for Agent 2, 3 or 4: returns a prepared result."""

    name = "Fixed Agent"

    def __init__(self, result):
        self.result = result

    def analyse(self, data):
        return self.result


class Broken:
    name = "Broken Agent"

    def analyse(self, data):
        raise RuntimeError("model crashed")


class Data:
    ticker = "TEST"


def test_decide_runs_all_three_agents():
    agent = DecisionMakingAgent(technical=Fixed(technical(BUY, 60, 0.4)),
                                fundamental=Fixed(fundamental(BUY, 60, 0.4)),
                                sentiment=Fixed(sentiment(BUY, 60, 0.4)))
    decision = agent.decide(Data())

    assert (decision.signal, decision.confidence) == (BUY, pytest.approx(60.0))
    assert decision.technical is agent._agents["technical"].result


def test_a_failing_agent_does_not_stop_the_decision():
    agent = DecisionMakingAgent(technical=Fixed(technical(BUY, 60, 0.4)),
                                fundamental=Fixed(fundamental(BUY, 60, 0.4)),
                                sentiment=Broken())
    decision = agent.decide(Data())

    # Same as a missing Sentiment result.
    assert decision.signal == BUY
    assert decision.confidence == pytest.approx(45.0)
    assert decision.sentiment is None
    assert by_agent(decision).at["sentiment", "status"] == "missing"
    assert decision.warnings[0] == ("TEST: the Sentiment Analysis Agent failed "
                                    "(RuntimeError: model crashed).")


def test_every_agent_failing_gives_hold_with_zero_confidence():
    agent = DecisionMakingAgent(technical=Broken(), fundamental=Broken(),
                                sentiment=Broken())
    decision = agent.decide(Data())

    assert (decision.ticker, decision.signal, decision.confidence) == ("TEST", HOLD, 0.0)
    assert sum("failed" in w for w in decision.warnings) == 3


# --- learnt weights (Agent 7) ------------------------------------------------------

def test_custom_weights_change_the_decision():
    results = (technical(BUY, 60, 0.4), fundamental(SELL, 60, -0.4),
               sentiment(HOLD, 60, 0.0))
    trust_price = DecisionMakingAgent(
        weights={"technical": 0.8, "fundamental": 0.1, "sentiment": 0.1})
    trust_statements = DecisionMakingAgent(
        weights={"technical": 0.1, "fundamental": 0.8, "sentiment": 0.1})

    assert trust_price.combine(*results).signal == BUY
    assert trust_statements.combine(*results).signal == SELL


def test_an_agent_with_weight_zero_is_left_out_without_a_warning():
    agent = DecisionMakingAgent(weights={"technical": 1, "fundamental": 1, "sentiment": 0})
    decision = agent.combine(technical(BUY, 60, 0.4), fundamental(BUY, 60, 0.4),
                             sentiment(SELL, 90, -0.9))

    assert decision.signal == BUY
    assert decision.confidence == pytest.approx(60.0)
    assert by_agent(decision).at["sentiment", "status"] == "weight 0"
    assert not any("no usable result" in w for w in decision.warnings)


# --- result ---------------------------------------------------------------------

def test_decision_score_shows_direction_not_certainty(agent):
    # A strongly bullish view from one unsure agent: high decision score,
    # low confidence.
    decision = agent.combine(technical(BUY, 20, 0.8), None, None)

    assert decision.decision_score == pytest.approx(90.0)
    assert decision.confidence == pytest.approx(7.0)


def test_result_shape_and_explanation(agent):
    decision = agent.combine(
        technical(BUY, 65, 0.36, warnings=["TEST: old prices."]),
        fundamental(BUY, 50, 0.24, warnings=["TEST: old prices."]),
        sentiment(HOLD, 29, 0.10, warnings=["TEST: little news."]))

    assert list(decision.contributions.columns) == CONTRIBUTION_COLUMNS
    assert len(decision.explanation) == 3
    rows = by_agent(decision)
    # The agent's own sentence that pushed hardest is quoted; a HOLD has none.
    assert rows.at["technical", "reason"] == "main reason (BUY)"
    assert "Strongest point: main reason (BUY)" in rows.at["technical", "comment"]
    assert pd.isna(rows.at["sentiment", "reason"])
    assert "Strongest point" not in rows.at["sentiment", "comment"]
    # The agents' warnings are passed on once each.
    assert decision.warnings == ["TEST: old prices.", "TEST: little news."]
    assert decision.summary.startswith("TEST: BUY with")
    assert -1 <= decision.score <= 1
