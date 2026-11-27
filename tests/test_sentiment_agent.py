"""Tests for the Sentiment Analysis Agent (Agent 4), offline.

A small fake model stands in for FinBERT: it reads a few keywords, so every
expected result can be worked out by hand.
"""

import numpy as np
import pandas as pd
import pytest

from stockmind import config
from stockmind.agents.data_collection import CompanyData, SectorPeers
from stockmind.agents.sentiment_analysis import (
    HISTORY_COLUMNS,
    SIGNAL_COLUMNS,
    SentimentAnalysisAgent,
    recency_weights,
)
from stockmind.analysis.scoring import BUY, HOLD, SELL
from stockmind.analysis.sentiment import ModelUnavailableError, SentimentScorer, normalise
from stockmind.data import news

NOW = pd.Timestamp("2026-11-25 12:00")
POSITIVE, NEGATIVE, NEUTRAL = (0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9)


class FakeModel:
    """Positive for "profit", "discovery", "gains"; negative for "loss",
    "falls", "sheds"; otherwise neutral. Records what it was asked."""

    name = "fake-finbert"

    def __init__(self):
        self.calls = []

    def classify(self, texts):
        self.calls.append(list(texts))
        def judge(text):
            if any(word in text for word in ("profit", "discovery", "gains")):
                return POSITIVE
            if any(word in text for word in ("loss", "falls", "sheds")):
                return NEGATIVE
            return NEUTRAL
        return np.array([judge(text) for text in texts])


class BrokenModel:
    name = "fake-finbert"  # same name: shares the remembered scores

    def classify(self, texts):
        raise ModelUnavailableError("no internet on first download")


@pytest.fixture
def model():
    return FakeModel()


@pytest.fixture
def agent(model, tmp_path):
    return SentimentAnalysisAgent(SentimentScorer(model, tmp_path / "scores"))


def item(title, days_ago, relevance="company", kind=None, informative=True):
    return {
        "article_id": f"id-{title}-{days_ago}-{relevance}",
        "published_at": NOW - pd.Timedelta(days=days_ago),
        "kind": kind or ("announcement" if relevance == "company" else "article"),
        "relevance": relevance, "category": None, "informative": informative,
        "title": title, "summary": None, "source": "PSX", "url": None,
    }


def company(*items, ticker="OGDC"):
    """CompanyData with only the news filled in."""
    empty = pd.DataFrame()
    return CompanyData(
        ticker=ticker, prices=empty, price_issues=empty, profile={},
        financials=empty, ratios=empty, income_statement=empty,
        balance_sheet=empty, payouts=empty, dividend_history=empty,
        peers=SectorPeers.empty(),
        news=pd.DataFrame(list(items), columns=news.NEWS_COLUMNS),
    )


def full_coverage(company_title, sector_title, market_title, copies=4):
    """Enough fresh news for every signal to count fully."""
    return [item(f"{title} {i}", days_ago=0, relevance=relevance)
            for relevance, title in (("company", company_title),
                                     ("sector", sector_title),
                                     ("market", market_title))
            for i in range(copies)]


# --- the scorer -------------------------------------------------------------

def test_each_text_is_scored_once_and_remembered(model, tmp_path):
    scorer = SentimentScorer(model, tmp_path)
    first, _ = scorer.score(["Record PROFIT", "Plant shut down"])
    again, _ = SentimentScorer(model, tmp_path).score(["record profit", "New text"])

    assert first["sentiment_label"].tolist() == ["positive", "neutral"]
    assert first.loc[0, "sentiment"] == pytest.approx(0.85)
    assert again.loc[0, "sentiment"] == pytest.approx(0.85)
    # Case and spacing do not matter; only the new text reached the model.
    assert model.calls == [["record profit", "plant shut down"], ["new text"]]


def test_unavailable_model_keeps_known_scores(model, tmp_path):
    SentimentScorer(model, tmp_path).score(["record profit"])
    scores, warnings = SentimentScorer(BrokenModel(), tmp_path).score(
        ["record profit", "never seen"])

    assert scores.loc[0, "sentiment_label"] == "positive"
    assert pd.isna(scores.loc[1, "sentiment"])
    assert "1 news items were not analysed" in warnings[0]


@pytest.mark.parametrize("text, expected", [
    ("OGDC profit surges 53pc", "ogdc profit surges 53 percent"),
    ("Exports up 12 PC in  October", "exports up 12 percent in october"),
    ("PC sales rise", "pc sales rise"),           # not after a number
    ("Topcon wins 5pcs order", "topcon wins 5pcs order"),
])
def test_text_is_normalised_for_finbert(text, expected):
    assert normalise(text) == expected


# --- the agent --------------------------------------------------------------

def test_positive_news_gives_buy(agent):
    result = agent.analyse(company(*full_coverage(
        "OGDC profit rises", "Oil output gains", "KSE-100 gains")), as_of=NOW)

    assert (result.signal, result.sentiment) == (BUY, "positive")
    assert list(result.signals.columns) == SIGNAL_COLUMNS
    # 0.85 average sentiment / 0.5 scale -> capped at +1 for every signal.
    assert result.score == 1.0 and result.sentiment_score == 100.0
    assert result.confidence == 100.0
    assert result.warnings == []
    assert "Most positive" in result.explanation[0]


def test_negative_news_gives_sell(agent):
    result = agent.analyse(company(*full_coverage(
        "OGDC loss widens", "Oil output falls", "KSE-100 sheds 1,000 points")), as_of=NOW)
    assert (result.signal, result.sentiment) == (SELL, "negative")
    assert "Most negative" in result.explanation[0]


def test_neutral_news_gives_a_confident_hold(agent):
    result = agent.analyse(company(*full_coverage(
        "Board approves", "Oil output steady", "Market closes")), as_of=NOW)
    assert (result.signal, result.score, result.confidence) == (HOLD, 0.0, 100.0)


def test_same_output_shape_as_agents_2_and_3(agent):
    result = agent.analyse(company(item("Gas discovery", 1)), as_of=NOW)
    for name in ("ticker", "signal", "confidence", "score", "period",
                 "summary", "warnings", "explanation"):
        assert hasattr(result, name)
    assert result.period == "news to 2026-11-24"
    assert result.summary.endswith(f"{result.signal} with {result.confidence:.0f}% confidence.")


def test_no_news_is_hold_with_zero_confidence(agent):
    result = agent.analyse(company(), as_of=NOW)
    assert (result.signal, result.confidence) == (HOLD, 0.0)
    assert result.signals.empty and result.history.empty
    assert "no news available" in result.warnings[0]


def test_procedural_notices_are_ignored(agent, model):
    result = agent.analyse(company(
        item("Board meeting to discuss profit", 1, informative=False),
        item("Plant maintenance completed", 1)), as_of=NOW)

    assert result.articles["title"].tolist() == ["Plant maintenance completed"]
    assert model.calls == [["plant maintenance completed"]]


def test_recent_news_counts_more_than_old_news(agent):
    result = agent.analyse(company(
        item("Profit rises", days_ago=1),
        item("Loss reported", days_ago=50)), as_of=NOW)
    recent = result.signals.set_index("signal").loc["company_recent"]

    weights = recency_weights(pd.Series([1.0, 50.0]), half_life=14)
    expected = (0.85 * weights[0] - 0.85 * weights[1]) / weights.sum()
    assert recent["value"] == pytest.approx(expected, abs=1e-3)
    assert recent["vote"] == BUY


def test_little_news_lowers_confidence(agent):
    one = agent.analyse(company(item("Profit rises", 0.1)), as_of=NOW)
    many = agent.analyse(company(*[item(f"Profit rises {i}", 0.1) for i in range(4)]),
                         as_of=NOW)

    assert one.signal == many.signal == BUY
    assert one.confidence < many.confidence
    evidence = one.signals.set_index("signal")["evidence"]
    assert evidence["company_recent"] == pytest.approx(1 / config.SENTIMENT_FULL_EVIDENCE,
                                                       abs=0.01)


def test_missing_sector_and_market_news_is_reported(agent):
    result = agent.analyse(company(item("Profit rises", 1)), as_of=NOW)

    items = result.signals.set_index("signal")["items"]
    assert items["sector"] == items["market"] == 0
    assert result.warnings == [
        "OGDC: sector unknown, so sector news is not used.",
        "OGDC: no market headlines in the last 14 days; no headlines are stored "
        "yet (the press archive may have been unavailable; Agent 1 adds more each run).",
    ]


def test_sector_without_keywords_is_named(agent):
    data = company(item("Profit rises", 1), item("KSE-100 gains", 1, "market"))
    data.profile = {"sector": "CABLE & ELECTRICAL GOODS"}
    with pytest.MonkeyPatch.context() as patch:
        patch.delitem(config.NEWS_SECTOR_KEYWORDS, "CABLE & ELECTRICAL GOODS")
        result = agent.analyse(data, as_of=NOW)

    assert result.warnings == [
        "OGDC: sector news is not used: no keywords for its sector "
        "(CABLE & ELECTRICAL GOODS) in config.NEWS_SECTOR_KEYWORDS."]


def test_short_headline_history_is_named_only_when_it_is_the_cause(agent):
    data = company(item("Profit rises", 1), item("KSE-100 gains", 3, "market"))
    data.profile = {"sector": "CEMENT"}
    short = agent.analyse(data, as_of=NOW)
    assert short.warnings == [
        f"OGDC: no CEMENT sector headlines in the last 30 days; headlines are stored "
        f"only since {NOW - pd.Timedelta(days=3):%Y-%m-%d} (the press archive may have "
        "been unavailable; Agent 1 adds more each run)."]

    data.news = pd.concat([data.news, pd.DataFrame(
        [item("KSE-100 falls", 200, "market")], columns=news.NEWS_COLUMNS)])
    long = agent.analyse(data, as_of=NOW)
    assert long.warnings == ["OGDC: no CEMENT sector headlines in the last 30 days."]


def test_old_company_news_only_counts_in_the_long_term_signal(agent):
    result = agent.analyse(company(item("Profit rises", 200)), as_of=NOW)
    items = result.signals.set_index("signal")["items"]
    assert (items["company_recent"], items["company_long_term"]) == (0, 1)
    assert any("no informative company news in the last 60 days;" in w
               for w in result.warnings)


def test_procedural_notices_are_named_as_the_reason(agent):
    result = agent.analyse(company(item("Profit rises", 200),
                                   item("Notice of Board Meeting", 5, informative=False)),
                           as_of=NOW)
    assert any("no informative company news in the last 60 days (only 1 procedural "
               "notice, e.g. board meetings, which carry no sentiment)" in w
               for w in result.warnings)


def test_as_of_ignores_later_news(agent):
    news_items = [item("Loss reported", days_ago=30), item("Profit rises", days_ago=1)]
    earlier = agent.analyse(company(*news_items), as_of=NOW - pd.Timedelta(days=10))

    assert earlier.articles["title"].tolist() == ["Loss reported"]
    assert earlier.signals.set_index("signal").loc["company_recent", "vote"] == SELL


def test_unavailable_model_still_gives_a_result(tmp_path):
    agent = SentimentAnalysisAgent(SentimentScorer(BrokenModel(), tmp_path))
    result = agent.analyse(company(item("Profit rises", 1)), as_of=NOW)
    assert (result.signal, result.confidence) == (HOLD, 0.0)
    assert any("FinBERT is unavailable" in w for w in result.warnings)


def test_monthly_history(agent):
    result = agent.analyse(company(
        item("Profit rises", 1), item("Loss reported", 2),
        item("Gas discovery", 40), item("Oil output gains", 1, relevance="sector")),
        as_of=NOW)

    history = result.history
    assert list(history.columns) == HISTORY_COLUMNS
    november = history[(history["month"] == "2026-11-01")
                       & (history["relevance"] == "company")].iloc[0]
    assert (november["items"], november["positive"], november["negative"]) == (2, 1, 1)
    assert november["sentiment"] == pytest.approx(0.0)
    assert len(history) == 3  # Oct company, Nov company, Nov sector


def test_recency_weights_halve_every_half_life():
    weights = recency_weights(pd.Series([0.0, 7.0, 14.0, -1.0]), half_life=7)
    assert weights.tolist() == [1.0, 0.5, 0.25, 1.0]


def test_summary_says_when_there_is_no_company_news(agent):
    result = agent.analyse(company(*[item(f"Urea offtake gains {i}", 1, "sector")
                                     for i in range(4)]), as_of=NOW)

    assert result.signal == BUY
    assert ("There is no informative news about OGDC itself, so this reflects "
            "only sector news.") in result.summary


# --- market news is context only ----------------------------------------------

def market(title, copies=4, days_ago=0):
    return [item(f"{title} {i}", days_ago, "market") for i in range(copies)]


def test_market_news_alone_is_not_enough_evidence(agent):
    result = agent.analyse(company(*market("KSE-100 sheds 1,000 points")), as_of=NOW)

    assert (result.signal, result.confidence, result.score) == (HOLD, 0.0, 0.0)
    assert (result.sentiment, result.sentiment_score) == ("unknown", 50.0)
    assert ("OGDC: no informative company or sector news, so there is not enough "
            "evidence for a sentiment signal (market news is context only).") in result.warnings
    # The market news is still scored and explained.
    row = result.signals.set_index("signal").loc["market"]
    assert (row["items"], row["vote"], row["weight"]) == (4, SELL, 0.0)
    assert "Context only" in row["comment"]
    assert "Market news (context only, not part of the signal): negative" in result.summary
    assert result.summary.endswith("Signal HOLD with 0% confidence.")


@pytest.mark.parametrize("market_news", [
    [],
    market("KSE-100 sheds 1,000 points"),   # a market crash
    market("KSE-100 gains in record rally"),
])
def test_market_news_never_changes_the_company_signal(agent, market_news):
    own_news = [item(f"OGDC profit rises {i}", 1) for i in range(2)]
    own_news += [item("Oil output falls", 2, "sector")]
    baseline = agent.analyse(company(*own_news), as_of=NOW)
    result = agent.analyse(company(*own_news, *market_news), as_of=NOW)

    assert (result.signal, result.score, result.confidence, result.sentiment) == (
        baseline.signal, baseline.score, baseline.confidence, baseline.sentiment)


def test_market_crash_is_reported_alongside_good_company_news(agent):
    result = agent.analyse(company(*[item(f"OGDC profit rises {i}", 0) for i in range(4)],
                                   *market("KSE-100 sheds 1,000 points")), as_of=NOW)

    assert (result.signal, result.sentiment) == (BUY, "positive")
    assert "Market news (context only, not part of the signal): negative" in result.summary
    assert any("Context only" in sentence for sentence in result.explanation)


def test_decision_agent_treats_market_only_sentiment_as_no_evidence(agent):
    from stockmind.agents.decision_making import DecisionMakingAgent

    sentiment = agent.analyse(company(*market("KSE-100 sheds 1,000 points")), as_of=NOW)
    decision = DecisionMakingAgent().combine(sentiment=sentiment)

    row = decision.contributions.set_index("agent").loc["sentiment"]
    assert (row["status"], row["effective_weight"]) == ("no evidence", 0.0)
    assert (decision.signal, decision.confidence) == (HOLD, 0.0)

    # Company news is evidence: Agent 5 uses it.
    sentiment = agent.analyse(company(*[item(f"OGDC profit rises {i}", 0) for i in range(4)],
                                      *market("KSE-100 sheds 1,000 points")), as_of=NOW)
    decision = DecisionMakingAgent().combine(sentiment=sentiment)
    assert decision.contributions.set_index("agent").at["sentiment", "status"] == "used"
    assert decision.signal == BUY
