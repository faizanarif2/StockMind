"""Agent 4 - Sentiment Analysis Agent.

Reads the news Agent 1 collected about a company, its sector and the
market, judges each headline with FinBERT (positive, negative or neutral),
and returns a Buy / Hold / Sell signal with a confidence and a plain-English
explanation for Agent 5. It answers "what is the news saying?"; Agent 2
answers "what is the price doing?" and Agent 3 "how healthy is the company?".

It fetches nothing itself. Its result has the same shape and the same signal
and confidence rules as Agents 2 and 3 (analysis/scoring.py), so the
Decision Agent can combine all three directly.

How the signal is made
----------------------
1. News. Agent 1's `news` table: PSX announcements and business-news
   headlines, each tagged company / sector / market. Only `informative`
   items are used: a "Notice of Board Meeting" gives FinBERT nothing to read.

2. Sentiment of each item (analysis/sentiment.py): FinBERT reads the
   headline and gives P(positive), P(negative), P(neutral). The item's
   sentiment is P(positive) - P(negative), from -1 to +1. Each headline is
   analysed once; the score is remembered.

3. Signal scores, from -1 (negative) to +1 (positive), one per signal in
   config.SENTIMENT_SIGNALS:
     company_recent     the company's own news, last 60 days
     company_long_term  the company's news over the last year
     sector             industry headlines, last 30 days
     market             PSX market headlines, last 14 days
   A signal's value is the recency-weighted average sentiment of its items
   (an item's weight halves every half-life, so recent news counts most, as
   the proposal plans). Its score is that average / config.SENTIMENT_SCALE,
   capped at +/-1. Its evidence (0-1) is the recency-weighted number of items
   / config.SENTIMENT_FULL_EVIDENCE: one fresh headline counts only partly.
   A signal with no items in its window is skipped.

4. Overall score = average of the signal scores, weighted by weight x
   evidence.
       score >= +T  -> BUY      score <= -T  -> SELL     otherwise HOLD
   with T = config.SENTIMENT_SIGNAL_THRESHOLD. The sentiment label is
   positive / neutral / negative accordingly.

5. Confidence (0-100), the same formula as Agents 2 and 3:
       confidence = 100 x coverage x (agreement + strength) / 2
   coverage is the share of the weight backed by evidence, so little or no
   news means low confidence (not a confident HOLD).

   Sentiment score (0-100) = 50 x (score + 1), for display.

Timing. PSX announcements appear the same day; the press usually reports the
next morning, after the price has moved. News sentiment is therefore a trend
over days and weeks, not a same-day trigger. `as_of` lets a backtest (Agent 7)
ask what the sentiment was on an earlier date, using only news published by
then.
"""

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from stockmind import config
from stockmind.agents.data_collection import CompanyData
from stockmind.analysis.scoring import (BUY, HOLD, SELL, clip, combine,
                                        signal_confidence, vote)
from stockmind.analysis.sentiment import SCORE_COLUMNS, SentimentScorer
from stockmind.data.news import NEWS_COLUMNS


SIGNAL_LABELS: dict[str, str] = {
    "company_recent": "Company news (recent)",
    "company_long_term": "Company news (past year)",
    "sector": "Sector news",
    "market": "Market news",
}

SIGNAL_COLUMNS = [
    "signal", "label", "items", "positive", "negative", "neutral",
    "value", "score", "evidence", "weight", "vote", "comment",
]

HISTORY_COLUMNS = ["month", "relevance", "items", "positive", "negative",
                   "neutral", "sentiment"]

SENTIMENT_LABEL = {BUY: "positive", SELL: "negative", HOLD: "neutral"}
DIRECTION = {BUY: "positive", SELL: "negative", HOLD: "neutral"}


@dataclass
class SentimentAnalysis:
    """Agent 4's result for one ticker."""

    ticker: str
    signal: str                 # "BUY", "HOLD" or "SELL"
    confidence: float           # 0-100
    score: float                # -1 (negative) to +1 (positive)
    sentiment_score: float      # 0-100
    sentiment: str              # "positive", "neutral" or "negative"
    period: str | None          # news used, e.g. "news to 2026-11-24"
    signals: pd.DataFrame       # one row per signal, see SIGNAL_COLUMNS
    articles: pd.DataFrame      # every scored news item (news + SCORE_COLUMNS)
    history: pd.DataFrame       # monthly sentiment by relevance (for charts)
    summary: str                # plain-English overview
    warnings: list[str] = field(default_factory=list)

    @property
    def explanation(self) -> list[str]:
        """One sentence per signal, for the report."""
        return [comment for comment in self.signals["comment"] if comment]


def recency_weights(ages_in_days: pd.Series, half_life: float) -> pd.Series:
    """1 for today's news, 0.5 after one half-life, 0.25 after two, ..."""
    return 0.5 ** (ages_in_days.clip(lower=0) / half_life)


def pakistan_now() -> datetime:
    """Current Pakistan time without a time zone, like Agent 1's dates."""
    return pd.Timestamp.now(tz=config.MARKET_TIMEZONE).tz_localize(None).to_pydatetime()


def _quote(title: str, limit: int = 90) -> str:
    title = " ".join(str(title).split())
    return f'"{title[:limit - 3]}..."' if len(title) > limit else f'"{title}"'


class SentimentAnalysisAgent:
    """Judge the tone of the news about a company, its sector and the market."""

    name = "Sentiment Analysis Agent"

    def __init__(self, scorer: SentimentScorer | None = None) -> None:
        # The scorer loads FinBERT only when a headline is new to it.
        self._scorer = scorer or SentimentScorer()

    def analyse(
        self, data: CompanyData, as_of: datetime | None = None
    ) -> SentimentAnalysis:
        """
        Analyse one company from Agent 1's output.

        Never raises for missing news or an unavailable model: signals
        without news are skipped, which lowers the confidence, and a warning
        explains why.
        """
        as_of = pd.Timestamp(as_of or pakistan_now())
        news = data.news
        if news is None or news.empty:
            return self._no_news(data.ticker, [
                f"{data.ticker}: no news available; sentiment analysis is "
                "not possible."])

        items = news[news["informative"].astype(bool)
                     & (news["published_at"] <= as_of)].reset_index(drop=True)
        scores, warnings = self._scorer.score(items["title"].tolist())
        warnings = [f"{data.ticker}: {warning}" for warning in warnings]
        articles = pd.concat([items, scores], axis=1)
        articles = articles[articles["sentiment"].notna()].reset_index(drop=True)
        if articles.empty:
            return self._no_news(data.ticker, warnings + [
                f"{data.ticker}: no informative news could be analysed."])

        signals = self._score_signals(articles, as_of)
        warnings += self._coverage_warnings(data.ticker, signals)
        score, coverage, agreement = combine(signals, config.SENTIMENT_SIGNAL_THRESHOLD)
        signal, confidence = signal_confidence(
            score, coverage, agreement,
            config.SENTIMENT_SIGNAL_THRESHOLD, config.SENTIMENT_FULL_STRENGTH)
        if coverage == 0:
            signal, confidence = HOLD, 0.0

        latest = articles["published_at"].max()
        result = SentimentAnalysis(
            ticker=data.ticker,
            signal=signal,
            confidence=confidence,
            score=round(score, 3),
            sentiment_score=round(50 * (score + 1), 1),
            sentiment=SENTIMENT_LABEL[signal],
            period=f"news to {latest:%Y-%m-%d}",
            signals=signals,
            articles=articles,
            history=self.monthly_history(articles),
            summary="",
            warnings=warnings,
        )
        result.summary = self._summarise(result, coverage)
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _score_signals(articles: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        """Score each signal from the items in its window."""
        ages = (as_of - articles["published_at"]) / pd.Timedelta(days=1)
        rows = []
        for name, (weight, relevance, window, half_life) in config.SENTIMENT_SIGNALS.items():
            label = SIGNAL_LABELS[name]
            chosen = (articles["relevance"] == relevance) & (ages <= window)
            group = articles[chosen]
            row = {"signal": name, "label": label, "items": len(group),
                   "positive": 0, "negative": 0, "neutral": 0, "value": None,
                   "score": None, "evidence": 0.0, "weight": weight, "vote": None}
            if group.empty:
                row["comment"] = f"{label}: no news in the last {window} days."
                rows.append(row)
                continue

            weights = recency_weights(ages[chosen], half_life)
            value = float((group["sentiment"] * weights).sum() / weights.sum())
            score = clip(value / config.SENTIMENT_SCALE)
            counts = group["sentiment_label"].value_counts()
            row.update({
                "positive": int(counts.get("positive", 0)),
                "negative": int(counts.get("negative", 0)),
                "neutral": int(counts.get("neutral", 0)),
                "value": round(value, 3),
                "score": round(score, 3),
                "evidence": round(min(1.0, weights.sum() / config.SENTIMENT_FULL_EVIDENCE), 3),
                "vote": vote(score, config.SENTIMENT_SIGNAL_THRESHOLD),
            })
            row["comment"] = SentimentAnalysisAgent._describe(row, window, group, weights)
            rows.append(row)
        return pd.DataFrame(rows, columns=SIGNAL_COLUMNS)

    @staticmethod
    def _describe(row: dict, window: int, group: pd.DataFrame, weights: pd.Series) -> str:
        """One sentence: counts, average, and the headline that weighed most."""
        count = row["items"]
        comment = (f"{row['label']}: {count} item{'s' if count != 1 else ''} in "
                   f"the last {window} days ({row['positive']} positive, "
                   f"{row['negative']} negative, {row['neutral']} neutral); "
                   f"recency-weighted sentiment {row['value']:+.2f} "
                   f"({DIRECTION[row['vote']]}).")
        # The headline that pushed the result most in its direction.
        pull = group["sentiment"] * weights
        if row["vote"] == BUY or (row["vote"] == HOLD and row["value"] >= 0):
            top, word = pull.idxmax(), "Most positive"
        else:
            top, word = pull.idxmin(), "Most negative"
        if abs(group.at[top, "sentiment"]) >= 0.5:
            comment += (f" {word}: {_quote(group.at[top, 'title'])} "
                        f"({group.at[top, 'published_at']:%Y-%m-%d}).")
        return comment

    @staticmethod
    def _coverage_warnings(ticker: str, signals: pd.DataFrame) -> list[str]:
        missing = signals.loc[signals["items"] == 0, "signal"].tolist()
        warnings = []
        headlines = [name for name in ("sector", "market") if name in missing]
        if headlines:
            warnings.append(
                f"{ticker}: no recent {' or '.join(headlines)} headlines. Agent 1 "
                "keeps headlines from the day it first runs, so run it regularly "
                "(at least daily) to build them up.")
        if "company_recent" in missing:
            warnings.append(
                f"{ticker}: no company news in the last "
                f"{config.SENTIMENT_SIGNALS['company_recent'][2]} days; the "
                "sentiment signal rests on older or wider news.")
        return warnings

    @staticmethod
    def monthly_history(articles: pd.DataFrame) -> pd.DataFrame:
        """Monthly sentiment for each relevance, oldest first (for charts)."""
        if articles.empty:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        table = articles.assign(
            month=articles["published_at"].dt.to_period("M").dt.to_timestamp())
        grouped = table.groupby(["month", "relevance"])
        history = grouped.agg(items=("sentiment", "size"),
                              sentiment=("sentiment", "mean")).reset_index()
        labels = (table.groupby(["month", "relevance"])["sentiment_label"]
                  .value_counts().unstack(fill_value=0)
                  .reindex(columns=["positive", "negative", "neutral"], fill_value=0)
                  .reset_index())
        history = history.merge(labels, on=["month", "relevance"])
        history["sentiment"] = history["sentiment"].round(3)
        return history[HISTORY_COLUMNS].sort_values(["month", "relevance"]).reset_index(drop=True)

    @staticmethod
    def _no_news(ticker: str, warnings: list[str]) -> SentimentAnalysis:
        result = SentimentAnalysis(
            ticker=ticker, signal=HOLD, confidence=0.0, score=0.0,
            sentiment_score=50.0, sentiment="neutral", period=None,
            signals=pd.DataFrame(columns=SIGNAL_COLUMNS),
            articles=pd.DataFrame(columns=[*NEWS_COLUMNS, *SCORE_COLUMNS]),
            history=pd.DataFrame(columns=HISTORY_COLUMNS),
            summary="", warnings=warnings,
        )
        result.summary = SentimentAnalysisAgent._summarise(result, coverage=0)
        return result

    @staticmethod
    def _summarise(result: SentimentAnalysis, coverage: float) -> str:
        if coverage == 0:
            return (f"{result.ticker}: not enough news for a sentiment "
                    "assessment. Signal HOLD with 0% confidence.")

        scored = result.signals[result.signals["score"].notna()]
        positive = scored.loc[scored["vote"] == BUY, "label"].tolist()
        negative = scored.loc[scored["vote"] == SELL, "label"].tolist()
        parts = [
            f"{result.ticker}'s news sentiment is {result.sentiment} "
            f"(sentiment score {result.sentiment_score:.0f}/100, "
            f"{result.period})."
        ]
        if positive:
            parts.append("Positive: " + ", ".join(positive) + ".")
        if negative:
            parts.append("Negative: " + ", ".join(negative) + ".")
        parts.append(f"Sentiment signal: {result.signal} with "
                     f"{result.confidence:.0f}% confidence.")
        return " ".join(parts)
