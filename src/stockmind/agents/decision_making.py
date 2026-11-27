"""Agent 5 - Decision Making Agent.

Combines the signals of the Technical (Agent 2), Fundamental (Agent 3) and
Sentiment (Agent 4) Analysis Agents into one final Buy / Hold / Sell
decision with a confidence and a plain-English explanation. It answers
"taking everything together, what should be done?".

It only combines the three results; it fetches and calculates nothing
else, so it runs instantly and can be tested offline. All three agents
share one result shape (signal, score -1 to +1, confidence 0-100), so they
can be combined directly.

How the decision is made
------------------------
1. Weights. Each agent has a base weight (config.DECISION_AGENT_WEIGHTS,
   rescaled to add up to 1): how much its view counts in general. Agent 7
   (Learning) can later pass weights learnt from backtests instead.

2. Each agent's weight is scaled by its own confidence:
       effective weight = base weight x confidence / 100
   so an agent that is only 15% sure barely moves the decision, and an
   agent with no result, or 0% confidence, does not move it at all.

3. Decision score = average of the agents' scores, weighted by their
   effective weights. Only agents with evidence take part, so a missing
   agent does not pull the score towards zero.
       score >= +T  -> BUY      score <= -T  -> SELL     otherwise HOLD
   with T = config.DECISION_SIGNAL_THRESHOLD, the same rule as Agents 2-4.

4. Confidence (0-100), deliberately conservative. HOLD lies between BUY
   and SELL, so each agent's effective weight counts by how far its signal
   is from the decision:
       same signal       x 1     supports
       one step away     x P     partly supports (HOLD for a BUY or SELL
                                 decision; BUY or SELL for a HOLD decision)
       opposite signal   x -1    opposes (BUY against SELL)
       confidence = 100 x (sum of those) / total base weight
   with P = config.DECISION_PARTIAL_SUPPORT (0.5). A decision just inside
   the HOLD zone between two BUYs is therefore partly supported, not 0%.
   The total base weight includes missing agents. Unlike the score (step
   3), confidence is NOT rescaled when an agent is missing: an agent with
   no result counts as no support, so a missing 25% agent caps confidence
   at 75%. No agent counts more than its own effective weight, so the
   decision is never more confident than the weighted confidence of the
   agents that are there; conflicts lower it, and below 0 it becomes 0.

   Decision score (0-100) = 50 x (score + 1), like the agents' technical,
   health and sentiment scores. It shows the DIRECTION of the combined view
   (0 strongly bearish, 50 neutral, 100 strongly bullish; BUY from 60, SELL
   from 40 with the default threshold), not how much to trust it - that is
   the confidence. A score of 80 at 20% confidence is a bullish view with
   little support behind it.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from stockmind import config
from stockmind.agents.data_collection import CompanyData
from stockmind.agents.fundamental_analysis import (FundamentalAnalysis,
                                                   FundamentalAnalysisAgent)
from stockmind.agents.sentiment_analysis import (SentimentAnalysis,
                                                 SentimentAnalysisAgent)
from stockmind.agents.technical_analysis import (TechnicalAnalysis,
                                                 TechnicalAnalysisAgent)
from stockmind.analysis.scoring import BUY, HOLD, SELL, vote

logger = logging.getLogger(__name__)

AGENTS = ("technical", "fundamental", "sentiment")

AGENT_LABELS: dict[str, str] = {
    "technical": "Technical Analysis",
    "fundamental": "Fundamental Analysis",
    "sentiment": "Sentiment Analysis",
}

CONTRIBUTION_COLUMNS = [
    "agent", "label", "status", "signal", "confidence", "score", "period",
    "weight", "effective_weight", "share", "relation", "reason", "comment",
]

AnalysisResult = TechnicalAnalysis | FundamentalAnalysis | SentimentAnalysis


@dataclass
class Decision:
    """Agent 5's result for one ticker."""

    ticker: str
    signal: str                 # "BUY", "HOLD" or "SELL"
    confidence: float           # 0-100: how much to trust the decision
    score: float                # -1 (bearish) to +1 (bullish)
    decision_score: float       # 0-100: direction of the combined view
    agreement: str              # "agree", "mixed", "conflict" or "none"
    contributions: pd.DataFrame  # one row per agent, see CONTRIBUTION_COLUMNS
    weights: dict[str, float]   # base weights used (add up to 1)
    technical: TechnicalAnalysis | None
    fundamental: FundamentalAnalysis | None
    sentiment: SentimentAnalysis | None
    summary: str                # plain-English overview
    warnings: list[str] = field(default_factory=list)

    @property
    def explanation(self) -> list[str]:
        """One sentence per agent, for the report."""
        return [comment for comment in self.contributions["comment"] if comment]


def normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    """Check the base weights and rescale them to add up to 1."""
    if set(weights) != set(AGENTS):
        raise ValueError(f"weights must be given for exactly {', '.join(AGENTS)}; "
                         f"got {', '.join(sorted(weights)) or 'none'}")
    if any(not weight >= 0 for weight in weights.values()):
        raise ValueError("weights must be 0 or more")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("at least one weight must be above 0")
    return {name: weights[name] / total for name in AGENTS}


def relation(agent_signal: str, decision: str) -> str:
    """How an agent's signal relates to the decision."""
    if agent_signal == decision:
        return "supports"
    if {agent_signal, decision} == {BUY, SELL}:
        return "opposes"
    return "partly supports"


# How much of an agent's effective weight counts towards the confidence.
SUPPORT = {"supports": 1.0, "opposes": -1.0}


def strongest_reason(result: AnalysisResult) -> str | None:
    """
    The agent's own sentence that pushed hardest towards its BUY or SELL
    signal: the signal (or ratio) with the largest |score| x weight x
    evidence among those voting the same way. None for a HOLD.
    """
    if result.signal == HOLD:
        return None
    table = result.metrics if isinstance(result, FundamentalAnalysis) else result.signals
    table = table[(table["vote"] == result.signal) & table["score"].notna()]
    if table.empty:
        return None
    evidence = table["evidence"].astype(float) if "evidence" in table else 1.0
    pull = table["score"].astype(float).abs() * table["weight"].astype(float) * evidence
    return table.at[pull.idxmax(), "comment"]


class DecisionMakingAgent:
    """Combine Agents 2-4 into one Buy / Hold / Sell decision."""

    name = "Decision Making Agent"

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        technical: TechnicalAnalysisAgent | None = None,
        fundamental: FundamentalAnalysisAgent | None = None,
        sentiment: SentimentAnalysisAgent | None = None,
    ) -> None:
        self.weights = normalise_weights(weights or config.DECISION_AGENT_WEIGHTS)
        if not 0 <= config.DECISION_PARTIAL_SUPPORT <= 1:
            # Above 1 an agent could add more than its own confidence.
            raise ValueError("config.DECISION_PARTIAL_SUPPORT must be between 0 and 1")
        self._agents = {
            "technical": technical or TechnicalAnalysisAgent(),
            "fundamental": fundamental or FundamentalAnalysisAgent(),
            "sentiment": sentiment or SentimentAnalysisAgent(),
        }

    def decide(self, data: CompanyData) -> Decision:
        """
        Run Agents 2-4 on Agent 1's output and combine their results.

        An agent that fails does not stop the decision: it counts as
        missing (no support, see the module docstring) and a warning says
        why.
        """
        results: dict[str, AnalysisResult | None] = {}
        failures = []
        for name, agent in self._agents.items():
            try:
                results[name] = agent.analyse(data)
            except Exception as exc:  # one agent must not stop the decision
                logger.exception("%s failed for %s", agent.name, data.ticker)
                results[name] = None
                failures.append(f"{data.ticker}: the {AGENT_LABELS[name]} Agent "
                                f"failed ({type(exc).__name__}: {exc}).")
        decision = self.combine(**results, ticker=data.ticker)
        decision.warnings = failures + decision.warnings
        return decision

    def combine(
        self,
        technical: TechnicalAnalysis | None = None,
        fundamental: FundamentalAnalysis | None = None,
        sentiment: SentimentAnalysis | None = None,
        ticker: str | None = None,
    ) -> Decision:
        """
        Combine results that have already been calculated. Any of them may
        be None (missing). `ticker` is needed only when all three are.
        """
        results = {"technical": technical, "fundamental": fundamental,
                   "sentiment": sentiment}
        tickers = {result.ticker for result in results.values() if result is not None}
        if ticker is not None:
            tickers.add(ticker)
        if len(tickers) != 1:
            raise ValueError("results must all be for one ticker; got "
                             + (", ".join(sorted(tickers)) or "no ticker"))
        ticker = tickers.pop()

        rows = [self._contribution(name, results[name]) for name in AGENTS]
        table = pd.DataFrame(rows, columns=CONTRIBUTION_COLUMNS)
        used = table[table["status"] == "used"]
        used_weight = float(used["effective_weight"].sum())

        if used_weight == 0:
            score, signal, confidence = 0.0, HOLD, 0.0
        else:
            score = float((used["score"].astype(float) * used["effective_weight"]).sum()
                          / used_weight)
            signal = vote(score, config.DECISION_SIGNAL_THRESHOLD)
            table.loc[used.index, "share"] = (used["effective_weight"] / used_weight).round(3)
            table.loc[used.index, "relation"] = [relation(s, signal) for s in used["signal"]]
            factor = table.loc[used.index, "relation"].map(
                lambda r: SUPPORT.get(r, config.DECISION_PARTIAL_SUPPORT))
            net = float((factor * used["effective_weight"]).sum())
            # sum(self.weights) is 1, so this is the share of ALL the weight,
            # missing agents included (see the module docstring).
            confidence = round(100 * max(0.0, net) / sum(self.weights.values()), 1)

        table["effective_weight"] = table["effective_weight"].round(3)
        table["comment"] = [self._describe(row, signal) for row in table.itertuples()]
        decision = Decision(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            score=round(score, 3),
            decision_score=round(50 * (score + 1), 1),
            agreement=self._agreement(table),
            contributions=table,
            weights=dict(self.weights),
            technical=technical,
            fundamental=fundamental,
            sentiment=sentiment,
            summary="",
            warnings=self._warnings(ticker, table, results),
        )
        decision.summary = self._summarise(decision)
        return decision

    # ------------------------------------------------------------------

    def _contribution(self, name: str, result: AnalysisResult | None) -> dict:
        weight = self.weights[name]
        row = {"agent": name, "label": AGENT_LABELS[name], "weight": round(weight, 3),
               "status": "missing", "signal": None, "confidence": None, "score": None,
               "period": None, "effective_weight": 0.0, "share": 0.0,
               "relation": "not used", "reason": None, "comment": ""}
        if result is None:
            return row
        row.update({
            "signal": result.signal, "confidence": result.confidence,
            "score": result.score, "period": result.period,
            "effective_weight": weight * result.confidence / 100,
            "reason": strongest_reason(result),
        })
        # 0% confidence (e.g. no prices, no statements, no news): present but
        # it carries no evidence, so it is treated like a missing agent.
        if weight == 0:
            row["status"] = "weight 0"
        elif result.confidence > 0:
            row["status"] = "used"
        else:
            row["status"] = "no evidence"
        return row

    @staticmethod
    def _describe(row, decision: str) -> str:
        """One sentence: what the agent said and how it bears on the decision."""
        if row.status == "missing":
            return (f"{row.label}: no result, so its {row.weight:.0%} weight "
                    "counts as no support.")
        said = (f"{row.label}: {row.signal} with {row.confidence:.0f}% confidence "
                f"(score {row.score:+.2f}, weight {row.weight:.0%})")
        if row.status == "no evidence":
            return f"{said}; no evidence behind it, so it is not used."
        if row.status == "weight 0":
            return f"{said}; its weight is 0, so it is not used."
        effect = {"supports": f"supports the {decision}",
                  "opposes": f"goes against the {decision}",
                  "partly supports": f"partly supports the {decision}"}[row.relation]
        sentence = f"{said}; {effect} ({row.share:.0%} of the decision)."
        if isinstance(row.reason, str) and row.reason:  # missing is NaN in a table
            sentence += f" Strongest point: {row.reason}"
        return sentence

    @staticmethod
    def _agreement(table: pd.DataFrame) -> str:
        votes = set(table.loc[table["status"] == "used", "signal"])
        if not votes:
            return "none"
        if {BUY, SELL} <= votes:
            return "conflict"
        return "agree" if len(votes) == 1 else "mixed"

    @staticmethod
    def _warnings(ticker: str, table: pd.DataFrame, results: dict) -> list[str]:
        """Decision-level warnings, then each agent's own (without repeats)."""
        warnings = []
        unused = table[table["status"].isin(["missing", "no evidence"])]
        if not unused.empty and (table["status"] == "used").any():
            labels = " and ".join(unused["label"])
            cap = 100 * (1 - unused["weight"].sum())
            warnings.append(f"{ticker}: no usable result from {labels}; the decision "
                            f"rests on the other agents and its confidence is at "
                            f"most {cap:.0f}%.")
        used = table[table["status"] == "used"]
        buyers = used.loc[used["signal"] == BUY, "label"].tolist()
        sellers = used.loc[used["signal"] == SELL, "label"].tolist()
        if buyers and sellers:
            warnings.append(f"{ticker}: the agents conflict: {' and '.join(buyers)} "
                            f"{'say' if len(buyers) > 1 else 'says'} BUY, "
                            f"{' and '.join(sellers)} "
                            f"{'say' if len(sellers) > 1 else 'says'} SELL; "
                            "the decision is less certain.")
        for name in AGENTS:
            if results[name] is not None:
                warnings += results[name].warnings
        return list(dict.fromkeys(warnings))

    @staticmethod
    def _summarise(decision: Decision) -> str:
        table = decision.contributions
        if decision.agreement == "none":
            return (f"{decision.ticker}: none of the analysis agents had usable "
                    "evidence, so there is no basis for a decision. Decision HOLD "
                    "with 0% confidence.")

        def names(relation_: str) -> str:
            return ", ".join(table.loc[table["relation"] == relation_, "label"])

        parts = [f"{decision.ticker}: {decision.signal} with "
                 f"{decision.confidence:.0f}% confidence (decision score "
                 f"{decision.decision_score:.0f}/100)."]
        used = table[table["status"] == "used"]
        if decision.agreement == "agree" and len(used) > 1:
            everyone = "Both agents" if len(used) == 2 else f"All {len(used)} agents"
            parts.append(f"{everyone} with evidence say {decision.signal}.")
        elif decision.agreement != "agree":
            if decision.agreement == "conflict":
                parts.append("The agents conflict.")
            for word, relation_ in (("Supports", "supports"),
                                    ("Partly supports", "partly supports"),
                                    ("Against", "opposes")):
                if names(relation_):
                    parts.append(f"{word}: {names(relation_)}.")
        unused = table.loc[table["status"].isin(["missing", "no evidence"]),
                           "label"].tolist()
        if unused:
            parts.append("No usable result from " + " and ".join(unused) + ".")
        return " ".join(parts)
