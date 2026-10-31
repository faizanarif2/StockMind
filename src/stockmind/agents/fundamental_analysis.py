"""Agent 3 - Fundamental Analysis Agent.

Judges the financial health of a company from the data Agent 1 collected,
compares it with the company's sector, and returns a Buy / Hold / Sell
signal with a confidence and a plain-English explanation for Agent 5.

It only calculates - it never fetches anything - so it runs in milliseconds
and can be tested offline.

How the signal is made
----------------------
1. Ratios. The same ratios are calculated for the company and for each
   sector peer from their consolidated statements (see analysis/ratios.py):
   P/E, EPS growth, ROE, revenue growth, net margin, debt-to-equity,
   price-to-book, and dividend yield.

2. Industry median. For each ratio, the median of the peers' values is the
   industry benchmark. The median is used, not the mean, because one
   outlier (e.g. a P/E of 500 when earnings are tiny) would distort a mean.
   Peers whose statements are too old are left out.

3. Ratio scores, from -1 (weak) to +1 (strong). Each ratio gets an
   absolute score from fixed benchmarks and an industry score from its gap
   to the sector median; its score is the average of the two. Benchmarks
   and weights are in config.FUNDAMENTAL_METRICS. A loss-making company
   (EPS <= 0) gets -1 for P/E.

4. Overall score = weighted average of the ratio scores.
       score >= +T  -> BUY      score <= -T  -> SELL     otherwise HOLD
   with T = config.FUNDAMENTAL_SIGNAL_THRESHOLD. Each ratio also "votes"
   BUY / HOLD / SELL with the same rule.

5. Confidence (0-100), defined so it is reproducible:
       confidence = 100 x coverage x (agreement + strength) / 2
   coverage   share of the applicable weight whose ratio could be
              calculated (missing data lowers confidence)
   agreement  share of that weight whose ratio votes the same way as the
              overall signal
   strength   for BUY / SELL: |score| / config.FUNDAMENTAL_FULL_STRENGTH,
              capped at 1; for HOLD: 1 - |score| / T (a score of exactly 0
              is the clearest HOLD)

   Health score (0-100) = 50 x (score + 1), for display.
"""

from dataclasses import dataclass, field
from statistics import median

import pandas as pd

from stockmind import config
from stockmind.agents.data_collection import CompanyData
from stockmind.analysis.ratios import (
    compute_ratios,
    latest_period,
    to_number,
    trailing_dividend_yield,
)


BUY, HOLD, SELL = "BUY", "HOLD", "SELL"

# How each ratio is shown: label and unit. Ratios not in
# config.FUNDAMENTAL_METRICS (EPS, 3-year revenue growth) are shown for
# context but not scored.
METRIC_LABELS: dict[str, tuple[str, str]] = {
    "pe_ratio": ("P/E ratio", "x"),
    "eps": ("EPS", "PKR"),
    "eps_growth": ("EPS growth", "%"),
    "roe": ("Return on equity", "%"),
    "revenue_growth": ("Revenue growth", "%"),
    "revenue_cagr_3y": ("Revenue growth (3-year average)", "%"),
    "net_margin": ("Net profit margin", "%"),
    "debt_to_equity": ("Debt-to-equity", "x"),
    "price_to_book": ("Price-to-book", "x"),
    "dividend_yield": ("Dividend yield", "%"),
}

# EPS depends on how many shares a company has issued, so comparing it with
# other companies' EPS means nothing. Its growth is compared instead.
NOT_COMPARED_WITH_PEERS = {"eps"}

METRIC_COLUMNS = [
    "metric", "label", "value", "unit", "peer_median", "peer_count",
    "absolute_score", "industry_score", "score", "weight", "vote", "comment",
]


@dataclass
class FundamentalAnalysis:
    """Agent 3's result for one ticker."""

    ticker: str
    signal: str                # "BUY", "HOLD" or "SELL"
    confidence: float          # 0-100
    score: float               # -1 (weak) to +1 (strong)
    health_score: float        # 0-100
    period: str | None         # statements used, e.g. "TTM to 2026-09-30"
    sector: str | None
    peers: list[str]           # peers used for the industry medians
    metrics: pd.DataFrame      # one row per ratio, see METRIC_COLUMNS
    peer_ratios: pd.DataFrame  # the same ratios for every peer
    summary: str               # plain-English overview
    warnings: list[str] = field(default_factory=list)

    @property
    def explanation(self) -> list[str]:
        """One sentence per ratio, for the report."""
        return [comment for comment in self.metrics["comment"] if comment]


def _fmt(value: float, unit: str) -> str:
    if unit == "x":
        return f"{value:.2f}x"
    if unit == "%":
        return f"{value:.1f}%"
    return f"PKR {value:,.2f}"


def _clip(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _vote(score: float) -> str:
    threshold = config.FUNDAMENTAL_SIGNAL_THRESHOLD
    if score >= threshold:
        return BUY
    if score <= -threshold:
        return SELL
    return HOLD


def absolute_score(value: float, bad: float, good: float) -> float:
    """-1 at `bad`, +1 at `good`, linear in between, capped at the ends."""
    return _clip(-1 + 2 * (value - bad) / (good - bad))


def industry_score(value: float, peer_median: float, bad: float, good: float) -> float:
    """+1 when better than the median by half the bad-to-good span."""
    return _clip(2 * (value - peer_median) / (good - bad))


def signal_confidence(
    score: float, coverage: float, agreement: float
) -> tuple[str, float]:
    """The overall signal and its confidence (see module docstring)."""
    signal = _vote(score)
    threshold = config.FUNDAMENTAL_SIGNAL_THRESHOLD
    if signal == HOLD:
        strength = 1 - abs(score) / threshold
    else:
        strength = min(1.0, abs(score) / config.FUNDAMENTAL_FULL_STRENGTH)
    return signal, round(100 * coverage * (agreement + strength) / 2, 1)


class FundamentalAnalysisAgent:
    """Score a company's financial health against its sector."""

    name = "Fundamental Analysis Agent"

    def analyse(self, data: CompanyData) -> FundamentalAnalysis:
        """
        Analyse one company from Agent 1's output.

        Never raises for missing data: ratios that cannot be calculated are
        skipped, which lowers the confidence, and a warning explains why.
        """
        warnings = []
        price, as_of = self._latest_price(data)

        company = compute_ratios(
            data.income_statement, data.balance_sheet, price,
            trailing_dividend_yield(data.dividend_history, price, as_of),
        )
        if company["period_end"] is None:
            warnings.append(
                f"{data.ticker}: no financial statements available; "
                "fundamental analysis is not possible."
            )

        peer_ratios, peer_warnings = self._peer_ratios(data, company["period_end"])
        warnings += peer_warnings
        usable = peer_ratios["ticker"].tolist() if not peer_ratios.empty else []
        if len(usable) < config.MIN_PEERS_FOR_COMPARISON:
            warnings.append(
                f"{data.ticker}: fewer than {config.MIN_PEERS_FOR_COMPARISON} "
                "sector peers with recent statements; ratios are judged on "
                "absolute benchmarks only."
            )

        metrics = self._score_metrics(company, peer_ratios)
        score, coverage, agreement = self._combine(metrics)
        signal, confidence = signal_confidence(score, coverage, agreement)
        if coverage == 0:
            signal, confidence = HOLD, 0.0

        period = (f"{company['period']} to {company['period_end']:%Y-%m-%d}"
                  if company["period_end"] is not None else None)
        result = FundamentalAnalysis(
            ticker=data.ticker,
            signal=signal,
            confidence=confidence,
            score=round(score, 3),
            health_score=round(50 * (score + 1), 1),
            period=period,
            sector=data.peers.sector,
            peers=usable,
            metrics=metrics,
            peer_ratios=peer_ratios,
            summary="",
            warnings=warnings,
        )
        result.summary = self._summarise(result, coverage)
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _latest_price(data: CompanyData) -> tuple[float | None, pd.Timestamp | None]:
        """Latest validated close and its date (prices are sorted by date)."""
        if data.prices.empty:
            return None, None
        last = data.prices.iloc[-1]
        return to_number(last["close"]), last["date"]

    @staticmethod
    def _peer_ratios(
        data: CompanyData, company_period_end: pd.Timestamp | None
    ) -> tuple[pd.DataFrame, list[str]]:
        """The same ratios for every peer with recent statements."""
        peers, warnings, rows = data.peers, [], []
        if peers.income_statement.empty or peers.market.empty:
            return pd.DataFrame(), warnings

        market = peers.market.set_index("symbol")
        for ticker in peers.tickers:
            income = peers.income_statement[peers.income_statement["ticker"] == ticker]
            balance = (peers.balance_sheet[peers.balance_sheet["ticker"] == ticker]
                       if not peers.balance_sheet.empty else peers.balance_sheet)
            _, period_end = latest_period(income)
            if period_end is None:
                continue
            if (company_period_end is not None and
                    period_end < company_period_end - config.PEER_MAX_STATEMENT_LAG):
                warnings.append(
                    f"{ticker} left out of the sector comparison: its latest "
                    f"statements end {period_end:%Y-%m-%d}."
                )
                continue

            listing = market.loc[ticker] if ticker in market.index else {}
            ratios = compute_ratios(
                income, balance,
                price=to_number(listing.get("price")),
                # PSX's own trailing yield; peers' dividend histories are
                # not collected.
                dividend_yield=to_number(listing.get("dividend_yield")),
            )
            rows.append({"ticker": ticker, **ratios})

        return pd.DataFrame(rows), warnings

    @staticmethod
    def _score_metrics(company: dict, peer_ratios: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for metric, (label, unit) in METRIC_LABELS.items():
            value = company[metric]
            weight, bad, good = config.FUNDAMENTAL_METRICS.get(metric, (0.0, None, None))
            peer_values = (
                [v for v in map(to_number, peer_ratios[metric]) if v is not None]
                if not peer_ratios.empty and metric not in NOT_COMPARED_WITH_PEERS
                else []
            )
            peer_median = (median(peer_values)
                           if len(peer_values) >= config.MIN_PEERS_FOR_COMPARISON
                           else None)

            row = {"metric": metric, "label": label, "value": value, "unit": unit,
                   "peer_median": peer_median, "peer_count": len(peer_values),
                   "absolute_score": None, "industry_score": None, "score": None,
                   "weight": weight, "vote": None}

            if metric == "debt_to_equity" and company["is_bank"]:
                row["weight"] = 0.0
                row["comment"] = (f"{label} is not used for banks: deposits are "
                                  "their business, not a sign of risky borrowing.")
            elif metric == "pe_ratio" and company["eps"] is not None and company["eps"] <= 0:
                # No profit: P/E is meaningless, and that is a weakness.
                row.update(absolute_score=-1.0, score=-1.0, vote=SELL,
                           comment=f"{label} is not meaningful: the company has "
                                   "no profit (EPS of zero or below).")
            elif value is None:
                row["comment"] = f"{label} could not be calculated (data not available)."
            elif weight == 0:
                row["comment"] = FundamentalAnalysisAgent._describe(
                    label, unit, value, peer_median)
            else:
                abs_score = absolute_score(value, bad, good)
                ind_score = (industry_score(value, peer_median, bad, good)
                             if peer_median is not None else None)
                score = abs_score if ind_score is None else (abs_score + ind_score) / 2
                row.update(absolute_score=round(abs_score, 3),
                           industry_score=None if ind_score is None else round(ind_score, 3),
                           score=round(score, 3), vote=_vote(score))
                row["comment"] = FundamentalAnalysisAgent._describe(
                    label, unit, value, peer_median, ind_score, abs_score)
            rows.append(row)

        return pd.DataFrame(rows, columns=METRIC_COLUMNS)

    @staticmethod
    def _describe(label, unit, value, peer_median,
                  ind_score=None, abs_score=None) -> str:
        text = f"{label} is {_fmt(value, unit)}"
        if peer_median is not None:
            position = ("above" if value > peer_median else
                        "below" if value < peer_median else "equal to")
            text += f", {position} the sector median of {_fmt(peer_median, unit)}"
            if ind_score is not None:
                text += {BUY: " (better)", SELL: " (worse)",
                         HOLD: " (similar)"}[_vote(ind_score)]
        if abs_score is not None:
            judgement = {BUY: "strong", SELL: "weak", HOLD: "average"}[_vote(abs_score)]
            text += f"; {judgement} on its own"
        return text + "."

    @staticmethod
    def _combine(metrics: pd.DataFrame) -> tuple[float, float, float]:
        """Weighted score, coverage, and agreement (see module docstring)."""
        applicable = metrics[metrics["weight"] > 0]
        scored = applicable[applicable["score"].notna()]
        total_weight = applicable["weight"].sum()
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
    def _summarise(result: FundamentalAnalysis, coverage: float) -> str:
        if coverage == 0:
            return (f"{result.ticker}: not enough financial data for a "
                    "fundamental assessment. Signal HOLD with 0% confidence.")

        health = {BUY: "strong", SELL: "weak", HOLD: "mixed"}[result.signal]
        scored = result.metrics[result.metrics["score"].notna()]
        strengths = scored.loc[scored["vote"] == BUY, "label"].tolist()
        weaknesses = scored.loc[scored["vote"] == SELL, "label"].tolist()

        parts = [
            f"{result.ticker}'s fundamentals look {health} "
            f"(health score {result.health_score:.0f}/100, {result.period})."
        ]
        if strengths:
            parts.append("Strengths: " + ", ".join(strengths) + ".")
        if weaknesses:
            parts.append("Weaknesses: " + ", ".join(weaknesses) + ".")
        if result.peers:
            parts.append(f"Compared with {', '.join(result.peers)} "
                         f"({result.sector}).")
        parts.append(f"Fundamental signal: {result.signal} with "
                     f"{result.confidence:.0f}% confidence.")
        return " ".join(parts)
