"""Tests for the Fundamental Analysis Agent.

Agent 1's output is built by hand from small made-up statements, so these
tests run offline and the expected ratios are easy to check.
"""

import pandas as pd
import pytest

from stockmind import config
from stockmind.agents.data_collection import CompanyData, SectorPeers
from stockmind.agents.fundamental_analysis import (
    BUY,
    HOLD,
    SELL,
    FundamentalAnalysisAgent,
    absolute_score,
    industry_score,
    signal_confidence,
)


PERIOD_END = "2026-09-30"


def company(net_income, revenue, eps, eps_growth, revenue_growth, equity,
            debt, book_value_per_share, period_end=PERIOD_END, bank=False):
    """Income statement and balance sheet with one TTM column."""
    income = {"revenue": revenue, "net_income": net_income, "eps_basic": eps,
              "eps_growth": eps_growth, "revenue_growth": revenue_growth}
    balance = {"shareholders_equity": equity, "total_debt": debt,
               "book_value_per_share": book_value_per_share}
    if bank:
        balance["total_deposits"] = equity * 10

    def long(values):
        return pd.DataFrame([
            {"fiscal_year": "TTM", "period_end": pd.Timestamp(period_end),
             "metric": metric, "value": value, "unit": ""}
            for metric, value in values.items()
        ])
    return long(income), long(balance)


# Price 150: P/E 5, ROE 30%, margin 30%, D/E 0.1, P/B 0.75
STRONG = company(net_income=300, revenue=1_000, eps=30.0, eps_growth=25.0,
                 revenue_growth=20.0, equity=1_000, debt=100,
                 book_value_per_share=200.0)
# Price 150: P/E 30, ROE 2%, margin 2%, D/E 2.5, P/B 5
WEAK = company(net_income=20, revenue=1_000, eps=5.0, eps_growth=-30.0,
               revenue_growth=-10.0, equity=1_000, debt=2_500,
               book_value_per_share=30.0)
# Price 150: P/E 15, ROE 10%, margin 10%, D/E 1.0, P/B 2
AVERAGE = company(net_income=100, revenue=1_000, eps=10.0, eps_growth=5.0,
                  revenue_growth=5.0, equity=1_000, debt=1_000,
                  book_value_per_share=75.0)


def make_data(statements, peers=None, price=150.0, dividends=None,
              ticker="OGDC") -> CompanyData:
    """
    Agent 1-style output. `peers` maps ticker -> (statements, price,
    dividend_yield); `dividends` is a list of (ex_date, cash_per_share).
    """
    income, balance = statements
    peers = peers or {}
    dividends = dividends if dividends is not None else [("2026-09-01", 15.0)]

    def stack(index):
        frames = [s[index].assign(ticker=t) for t, (s, _, _) in peers.items()]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return CompanyData(
        ticker=ticker,
        prices=pd.DataFrame({"date": pd.to_datetime(["2026-10-29", "2026-10-30"]),
                             "close": [149.0, price]}),
        price_issues=pd.DataFrame(),
        profile={},
        financials=pd.DataFrame(),
        ratios=pd.DataFrame(),
        income_statement=income,
        balance_sheet=balance,
        payouts=pd.DataFrame(),
        dividend_history=pd.DataFrame({
            "ex_date": pd.to_datetime([d for d, _ in dividends]),
            "cash_per_share": [c for _, c in dividends],
        }),
        peers=SectorPeers(
            sector="E&P",
            tickers=list(peers),
            market=pd.DataFrame({
                "symbol": list(peers),
                "price": [p for _, p, _ in peers.values()],
                "dividend_yield": [y for _, _, y in peers.values()],
            }),
            income_statement=stack(0),
            balance_sheet=stack(1),
        ),
    )


def metric(result, name):
    return result.metrics.set_index("metric").loc[name]


@pytest.fixture
def agent():
    return FundamentalAnalysisAgent()


AVERAGE_PEERS = {"MARI": (AVERAGE, 150.0, 4.0), "PPL": (AVERAGE, 150.0, 4.0),
                 "POL": (AVERAGE, 150.0, 4.0)}


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def test_strong_company_in_average_sector_is_a_buy(agent):
    result = agent.analyse(make_data(STRONG, AVERAGE_PEERS))

    assert result.signal == BUY
    assert result.confidence > 80
    assert result.health_score > 80
    assert result.peers == ["MARI", "PPL", "POL"]
    assert result.warnings == []


def test_weak_company_is_a_sell(agent):
    result = agent.analyse(make_data(WEAK, AVERAGE_PEERS, dividends=[]))

    assert result.signal == SELL
    assert result.score < -config.FUNDAMENTAL_SIGNAL_THRESHOLD
    assert "Weaknesses" in result.summary


def test_company_like_its_sector_is_a_hold(agent):
    result = agent.analyse(make_data(AVERAGE, AVERAGE_PEERS,
                                     dividends=[("2026-09-01", 7.5)]))

    assert result.signal == HOLD
    assert metric(result, "roe")["industry_score"] == 0


def test_ratios_and_peer_medians(agent):
    result = agent.analyse(make_data(STRONG, AVERAGE_PEERS))

    pe = metric(result, "pe_ratio")
    assert pe["value"] == pytest.approx(5.0)
    assert pe["peer_median"] == pytest.approx(15.0)
    assert pe["peer_count"] == 3
    assert metric(result, "roe")["value"] == pytest.approx(30.0)
    assert metric(result, "dividend_yield")["value"] == pytest.approx(10.0)
    assert metric(result, "dividend_yield")["peer_median"] == pytest.approx(4.0)


def test_peer_median_ignores_one_outlier(agent):
    peers = {**AVERAGE_PEERS, "XYZ": (company(
        net_income=1, revenue=1_000, eps=0.1, eps_growth=0, revenue_growth=0,
        equity=1_000, debt=0, book_value_per_share=75.0), 150.0, 0.0)}

    result = agent.analyse(make_data(STRONG, peers))

    # Peer P/Es are 15, 15, 15 and 1,500. The median stays 15.
    assert metric(result, "pe_ratio")["peer_median"] == pytest.approx(15.0)


def test_eps_is_shown_but_not_compared_with_peers(agent):
    result = agent.analyse(make_data(STRONG, AVERAGE_PEERS))

    eps = metric(result, "eps")
    assert eps["value"] == 30.0
    assert pd.isna(eps["peer_median"])
    assert eps["weight"] == 0


# ---------------------------------------------------------------------------
# Special cases
# ---------------------------------------------------------------------------

def test_loss_making_company_scores_lowest_on_pe(agent):
    loss = company(net_income=-50, revenue=1_000, eps=-5.0, eps_growth=-50.0,
                   revenue_growth=0.0, equity=1_000, debt=500,
                   book_value_per_share=75.0)

    result = agent.analyse(make_data(loss, AVERAGE_PEERS))

    pe = metric(result, "pe_ratio")
    assert pe["score"] == -1
    assert "no profit" in pe["comment"]


def test_banks_skip_debt_to_equity(agent):
    bank = company(net_income=300, revenue=1_000, eps=30.0, eps_growth=25.0,
                   revenue_growth=20.0, equity=1_000, debt=8_000,
                   book_value_per_share=200.0, bank=True)

    result = agent.analyse(make_data(bank, AVERAGE_PEERS))

    debt = metric(result, "debt_to_equity")
    assert debt["weight"] == 0
    assert "not used for banks" in debt["comment"]
    # A non-applicable ratio does not lower the confidence.
    assert result.signal == BUY
    assert result.confidence > 80


def test_without_enough_peers_absolute_benchmarks_are_used(agent):
    result = agent.analyse(make_data(STRONG, {"MARI": AVERAGE_PEERS["MARI"]}))

    assert result.signal == BUY
    assert metric(result, "roe")["industry_score"] is None
    assert metric(result, "roe")["score"] == metric(result, "roe")["absolute_score"]
    assert any("absolute benchmarks only" in w for w in result.warnings)


def test_peers_with_old_statements_are_left_out(agent):
    stale = company(net_income=100, revenue=1_000, eps=10.0, eps_growth=5.0,
                    revenue_growth=5.0, equity=1_000, debt=1_000,
                    book_value_per_share=75.0, period_end="2024-06-30")
    peers = {**AVERAGE_PEERS, "KEL": (stale, 5.0, 0.0)}

    result = agent.analyse(make_data(STRONG, peers))

    assert "KEL" not in result.peers
    assert any("KEL left out" in w for w in result.warnings)


def test_missing_ratios_lower_the_confidence(agent):
    income, balance = STRONG
    full = agent.analyse(make_data(STRONG, AVERAGE_PEERS))
    partial = agent.analyse(make_data((income, balance.iloc[0:0]), AVERAGE_PEERS))

    assert partial.signal == BUY
    assert partial.confidence < full.confidence
    assert "could not be calculated" in metric(partial, "roe")["comment"]


def test_no_statements_gives_hold_with_zero_confidence(agent):
    empty = (pd.DataFrame(), pd.DataFrame())
    result = agent.analyse(make_data(empty, dividends=[]))

    assert (result.signal, result.confidence) == (HOLD, 0.0)
    assert any("no financial statements" in w for w in result.warnings)
    assert "not enough financial data" in result.summary


def test_every_ratio_is_explained(agent):
    result = agent.analyse(make_data(STRONG, AVERAGE_PEERS))

    assert len(result.explanation) == len(result.metrics)
    assert "sector median" in metric(result, "roe")["comment"]
    assert "Fundamental signal: BUY" in result.summary


# ---------------------------------------------------------------------------
# Scoring rules
# ---------------------------------------------------------------------------

def test_weights_add_up_to_one():
    total = sum(weight for weight, _, _ in config.FUNDAMENTAL_METRICS.values())
    assert total == pytest.approx(1.0)


@pytest.mark.parametrize(
    "value, bad, good, expected",
    [(25, 5, 25, 1), (5, 5, 25, -1), (15, 5, 25, 0), (100, 5, 25, 1),
     # lower is better: P/E with bad 15, good 5
     (5, 15, 5, 1), (10, 15, 5, 0), (30, 15, 5, -1)],
)
def test_absolute_score(value, bad, good, expected):
    assert absolute_score(value, bad, good) == pytest.approx(expected)


def test_industry_score_is_full_at_half_the_span():
    # ROE span is 20 points: 10 points above the median is +1.
    assert industry_score(30, 20, 5, 25) == pytest.approx(1)
    assert industry_score(15, 20, 5, 25) == pytest.approx(-0.5)
    # P/E: lower than the median is better.
    assert industry_score(5, 10, 15, 5) == pytest.approx(1)


@pytest.mark.parametrize(
    "score, coverage, agreement, expected",
    [(0.6, 1.0, 1.0, (BUY, 100.0)),
     (0.3, 1.0, 1.0, (BUY, 75.0)),    # strength 0.3 / 0.6 = 0.5
     (-0.6, 0.5, 1.0, (SELL, 50.0)),  # half the data missing
     (0.0, 1.0, 0.5, (HOLD, 75.0)),   # clearest HOLD, half the ratios agree
     (0.1, 1.0, 1.0, (HOLD, 75.0))],  # strength 1 - 0.1 / 0.2 = 0.5
)
def test_signal_confidence(score, coverage, agreement, expected):
    assert signal_confidence(score, coverage, agreement) == expected
