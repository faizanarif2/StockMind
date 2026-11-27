"""Tests for the financial ratio calculations, using small made-up statements."""

import pandas as pd
import pytest

from stockmind.analysis.ratios import (
    compute_ratios,
    revenue_cagr,
    trailing_dividend_yield,
)


def statement(columns: dict[str, tuple[str, dict]]) -> pd.DataFrame:
    """Long-format statement like Agent 1's: {label: (period_end, {metric: value})}."""
    return pd.DataFrame([
        {"fiscal_year": label, "period_end": pd.Timestamp(end),
         "metric": metric, "value": value, "unit": ""}
        for label, (end, values) in columns.items()
        for metric, value in values.items()
    ])


INCOME = statement({
    "TTM": ("2026-09-30", {"revenue": 1_000, "net_income": 150,
                           "net_income_to_common": 140, "eps_basic": 14.0,
                           "eps_growth": 12.0, "revenue_growth": 8.0}),
    "FY 2025": ("2025-12-31", {"revenue": 900, "net_income": 120, "eps_basic": 12.0}),
    "FY 2024": ("2024-12-31", {"revenue": 800}),
    "FY 2023": ("2023-12-31", {"revenue": 700}),
    "FY 2022": ("2022-12-31", {"revenue": 600}),
})
BALANCE = statement({
    "TTM": ("2026-09-30", {"shareholders_equity": 1_100, "total_common_equity": 1_000,
                           "total_debt": 550, "book_value_per_share": 100.0}),
    "FY 2025": ("2025-12-31", {"shareholders_equity": 900, "total_debt": 100,
                               "book_value_per_share": 90.0}),
})


def test_latest_period_and_matching_balance_sheet_are_used():
    ratios = compute_ratios(INCOME, BALANCE, price=140.0)

    assert ratios["period"] == "TTM"
    assert ratios["period_end"] == pd.Timestamp("2026-09-30")
    assert ratios["eps"] == 14.0
    assert ratios["pe_ratio"] == pytest.approx(10.0)
    assert ratios["price_to_book"] == pytest.approx(1.4)
    assert ratios["eps_growth"] == 12.0
    assert ratios["revenue_growth"] == 8.0


def test_roe_uses_profit_and_equity_of_ordinary_shareholders():
    # 140 to common / 1,000 common equity (minority interest excluded)
    assert compute_ratios(INCOME, BALANCE, 140.0)["roe"] == pytest.approx(14.0)


def test_net_margin_and_debt_to_equity():
    ratios = compute_ratios(INCOME, BALANCE, 140.0)

    assert ratios["net_margin"] == pytest.approx(14.0)
    assert ratios["debt_to_equity"] == pytest.approx(0.5)  # 550 / 1,100
    assert ratios["is_bank"] is False


def test_balance_sheet_falls_back_to_latest_earlier_date():
    balance = BALANCE[BALANCE["fiscal_year"] == "FY 2025"]
    ratios = compute_ratios(INCOME, balance, 140.0)

    assert ratios["roe"] == pytest.approx(140 / 900 * 100)


def test_loss_making_company_has_no_pe():
    income = INCOME.copy()
    income.loc[income["metric"] == "eps_basic", "value"] = -3.0

    assert compute_ratios(income, BALANCE, 140.0)["pe_ratio"] is None


def test_banks_have_no_debt_to_equity():
    balance = pd.concat([BALANCE, statement(
        {"TTM": ("2026-09-30", {"total_deposits": 9_000})})])

    ratios = compute_ratios(INCOME, balance, 140.0)

    assert ratios["is_bank"] is True
    assert ratios["debt_to_equity"] is None
    assert ratios["roe"] is not None


def test_missing_figures_stay_missing():
    ratios = compute_ratios(INCOME, BALANCE.iloc[0:0], price=None)

    assert ratios["roe"] is None
    assert ratios["pe_ratio"] is None
    assert ratios["debt_to_equity"] is None
    assert ratios["eps"] == 14.0


def test_no_statements_gives_no_period():
    ratios = compute_ratios(pd.DataFrame(), pd.DataFrame(), 100.0)
    assert ratios["period_end"] is None
    assert ratios["roe"] is None


def test_revenue_cagr_uses_fiscal_years_three_years_apart():
    # FY 2025 900 vs FY 2022 600 -> (1.5 ** (1/3) - 1); TTM is ignored.
    assert revenue_cagr(INCOME) == pytest.approx((1.5 ** (1 / 3) - 1) * 100)


def test_revenue_cagr_needs_the_year_three_years_back():
    income = INCOME[INCOME["fiscal_year"] != "FY 2022"]
    assert revenue_cagr(income) is None


DIVIDENDS = pd.DataFrame({
    "ex_date": pd.to_datetime(["2026-12-01", "2026-09-01", "2026-03-01", "2025-09-01"]),
    "cash_per_share": [99.0, 4.0, 6.0, 5.0],
})


def test_trailing_dividend_yield_counts_last_twelve_months_only():
    # 2026-09-01 and 2026-03-01 count; the announced 2026-12-01 and the
    # 2025-09-01 dividend do not.
    result = trailing_dividend_yield(DIVIDENDS, 100.0, pd.Timestamp("2026-10-30"))
    assert result == pytest.approx(10.0)


def test_trailing_dividend_yield_is_zero_after_dividends_stop():
    result = trailing_dividend_yield(DIVIDENDS, 100.0, pd.Timestamp("2028-01-01"))
    assert result == 0.0


def test_trailing_dividend_yield_is_missing_without_history():
    empty = DIVIDENDS.iloc[0:0]
    assert trailing_dividend_yield(empty, 100.0, pd.Timestamp("2026-10-30")) is None
