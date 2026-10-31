"""Financial ratios from Agent 1's statements.

The Fundamental Agent needs the same ratios for the company and for each of
its sector peers, so they are calculated here by one set of functions that
work on any company's statements.

Which figures are used
    Income statement and balance sheet both come from stockanalysis.com, so
    they are consolidated and consistent with each other (see README).
    The most recent period is used: "TTM" (last twelve months) when the
    company has published quarterly results since its year end, otherwise
    the latest fiscal year. The balance sheet is taken at the same date.

    Ratio            Formula
    ---------------  ---------------------------------------------------
    eps              basic EPS, PKR
    pe_ratio         price / EPS (missing if EPS <= 0: loss-making)
    price_to_book    price / book value per share
    roe              net income to common / common equity, %
    eps_growth       EPS vs the previous 12 months, %
    revenue_growth   revenue vs the previous 12 months, %
    revenue_cagr_3y  compound annual revenue growth over 3 fiscal years, %
    net_margin       net income / revenue, %
    debt_to_equity   total debt / shareholders' equity, x (not for banks)
    dividend_yield   cash dividends of the last 12 months / price, %

A figure that is missing at the source stays missing (None). It is never
replaced by zero, so a missing ratio cannot look like a bad one.
"""

import math

import pandas as pd


RATIO_NAMES = (
    "eps", "pe_ratio", "price_to_book", "roe", "eps_growth",
    "revenue_growth", "revenue_cagr_3y", "net_margin", "debt_to_equity",
    "dividend_yield",
)


def to_number(value) -> float | None:
    """float, or None for missing / NaN / pandas NA / non-numeric values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _divide(numerator: float | None, denominator: float | None,
            scale: float = 1) -> float | None:
    """numerator / denominator, or None if either is missing or the
    denominator is not positive."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator * scale


def _dated(statement: pd.DataFrame) -> pd.DataFrame:
    if statement.empty or "period_end" not in statement:
        return statement.iloc[0:0]
    return statement[statement["period_end"].notna()]


def latest_period(statement: pd.DataFrame) -> tuple[str | None, pd.Timestamp | None]:
    """The fiscal-year label and end date of the most recent column."""
    dated = _dated(statement)
    if dated.empty:
        return None, None
    row = dated.loc[dated["period_end"].idxmax()]
    return row["fiscal_year"], row["period_end"]


def values_at(statement: pd.DataFrame, period_end: pd.Timestamp,
              on_or_before: bool = False) -> dict[str, float]:
    """
    metric -> value for one column of a long-format statement.

    With on_or_before=True the latest column ending on or before the date is
    used, e.g. when the balance sheet has no column for that exact date.
    """
    dated = _dated(statement)
    if on_or_before:
        earlier = dated.loc[dated["period_end"] <= period_end, "period_end"]
        if earlier.empty:
            return {}
        period_end = earlier.max()
    rows = dated[dated["period_end"] == period_end].drop_duplicates("metric")
    return {metric: value for metric, value in zip(rows["metric"], rows["value"])
            if to_number(value) is not None}


def _first(values: dict[str, float], *metrics: str) -> float | None:
    """The first of several alternative line items that is present."""
    for metric in metrics:
        if metric in values:
            return float(values[metric])
    return None


def revenue_cagr(income_statement: pd.DataFrame, years: int = 3) -> float | None:
    """
    Compound annual revenue growth over `years` fiscal years, in %.

    Uses fiscal-year columns only (not TTM). Missing if there is no revenue
    figure exactly `years` years before the latest one, or if either is <= 0.
    """
    dated = _dated(income_statement)
    revenue = dated[
        (dated["metric"] == "revenue")
        & dated["fiscal_year"].astype(str).str.startswith("FY")
    ]
    if revenue.empty:
        return None

    by_year = dict(zip(revenue["period_end"].dt.year, revenue["value"]))
    latest_year = max(by_year)
    end = to_number(by_year[latest_year])
    start = to_number(by_year.get(latest_year - years))
    if start is None or end is None or start <= 0 or end <= 0:
        return None
    return ((end / start) ** (1 / years) - 1) * 100


def trailing_dividend_yield(
    dividend_history: pd.DataFrame,
    price: float | None,
    as_of: pd.Timestamp | None,
) -> float | None:
    """
    Cash dividends with an ex-date in the 12 months up to `as_of`, as a
    percentage of the price.

    Returns 0 for a company with a dividend history but no dividend in the
    last 12 months, and None if there is no history at all. Announced
    dividends whose ex-date is still in the future are not counted.
    """
    if dividend_history.empty or not price or as_of is None:
        return None
    ex_dates = dividend_history["ex_date"]
    window = dividend_history[
        (ex_dates <= as_of) & (ex_dates > as_of - pd.DateOffset(years=1))
    ]
    return float(window["cash_per_share"].sum()) / price * 100


def compute_ratios(
    income_statement: pd.DataFrame,
    balance_sheet: pd.DataFrame,
    price: float | None,
    dividend_yield: float | None = None,
) -> dict:
    """
    Ratios for one company from its latest statements.

    Args:
        income_statement, balance_sheet: Agent 1's long-format statements.
        price: Current share price in PKR.
        dividend_yield: Already-calculated dividend yield in %, if known.
            (stockanalysis.com's TTM dividend per share can be incomplete,
            so the yield comes from the dividend history or PSX instead.)

    Returns:
        {"period": "TTM", "period_end": Timestamp, "is_bank": bool,
         "eps": ..., "pe_ratio": ..., ...} with None for missing ratios.
    """
    period, period_end = latest_period(income_statement)
    ratios = {"period": period, "period_end": period_end, "is_bank": False}
    ratios.update(dict.fromkeys(RATIO_NAMES))
    ratios["dividend_yield"] = dividend_yield
    if period_end is None:
        return ratios

    income = values_at(income_statement, period_end)
    balance = values_at(balance_sheet, period_end, on_or_before=True)

    # Banks report deposits; for them borrowing is the business itself, so
    # debt-to-equity says nothing about financial health.
    ratios["is_bank"] = "total_deposits" in balance

    net_income = _first(income, "net_income_to_common", "net_income")
    revenue = _first(income, "revenue")
    eps = _first(income, "eps_basic", "eps_diluted")
    common_equity = _first(balance, "total_common_equity", "shareholders_equity")
    equity = _first(balance, "shareholders_equity", "total_common_equity")

    ratios["eps"] = eps
    ratios["pe_ratio"] = _divide(price, eps)
    ratios["price_to_book"] = _divide(price, _first(balance, "book_value_per_share"))
    ratios["roe"] = _divide(net_income, common_equity, 100)
    ratios["eps_growth"] = _first(income, "eps_growth")
    ratios["revenue_growth"] = _first(income, "revenue_growth")
    ratios["revenue_cagr_3y"] = revenue_cagr(income_statement)
    ratios["net_margin"] = _divide(net_income, revenue, 100)
    if not ratios["is_bank"]:
        ratios["debt_to_equity"] = _divide(_first(balance, "total_debt"), equity)

    return ratios
