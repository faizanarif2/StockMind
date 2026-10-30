"""Tests for stockanalysis.com parsing, using small saved HTML snippets."""

import pandas as pd

from stockmind.data.stockanalysis import parse_dividend_history, parse_statement


STATEMENT_HTML = """
<table>
  <tr><th>Fiscal Year</th><th>TTM</th><th>FY 2025</th></tr>
  <tr><td>Period Ending</td><td>Jun '26 Jun 30, 2026</td><td>Dec '25 Dec 31, 2025</td></tr>
  <tr><td>Net Income</td><td>42,672</td><td>55,633</td></tr>
  <tr><td>Net Income Growth</td><td>5.15%</td><td>-</td></tr>
  <tr><td>Shares Outstanding (Basic)</td><td>1,204</td><td>1,204</td></tr>
  <tr><td>EPS (Basic)</td><td>35.45</td><td>46.20</td></tr>
</table>
<table>
  <tr><th>Fiscal Year</th><th>TTM</th><th>FY 2025</th></tr>
  <tr><td>Period Ending</td><td>Jun '26 Jun 30, 2026</td><td>Dec '25 Dec 31, 2025</td></tr>
  <tr><td>Book Value Per Share</td><td>251.70</td><td>240.10</td></tr>
</table>
<table><tr><th>Unrelated</th></tr><tr><td>ignored</td></tr></table>
"""

DIVIDEND_HTML = """
<table>
  <tr><th>Ex-Dividend Date</th><th>Cash Amount</th><th>Record Date</th><th>Pay Date</th></tr>
  <tr><td>Oct 8, 2026</td><td>6.000 PKR</td><td>Oct 8, 2026</td><td>Oct 29, 2026</td></tr>
  <tr><td>May 11, 2026</td><td>3.250 PKR</td><td>May 11, 2026</td><td>May 22, 2026</td></tr>
</table>
"""


def _value(df, year, metric):
    return df.query("fiscal_year == @year and metric == @metric")


def test_amounts_are_converted_from_millions_to_thousands():
    statement = parse_statement(STATEMENT_HTML)
    row = _value(statement, "FY 2025", "net_income")

    assert row["value"].item() == 55_633_000
    assert row["unit"].item() == "PKR thousands"
    assert row["period_end"].item() == pd.Timestamp("2025-12-31")


def test_per_share_percent_and_share_count_units():
    statement = parse_statement(STATEMENT_HTML)

    eps = _value(statement, "TTM", "eps_basic")
    assert (eps["value"].item(), eps["unit"].item()) == (35.45, "PKR per share")

    growth = _value(statement, "TTM", "net_income_growth")
    assert (growth["value"].item(), growth["unit"].item()) == (5.15, "%")

    shares = _value(statement, "FY 2025", "shares_outstanding_basic")
    assert (shares["value"].item(), shares["unit"].item()) == (1_204_000_000, "shares")

    bvps = _value(statement, "FY 2025", "book_value_per_share")
    assert bvps["unit"].item() == "PKR per share"


def test_dashes_are_missing_not_zero():
    statement = parse_statement(STATEMENT_HTML)
    assert _value(statement, "FY 2025", "net_income_growth").empty


def test_dividend_history():
    history = parse_dividend_history(DIVIDEND_HTML)

    assert list(history["cash_per_share"]) == [6.0, 3.25]
    assert history["ex_date"].iloc[0] == pd.Timestamp("2026-10-08")
    assert history["pay_date"].iloc[1] == pd.Timestamp("2026-05-22")


def test_pages_without_tables_give_empty_frames():
    assert parse_statement("<html></html>").empty
    assert parse_dividend_history("<html></html>").empty
