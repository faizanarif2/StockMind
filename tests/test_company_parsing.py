"""Tests for PSX company-page parsing, using small saved HTML snippets."""

import pandas as pd
import pytest

from stockmind.data.company import (
    add_cash_per_share,
    parse_financials,
    parse_number,
    parse_payout_details,
    parse_payouts,
    parse_profile,
    parse_ratios,
)


COMPANY_HTML = """
<div class="quote__details">
  <div class="quote__name">The Hub Power Company Limited<span>XD</span></div>
  <div class="quote__sector">POWER GENERATION &amp; DISTRIBUTION</div>
  <div class="quote__close">Rs.193.18</div>
  <div class="stats">
    <div class="stats_item"><div class="stats_label">LDCP</div><div class="stats_value">194.04</div></div>
    <div class="stats_item"><div class="stats_label">P/E Ratio (TTM) **</div><div class="stats_value">7.10</div></div>
    <div class="stats_item"><div class="stats_label">52-WEEK RANGE ^</div><div class="stats_value">120.50 — 210.00</div></div>
    <div class="stats_item"><div class="stats_label">1-Year Change * ^</div><div class="stats_value">(4.50%)</div></div>
  </div>
  <div class="stats">
    <div class="stats_item"><div class="stats_label">LDCP</div><div class="stats_value">999.00</div></div>
  </div>
</div>
<div id="profile">
  <div class="profile__item"><div class="item__head">BUSINESS DESCRIPTION</div><p>Power producer.</p></div>
  <div class="profile__item"><div class="item__head">Fiscal Year End</div><p>June</p></div>
</div>
<div id="equity">
  <div class="stats_item"><div class="stats_label">Market Cap (000's)</div><div class="stats_value">250,584,286.99</div></div>
  <div class="stats_item"><div class="stats_label">Shares</div><div class="stats_value">1,297,154,400</div></div>
  <div class="stats_item"><div class="stats_label">Face Value</div><div class="stats_value">10.00</div></div>
  <div class="stats_item"><div class="stats_label">Free Float</div><div class="stats_value">972,865,790</div></div>
  <div class="stats_item"><div class="stats_label">Free Float</div><div class="stats_value">75.00%</div></div>
</div>
<div id="financials">
  <div class="tabs__panel" data-name="Annual"><table>
    <tr><th></th><th>2026</th><th>2025</th></tr>
    <tr><td>Sales</td><td></td><td>13,209,513</td></tr>
    <tr><td>Profit after Taxation</td><td>24,467,399</td><td>19,079,478</td></tr>
    <tr><td>EPS</td><td>18.86</td><td>14.71</td></tr>
  </table></div>
  <div class="tabs__panel" data-name="Quarterly"><table>
    <tr><th></th><th>Q1 2027</th></tr>
    <tr><td>EPS</td><td>6.26</td></tr>
  </table></div>
</div>
<div id="ratios"><table>
  <tr><th></th><th>2026</th><th>2025</th></tr>
  <tr><td>Net Profit Margin (%)</td><td></td><td>144.44</td></tr>
  <tr><td>EPS Growth (%)</td><td>28.21</td><td>(43.68)</td></tr>
</table></div>
"""

PAYOUTS_HTML = """
<table class="tbl">
  <tr><th>Date</th><th>Financial Results</th><th>Details</th><th>Book Closure</th></tr>
  <tr><td>September 8, 2026 12:51 PM</td><td>30/06/2026(YR)</td>
      <td> 60% (F) (D) </td><td>09/10/2026  - 16/10/2026 </td></tr>
  <tr><td>April 29, 2026 3:56 PM</td><td>31/03/2026(IIIQ)</td>
      <td> 32.50%(iii) (D) </td><td>12/05/2026  - 13/05/2026 </td></tr>
</table>
"""


@pytest.mark.parametrize(
    "text, expected",
    [("1,234.50", 1234.5), ("(18.71)", -18.71), ("15.00%", 15.0),
     ("Rs.306.78", 306.78), ("", None), ("-", None), (None, None)],
)
def test_parse_number(text, expected):
    assert parse_number(text) == expected


def test_profile_extracts_company_equity_and_market_data():
    profile = parse_profile(COMPANY_HTML, "HUBC")

    assert profile["name"] == "The Hub Power Company Limited"  # no "XD"
    assert profile["fiscal_year_end"] == "June"
    assert profile["equity"] == {
        "market_cap_thousands": 250584286.99,
        "shares_outstanding": 1297154400.0,
        "face_value": 10.0,
        "free_float_shares": 972865790.0,
        "free_float_pct": 75.0,
    }
    market = profile["market"]
    assert market["price"] == 193.18
    assert market["last_close"] == 194.04  # first (regular market) block wins
    assert market["pe_ratio_ttm"] == 7.10
    assert (market["week_52_low"], market["week_52_high"]) == (120.5, 210.0)
    assert market["change_1y_pct"] == -4.5


def test_profile_without_company_name_is_rejected():
    with pytest.raises(ValueError):
        parse_profile("<html><body>No record found</body></html>", "XYZ")


def test_financials_are_long_format_and_skip_blank_cells():
    financials = parse_financials(COMPANY_HTML)

    annual = financials[financials["period_type"] == "annual"]
    # 2026 sales is blank on PSX, so it must be missing - not zero.
    assert not ((annual["period"] == "2026") & (annual["metric"] == "sales")).any()
    eps_2026 = annual.query("period == '2026' and metric == 'eps'")
    assert eps_2026["value"].item() == 18.86
    assert eps_2026["unit"].item() == "PKR per share"
    assert set(financials["period_type"]) == {"annual", "quarterly"}


def test_ratios_mark_percentages_and_negative_values():
    ratios = parse_ratios(COMPANY_HTML)

    growth = ratios.query("period == '2025' and metric == 'eps_growth_pct'")
    assert growth["value"].item() == -43.68
    assert len(ratios) == 3  # blank 2026 margin skipped


@pytest.mark.parametrize(
    "details, cash, bonus, final",
    [("60% (F) (D)", 60.0, None, True),
     ("32.50%(iii) (D)", 32.5, None, False),
     ("20% (D) 10% (B)", 20.0, 10.0, False)],
)
def test_parse_payout_details(details, cash, bonus, final):
    parsed = parse_payout_details(details)
    assert parsed["cash_dividend_pct"] == cash
    assert parsed["bonus_pct"] == bonus
    assert parsed["is_final"] is final


def test_payouts_dates_and_periods():
    first = parse_payouts(PAYOUTS_HTML).iloc[0]

    assert first["cash_dividend_pct"] == 60.0
    assert first["period_end"] == pd.Timestamp("2026-06-30")
    assert first["period_label"] == "YR"
    assert first["book_closure_start"] == pd.Timestamp("2026-10-09")
    assert first["book_closure_end"] == pd.Timestamp("2026-10-16")


def test_cash_per_share_uses_face_value():
    payouts = add_cash_per_share(parse_payouts(PAYOUTS_HTML), face_value=10.0)
    assert list(payouts["cash_dividend_per_share"]) == [6.0, 3.25]


def test_cash_per_share_is_not_guessed_without_face_value():
    payouts = add_cash_per_share(parse_payouts(PAYOUTS_HTML), face_value=None)
    assert payouts["cash_dividend_per_share"].isna().all()


def test_empty_payouts_table():
    html = "<table><tr><th>Date</th></tr></table>"
    assert parse_payouts(html).empty
