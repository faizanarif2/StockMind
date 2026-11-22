"""Tests for the bonus-share price adjustment."""

import pandas as pd

from stockmind.analysis.price_adjustments import adjust_for_bonus, bonus_events


# Mon 3 Nov - Fri 14 Nov 2025, trading days only.
DAYS = pd.bdate_range("2025-11-03", "2025-11-14")
# A 10% bonus: price falls from 110 to 100 on the ex-date (Fri 7 Nov).
PRICES = pd.DataFrame({
    "date": DAYS,
    "open": [110.0] * 4 + [100.0] * 6,
    "high": [111.0] * 4 + [101.0] * 6,
    "low": [109.0] * 4 + [99.0] * 6,
    "close": [110.0] * 4 + [100.0] * 6,
    "volume": [1000] * 10,
})


def payouts(bonus_pct=10.0, closure="2025-11-10", right_pct=None):
    # Book closure starts Mon 10 Nov, so the ex-date is Fri 7 Nov.
    return pd.DataFrame({
        "details": ["10% (B)"],
        "bonus_pct": [bonus_pct],
        "right_pct": [right_pct],
        "book_closure_start": [pd.Timestamp(closure)],
    })


def test_ex_date_is_last_trading_day_before_book_closure():
    events = bonus_events(payouts(), PRICES["date"])

    assert events["ex_date"].item() == pd.Timestamp("2025-11-07")
    assert events["factor"].item() == 1.1


def test_prices_before_ex_date_are_divided_by_the_factor():
    adjusted, notes, warnings = adjust_for_bonus(PRICES, payouts())

    assert adjusted["close"].round(6).tolist() == [100.0] * 10  # no fake drop
    assert adjusted.loc[0, "volume"] == 1100
    assert adjusted.loc[9, "volume"] == 1000
    assert "10% bonus" in notes[0]
    assert warnings == []


def test_future_book_closure_is_not_applied_yet():
    # Book closure two weeks after our last price: not ex-bonus yet.
    adjusted, notes, _ = adjust_for_bonus(PRICES, payouts(closure="2025-11-28"))

    assert adjusted["close"].tolist() == PRICES["close"].tolist()
    assert notes == []


def test_cash_dividends_and_empty_payouts_change_nothing():
    cash_only = payouts(bonus_pct=None)
    for table in (cash_only, pd.DataFrame()):
        adjusted, notes, _ = adjust_for_bonus(PRICES, table)
        assert adjusted["close"].tolist() == PRICES["close"].tolist()
        assert notes == []


def test_right_issue_gives_a_warning():
    _, _, warnings = adjust_for_bonus(PRICES, payouts(bonus_pct=None, right_pct=25.0))
    assert "right issue" in warnings[0]
