"""Adjust historical prices for bonus shares.

Why: PSX prices are not adjusted. When a company issues a 10% bonus, every
shareholder gets 1 extra share per 10, so the price drops by about 1/1.1
overnight even though nobody lost money. Indicators would read that drop as
a crash. Adjusting divides every price before the ex-date by 1.1 (and
multiplies volume by 1.1), so the history is comparable with today's price.

Ex-date: the first day a buyer no longer gets the bonus. On PSX it is the
last trading day before book closure starts (checked against published
ex-dates for FFC, MEBL, OGDC and HUBC).

Bonus history comes from PSX payouts, which cover about the last 2 years.
That is enough for the technical indicators, which look back at most
200 trading days. Cash dividends are not adjusted, as is standard for
technical analysis. Right issues are not adjusted (that needs the
subscription price) and produce a warning instead.
"""

import pandas as pd


# Book closure can start a few calendar days after the ex-date (weekends,
# holidays). A larger gap means the ex-date has not been reached yet.
MAX_EX_DATE_GAP = pd.Timedelta(days=6)


def bonus_events(payouts: pd.DataFrame, trading_days: pd.Series) -> pd.DataFrame:
    """
    Bonus issues with their ex-date and adjustment factor.

    Columns: ex_date, bonus_pct, factor, details
    Only events whose ex-date falls within the price history are returned.
    """
    columns = ["ex_date", "bonus_pct", "factor", "details"]
    if payouts.empty or "bonus_pct" not in payouts.columns:
        return pd.DataFrame(columns=columns)

    days = pd.Series(sorted(pd.to_datetime(trading_days).dropna().unique()))
    records = []
    for _, row in payouts.iterrows():
        bonus = pd.to_numeric(row.get("bonus_pct"), errors="coerce")
        closure = row.get("book_closure_start")
        if pd.isna(bonus) or bonus <= 0 or pd.isna(closure):
            continue
        before = days[days < closure]
        if before.empty or closure - before.iloc[-1] > MAX_EX_DATE_GAP:
            continue  # before our history, or not ex yet
        records.append({
            "ex_date": before.iloc[-1],
            "bonus_pct": float(bonus),
            "factor": 1 + float(bonus) / 100,
            "details": row.get("details"),
        })

    return pd.DataFrame(records, columns=columns)


def adjust_for_bonus(
    prices: pd.DataFrame, payouts: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Return (adjusted prices, notes on adjustments made, warnings).

    Prices on and after an ex-date are unchanged; earlier prices are divided
    by the bonus factor and volumes multiplied by it.
    """
    adjusted = prices.copy()
    notes, warnings = [], []

    events = bonus_events(payouts, prices["date"])
    for _, event in events.iterrows():
        earlier = adjusted["date"] < event["ex_date"]
        for column in ("open", "high", "low", "close"):
            if column in adjusted.columns:
                adjusted.loc[earlier, column] = adjusted.loc[earlier, column] / event["factor"]
        if "volume" in adjusted.columns:
            adjusted["volume"] = adjusted["volume"].astype(float)
            adjusted.loc[earlier, "volume"] = adjusted.loc[earlier, "volume"] * event["factor"]
        notes.append(
            f"Prices before {event['ex_date']:%Y-%m-%d} adjusted for a "
            f"{event['bonus_pct']:g}% bonus issue (divided by {event['factor']:g})."
        )

    if not payouts.empty and "right_pct" in payouts.columns:
        rights = pd.to_numeric(payouts["right_pct"], errors="coerce")
        if (rights > 0).any():
            warnings.append(
                "A right issue was announced; prices are not adjusted for it, "
                "so indicators around its ex-date may be distorted."
            )

    return adjusted, notes, warnings
