"""Tests for the Market Watch reader (offline, with saved HTML)."""

from datetime import datetime

import pandas as pd

from stockmind.data.live_prices import LiveSnapshot, overlay_bar, parse_market_watch
from stockmind.data.market_clock import PKT, MarketState


def cell(value) -> str:
    return f'<td class="right" data-order="{value}">{value}</td>'


def row(symbol, ldcp, o, h, low, c, volume, badge="") -> str:
    return (f'<tr><td data-order="{symbol}"><a><strong>{symbol}</strong></a>{badge}</td>'
            f'<td>0820</td><td>KMI30</td>' + cell(ldcp) + cell(o) + cell(h) + cell(low)
            + cell(c) + cell(round(c - ldcp, 2)) + cell(0) + cell(volume) + '</tr>')


HTML = ("<table><thead><tr><th>SYMBOL</th></tr></thead><tbody>"
        + row("OGDCXD", 312.79, 313.6, 315.48, 311.91, 312.55, 1245401)
        + row("PASL", 11.82, 10.64, 11.50, 10.64, 10.64, 90168474, badge=" <span>NC</span>")
        + row("BROKEN", 10, 12, 11, 9, 10, 5)  # high below open: inconsistent
        + "</tbody></table>")

AS_OF = datetime(2026, 11, 23, 11, 0, tzinfo=PKT)


def snapshot(state="Open", as_of=AS_OF, volume=1245401 + 90168474 + 5) -> LiveSnapshot:
    return LiveSnapshot(
        prices=parse_market_watch(HTML),
        state=MarketState(state, as_of, volume, 1000, AS_OF),
        fetched_at=AS_OF,
    )


def test_market_watch_is_parsed_from_data_attributes():
    table = parse_market_watch(HTML).set_index("symbol")

    assert list(table.index) == ["OGDCXD", "PASL", "BROKEN"]  # badge ignored
    assert table.loc["OGDCXD", "close"] == 312.55
    assert table.loc["PASL", "volume"] == 90168474


def test_bar_matches_entitlement_suffix_and_uses_the_as_of_day():
    bar = snapshot().bar("OGDC")
    assert bar == {"date": pd.Timestamp("2026-11-23"), "open": 313.6, "high": 315.48,
                   "low": 311.91, "close": 312.55, "volume": 1245401}


def test_inconsistent_or_missing_bars_are_not_used():
    assert snapshot().bar("BROKEN") is None
    assert snapshot().bar("NOPE") is None


def test_a_consistent_recent_snapshot_is_accepted():
    assert snapshot().problem(AS_OF.replace(minute=10)) is None


def test_a_stale_as_of_time_is_rejected_while_trading():
    problem = snapshot().problem(AS_OF.replace(minute=30))
    assert "older than 20 minutes" in problem


def test_an_old_as_of_time_is_fine_after_the_close():
    assert snapshot(state="Closed").problem(AS_OF.replace(hour=18)) is None


def test_a_volume_mismatch_is_rejected():
    problem = snapshot(volume=50_000_000).problem(AS_OF)
    assert "differs from PSX's regular-market volume" in problem


def test_no_as_of_time_is_rejected():
    assert "no 'As of' time" in snapshot(as_of=None).problem(AS_OF)


def test_overlay_replaces_or_adds_todays_row():
    history = pd.DataFrame({
        "date": pd.to_datetime(["2026-11-20", "2026-11-23"]),
        "open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0],
        "close": [1.0, 2.0], "volume": [10, 20], "is_valid": [True, True],
    })
    bar = snapshot().bar("OGDC")

    result = overlay_bar(history, bar)

    assert len(result) == 2
    assert result.iloc[-1]["close"] == 312.55 and bool(result.iloc[-1]["is_valid"])
    assert len(overlay_bar(history.iloc[:1], bar)) == 2  # added when missing
