"""Tests for the market clock (market-aware freshness).

All times are fixed Pakistan times. 2026-11-20 is a Friday, 2026-11-23 a
Monday. PSX's live state is faked, so these tests are offline.
"""

from datetime import date, datetime, timedelta

import pytest

from stockmind.data.market_clock import (
    PKT,
    MarketClock,
    MarketState,
    parse_market_state,
)


def at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, minute, tzinfo=PKT)


class FakeState:
    """PSX's market state as a function; counts reads. word=None: unreadable."""

    def __init__(self, word=None, as_of=None):
        self.word, self.as_of, self.reads = word, as_of, 0

    def __call__(self):
        self.reads += 1
        if self.word is None:
            raise ConnectionError("PSX unreachable")
        return MarketState(self.word, self.as_of, 1e6, 1e4, self.as_of or at(23, 9))


def clock_at(now: datetime, state: FakeState | None = None) -> MarketClock:
    return MarketClock(fetch_state=state or FakeState(), now=lambda: now)


# -- schedule ---------------------------------------------------------------

def test_weekends_have_no_window_and_friday_runs_later():
    assert MarketClock.window(date(2026, 11, 21)) is None  # Saturday
    assert MarketClock.window(date(2026, 11, 20))[1] == at(20, 17, 15)
    assert MarketClock.window(date(2026, 11, 23))[1] == at(23, 16, 15)


def test_last_window_end_skips_the_weekend():
    assert clock_at(at(23, 8)).last_window_end() == at(20, 17, 15)


def test_day_is_final_once_its_window_has_ended():
    clock = clock_at(at(23, 11))
    assert not clock.day_is_final(date(2026, 11, 23))
    assert clock.day_is_final(date(2026, 11, 20))
    assert clock_at(at(23, 16, 30)).day_is_final(date(2026, 11, 23))


# -- freshness --------------------------------------------------------------

def test_nothing_is_fetched_overnight_or_at_the_weekend():
    state = FakeState("Open")
    assert clock_at(at(24, 8), state).is_fresh(at(23, 16, 30))   # Tue morning
    assert clock_at(at(22, 12), state).is_fresh(at(20, 17, 20))  # Sunday
    assert state.reads == 0  # no request to PSX outside the schedule


def test_data_from_before_a_session_ended_is_refreshed_after_it():
    assert not clock_at(at(23, 16, 20)).is_fresh(at(23, 15, 0))


def test_refreshes_every_interval_while_trading():
    state = FakeState("Open", as_of=at(23, 10, 15))
    assert clock_at(at(23, 10, 10), state).is_fresh(at(23, 10, 0))
    assert not clock_at(at(23, 10, 20), state).is_fresh(at(23, 10, 0))


def test_first_check_of_a_trading_day_refreshes():
    state = FakeState("Open", as_of=at(23, 9, 40))
    assert not clock_at(at(23, 9, 45), state).is_fresh(at(20, 17, 20))


def test_holiday_needs_no_refresh():
    # Closed all day; PSX's last update is still Friday's close.
    state = FakeState("Closed", as_of=at(20, 16, 50))
    clock = clock_at(at(23, 11), state)
    assert clock.is_fresh(at(20, 17, 20))
    assert clock.is_fresh(at(20, 17, 20))
    assert state.reads == 1  # the state is re-read at most every 5 minutes


def test_closing_prices_are_fetched_once_after_the_close():
    state = FakeState("Closed", as_of=at(23, 15, 45))
    assert not clock_at(at(23, 15, 50), state).is_fresh(at(23, 15, 20))
    assert clock_at(at(23, 16, 5), state).is_fresh(at(23, 15, 46))


def test_unreadable_state_counts_as_trading():
    # Refresh rather than risk serving stale data.
    assert not clock_at(at(23, 11), FakeState(None)).is_fresh(at(23, 10, 30))


def test_without_interval_only_session_ends_matter():
    clock = clock_at(at(23, 11), FakeState("Open", as_of=at(23, 11)))
    assert clock.is_fresh(at(23, 8), interval=None)
    assert not clock.is_fresh(at(20, 12), interval=None)  # before Friday's end


def test_naive_times_are_read_as_this_computers_local_time():
    fetched = at(23, 10, 0).astimezone().replace(tzinfo=None)  # naive local
    state = FakeState("Open", as_of=at(23, 10, 5))
    assert clock_at(at(23, 10, 10), state).is_fresh(fetched)
    assert not clock_at(at(23, 10, 20), state).is_fresh(fetched)


# -- PSX market state -------------------------------------------------------

HOMEPAGE = """
<div class="glide__slide" data-key="REG"><a class="markets__item">
  <div class="markets__item__title c0">Regular</div>
  <div class="markets__item__stat"><div class="markets__item__stat__label">State</div><div>Closed</div></div>
  <div class="markets__item__stat"><div class="markets__item__stat__label">Trades</div><div>329,931</div></div>
  <div class="markets__item__stat"><div class="markets__item__stat__label">Volume</div><div>620,697,365</div></div>
</a></div>
<div class="glide__slide" data-key="MTS"><div class="markets__item__stat">
  <div class="markets__item__stat__label">State</div><div>Suspended</div></div></div>
<div class="tabs__panel marketIndices__details" data-name="KSE100" data-type="1"
     data-date="2026-11-20 16:50:00" data-close="177382.57"></div>
"""


def test_market_state_is_read_from_the_homepage():
    state = parse_market_state(HOMEPAGE)
    assert state.state == "Closed"
    assert state.volume == 620_697_365
    assert state.trades == 329_931
    assert state.as_of == at(20, 16, 50)
    assert state.is_trading is False


@pytest.mark.parametrize("word, trading", [
    ("Closed", False), ("Suspended", False), ("Open", True),
    ("Pre-Open", True), ("Post-Close", True), (None, None),
])
def test_state_words(word, trading):
    assert MarketState(word, None, None, None, at(23, 9)).is_trading is trading


def test_missing_board_gives_unknown_state():
    state = parse_market_state("<html></html>")
    assert state.state is None and state.as_of is None
    assert state.checked_at.tzinfo is not None
    assert timedelta(0) <= datetime.now(PKT) - state.checked_at < timedelta(minutes=1)
