"""Market clock: decides when market-driven data needs refreshing.

A fixed expiry (e.g. "prices are fresh for 1 hour") is wrong in both
directions: during a session a crash can go unseen for an hour, and at
night, at weekends and on holidays the same unchanged data is downloaded
again every hour. The clock instead asks "has the market traded since this
data was fetched?".

Two sources of truth, both in Pakistan time (config.MARKET_TIMEZONE):

1. A broad weekly schedule (config.TRADING_WINDOWS): when the market MAY be
   trading. Outside it the answer is "no" without any request.
2. PSX's own live market state, read from the PSX Data Portal homepage
   (the "State" of the regular market, plus its "As of" time and volume).
   It is only read inside the schedule, at most every
   config.MARKET_STATE_MAX_AGE. It catches holidays, moon-sighted Eid
   dates, Ramadan hours and unscheduled closures with no calendar.

If the live state cannot be read, the clock assumes the market may be
trading, so it refreshes more often rather than ever serving stale data.

Freshness rule for data fetched at time F, checked at time T:
    - a session's window ended between F and T     -> stale (get final data)
    - T outside any window                          -> fresh
    - T - F <= interval                             -> fresh
    - PSX says it is trading                        -> stale
    - not trading: stale only if PSX's "As of" time is later than F (e.g.
      the closing prices of a session that has just ended); a holiday has
      no new update, so nothing is fetched again
"""

import re
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
from psxdata import BaseScraper
from psxdata.constants import BASE_URL

from stockmind import config


PKT = ZoneInfo(config.MARKET_TIMEZONE)

# The index whose "As of" time stands for the market's last update.
AS_OF_INDEX = "KSE100"


def to_pkt(moment: datetime) -> datetime:
    """Pakistan time. A naive datetime is taken as this computer's local time
    (that is what datetime.now() returns, e.g. in the response cache)."""
    return moment.astimezone(PKT)


# ---------------------------------------------------------------------------
# PSX live market state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketState:
    """PSX's own view of the market, from the Data Portal homepage."""

    state: str | None          # PSX's word for the board, e.g. "Closed"
    as_of: datetime | None     # last market update (KSE-100 "As of"), PKT
    volume: float | None       # shares traded today on the board
    trades: float | None       # number of trades today on the board
    checked_at: datetime       # when the homepage was read, PKT

    @property
    def is_trading(self) -> bool | None:
        """False for a closed state, True for any other, None if unknown."""
        if not self.state:
            return None
        return self.state.strip().upper() not in config.MARKET_CLOSED_STATES


def _number(text: str | None) -> float | None:
    try:
        return float(str(text).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def parse_market_state(
    html: str,
    board: str = config.MARKET_STATE_BOARD,
    checked_at: datetime | None = None,
) -> MarketState:
    """Read the board's State, trades and volume and the KSE-100 "As of" time."""
    soup = BeautifulSoup(html, "lxml")
    stats: dict[str, str] = {}
    slide = soup.select_one(f'.glide__slide[data-key="{board}"]')
    if slide is not None:
        for stat in slide.select(".markets__item__stat"):
            label = stat.select_one(".markets__item__stat__label")
            values = stat.find_all("div")
            if label is not None and values:
                stats[label.get_text(strip=True).lower()] = values[-1].get_text(strip=True)

    as_of = None
    index = soup.select_one(f'[data-name="{AS_OF_INDEX}"][data-date]')
    if index is not None and re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", index["data-date"]):
        as_of = datetime.fromisoformat(index["data-date"]).replace(tzinfo=PKT)

    return MarketState(
        state=stats.get("state") or None,
        as_of=as_of,
        volume=_number(stats.get("volume")),
        trades=_number(stats.get("trades")),
        checked_at=to_pkt(checked_at or datetime.now(PKT)),
    )


class HomePageScraper(BaseScraper):
    """The PSX Data Portal homepage (psxdata handles token and retries)."""

    def homepage(self) -> str:
        return self._request("GET", f"{BASE_URL}/").text


_scraper: HomePageScraper | None = None


def fetch_market_state() -> MarketState:
    """Download and parse PSX's live market state."""
    global _scraper
    if _scraper is None:
        _scraper = HomePageScraper()
    state = parse_market_state(_scraper.homepage())
    if state.state is None:
        raise ValueError("PSX homepage has no market state (page layout changed?).")
    return state


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

class MarketClock:
    """Market-aware freshness decisions (see module docstring)."""

    def __init__(self, fetch_state=None, now=None) -> None:
        """
        Args:
            fetch_state: Function returning a MarketState (default: read
                the PSX homepage). Errors mean "state unknown".
            now: Function returning the current time (for tests).
        """
        self._fetch_state = fetch_state
        self._now = now or (lambda: datetime.now(PKT))
        self._lock = threading.Lock()
        self._state: MarketState | None = None
        self._state_read_at: datetime | None = None

    def now(self) -> datetime:
        return to_pkt(self._now())

    # -- schedule -----------------------------------------------------

    @staticmethod
    def window(day: date) -> tuple[datetime, datetime] | None:
        """The day's trading window, or None (weekend)."""
        hours = config.TRADING_WINDOWS.get(day.weekday())
        if hours is None:
            return None
        start, end = hours
        return (datetime.combine(day, start, PKT), datetime.combine(day, end, PKT))

    def in_window(self, moment: datetime | None = None) -> bool:
        moment = to_pkt(moment) if moment else self.now()
        window = self.window(moment.date())
        return window is not None and window[0] <= moment < window[1]

    def last_window_end(self, moment: datetime | None = None) -> datetime:
        """The end of the most recent trading window at or before `moment`."""
        moment = to_pkt(moment) if moment else self.now()
        for days_back in range(8):
            window = self.window(moment.date() - timedelta(days=days_back))
            if window is not None and window[1] <= moment:
                return window[1]
        return moment - timedelta(days=7)  # no windows configured

    def day_is_final(self, day: date, moment: datetime | None = None) -> bool:
        """True once a day's session can no longer change its prices."""
        moment = to_pkt(moment) if moment else self.now()
        if day < moment.date():
            return True
        if day > moment.date():
            return False
        window = self.window(day)
        return window is None or moment >= window[1]

    # -- live state ---------------------------------------------------

    def state(self) -> MarketState | None:
        """PSX's live market state, re-read at most every MARKET_STATE_MAX_AGE.
        None if it cannot be read (the failure is remembered just as long)."""
        with self._lock:
            now = self.now()
            if (self._state_read_at is not None
                    and now - self._state_read_at <= config.MARKET_STATE_MAX_AGE):
                return self._state
            try:
                self._state = (self._fetch_state or fetch_market_state)()
            except Exception:  # any failure just means "unknown"
                self._state = None
            self._state_read_at = now
            return self._state

    def is_trading(self, moment: datetime | None = None) -> bool:
        """Whether the market may be trading now. Outside the schedule this
        is False without any request; inside it PSX's state decides, and an
        unknown state counts as trading (refresh rather than go stale)."""
        if not self.in_window(moment):
            return False
        state = self.state()
        trading = state.is_trading if state is not None else None
        return True if trading is None else trading

    # -- freshness ----------------------------------------------------

    def is_fresh(
        self,
        fetched_at: datetime,
        interval: timedelta | None = config.SESSION_REFRESH,
        moment: datetime | None = None,
    ) -> bool:
        """
        Whether data fetched at `fetched_at` is still current.

        `interval` is how often to refresh during a session; None means
        "not during a session, only once each session has ended" (used for
        full price history when Market Watch supplies the live bar).
        """
        now = to_pkt(moment) if moment else self.now()
        fetched = to_pkt(fetched_at)

        if fetched < self.last_window_end(now):
            return False          # a session ended since: get its final data
        if not self.in_window(now) or interval is None:
            return True
        if now - fetched <= interval:
            return True
        if self.is_trading(now):
            return False          # trading and older than the interval
        # Not trading now (before the open, after the close, a holiday):
        # fresh unless PSX has published an update since we fetched, e.g.
        # the closing prices of a session that has just ended.
        state = self.state()
        if state is not None and state.as_of is not None:
            return state.as_of <= fetched
        window_start = self.window(now.date())[0]
        return fetched < window_start
