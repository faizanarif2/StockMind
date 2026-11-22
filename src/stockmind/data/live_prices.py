"""Today's prices for every company from PSX Market Watch, in one request.

PSX's historical endpoint always returns a company's full 10-year history,
so refreshing prices every 15 minutes during a session would mean about 30
full downloads each time. The Market Watch page instead lists today's open,
high, low, current price and volume of every listed company (~500 rows)
in a single request.

Market Watch shows no timestamp of its own. Each read therefore also reads
PSX's market state (market_clock.fetch_market_state) and is only trusted
when:
    - PSX's "As of" time is recent while the market is trading
      (config.LIVE_PRICES_MAX_LAG), so a stuck feed is never taken as live;
    - Market Watch's total volume matches PSX's regular-market volume
      (within config.LIVE_PRICES_VOLUME_TOLERANCE). This proves the table
      is the same snapshot PSX reports, not an older cached page. On
      2026-11-20 after the close both were exactly 620,697,365 shares.

After the close, the same values matched the official daily history for
every company checked (OGDC, FFC, MEBL, LUCK, HUBC).

Use is switched on by config.LIVE_PRICES_ENABLED once behaviour during a
live session has been checked with scripts/check_market_session.py.
"""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup
from psxdata import BaseScraper
from psxdata.constants import BASE_URL

from stockmind import config
from stockmind.data.market_clock import MarketState, fetch_market_state, to_pkt
from stockmind.data.tickers import ENTITLEMENT_SUFFIXES


LIVE_COLUMNS = ["symbol", "ldcp", "open", "high", "low", "close", "volume"]

# Market Watch column order: SYMBOL, SECTOR, LISTED IN, LDCP, OPEN, HIGH,
# LOW, CURRENT, CHANGE, CHANGE (%), VOLUME.
_POSITIONS = {"ldcp": 3, "open": 4, "high": 5, "low": 6, "close": 7, "volume": 10}


def parse_market_watch(html: str) -> pd.DataFrame:
    """One row per company: symbol, ldcp (last close), open, high, low,
    close (current price), volume. Numbers come from PSX's own data-order
    attributes, so formatting (commas, arrows, badges) does not matter."""
    soup = BeautifulSoup(html, "lxml")
    records = []
    for row in soup.select("table tbody tr"):
        cells = row.find_all("td")
        if len(cells) <= max(_POSITIONS.values()):
            continue
        symbol = cells[0].get("data-order") or cells[0].get_text(" ", strip=True).split()[0]
        record = {"symbol": symbol.strip().upper()}
        for name, position in _POSITIONS.items():
            cell = cells[position]
            raw = cell.get("data-order", cell.get_text(strip=True))
            record[name] = pd.to_numeric(str(raw).replace(",", ""), errors="coerce")
        records.append(record)
    return pd.DataFrame(records, columns=LIVE_COLUMNS)


@dataclass(frozen=True)
class LiveSnapshot:
    """Market Watch plus the PSX market state read just before it."""

    prices: pd.DataFrame
    state: MarketState
    fetched_at: datetime

    @property
    def total_volume(self) -> float:
        return float(self.prices["volume"].sum()) if not self.prices.empty else 0.0

    def problem(self, now: datetime | None = None) -> str | None:
        """Why this snapshot must not be used, or None if it can be."""
        now = to_pkt(now or self.fetched_at)
        if self.prices.empty:
            return "Market Watch returned no prices"
        if self.state.as_of is None:
            return "PSX gave no 'As of' time, so freshness cannot be confirmed"
        if self.state.is_trading and now - self.state.as_of > config.LIVE_PRICES_MAX_LAG:
            return (f"PSX's last update ({self.state.as_of:%H:%M}) is older than "
                    f"{int(config.LIVE_PRICES_MAX_LAG.total_seconds() // 60)} minutes "
                    "while the market is trading")
        expected = self.state.volume
        if not expected:
            return "PSX gave no regular-market volume to check Market Watch against"
        gap = abs(self.total_volume - expected) / expected
        if gap > config.LIVE_PRICES_VOLUME_TOLERANCE:
            return (f"Market Watch total volume ({self.total_volume:,.0f}) differs "
                    f"from PSX's regular-market volume ({expected:,.0f}) by {gap:.1%}")
        return None

    def bar(self, ticker: str) -> dict | None:
        """Today's price bar for one company (date = PSX's "As of" day), or
        None if it is not listed or its values are inconsistent."""
        if self.state.as_of is None or self.prices.empty:
            return None
        candidates = {ticker.upper()} | {ticker.upper() + s for s in ENTITLEMENT_SUFFIXES}
        rows = self.prices[self.prices["symbol"].isin(candidates)]
        if rows.empty:
            return None
        row = rows.iloc[0]
        values = {name: row[name] for name in ("open", "high", "low", "close", "volume")}
        if any(pd.isna(v) for v in values.values()):
            return None
        if values["volume"] <= 0 or min(values["open"], values["low"], values["close"]) <= 0:
            return None  # no trades yet today
        if not (values["low"] <= min(values["open"], values["close"])
                and values["high"] >= max(values["open"], values["close"])):
            return None
        values["volume"] = int(values["volume"])
        return {"date": pd.Timestamp(self.state.as_of.date()), **values}


class MarketWatchScraper(BaseScraper):
    def market_watch(self) -> str:
        return self._request("GET", f"{BASE_URL}/market-watch").text


_scraper: MarketWatchScraper | None = None


def fetch_live_snapshot() -> LiveSnapshot:
    """Read PSX's market state, then Market Watch."""
    global _scraper
    if _scraper is None:
        _scraper = MarketWatchScraper()
    state = fetch_market_state()
    prices = parse_market_watch(_scraper.market_watch())
    return LiveSnapshot(prices=prices, state=state, fetched_at=to_pkt(datetime.now()))


def overlay_bar(prices: pd.DataFrame, bar: dict) -> pd.DataFrame:
    """Replace (or add) the row for the bar's date with Market Watch values.
    Other columns keep their meaning: not flagged, valid."""
    prices = prices[prices["date"] != bar["date"]] if not prices.empty else prices
    row = pd.DataFrame([bar])
    defaults = {"source_is_anomaly": False, "anomaly_flags": "", "is_valid": True}
    for column, value in defaults.items():
        if column in prices.columns:
            row[column] = value
    combined = pd.concat([prices, row], ignore_index=True)
    return combined.sort_values("date").reset_index(drop=True)
