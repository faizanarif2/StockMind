"""Shared test setup: keep every test offline and away from real user data.

- The permanent history store is redirected to a temporary folder, so no
  test can ever write into ~/.stockmind/data.
- PSX's live market state is "unavailable", so the market clock never
  makes a network request during tests and falls back to its schedule.
- News sources are "unavailable" too, unless a test replaces them.
"""

import pytest
from psxdata.exceptions import PSXConnectionError

from stockmind import config
from stockmind.data import market_clock, news


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    directory = tmp_path / "history"
    monkeypatch.setattr(config, "HISTORY_DIR", directory)
    return directory


@pytest.fixture(autouse=True)
def offline_market_state(monkeypatch):
    def offline():
        raise PSXConnectionError("tests run offline")
    monkeypatch.setattr(market_clock, "fetch_market_state", offline)


@pytest.fixture(autouse=True)
def offline_news(monkeypatch):
    """News sources are unavailable unless a test provides fakes."""
    def offline(*args, **kwargs):
        raise PSXConnectionError("tests run offline")
    for name in ("fetch_announcements", "fetch_announcement_history", "fetch_feeds"):
        monkeypatch.setattr(news, name, offline)
