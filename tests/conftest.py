"""Shared test setup: keep every test offline and away from real user data.

- The permanent history store is redirected to a temporary folder, so no
  test can ever write into ~/.stockmind/data.
- PSX's live market state is "unavailable", so the market clock never
  makes a network request during tests and falls back to its schedule.
"""

import pytest
from psxdata.exceptions import PSXConnectionError

from stockmind import config
from stockmind.data import market_clock


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
