"""Tests for the permanent history store."""

from datetime import date

import pandas as pd
import pytest

from stockmind import config
from stockmind.data.history_store import HistoryStore


def prices(days: list[int], closes: list[float] | None = None) -> pd.DataFrame:
    """Daily prices for 2026-11-<day>."""
    closes = closes or [100.0 + d for d in days]
    return pd.DataFrame({
        "date": pd.to_datetime([f"2026-11-{d:02d}" for d in days]),
        "open": closes, "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes], "close": closes,
        "volume": pd.array([1000] * len(days), dtype="Int64"),
    })


@pytest.fixture
def store(tmp_path):
    return HistoryStore(tmp_path / "store")


def stored_days(store, ticker="OGDC") -> list[int]:
    frame, _ = store.load(ticker, "prices")
    return [d.day for d in frame["date"]]


def test_first_download_is_stored(store):
    result = store.merge("ogdc", "prices", prices([5, 3, 4]))

    assert result.added == 3 and result.written
    assert stored_days(store) == [3, 4, 5]  # sorted, ticker upper-cased
    assert store.path("OGDC", "prices").exists()


def test_records_that_leave_the_source_window_are_kept(store):
    store.merge("OGDC", "prices", prices([2, 3, 4, 5]))
    result = store.merge("OGDC", "prices", prices([4, 5, 6, 9]))  # window moved

    assert result.added == 2
    assert stored_days(store) == [2, 3, 4, 5, 6, 9]
    assert [d.day for d in result.data["date"]] == [2, 3, 4, 5, 6, 9]


def test_same_download_twice_writes_nothing(store):
    store.merge("OGDC", "prices", prices([2, 3]))
    result = store.merge("OGDC", "prices", prices([2, 3]))

    assert result.added == 0 and not result.written
    assert not (store.directory / "_backups").exists()


def test_duplicate_keys_in_a_download_become_one_record(store):
    download = pd.concat([prices([2, 3]), prices([3], [999.0])], ignore_index=True)
    store.merge("OGDC", "prices", download)

    frame, _ = store.load("OGDC", "prices")
    assert len(frame) == 2
    assert frame.loc[frame["date"].dt.day == 3, "close"].item() == 999.0  # last wins


def test_corrections_are_applied_and_logged(store):
    store.merge("OGDC", "prices", prices(list(range(1, 21))))
    corrected = prices(list(range(1, 21)))
    corrected.loc[corrected["date"].dt.day == 10, "close"] = 123.45

    result = store.merge("OGDC", "prices", corrected)

    assert result.corrected == 1 and result.written
    frame, _ = store.load("OGDC", "prices")
    assert frame.loc[frame["date"].dt.day == 10, "close"].item() == 123.45
    log = pd.read_csv(store.corrections_log, dtype=str)
    assert log[["ticker", "dataset", "column"]].values.tolist() == [["OGDC", "prices", "close"]]
    assert log["record"].item() == "2026-11-10T00:00:00"  # readable, not internal
    assert log["old_value"].item() == "110.0" and log["new_value"].item() == "123.45"


def test_an_unchanged_download_is_not_merged_again(store, monkeypatch):
    store.merge("OGDC", "prices", prices([1, 2, 3]))
    store.merge("OGDC", "prices", prices([1, 2, 3]))  # remembered

    def must_not_merge(*args):
        raise AssertionError("merged again")
    monkeypatch.setattr(store, "_merge", must_not_merge)
    result = store.merge("OGDC", "prices", prices([1, 2, 3]))

    assert [d.day for d in result.data["date"]] == [1, 2, 3]


def test_suspicious_mass_change_is_refused(store):
    store.merge("OGDC", "prices", prices([1, 2, 3, 4, 5, 6, 7]))
    before = store.path("OGDC", "prices").read_bytes()
    halved = prices([1, 2, 3, 4, 5, 6, 7, 8], [50.0] * 8)  # e.g. a unit change

    result = store.merge("OGDC", "prices", halved)

    assert not result.written
    assert "left unchanged" in result.warnings[0]
    assert store.path("OGDC", "prices").read_bytes() == before
    assert result.data["close"].tolist() == [50.0] * 8  # the download is passed on
    assert not store.corrections_log.exists()


def test_an_empty_download_changes_nothing(store):
    store.merge("OGDC", "prices", prices([2, 3]))
    result = store.merge("OGDC", "prices", pd.DataFrame())

    assert not result.written
    assert stored_days(store) == [2, 3]
    assert [d.day for d in result.data["date"]] == [2, 3]


def test_previous_versions_are_backed_up_and_pruned(store):
    for last_day in range(2, 8):
        store.merge("OGDC", "prices", prices(list(range(1, last_day + 1))))

    backups = list((store.directory / "_backups" / "prices" / "OGDC").glob("*.parquet"))
    assert len(backups) == config.HISTORY_BACKUPS_KEPT
    leftovers = [p for p in store.path("OGDC", "prices").parent.iterdir()
                 if p.suffix == ".tmp" or p.name.startswith(".")]
    assert leftovers == []  # no half-written temporary files


def test_corrupt_file_is_restored_from_the_newest_backup(store):
    store.merge("OGDC", "prices", prices([1, 2]))
    store.merge("OGDC", "prices", prices([1, 2, 3]))  # backup holds days 1-2
    store.path("OGDC", "prices").write_bytes(b"not a parquet file")

    frame, warnings = store.load("OGDC", "prices")

    assert [d.day for d in frame["date"]] == [1, 2]
    assert "restored from backup" in warnings[0]
    corrupt = list(store.path("OGDC", "prices").parent.glob("*.corrupt-*"))
    assert len(corrupt) == 1  # moved aside, never deleted
    assert stored_days(store) == [1, 2]


def test_corrupt_file_without_backup_is_kept_aside(store):
    store.merge("OGDC", "prices", prices([1]))
    store.path("OGDC", "prices").write_bytes(b"garbage")

    frame, warnings = store.load("OGDC", "prices")

    assert frame.empty
    assert "no readable backup" in warnings[0]
    assert len(list(store.path("OGDC", "prices").parent.glob("*.corrupt-*"))) == 1


def statement(columns: dict[str, str], value: float = 1.0) -> pd.DataFrame:
    """columns: fiscal_year label -> period_end; two metrics each."""
    rows = [{"fiscal_year": fy, "period_end": pd.Timestamp(end), "metric": m,
             "value": value, "unit": "PKR thousands"}
            for fy, end in columns.items() for m in ("revenue", "net_income")]
    return pd.DataFrame(rows)


def test_statements_keep_old_years_and_replace_ttm(store):
    store.merge("OGDC", "income_statement",
                statement({"FY 2021": "2021-06-30", "FY 2025": "2025-06-30",
                           "TTM": "2026-03-31"}))
    result = store.merge("OGDC", "income_statement",
                         statement({"FY 2025": "2025-06-30", "FY 2026": "2026-06-30",
                                    "TTM": "2026-09-30"}))

    frame, _ = store.load("OGDC", "income_statement")
    columns = sorted(set(zip(frame["fiscal_year"], frame["period_end"].dt.strftime("%Y-%m"))))
    assert columns == [("FY 2021", "2021-06"), ("FY 2025", "2025-06"),
                       ("FY 2026", "2026-06"), ("TTM", "2026-09")]
    assert result.added == 2  # FY 2026's two metrics; TTM is a snapshot


def test_new_reporting_period_is_reported(store):
    def financials(periods):
        return pd.DataFrame({"period_type": "quarterly", "period": periods,
                             "metric": "eps", "value": 1.0, "unit": "PKR per share"})
    store.merge("OGDC", "financials", financials(["Q3 2026", "Q1 2027"]))
    result = store.merge("OGDC", "financials", financials(["Q1 2027", "Q2 2027"]))

    assert result.new_periods == ["Q2 2027"]


def test_retain_stores_only_finished_sessions(store):
    value = {"prices": prices([20, 23]), "price_issues": prices([])}

    returned, info = store.retain("OGDC", "prices", value,
                                  day_is_final=lambda d: d < date(2026, 11, 23))

    assert [d.day for d in returned["prices"]["date"]] == [20, 23]  # both passed on
    assert stored_days(store) == [20]                                # only final stored
    assert info["added"] == {"prices": 1}


def test_retain_keeps_the_shape_of_each_source(store):
    page = {"profile": {"name": "OGDC"},
            "financials": pd.DataFrame({"period_type": ["annual"], "period": ["2026"],
                                        "metric": ["eps"], "value": [1.0]}),
            "ratios": pd.DataFrame({"period": ["2026"], "metric": ["pe"], "value": [5.0]})}

    returned, _ = store.retain("OGDC", "company_page", page, lambda d: True)

    assert returned["profile"] == {"name": "OGDC"}
    assert len(returned["financials"]) == 1 and len(returned["ratios"]) == 1
    assert store.stored_value("OGDC", "company_page")["financials"]["metric"].tolist() == ["eps"]
    assert store.stored_value("OGDC", "market_listing") is None


def test_a_later_run_recognises_an_already_merged_download(store, monkeypatch):
    store.merge("OGDC", "prices", prices([1, 2, 3]))
    later_run = HistoryStore(store.directory)  # e.g. the next time a script runs

    def must_not_merge(*args):
        raise AssertionError("merged again")
    monkeypatch.setattr(later_run, "_merge", must_not_merge)

    assert len(later_run.merge("OGDC", "prices", prices([1, 2, 3])).data) == 3


def test_a_changed_file_is_merged_again_despite_the_marker(store):
    store.merge("OGDC", "prices", prices([1, 2, 3]))
    other_run = HistoryStore(store.directory)
    other_run.merge("OGDC", "prices", prices([3, 4]))  # file changes

    result = HistoryStore(store.directory).merge("OGDC", "prices", prices([1, 2, 3]))

    assert [d.day for d in result.data["date"]] == [1, 2, 3, 4]
