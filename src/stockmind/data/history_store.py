"""Permanent history store: every record ever collected, kept safely.

Why, next to the cache?
    The response cache (cache.py) keeps only the LATEST download of each
    source and is safe to delete. But every source only shows a rolling
    window - PSX prices cover 10 years, PSX payouts about 2 years,
    stockanalysis.com 5 fiscal years - so each refresh used to drop the
    oldest records for good. The store keeps them. The cache is unchanged:
    it still decides when to download; the store decides what is kept.

What is stored (one Parquet file per company and dataset):

    dataset           key (one record per key)          updated how
    ----------------  --------------------------------  --------------------
    prices            date                              new days added; only
                                                        finished sessions
    payouts           announced_at, details             new announcements
    dividend_history  ex_date                           new dividends
    income_statement  fiscal_year, period_end, metric   new periods added;
    balance_sheet     fiscal_year, period_end, metric   "TTM" replaced
    financials (PSX)  period_type, period, metric       new periods added
    ratios (PSX)      period, metric                    new periods added

Snapshots (company profile, market listing, index list) stay in the cache
only: they describe "now" and have no history to keep.

Protection against loss and corruption
    - Merge, never replace: records are only added or corrected, never
      removed, and an empty or failed download changes nothing.
    - One record per key: incoming duplicates are dropped and the key is
      enforced on every write and read.
    - Corrections are logged: when a source changes a record we already
      have, the new value is used and old/new are appended to
      corrections.csv.
    - Suspicious updates are refused: if a download would change more than
      config.HISTORY_MAX_CORRECTION_SHARE of the overlapping records (e.g.
      a unit change or a broken page), nothing is written and a warning is
      returned; the download itself is still passed on, as before.
    - Safe writes: a new version is written to a temporary file, read back
      and checked, then swapped in with an atomic rename, so a crash never
      leaves a half-written file. The previous version is copied to
      _backups/ first (config.HISTORY_BACKUPS_KEPT kept).
    - Corrupt files are recovered: if a file cannot be read, it is moved
      aside (never deleted) and the newest readable backup is restored.
    - Locking: writes to one file are serialised across threads and
      processes (diskcache's lock), so two runs cannot interleave.
"""

import csv
import hashlib
import os
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import diskcache
import numpy as np
import pandas as pd

from stockmind import config


@dataclass(frozen=True)
class DatasetSpec:
    """How one dataset is identified, compared and updated."""

    key: tuple[str, ...]
    # Columns compared to detect corrections (None: every non-key column).
    compare: tuple[str, ...] | None = None
    # Rows that are a "current snapshot" (e.g. TTM): replaced by each
    # download instead of kept, and not counted as corrections.
    is_snapshot: Callable[[pd.DataFrame], pd.Series] | None = None
    # Column naming the reporting period, to notice when a new one appears.
    period: str | None = None


def _is_ttm(frame: pd.DataFrame) -> pd.Series:
    return frame["fiscal_year"].astype("string").str.upper().eq("TTM")


STATEMENT = DatasetSpec(key=("fiscal_year", "period_end", "metric"),
                        compare=("value",), is_snapshot=_is_ttm,
                        period="period_end")

DATASETS: dict[str, DatasetSpec] = {
    "prices": DatasetSpec(key=("date",),
                          compare=("open", "high", "low", "close", "volume")),
    "payouts": DatasetSpec(key=("announced_at", "details")),
    "dividend_history": DatasetSpec(key=("ex_date",)),
    "income_statement": STATEMENT,
    "balance_sheet": STATEMENT,
    "financials": DatasetSpec(key=("period_type", "period", "metric"),
                              compare=("value",), period="period"),
    "ratios": DatasetSpec(key=("period", "metric"), compare=("value",),
                          period="period"),
    # News (see news.py): one record per article id, never removed.
    "announcements": DatasetSpec(key=("article_id",), compare=("title",)),
    "articles": DatasetSpec(key=("article_id",), compare=("title",)),
}

# Which datasets each Agent 1 source carries. A string means the source's
# value is a dict and the dataset is under that key.
SOURCE_DATASETS: dict[str, tuple[tuple[str, str | None], ...]] = {
    "prices": (("prices", "prices"),),
    "company_page": (("financials", "financials"), ("ratios", "ratios")),
    "payouts": (("payouts", None),),
    "dividend_history": (("dividend_history", None),),
    "income_statement": (("income_statement", None),),
    "balance_sheet": (("balance_sheet", None),),
    "announcements": (("announcements", None),),
    "articles": (("articles", None),),
}

CORRECTION_COLUMNS = ["logged_at", "ticker", "dataset", "record", "column",
                      "old_value", "new_value"]


@dataclass
class MergeResult:
    """The merged dataset and what changed."""

    data: pd.DataFrame
    added: int = 0
    corrected: int = 0
    written: bool = False
    new_periods: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _key_text(frame: pd.DataFrame, key: tuple[str, ...]) -> pd.Series:
    """One string per row identifying its key, stable across file round trips."""
    parts = []
    for column in key:
        values = frame[column] if column in frame else pd.Series(None, index=frame.index)
        if pd.api.types.is_datetime64_any_dtype(values):
            # Whole seconds since 1970: independent of the stored time unit
            # (ns / us after a file round trip) and much faster than strftime.
            seconds = values.to_numpy(dtype="datetime64[s]").astype("int64").astype(str)
            text = pd.Series(np.where(values.isna().to_numpy(), "NaT", seconds),
                             index=values.index, dtype="string")
        else:
            text = values.astype("string").fillna("<NA>")
        parts.append(text)
    combined = parts[0]
    for part in parts[1:]:
        combined = combined + "|" + part
    return combined


def _equal(old: pd.Series, new: pd.Series) -> pd.Series:
    """Element-wise equality: two missing values are equal, and numbers are
    compared allowing for floating-point rounding."""
    both_missing = old.isna().to_numpy() & new.isna().to_numpy()
    if pd.api.types.is_numeric_dtype(old) and pd.api.types.is_numeric_dtype(new):
        a = old.astype("float64").to_numpy()
        b = new.astype("float64").to_numpy()
        same = np.isclose(a, b, rtol=1e-9, atol=1e-9)
    else:
        same = (old.map(_text).to_numpy() == new.map(_text).to_numpy())
    return pd.Series(same | both_missing, index=old.index)


def _differences(
    stored: pd.DataFrame, incoming: pd.DataFrame, spec: "DatasetSpec",
    count_missing: bool = False,
) -> list[dict]:
    """
    One entry per (record, column) whose value differs between the stored
    and the incoming version of the same key. With count_missing=True, a key
    present on only one side also counts as a difference.
    """
    stored_keys = _key_text(stored, spec.key)
    incoming_keys = _key_text(incoming, spec.key)
    if count_missing and set(stored_keys) != set(incoming_keys):
        return [{"key": "*", "record": "*", "column": "*",
                 "old_value": "", "new_value": ""}]

    incoming = incoming[incoming_keys.isin(set(stored_keys))]
    if incoming.empty:
        return []
    keys = _key_text(incoming, spec.key).to_numpy()
    old = stored.set_index(stored_keys.to_numpy()).reindex(keys)
    new = incoming.set_index(keys)
    columns = spec.compare or [c for c in incoming.columns if c not in spec.key]

    differences = []
    for column in columns:
        if column not in old.columns or column not in new.columns:
            continue
        changed = ~_equal(old[column], new[column])
        for key in changed.index[changed.to_numpy()]:
            differences.append({
                "key": key, "column": column,
                "record": "|".join(_text(new.at[key, k]) for k in spec.key),
                "old_value": _text(old.at[key, column]),
                "new_value": _text(new.at[key, column]),
            })
    return differences


def _fingerprint(frame: pd.DataFrame) -> str | None:
    """A content hash of a download, stable across runs, used to skip
    re-merging a download that has already been merged."""
    try:
        hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    except (TypeError, ValueError):
        return None
    digest = hashlib.sha1(hashed.tobytes())
    digest.update("|".join(map(str, frame.columns)).encode())
    return digest.hexdigest()


def _text(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class HistoryStore:
    """Keeps every collected record per company and dataset (see module docs)."""

    def __init__(self, directory: Path | str | None = None) -> None:
        self.directory = Path(directory or config.HISTORY_DIR)
        self._locks: diskcache.Cache | None = None  # opened on first write
        # (ticker, dataset) -> (download fingerprint, file mtime, merged data):
        # an unchanged download over an unchanged file needs no new merge.
        self._memo: dict[tuple[str, str], tuple] = {}

    # -- paths --------------------------------------------------------

    def path(self, ticker: str, dataset: str) -> Path:
        return self.directory / dataset / f"{ticker.upper()}.parquet"

    def _backup_dir(self, ticker: str, dataset: str) -> Path:
        return self.directory / "_backups" / dataset / ticker.upper()

    @property
    def corrections_log(self) -> Path:
        return self.directory / "corrections.csv"

    def _lock(self, name: str) -> diskcache.Lock:
        if self._locks is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._locks = diskcache.Cache(str(self.directory / "_locks"))
        return diskcache.Lock(self._locks, f"lock:{name}", expire=120)

    # -- reading ------------------------------------------------------

    def load(self, ticker: str, dataset: str) -> tuple[pd.DataFrame, list[str]]:
        """The stored records (empty if none) and any recovery warnings."""
        path = self.path(ticker, dataset)
        if not path.exists():
            return pd.DataFrame(), []
        try:
            return self._checked(pd.read_parquet(path), dataset), []
        except Exception as exc:  # unreadable: recover, never delete
            return self._recover(ticker, dataset, exc)

    def _checked(self, frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
        spec = DATASETS[dataset]
        missing = [column for column in spec.key if column not in frame.columns]
        if missing:
            raise ValueError(f"stored {dataset} has no key column(s) {missing}")
        return frame[~_key_text(frame, spec.key).duplicated(keep="last")]

    def _recover(
        self, ticker: str, dataset: str, error: Exception
    ) -> tuple[pd.DataFrame, list[str]]:
        path = self.path(ticker, dataset)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        aside = path.with_name(f"{path.stem}.corrupt-{stamp}.parquet")
        os.replace(path, aside)
        message = (f"{ticker}:{dataset}: stored history could not be read "
                   f"({type(error).__name__}); moved aside to {aside.name}")

        for backup in sorted(self._backup_dir(ticker, dataset).glob("*.parquet"),
                             reverse=True):
            try:
                frame = self._checked(pd.read_parquet(backup), dataset)
            except Exception:
                continue
            shutil.copy2(backup, path)
            return frame, [f"{message} and restored from backup {backup.name}."]
        return pd.DataFrame(), [f"{message}; no readable backup was found."]

    # -- writing ------------------------------------------------------

    def _write(self, ticker: str, dataset: str, frame: pd.DataFrame) -> None:
        """Back up the current version, then atomically replace it."""
        path = self.path(ticker, dataset)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            backups = self._backup_dir(ticker, dataset)
            backups.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
            shutil.copy2(path, backups / f"{stamp}.parquet")
            for old in sorted(backups.glob("*.parquet"))[:-config.HISTORY_BACKUPS_KEPT]:
                old.unlink()

        temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
        try:
            frame.to_parquet(temporary, index=False)
            if len(pd.read_parquet(temporary)) != len(frame):  # verify
                raise OSError(f"verification of {temporary.name} failed")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _log_corrections(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self._lock("corrections"):
            new_file = not self.corrections_log.exists()
            with open(self.corrections_log, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CORRECTION_COLUMNS)
                if new_file:
                    writer.writeheader()
                writer.writerows(rows)

    # -- merging ------------------------------------------------------

    def merge(self, ticker: str, dataset: str, incoming: pd.DataFrame) -> MergeResult:
        """
        Add a download to the stored records (see module docstring).

        Returns the merged records. They contain everything stored plus
        everything downloaded, one record per key, sorted by key.
        """
        spec = DATASETS[dataset]
        ticker = ticker.upper()
        if incoming is None or incoming.empty or any(c not in incoming for c in spec.key):
            stored, warnings = self.load(ticker, dataset)
            return MergeResult(stored if not stored.empty else incoming,
                               warnings=warnings)

        path = self.path(ticker, dataset)
        fingerprint = _fingerprint(incoming)
        if fingerprint is not None and path.exists():
            version = (fingerprint, path.stat().st_mtime_ns)
            memo = self._memo.get((ticker, dataset))
            if memo is not None and memo[:2] == version:
                return MergeResult(memo[2].copy())       # same run, nothing new
            if self._merged_marker(path) == version:     # merged by an earlier run
                stored, warnings = self.load(ticker, dataset)
                if not warnings:
                    self._memo[(ticker, dataset)] = (*version, stored.copy())
                    return MergeResult(stored)

        incoming = incoming[~_key_text(incoming, spec.key).duplicated(keep="last")]
        with self._lock(f"{dataset}:{ticker}"):
            stored, warnings = self.load(ticker, dataset)
            result = self._merge(ticker, dataset, spec, stored, incoming)
            result.warnings = warnings + result.warnings
            if result.written:
                self._write(ticker, dataset, result.data)
            if fingerprint is not None and path.exists() and not result.warnings:
                version = (fingerprint, path.stat().st_mtime_ns)
                self._memo[(ticker, dataset)] = (*version, result.data.copy())
                self._set_merged_marker(path, version)
        return result

    # A small text file next to each dataset records the fingerprint of the
    # last download merged into it and the dataset file's modification time.
    # If both still match, the download is already in the file.

    @staticmethod
    def _merged_marker(path: Path) -> tuple[str, int] | None:
        try:
            fingerprint, mtime = path.with_suffix(".merged").read_text().split()
            return fingerprint, int(mtime)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _set_merged_marker(path: Path, version: tuple[str, int]) -> None:
        marker = path.with_suffix(".merged")
        temporary = marker.with_name(f".{marker.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(f"{version[0]} {version[1]}")
        os.replace(temporary, marker)

    def _merge(
        self, ticker: str, dataset: str, spec: DatasetSpec,
        stored: pd.DataFrame, incoming: pd.DataFrame,
    ) -> MergeResult:
        if stored.empty:
            data = self._sorted(incoming, spec)
            return MergeResult(data, added=len(data), written=True)

        incoming_snapshot = (spec.is_snapshot(incoming) if spec.is_snapshot
                             else pd.Series(False, index=incoming.index))
        snapshot_changed = False
        if spec.is_snapshot is not None:
            stored_snapshot = spec.is_snapshot(stored)
            snapshot_changed = bool(_differences(
                stored[stored_snapshot], incoming[incoming_snapshot], spec,
                count_missing=True))
            stored = stored[~stored_snapshot]

        stored_keys = _key_text(stored, spec.key)
        incoming_keys = _key_text(incoming, spec.key)
        is_new = ~incoming_keys.isin(set(stored_keys)) & ~incoming_snapshot
        overlap = incoming[~is_new & ~incoming_snapshot]

        corrections = _differences(stored, overlap, spec)
        changed_keys = {c["key"] for c in corrections}
        if (len(changed_keys) >= config.HISTORY_MIN_SUSPICIOUS_CHANGES
                and len(changed_keys) > config.HISTORY_MAX_CORRECTION_SHARE * len(overlap)):
            return MergeResult(
                self._sorted(incoming, spec),
                warnings=[
                    f"{ticker}:{dataset}: the download would change "
                    f"{len(changed_keys)} of {len(overlap)} stored records, more "
                    "than expected for genuine corrections; the stored history "
                    "was left unchanged. Review the source before accepting it."
                ],
            )

        # Stored records the download does not include are kept as they are;
        # the download's version of every other record is used.
        kept = stored[~stored_keys.isin(set(incoming_keys))]
        data = self._sorted(pd.concat([kept, incoming], ignore_index=True), spec)

        new_periods = []
        if spec.period and spec.period in incoming and spec.period in stored:
            known = set(stored[spec.period].map(_text))
            new_periods = sorted({_text(v) for v in incoming.loc[is_new, spec.period]}
                                 - known - {""})

        logged_at = datetime.now().isoformat(timespec="seconds")
        self._log_corrections([
            {"logged_at": logged_at, "ticker": ticker, "dataset": dataset,
             "record": c["record"], "column": c["column"],
             "old_value": c["old_value"], "new_value": c["new_value"]}
            for c in corrections])
        return MergeResult(
            data,
            added=int(is_new.sum()),
            corrected=len(changed_keys),
            written=bool(is_new.any() or changed_keys or snapshot_changed),
            new_periods=new_periods,
        )

    @staticmethod
    def _sorted(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
        return (frame.assign(_key=_key_text(frame, spec.key))
                .sort_values("_key", kind="stable")
                .drop(columns="_key")
                .reset_index(drop=True))

    # -- Agent 1 sources ---------------------------------------------

    def retain(
        self,
        ticker: str,
        source: str,
        value,
        day_is_final: Callable[[date], bool],
    ) -> tuple[object, dict]:
        """
        Merge one Agent 1 source value into the store and return the value
        with the full retained history, in exactly the same shape.

        Prices: only days whose session has finished are stored; a day still
        in progress is returned as downloaded but not stored, so partial
        prices never enter the history.

        Returns (value, info) with info = {"warnings": [...],
        "new_periods": {dataset: [...]}, "added": {...}, "corrected": {...}}.
        """
        info = {"warnings": [], "new_periods": {}, "added": {}, "corrected": {}}
        if source not in SOURCE_DATASETS or value is None:
            return value, info

        value = dict(value) if isinstance(value, dict) else value
        for dataset, part in SOURCE_DATASETS[source]:
            frame = value[part] if part else value
            if not isinstance(frame, pd.DataFrame):
                continue

            in_progress = frame.iloc[0:0]
            if dataset == "prices" and not frame.empty and "date" in frame:
                final = frame["date"].map(lambda d: day_is_final(pd.Timestamp(d).date()))
                in_progress, frame = frame[~final], frame[final]

            result = self.merge(ticker, dataset, frame)
            merged = result.data
            if not in_progress.empty:
                merged = pd.concat([merged, in_progress], ignore_index=True)
                merged = self._sorted(merged, DATASETS[dataset])
            if part:
                value[part] = merged
            else:
                value = merged

            info["warnings"] += result.warnings
            info["added"][dataset] = result.added
            info["corrected"][dataset] = result.corrected
            if result.new_periods:
                info["new_periods"][dataset] = result.new_periods
        return value, info

    def stored_value(self, ticker: str, source: str):
        """The stored history in the shape of an Agent 1 source value, or
        None when nothing is stored. Used when a source is down and nothing
        is cached."""
        if source not in SOURCE_DATASETS:
            return None
        frames = {dataset: self.load(ticker, dataset)[0]
                  for dataset, _ in SOURCE_DATASETS[source]}
        if all(frame.empty for frame in frames.values()):
            return None
        if source == "prices":
            prices = frames["prices"]
            return {"prices": prices, "price_issues": prices.iloc[0:0]}
        if source == "company_page":
            return {"profile": {}, **frames}
        return frames[source]
