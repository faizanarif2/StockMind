"""Back up, restore, and check the permanent history store.

The history store (config.HISTORY_DIR, default ~/.stockmind/data) keeps
records that the sources no longer show, so it cannot be rebuilt by
downloading again. Back it up regularly, ideally to another drive or a
cloud-synced folder.

Usage:
    python scripts/backup_history.py backup                 # zip to ~/.stockmind/backups
    python scripts/backup_history.py backup --to D:/Backups # zip to another folder
    python scripts/backup_history.py check                  # read every file, report problems
    python scripts/backup_history.py restore FILE.zip       # restore a backup

Restoring never deletes anything: the current store is first renamed to
data.before-restore-<time> next to it.
"""

import argparse
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from stockmind import config
from stockmind.data.history_store import DATASETS, _key_text


SKIPPED = ("_locks",)  # lock database: recreated automatically


def _files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and not any(part in SKIPPED for part in path.relative_to(root).parts):
            yield path


def backup(destination: Path) -> Path:
    root = config.HISTORY_DIR
    if not root.exists():
        raise SystemExit(f"No history store at {root} yet.")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"stockmind-history-{datetime.now():%Y%m%d-%H%M%S}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in _files(root):
            archive.write(path, path.relative_to(root))
    with zipfile.ZipFile(target) as archive:  # verify before reporting success
        bad = archive.testzip()
        if bad:
            raise SystemExit(f"Backup {target} is damaged at {bad}.")
        count = len(archive.namelist())
    print(f"Backed up {count} files from {root} to {target}")
    return target


def check() -> int:
    """Read every dataset file; report unreadable files and duplicate keys."""
    root = config.HISTORY_DIR
    problems = 0
    for dataset, spec in DATASETS.items():
        for path in sorted((root / dataset).glob("*.parquet")):
            try:
                frame = pd.read_parquet(path)
            except Exception as exc:
                problems += 1
                print(f"UNREADABLE {path.relative_to(root)}: {type(exc).__name__}")
                continue
            duplicates = int(_key_text(frame, spec.key).duplicated().sum())
            status = "ok" if duplicates == 0 else f"{duplicates} DUPLICATE KEYS"
            problems += duplicates > 0
            print(f"{status:>16}  {path.relative_to(root)}  {len(frame)} records")
    log = root / "corrections.csv"
    if log.exists():
        print(f"Corrections logged: {len(pd.read_csv(log))} (see {log})")
    print("No problems found." if problems == 0 else f"{problems} problem(s) found.")
    return problems


def restore(archive_path: Path) -> None:
    root = config.HISTORY_DIR
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise SystemExit(f"{archive_path} is damaged at {bad}; nothing changed.")
        if root.exists():
            aside = root.with_name(f"{root.name}.before-restore-{datetime.now():%Y%m%d-%H%M%S}")
            shutil.move(str(root), str(aside))
            print(f"Current store moved aside to {aside}")
        archive.extractall(root)
    print(f"Restored {archive_path} into {root}")
    check()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("backup", help="zip the store")
    make.add_argument("--to", type=Path, default=Path.home() / ".stockmind" / "backups")
    commands.add_parser("check", help="read every file and report problems")
    undo = commands.add_parser("restore", help="restore a backup zip")
    undo.add_argument("archive", type=Path)
    args = parser.parse_args()

    if args.command == "backup":
        backup(args.to)
    elif args.command == "check":
        raise SystemExit(1 if check() else 0)
    else:
        restore(args.archive)


if __name__ == "__main__":
    main()
