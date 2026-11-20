"""Record how PSX's live data behaves during a trading session.

Market Watch (live prices) stays switched off (config.LIVE_PRICES_ENABLED)
until this has been run through a real session. Start it before the open
and leave it running; it stops by itself after the close.

Usage:
    python scripts/check_market_session.py                 # sample every 60 s until 17:30 PKT
    python scripts/check_market_session.py --interval 30 --until 12:00
    python scripts/check_market_session.py --report FILE   # summarise a finished log

Every sample records, in ~/.stockmind/session_checks/<date>.csv:
    - PSX's market State word (homepage, regular market) and its "As of" time
    - PSX's regular-market volume and trade count
    - Market Watch's total volume, and whether its table changed since the
      previous sample
    - Market Watch's bar for a few large companies
    - every 15 minutes, the last row of PSX's daily history for OGDC (does
      it contain a partial bar for today during the session?)

After the close it compares the final Market Watch values with the official
daily history row of each watched company, then writes a report next to
the CSV answering the questions the design depends on.
"""

import argparse
import csv
import hashlib
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from stockmind.data import live_prices
from stockmind.data.fetcher import fetch_historical_prices
from stockmind.data.market_clock import PKT, MarketClock


WATCH = ("OGDC", "FFC", "MEBL", "HUBC", "LUCK")
HISTORY_EVERY = timedelta(minutes=15)
OUT_DIR = Path.home() / ".stockmind" / "session_checks"


def sample(previous_hash: str | None) -> tuple[dict, str | None, live_prices.LiveSnapshot | None]:
    now = datetime.now(PKT)
    row = {"checked_at": now.isoformat(timespec="seconds")}
    try:
        snapshot = live_prices.fetch_live_snapshot()
    except Exception as exc:  # record, keep sampling
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row, previous_hash, None

    state = snapshot.state
    table_hash = hashlib.sha1(
        pd.util.hash_pandas_object(snapshot.prices, index=False).values.tobytes()
    ).hexdigest()
    row.update({
        "state": state.state,
        "as_of": state.as_of.isoformat() if state.as_of else "",
        "as_of_lag_min": round((now - state.as_of).total_seconds() / 60, 1) if state.as_of else "",
        "psx_volume": state.volume,
        "psx_trades": state.trades,
        "mw_rows": len(snapshot.prices),
        "mw_volume": snapshot.total_volume,
        "volume_gap_pct": (round(100 * (snapshot.total_volume - state.volume) / state.volume, 3)
                           if state.volume else ""),
        "mw_changed": "" if previous_hash is None else table_hash != previous_hash,
        "problem": snapshot.problem(now) or "",
    })
    for ticker in WATCH:
        bar = snapshot.bar(ticker)
        row[f"{ticker}_bar"] = ("" if bar is None else
                                f"{bar['date']:%Y-%m-%d} O{bar['open']} H{bar['high']} "
                                f"L{bar['low']} C{bar['close']} V{bar['volume']}")
    return row, table_hash, snapshot


def history_last_row(ticker: str = "OGDC") -> str:
    try:
        last = fetch_historical_prices(ticker).iloc[-1]
    except Exception as exc:
        return f"error {type(exc).__name__}"
    return (f"{last['date']:%Y-%m-%d} O{last['open']} H{last['high']} L{last['low']} "
            f"C{last['close']} V{last['volume']}")


def compare_with_history(snapshot: live_prices.LiveSnapshot) -> list[str]:
    lines = []
    for ticker in WATCH:
        bar = snapshot.bar(ticker)
        try:
            history = fetch_historical_prices(ticker)
        except Exception as exc:
            lines.append(f"{ticker}: history unavailable ({type(exc).__name__})")
            continue
        if bar is None:
            lines.append(f"{ticker}: no Market Watch bar")
            continue
        match = history[history["date"] == bar["date"]]
        if match.empty:
            lines.append(f"{ticker}: history has no row for {bar['date']:%Y-%m-%d} yet")
            continue
        official = match.iloc[-1]
        diffs = [f"{c} MW {bar[c]} vs history {official[c]}"
                 for c in ("open", "high", "low", "close", "volume")
                 if abs(float(bar[c]) - float(official[c])) > 1e-6]
        lines.append(f"{ticker}: " + ("matches the official row" if not diffs
                                      else "DIFFERS: " + "; ".join(diffs)))
    return lines


def report(csv_path: Path, comparison: list[str] | None = None) -> str:
    log = pd.read_csv(csv_path)
    ok = log[log.get("error", pd.Series(dtype=str)).isna()] if "error" in log else log
    lines = [f"# PSX session check {csv_path.stem}", "",
             f"Samples: {len(log)} ({len(log) - len(ok)} failed)", ""]

    lines.append("## Market State words seen (in order)")
    states = ok[["checked_at", "state"]]
    changes = states[states["state"] != states["state"].shift()]
    lines += [f"- {r.checked_at}: {r.state}" for r in changes.itertuples()] + [""]

    trading = ok[ok["state"].astype(str).str.upper() != "CLOSED"]
    lines.append("## How often PSX's 'As of' time advanced while not Closed")
    if not trading.empty:
        as_of = pd.to_datetime(trading["as_of"], errors="coerce").dropna().drop_duplicates()
        gaps = as_of.diff().dropna().dt.total_seconds() / 60
        lines.append(f"- distinct As-of times: {len(as_of)}; median gap "
                     f"{gaps.median():.1f} min, max gap {gaps.max():.1f} min"
                     if not gaps.empty else "- As-of never advanced")
        lag = pd.to_numeric(trading["as_of_lag_min"], errors="coerce")
        lines.append(f"- lag behind the clock: median {lag.median():.1f} min, "
                     f"max {lag.max():.1f} min")
        changed = trading["mw_changed"].astype(str).str.lower().eq("true")
        lines.append(f"- Market Watch table changed in {changed.sum()} of "
                     f"{len(trading)} samples")
        gap = pd.to_numeric(trading["volume_gap_pct"], errors="coerce").abs()
        lines.append(f"- Market Watch vs PSX volume gap: median {gap.median():.3f}%, "
                     f"max {gap.max():.3f}%")
        problems = trading["problem"].fillna("")
        lines.append(f"- samples that would be rejected: {(problems != '').sum()}")
        for text in problems[problems != ""].unique()[:5]:
            lines.append(f"  - {text}")
    else:
        lines.append("- the market was never open during sampling")
    lines.append("")

    lines.append("## OGDC daily history during the session")
    if "history_last_row" in ok:
        hist = ok[ok["history_last_row"].fillna("") != ""][["checked_at", "history_last_row"]]
        lines += [f"- {r.checked_at}: {r.history_last_row}" for r in hist.itertuples()]
    lines.append("")

    if comparison:
        lines.append("## Final Market Watch values vs official daily history")
        lines += [f"- {line}" for line in comparison]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interval", type=int, default=60, help="seconds between samples")
    parser.add_argument("--until", default="17:30", help="stop time, PKT (HH:MM)")
    parser.add_argument("--report", type=Path, help="summarise an existing CSV and exit")
    args = parser.parse_args()

    if args.report:
        print(report(args.report))
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(PKT)
    csv_path = OUT_DIR / f"{today:%Y-%m-%d}.csv"
    stop = datetime.combine(today.date(), datetime.strptime(args.until, "%H:%M").time(), PKT)
    clock = MarketClock()
    print(f"Sampling every {args.interval}s until {stop:%H:%M} PKT -> {csv_path}")

    fields, previous_hash, last_snapshot = None, None, None
    next_history = datetime.now(PKT)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        while datetime.now(PKT) < stop:
            row, previous_hash, snapshot = sample(previous_hash)
            last_snapshot = snapshot or last_snapshot
            row["history_last_row"] = ""
            if datetime.now(PKT) >= next_history and clock.in_window():
                row["history_last_row"] = history_last_row()
                next_history = datetime.now(PKT) + HISTORY_EVERY
            if fields is None:
                fields = list(row) + ["error"]
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                if handle.tell() == 0:
                    writer.writeheader()
            writer.writerow(row)
            handle.flush()
            print(f"{row['checked_at']}  {row.get('state', row.get('error'))}  "
                  f"as of {row.get('as_of', '')}  MW changed {row.get('mw_changed', '')}  "
                  f"vol gap {row.get('volume_gap_pct', '')}%  {row.get('problem', '')}")
            time.sleep(args.interval)

    comparison = compare_with_history(last_snapshot) if last_snapshot else None
    text = report(csv_path, comparison)
    report_path = csv_path.with_suffix(".md")
    report_path.write_text(text, encoding="utf-8")
    print("\n" + text + f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
