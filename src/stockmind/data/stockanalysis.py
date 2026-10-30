"""Financial statements and dividend history from stockanalysis.com.

Why a second source?
    The PSX Data Portal has no balance sheet (so no shareholders' equity,
    which ROE needs) and only shows about 2 years of payouts. Its results
    can also be the company's standalone figures (e.g. ENGROH), while
    equity here is consolidated - so ROE must take profit and equity from
    the same source.
    stockanalysis.com publishes PSX-listed companies' consolidated annual
    statements (last 5 fiscal years) and a longer cash-dividend history.
    Its robots.txt allows these pages.

Pages used:
    /quote/psx/<SYMBOL>/financials/income-statement/
    /quote/psx/<SYMBOL>/financials/balance-sheet/
    /quote/psx/<SYMBOL>/dividend/

Fetching and parsing are separate so the parsers can be tested offline.
"""

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from stockmind.data.company import parse_number


BASE_URL = "https://stockanalysis.com/quote/psx"
REQUEST_TIMEOUT = 30  # seconds

# The balance sheet is published in millions of PKR. We convert to
# thousands so it uses the same unit as the PSX financials.
MILLIONS_TO_THOUSANDS = 1_000

_DATE = re.compile(r"[A-Z][a-z]{2} \d{1,2}, \d{4}")  # e.g. "Jun 30, 2026"


def _session() -> requests.Session:
    """HTTP session with a browser User-Agent and automatic retries."""
    session = requests.Session()
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    )
    retries = Retry(total=3, backoff_factor=1,
                    status_forcelist=(429, 500, 502, 503, 504))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _get(path: str) -> str:
    response = _session().get(f"{BASE_URL}/{path}", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def fetch_balance_sheet(symbol: str) -> pd.DataFrame:
    return parse_statement(_get(f"{symbol}/financials/balance-sheet/"))


def fetch_income_statement(symbol: str) -> pd.DataFrame:
    return parse_statement(_get(f"{symbol}/financials/income-statement/"))


def fetch_dividend_history(symbol: str) -> pd.DataFrame:
    return parse_dividend_history(_get(f"{symbol}/dividend/"))


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _snake_case(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def _unit_and_scale(label: str, raw: str) -> tuple[str, float]:
    """Decide the unit of a statement row and how to convert it."""
    lowered = label.lower()
    if raw.endswith("%"):
        return "%", 1
    if "per share" in lowered or lowered.startswith("eps"):
        return "PKR per share", 1
    if "shares outstanding" in lowered:
        return "shares", 1_000_000  # published in millions of shares
    return "PKR thousands", MILLIONS_TO_THOUSANDS


def parse_statement(html: str) -> pd.DataFrame:
    """
    Parse an annual financial statement (balance sheet or income statement)
    into long format.

    Columns: fiscal_year, period_end, metric, value, unit
    e.g.  FY 2026 | 2026-06-30 | shareholders_equity | 1519547000 | PKR thousands

    "TTM" columns hold the latest twelve months (income statement) or the
    latest quarter-end balance (balance sheet).
    """
    columns = ["fiscal_year", "period_end", "metric", "value", "unit"]
    soup = BeautifulSoup(html, "lxml")
    records = []

    # The page splits a statement over several tables. Each starts with
    # the fiscal-year header row and a "Period Ending" row.
    for table in soup.find_all("table"):
        rows = [
            [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
            for row in table.find_all("tr")
        ]
        if len(rows) < 3 or rows[0][0] != "Fiscal Year":
            continue

        years = rows[0][1:]
        period_row = next((r for r in rows if r[0] == "Period Ending"), None)
        period_ends = [
            pd.to_datetime(match[-1]) if (match := _DATE.findall(cell)) else pd.NaT
            for cell in (period_row[1:] if period_row else [""] * len(years))
        ]

        for row in rows:
            label = row[0]
            if label in ("Fiscal Year", "Period Ending"):
                continue
            for year, period_end, raw in zip(years, period_ends, row[1:]):
                value = parse_number(raw)
                if value is None:
                    continue
                unit, scale = _unit_and_scale(label, raw)
                records.append({
                    "fiscal_year": year,
                    "period_end": period_end,
                    "metric": _snake_case(label),
                    "value": value * scale,
                    "unit": unit,
                })

    return pd.DataFrame(records, columns=columns)


def parse_dividend_history(html: str) -> pd.DataFrame:
    """
    Parse cash dividend history, newest first.

    Columns: ex_date, cash_per_share, record_date, pay_date
    The ex-date is the first day a buyer no longer receives the dividend.
    """
    columns = ["ex_date", "cash_per_share", "record_date", "pay_date"]
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    records = []

    if table is not None:
        for row in table.find_all("tr")[1:]:
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]
            if len(cells) < 4:
                continue
            records.append({
                "ex_date": pd.to_datetime(cells[0], errors="coerce"),
                "cash_per_share": parse_number(cells[1].replace("PKR", "")),
                "record_date": pd.to_datetime(cells[2], errors="coerce"),
                "pay_date": pd.to_datetime(cells[3], errors="coerce"),
            })

    return pd.DataFrame(records, columns=columns)
