"""Fetch and parse company information from the PSX Data Portal.

Source pages (official PSX data, https://dps.psx.com.pk):

    GET  /company/<SYMBOL>    profile, equity (shares, free float),
                              annual + quarterly financials, ratios,
                              market snapshot (P/E, 52-week range)
    POST /company/payouts     dividend / bonus / right announcements

psxdata does not expose these pages, so we parse them here. We reuse
psxdata's BaseScraper for the HTTP work because it already handles the PSX
request token, retries, and rate limiting.

Parsing is kept in separate pure functions (``parse_*``) so it can be tested
with saved HTML, without any network access.
"""

import re

import pandas as pd
from bs4 import BeautifulSoup
from psxdata import BaseScraper
from psxdata.constants import BASE_URL


# PSX reports amounts in thousands of rupees, except EPS.
AMOUNT_UNIT = "PKR thousands"
EPS_UNIT = "PKR per share"


class CompanyPageScraper(BaseScraper):
    """Download the raw HTML used by the parsers below."""

    def company_page(self, symbol: str) -> str:
        return self._request("GET", f"{BASE_URL}/company/{symbol}").text

    def payouts(self, symbol: str) -> str:
        return self._request(
            "POST", f"{BASE_URL}/company/payouts", data={"symbol": symbol}
        ).text


# Created on first use so importing this module never touches the network.
_scraper: CompanyPageScraper | None = None


def _get_scraper() -> CompanyPageScraper:
    global _scraper
    if _scraper is None:
        _scraper = CompanyPageScraper()
    return _scraper


def fetch_company(symbol: str) -> dict:
    """
    Fetch and parse the company page.

    Returns {"profile": dict, "financials": DataFrame, "ratios": DataFrame}.
    """
    html = _get_scraper().company_page(symbol)
    return {
        "profile": parse_profile(html, symbol),
        "financials": parse_financials(html),
        "ratios": parse_ratios(html),
    }


def fetch_payouts(symbol: str) -> pd.DataFrame:
    """Fetch and parse dividend / bonus / right-share announcements."""
    return parse_payouts(_get_scraper().payouts(symbol))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def parse_number(text: str | None) -> float | None:
    """
    Convert PSX number formats to float.

    "1,234.50" -> 1234.5,  "(18.71)" -> -18.71,  "15.00%" -> 15.0,
    "" or "-"  -> None
    """
    if text is None:
        return None

    cleaned = text.strip().replace(",", "").replace("%", "").replace("Rs.", "")
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()").strip()

    try:
        value = float(cleaned)
    except ValueError:
        return None

    return -value if negative else value


def _snake_case(label: str) -> str:
    """
    'Profit after Taxation'   -> 'profit_after_taxation'
    'Net Profit Margin (%)'   -> 'net_profit_margin_pct'
    """
    name = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"{name}_pct" if "(%)" in label else name


def _table_rows(table) -> list[list[str]]:
    return [
        [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        for row in table.find_all("tr")
    ]


def _table_to_long(table, **extra_columns) -> pd.DataFrame:
    """
    Turn a PSX "metric x period" table into one row per (period, metric).

    Long format makes it easy to merge new periods with stored history.
    Empty cells (e.g. banks have no "Sales") are skipped rather than
    stored as zero.
    """
    rows = _table_rows(table)
    if len(rows) < 2:
        return pd.DataFrame()

    periods = rows[0][1:]
    records = []

    for row in rows[1:]:
        metric = _snake_case(row[0])
        for period, cell in zip(periods, row[1:]):
            value = parse_number(cell)
            if value is not None:
                records.append(
                    {**extra_columns, "period": period,
                     "metric": metric, "value": value}
                )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Company page
# ---------------------------------------------------------------------------

def parse_profile(html: str, symbol: str) -> dict:
    """Extract company details, equity profile, and market snapshot."""
    soup = BeautifulSoup(html, "lxml")

    def text_of(selector: str) -> str | None:
        element = soup.select_one(selector)
        # The first string avoids badges such as "XD" inside the element.
        return next(element.stripped_strings, None) if element else None

    profile = {
        "ticker": symbol,
        "name": text_of(".quote__name"),
        "sector": text_of(".quote__sector"),
    }

    if not profile["name"]:
        raise ValueError(f"PSX company page for {symbol} has no company name.")

    # Profile items are "heading followed by paragraph" pairs.
    wanted = {
        "business description": "description",
        "fiscal year end": "fiscal_year_end",
        "website": "website",
        "auditor": "auditor",
    }
    for head in soup.select("#profile .item__head"):
        key = wanted.get(head.get_text(strip=True).lower())
        paragraph = head.find_next_sibling("p")
        if key and paragraph:
            profile[key] = paragraph.get_text(" ", strip=True)

    # Equity profile: market cap, shares outstanding, face value, free float.
    equity = {}
    for item in soup.select("#equity .stats_item"):
        label_el = item.select_one(".stats_label")
        value_el = item.select_one(".stats_value")
        if not label_el or not value_el:
            continue
        label = label_el.get_text("", strip=True).lower()
        raw_value = value_el.get_text(strip=True)
        value = parse_number(raw_value)

        if label.startswith("market cap"):
            equity["market_cap_thousands"] = value
        elif label == "shares":
            equity["shares_outstanding"] = value
        elif label == "face value":
            equity["face_value"] = value
        elif label == "free float":
            key = "free_float_pct" if raw_value.endswith("%") else "free_float_shares"
            equity[key] = value
    profile["equity"] = equity

    # Market snapshot. The page repeats some labels for other markets
    # (futures etc.), so we keep the first occurrence: the regular market.
    labels = {
        "ldcp": "last_close",
        "p/e ratio (ttm)": "pe_ratio_ttm",
        "52-week range": "week_52_range",
        "1-year change": "change_1y_pct",
        "ytd change": "change_ytd_pct",
    }
    market = {"price": parse_number(text_of(".quote__close"))}
    for item in soup.select(".stats_item"):
        label_el = item.select_one(".stats_label")
        value_el = item.select_one(".stats_value")
        if not label_el or not value_el:
            continue
        label = re.sub(r"[*^]", "", label_el.get_text(" ", strip=True)).strip().lower()
        key = labels.get(label)
        if key is None or key in market:
            continue
        value_text = value_el.get_text(" ", strip=True)
        if key == "week_52_range":
            numbers = re.findall(r"\d[\d,]*\.?\d*", value_text)
            if len(numbers) == 2:
                market["week_52_low"] = parse_number(numbers[0])
                market["week_52_high"] = parse_number(numbers[1])
            market[key] = value_text
        else:
            market[key] = parse_number(value_text)
    market.pop("week_52_range", None)
    profile["market"] = market

    return profile


def parse_financials(html: str) -> pd.DataFrame:
    """
    Extract annual and quarterly results in long format.

    Columns: period_type, period, metric, value, unit
    Metrics depend on the business, e.g. "sales" for most companies but
    "mark_up_earned" / "total_income" for banks; "eps" is always present.
    """
    soup = BeautifulSoup(html, "lxml")
    frames = []

    for panel in soup.select("#financials .tabs__panel"):
        table = panel.find("table")
        if table is None:
            continue
        period_type = panel.get("data-name", "").lower()  # annual / quarterly
        frames.append(_table_to_long(table, period_type=period_type))

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(
            columns=["period_type", "period", "metric", "value", "unit"]
        )

    financials = pd.concat(frames, ignore_index=True)
    financials["unit"] = financials["metric"].map(
        lambda metric: EPS_UNIT if metric == "eps" else AMOUNT_UNIT
    )
    return financials


def parse_ratios(html: str) -> pd.DataFrame:
    """
    Extract the annual ratios table (margins, EPS growth, PEG).

    Columns: period, metric, value
    Percent metrics keep the "_pct" suffix, e.g. net_profit_margin_pct.
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("#ratios table")

    if table is None:
        return pd.DataFrame(columns=["period", "metric", "value"])

    return _table_to_long(table)


# ---------------------------------------------------------------------------
# Dividends and other payouts
# ---------------------------------------------------------------------------

# One entitlement inside the "Details" column, e.g. "60% (F) (D)" or
# "32.50%(iii) (D)". The tags after the percentage say what it is:
# (D) cash dividend, (B) bonus shares, (R) right shares;
# (F) final, (i)/(ii)/(iii) interim number.
_ENTITLEMENT = re.compile(r"([\d.]+)\s*%\s*((?:\([^)]*\)\s*)*)")


def parse_payout_details(details: str) -> dict:
    """
    Split a payout description into cash / bonus / right percentages.

    "60% (F) (D)"          -> cash 60, final
    "20% (D) 10% (B)"      -> cash 20, bonus 10
    "40.50% (D)"           -> cash 40.5
    Percentages are of the share's face value.
    """
    result = {"cash_dividend_pct": None, "bonus_pct": None,
              "right_pct": None, "is_final": "(F)" in details.upper()}

    for percent, tags in _ENTITLEMENT.findall(details):
        tags = tags.upper()
        value = float(percent)
        if "(B)" in tags:
            result["bonus_pct"] = value
        elif "(R)" in tags:
            result["right_pct"] = value
        elif "(D)" in tags:
            result["cash_dividend_pct"] = value

    return result


def parse_payouts(html: str) -> pd.DataFrame:
    """Parse the payouts table into one row per announcement."""
    columns = [
        "announced_at", "period_end", "period_label", "details",
        "cash_dividend_pct", "bonus_pct", "right_pct", "is_final",
        "book_closure_start", "book_closure_end",
    ]
    soup = BeautifulSoup(html, "lxml")
    records = []

    for row in _table_rows(soup)[1:] if soup.find("table") else []:
        if len(row) < 4:
            continue
        announced, results_for, details, book_closure = row[:4]

        # "30/06/2026(YR)" -> period end + label (YR, HYR, IQ, IIIQ ...)
        period_match = re.match(r"([\d/]+)\s*\((\w+)\)", results_for)
        closure_dates = re.findall(r"\d{2}/\d{2}/\d{4}", book_closure)

        record = {
            "announced_at": pd.to_datetime(announced, errors="coerce"),
            "period_end": pd.to_datetime(
                period_match.group(1) if period_match else None,
                dayfirst=True, errors="coerce",
            ),
            "period_label": period_match.group(2) if period_match else None,
            "details": details,
            **parse_payout_details(details),
            "book_closure_start": pd.to_datetime(
                closure_dates[0] if closure_dates else None,
                dayfirst=True, errors="coerce",
            ),
            "book_closure_end": pd.to_datetime(
                closure_dates[-1] if closure_dates else None,
                dayfirst=True, errors="coerce",
            ),
        }
        records.append(record)

    return pd.DataFrame(records, columns=columns)


def add_cash_per_share(payouts: pd.DataFrame, face_value: float | None) -> pd.DataFrame:
    """
    Convert dividend percentages to rupees per share.

    PSX quotes dividends as a percentage of face value, so
    60% on a Rs 10 share = Rs 6.00 per share. If the face value is
    unknown the amount is left empty rather than guessed.
    """
    payouts = payouts.copy()
    cash_pct = pd.to_numeric(payouts["cash_dividend_pct"], errors="coerce")
    payouts["cash_dividend_per_share"] = (
        (cash_pct / 100 * face_value).round(4) if face_value else float("nan")
    )
    return payouts
