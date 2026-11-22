"""The stock universe: current index constituents and their categories.

The Stock Selection stage starts from every company in a PSX index
(config.SELECTION_INDEX, the KMI-30), fetched live so the system follows
PSX's semi-annual rebalancing on its own. Nothing here lists companies.

    psxdata.indices(index)   constituents and their index weights
    market listing           each company's PSX sector, P/E, dividend yield
                             (see market.py; shared with Agent 1's peers)
    config.SECTOR_CATEGORIES PSX sector -> broad category

Building the table is kept separate from fetching so it can be tested
offline.
"""

import pandas as pd
import psxdata

from stockmind import config
from stockmind.data.tickers import base_symbol


CONSTITUENT_COLUMNS = [
    "symbol", "name", "sector", "category", "idx_weight", "market_cap_m",
    "price", "pe_ratio", "dividend_yield",
]

UNCLASSIFIED = "Unclassified"


def fetch_index_table(index: str) -> pd.DataFrame:
    """Download an index's constituents (symbol, name, idx_weight, ...)."""
    table = psxdata.indices(index, cache=False)
    if table.empty:
        raise ValueError(f"PSX returned no constituents for {index}.")
    return table


def categorise(sector: str | None) -> tuple[str, str | None]:
    """
    Return (category, warning) for a PSX sector name.

    Unknown sectors are not guessed: they become their own category, with a
    warning so the mapping in config.SECTOR_CATEGORIES can be extended.
    """
    if sector is None or pd.isna(sector) or not str(sector).strip():
        return UNCLASSIFIED, "has no PSX sector; ranked in its own 'Unclassified' category."

    sector = str(sector).strip()
    category = config.SECTOR_CATEGORIES.get(sector.upper())
    if category is not None:
        return category, None
    return sector.title(), (
        f"is in PSX sector '{sector}', which is not in "
        "config.SECTOR_CATEGORIES; ranked as its own category."
    )


def build_constituents(
    index_table: pd.DataFrame, listing: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """
    One row per constituent (see CONSTITUENT_COLUMNS), largest weight first,
    plus warnings about companies whose category had to be improvised.

    Symbols with a temporary suffix (e.g. "HUBCXD") are matched to the base
    symbol, as in the market listing.
    """
    known = set(listing["symbol"]) if not listing.empty else set()
    table = pd.DataFrame({
        "symbol": index_table["symbol"].map(lambda s: base_symbol(s, known)),
        "index_name": index_table.get("name"),
        "idx_weight": pd.to_numeric(
            index_table["idx_weight"].astype("string").str.replace(",", ""),
            errors="coerce",
        ),
    }).drop_duplicates("symbol")

    details = (listing.drop_duplicates("symbol") if not listing.empty
               else pd.DataFrame(columns=["symbol", "name", "sector"]))
    table = table.merge(details, on="symbol", how="left")
    table["name"] = table["name"].fillna(table["index_name"])
    for column in CONSTITUENT_COLUMNS:
        if column not in table.columns:
            table[column] = float("nan")

    warnings = []
    categories = []
    for symbol, sector in zip(table["symbol"], table["sector"]):
        category, warning = categorise(sector)
        categories.append(category)
        if warning:
            warnings.append(f"{symbol} {warning}")
    table["category"] = categories

    table = table.sort_values("idx_weight", ascending=False, na_position="last")
    return table[CONSTITUENT_COLUMNS].reset_index(drop=True), warnings
