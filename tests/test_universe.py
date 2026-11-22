"""Tests for the index universe and automatic categorisation."""

import pandas as pd

from stockmind import config
from stockmind.data.universe import UNCLASSIFIED, build_constituents, categorise


INDEX = pd.DataFrame({
    # HUBC is trading ex-dividend, so the index lists it as "HUBCXD".
    "symbol": ["FFC", "HUBCXD", "MEBL", "NEWCO", "ODDCO"],
    "name": ["Fauji Fertilizer", "Hub Power", "Meezan Bank", "New Co", "Odd Co"],
    "idx_weight": ["12.34", "8.24", "9.95", "0.50", "0.10"],
})
LISTING = pd.DataFrame({
    "symbol": ["FFC", "HUBC", "MEBL", "NEWCO", "ODDCO", "UBL"],
    "name": ["FFC Ltd", "HUBC Ltd", "Meezan", "NewCo Ltd", "OddCo Ltd", "UBL"],
    "sector": ["FERTILIZER", "POWER GENERATION & DISTRIBUTION",
               "COMMERCIAL BANKS", "SPACE MINING", "", "COMMERCIAL BANKS"],
    "market_cap_m": [323_210.0, 215_861.0, 260_650.0, 1_000.0, 500.0, 900_000.0],
    "price": [571.3, 160.0, 547.5, 10.0, 5.0, 400.0],
    "pe_ratio": [10.24, 11.03, 10.26, float("nan"), 8.0, 7.0],
    "dividend_yield": [9.24, 9.93, 5.5, float("nan"), 0.0, 9.0],
})


def test_known_sectors_map_to_their_category():
    assert categorise("FERTILIZER") == ("Materials", None)
    assert categorise("commercial banks") == ("Financials", None)


def test_unknown_sector_becomes_its_own_category_with_a_warning():
    category, warning = categorise("SPACE MINING")
    assert category == "Space Mining"
    assert "not in config.SECTOR_CATEGORIES" in warning


def test_missing_sector_is_unclassified():
    for sector in (None, "", float("nan")):
        category, warning = categorise(sector)
        assert category == UNCLASSIFIED
        assert warning


def test_constituents_come_from_the_index_not_from_a_list():
    table, _ = build_constituents(INDEX, LISTING)

    # Only index members, UBL is not one; HUBCXD is matched to HUBC.
    assert set(table["symbol"]) == {"FFC", "HUBC", "MEBL", "NEWCO", "ODDCO"}
    assert table["symbol"].tolist()[:3] == ["FFC", "MEBL", "HUBC"]  # by weight


def test_constituents_carry_sector_category_and_valuation():
    table, warnings = build_constituents(INDEX, LISTING)
    rows = table.set_index("symbol")

    assert rows.loc["HUBC", "category"] == "Utilities"
    assert rows.loc["FFC", "category"] == "Materials"
    assert rows.loc["MEBL", "pe_ratio"] == 10.26
    assert rows.loc["NEWCO", "category"] == "Space Mining"
    assert rows.loc["ODDCO", "category"] == UNCLASSIFIED
    assert len(warnings) == 2
    assert any(w.startswith("NEWCO") for w in warnings)


def test_every_category_name_is_used_consistently():
    # A typo in the table would silently create an extra category.
    assert set(config.SECTOR_CATEGORIES.values()) == {
        "Energy", "Utilities", "Materials", "Industrials",
        "Consumer Discretionary", "Consumer Staples", "Health Care",
        "Financials", "Technology & Communication", "Real Estate",
        "Miscellaneous",
    }
