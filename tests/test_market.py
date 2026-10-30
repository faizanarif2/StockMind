"""Tests for the market listing and sector-peer selection."""

import pandas as pd

from stockmind.data.market import build_market_listing, find_sector_peers
from stockmind.data.tickers import base_symbol


SYMBOLS = pd.DataFrame({
    "symbol": ["MEBL", "HBL", "UBL", "MCB", "OGDC", "PPL", "MZNPETF", "HBLTFC"],
    "name": ["Meezan", "HBL", "UBL", "MCB", "OGDC", "PPL", "ETF", "TFC"],
    "sector_name": ["BANKS", "BANKS", "BANKS", "BANKS", "E&P", "E&P",
                    "ETF", "BONDS"],
    "is_etf": [False] * 6 + [True, False],
    "is_debt": [False] * 7 + [True],
})
ALL_SHARE = pd.DataFrame({
    # PPL is trading ex-dividend, so the index lists it as "PPLXD".
    "symbol": ["MEBL", "HBL", "UBL", "OGDC", "PPLXD"],
    "market_cap_m": ["955,249", "427,661", "1,025,940", "1,319,438", "151,460"],
    "current": ["528.61", "290.10", "410.00", "306.78", "213.93"],
})
SCREENER = pd.DataFrame({
    "symbol": ["MEBL", "HBL", "OGDC", "PPLXD"],
    "pe_ratio": [9.91, 0.00, 5.44, 6.10],
    "dividend_yield": [5.5, 7.2, 3.81, 4.0],
})


def listing():
    return build_market_listing(SYMBOLS, ALL_SHARE, SCREENER)


def test_listing_excludes_etfs_and_debt():
    assert set(listing()["symbol"]) == {"MEBL", "HBL", "UBL", "MCB", "OGDC", "PPL"}


def test_listing_matches_suffixed_symbols_and_parses_numbers():
    rows = listing().set_index("symbol")

    assert rows.loc["PPL", "market_cap_m"] == 151_460  # matched from "PPLXD"
    assert rows.loc["PPL", "pe_ratio"] == 6.10
    assert rows.loc["UBL", "market_cap_m"] == 1_025_940
    # PSX shows 0.00 when it has no P/E; that must not become a real zero.
    assert pd.isna(rows.loc["HBL", "pe_ratio"])
    # Companies missing from a source keep their row with empty values.
    assert pd.isna(rows.loc["MCB", "market_cap_m"])


def test_peers_are_largest_in_same_sector_excluding_company():
    assert find_sector_peers("MEBL", listing(), count=4) == ["UBL", "HBL"]


def test_peer_count_is_respected():
    assert find_sector_peers("MEBL", listing(), count=1) == ["UBL"]


def test_unknown_symbol_has_no_peers():
    assert find_sector_peers("XYZ", listing(), count=4) == []


def test_base_symbol_only_strips_suffix_for_known_bases():
    known = {"HUBC", "PPL"}
    assert base_symbol("HUBCXD", known) == "HUBC"
    assert base_symbol("ppl", known) == "PPL"
    assert base_symbol("ABCXD", known) == "ABCXD"  # base "ABC" is unknown
