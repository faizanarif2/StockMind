"""Market-wide listing of PSX equities and sector-peer selection.

The Fundamental Agent compares a company with the average of its industry,
so Agent 1 needs to know which other companies are in the same sector and
how large they are. Three PSX tables provide that (all via psxdata):

    symbols()           every listed symbol with its sector name
    indices("ALLSHR")   market cap and current price of every listed share
    screener()          P/E ratio and dividend yield

They are combined into one "market listing" table. Building it is kept
separate from fetching so it can be tested offline.
"""

import pandas as pd
import psxdata

from stockmind.data.tickers import base_symbol


LISTING_COLUMNS = [
    "symbol", "name", "sector", "market_cap_m", "price",
    "pe_ratio", "dividend_yield",
]


def fetch_market_listing() -> pd.DataFrame:
    """Download and combine the PSX tables into one listing."""
    return build_market_listing(
        symbols=psxdata.symbols(cache=False),
        all_share_index=psxdata.indices("ALLSHR", cache=False),
        screener=psxdata.screener(cache=False),
    )


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False),
        errors="coerce",
    )


def build_market_listing(
    symbols: pd.DataFrame,
    all_share_index: pd.DataFrame,
    screener: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per listed equity:
    symbol, name, sector, market_cap_m (PKR millions), price,
    pe_ratio, dividend_yield (%).

    ETFs and debt instruments are excluded. Index and screener rows that
    carry a temporary suffix (e.g. "HUBCXD") are matched to the base symbol.
    """
    equities = symbols.loc[
        ~symbols["is_etf"] & ~symbols["is_debt"],
        ["symbol", "name", "sector_name"],
    ].rename(columns={"sector_name": "sector"})
    known = set(equities["symbol"])

    index = pd.DataFrame({
        "symbol": all_share_index["symbol"].map(lambda s: base_symbol(s, known)),
        "market_cap_m": _numeric(all_share_index["market_cap_m"]),
        "price": _numeric(all_share_index["current"]),
    })
    valuation = pd.DataFrame({
        "symbol": screener["symbol"].map(lambda s: base_symbol(s, known)),
        # PSX shows 0.00 when it has no P/E, which is not a real P/E of zero.
        "pe_ratio": _numeric(screener["pe_ratio"]).replace(0, float("nan")),
        "dividend_yield": _numeric(screener["dividend_yield"]),
    })

    listing = (
        equities
        .merge(index.drop_duplicates("symbol"), on="symbol", how="left")
        .merge(valuation.drop_duplicates("symbol"), on="symbol", how="left")
    )
    return listing[LISTING_COLUMNS].reset_index(drop=True)


def find_sector_peers(symbol: str, listing: pd.DataFrame, count: int) -> list[str]:
    """
    Return the `count` largest other companies in the same PSX sector,
    ranked by market capitalisation.

    Returns an empty list if the symbol or its sector is not in the listing.
    """
    company = listing[listing["symbol"] == symbol]
    if company.empty:
        return []

    sector = company["sector"].iloc[0]
    peers = listing[
        (listing["sector"] == sector)
        & (listing["symbol"] != symbol)
        & listing["market_cap_m"].notna()
    ]
    return peers.nlargest(count, "market_cap_m")["symbol"].tolist()
