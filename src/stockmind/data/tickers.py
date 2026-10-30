"""Validate and normalise PSX ticker symbols."""

from stockmind.config import SUPPORTED_TICKERS


# PSX temporarily appends these to a symbol around corporate actions,
# e.g. "HUBCXD" means HUBC is trading ex-dividend. Data endpoints only
# accept the base symbol ("HUBC").
ENTITLEMENT_SUFFIXES = ("XD", "XB", "XR", "XA")


class UnsupportedTickerError(ValueError):
    """Raised when a ticker is not part of StockMind's supported universe."""


def base_symbol(symbol: str, known: set[str] | tuple[str, ...]) -> str:
    """
    Remove a temporary entitlement suffix if what remains is a known symbol.

    "HUBCXD" -> "HUBC" when "HUBC" is known. Symbols that merely end in
    these letters are left alone because their base is not a known symbol.
    """
    symbol = str(symbol).strip().upper()

    if symbol in known:
        return symbol

    for suffix in ENTITLEMENT_SUFFIXES:
        base = symbol.removesuffix(suffix)
        if base != symbol and base in known:
            return base

    return symbol


def resolve_ticker(
    ticker: str,
    supported: tuple[str, ...] = SUPPORTED_TICKERS,
) -> str:
    """
    Return the canonical PSX symbol for a user-supplied ticker.

    Examples:
        " ogdc "  -> "OGDC"
        "HUBCXD"  -> "HUBC"

    Raises:
        UnsupportedTickerError: If the ticker is empty or not supported.
    """
    symbol = base_symbol(ticker, supported)

    if not symbol:
        raise UnsupportedTickerError("Ticker cannot be empty.")

    if symbol in supported:
        return symbol

    raise UnsupportedTickerError(
        f"{symbol!r} is not supported. "
        f"StockMind covers: {', '.join(supported)}."
    )
