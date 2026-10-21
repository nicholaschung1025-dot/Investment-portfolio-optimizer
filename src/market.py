"""Market metadata: currency, board lot and exchange for each ticker.

Some of the inputs this package needs are properties of the listing rather
than of the company: which currency an asset trades in, and how many shares
make up one board lot. Those change rarely, so they live in a small built-in
table with two escape hatches -- a fallback rule based on the ticker suffix,
and a CSV file the user can supply to override anything.

Currency conversion lives here too, because a portfolio that spans markets
cannot add up market capitalizations expressed in different currencies.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.errors import OptimizationError

DEFAULT_HK_BOARD_LOT = 100
DEFAULT_US_BOARD_LOT = 1
DEFAULT_USD_HKD = 7.82


@dataclass(frozen=True)
class MarketInfo:
    """How a ticker is listed.

    Attributes:
        currency: Currency the asset is quoted in.
        board_lot: Number of shares that make up one board lot.
        market: Market code, ``"US"`` or ``"HK"``.
    """

    currency: str
    board_lot: int
    market: str


# Board lots are exchange rules, not company data, so they are stable enough
# to ship with the package. Anything not listed here falls back to the
# suffix rules below.
_BUILT_IN_MARKET_INFO: dict[str, MarketInfo] = {
    "0001.HK": MarketInfo("HKD", 500, "HK"),
    "0002.HK": MarketInfo("HKD", 500, "HK"),
    "0003.HK": MarketInfo("HKD", 500, "HK"),
    "0005.HK": MarketInfo("HKD", 500, "HK"),
    "0011.HK": MarketInfo("HKD", 500, "HK"),
    "0016.HK": MarketInfo("HKD", 500, "HK"),
    "0027.HK": MarketInfo("HKD", 500, "HK"),
    "0388.HK": MarketInfo("HKD", 100, "HK"),
    "0700.HK": MarketInfo("HKD", 100, "HK"),
    "0883.HK": MarketInfo("HKD", 500, "HK"),
    "0941.HK": MarketInfo("HKD", 500, "HK"),
    "1299.HK": MarketInfo("HKD", 500, "HK"),
    "1810.HK": MarketInfo("HKD", 200, "HK"),
    "2318.HK": MarketInfo("HKD", 100, "HK"),
    "3690.HK": MarketInfo("HKD", 100, "HK"),
    "9618.HK": MarketInfo("HKD", 50, "HK"),
    "9888.HK": MarketInfo("HKD", 100, "HK"),
    "9988.HK": MarketInfo("HKD", 100, "HK"),
    "9999.HK": MarketInfo("HKD", 100, "HK"),
}


class MarketMetadata:
    """Lookup table for the market a ticker belongs to.

    Args:
        entries: Optional overrides keyed by ticker.
    """

    def __init__(self, entries: Mapping[str, MarketInfo] | None = None) -> None:
        self._entries: dict[str, MarketInfo] = {
            key.upper(): value for key, value in (entries or {}).items()
        }

    def info(self, ticker: str) -> MarketInfo:
        """Return how ``ticker`` is listed.

        Args:
            ticker: Ticker symbol.

        Returns:
            The matching :class:`MarketInfo`. Unknown Hong Kong tickers fall
            back to HKD with the default board lot; everything else is treated
            as a US listing.
        """
        key = ticker.upper()
        if key in self._entries:
            return self._entries[key]
        if key in _BUILT_IN_MARKET_INFO:
            return _BUILT_IN_MARKET_INFO[key]
        if key.endswith(".HK"):
            return MarketInfo("HKD", DEFAULT_HK_BOARD_LOT, "HK")
        return MarketInfo("USD", DEFAULT_US_BOARD_LOT, "US")

    @classmethod
    def from_csv(cls, path: str | Path) -> "MarketMetadata":
        """Build metadata from a CSV file.

        The file needs the columns ``ticker``, ``currency``, ``board_lot`` and
        ``market``; extra columns are ignored.

        Args:
            path: Path of the CSV file.

        Returns:
            A :class:`MarketMetadata` holding the file's entries.

        Raises:
            OptimizationError: If the file cannot be read or a row is
                malformed.
        """
        entries: dict[str, MarketInfo] = {}
        try:
            with Path(path).open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    ticker = (row.get("ticker") or "").strip().upper()
                    currency = (row.get("currency") or "").strip().upper()
                    if not ticker or not currency:
                        raise OptimizationError(
                            f"market metadata row is missing a ticker or currency: {row!r}"
                        )
                    try:
                        board_lot = int(str(row.get("board_lot", "1")).strip())
                    except ValueError as error:
                        raise OptimizationError(
                            f"board lot for {ticker} is not a number: {row.get('board_lot')!r}"
                        ) from error
                    market = (row.get("market") or "US").strip().upper()
                    entries[ticker] = MarketInfo(currency, board_lot, market)
        except OSError as error:
            raise OptimizationError(f"could not read market metadata from {path}: {error}") from error
        return cls(entries)


@dataclass(frozen=True)
class FxRates:
    """Currency conversion into US dollars.

    Attributes:
        usd_per_unit: US dollar value of one unit of each currency.
    """

    usd_per_unit: dict[str, float]

    @classmethod
    def default(cls, usd_hkd: float = DEFAULT_USD_HKD) -> "FxRates":
        """Return the built-in rates.

        Args:
            usd_hkd: Hong Kong dollars per US dollar.

        Returns:
            The default :class:`FxRates`.
        """
        if usd_hkd <= 0:
            raise OptimizationError(f"the USD/HKD rate must be positive, got {usd_hkd}")
        return cls({"USD": 1.0, "HKD": 1.0 / usd_hkd})

    def to_usd(self, amount: float, currency: str) -> float:
        """Convert an amount into US dollars.

        Args:
            amount: Amount in ``currency``.
            currency: Currency of the amount.

        Returns:
            The amount in US dollars.

        Raises:
            OptimizationError: If the currency has no known rate.
        """
        key = currency.upper()
        if key not in self.usd_per_unit:
            raise OptimizationError(
                f"no exchange rate is known for {key!r}; "
                f"known currencies: {sorted(self.usd_per_unit)}"
            )
        return float(amount) * self.usd_per_unit[key]


def capitalization_weights(market_caps: pd.Series) -> pd.Series:
    """Turn market capitalizations into market-portfolio weights.

    Args:
        market_caps: Market capitalization per ticker, in any single currency.

    Returns:
        Weights that sum to 1. Tickers without a usable capitalization are
        dropped.

    Raises:
        OptimizationError: If no ticker has a positive capitalization.
    """
    caps = pd.Series(market_caps, dtype=float).dropna()
    caps = caps[caps > 0.0]
    if caps.empty:
        raise OptimizationError(
            "no usable market capitalizations, so the market portfolio cannot be built"
        )
    return caps / caps.sum()
