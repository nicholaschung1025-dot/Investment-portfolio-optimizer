"""Data ingestion and cleaning engine.

Fetches daily adjusted close prices from Yahoo Finance (a free, key-less
source) and turns the raw panel into a clean, aligned matrix ready for the
math engine. Mismatched trading calendars are handled by forward-filling and
by truncating to the common overlap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import pandas as pd

from src.math_engine import compute_log_returns, detect_zero_variance_assets

Downloader = Callable[..., pd.DataFrame]


class DataFetchError(RuntimeError):
    """Raised when prices cannot be fetched or no usable assets remain."""


@dataclass
class PreparedPrices:
    """Cleaned price panel plus a record of what was changed.

    Attributes:
        prices: Aligned, forward-filled price matrix with zero-variance assets
            removed.
        warnings: Human-readable notes about truncation and exclusions.
        excluded_assets: Tickers dropped because they never moved.
    """

    prices: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    excluded_assets: list[str] = field(default_factory=list)


def _extract_close(data: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Pull the per-ticker ``Close`` columns out of a yfinance response."""
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise DataFetchError("downloaded data contains no 'Close' column")
        close = data["Close"]
    else:
        if "Close" not in data.columns:
            raise DataFetchError("downloaded data contains no 'Close' column")
        close = data[["Close"]].copy()
        if len(tickers) == 1:
            close.columns = [tickers[0]]

    return close


def fetch_close_prices(
    tickers: Sequence[str],
    start: str,
    end: str,
    downloader: Downloader | None = None,
) -> pd.DataFrame:
    """Fetch daily adjusted close prices for ``tickers`` between two dates.

    Args:
        tickers: Ticker symbols to download.
        start: Inclusive start date (``YYYY-MM-DD``).
        end: Inclusive end date (``YYYY-MM-DD``).
        downloader: Optional callable used instead of ``yfinance.download``;
            injected in tests to avoid network access.

    Returns:
        DataFrame of close prices indexed by date with one column per ticker.

    Raises:
        DataFetchError: If the download is empty or no requested ticker came back.
    """
    if downloader is None:
        import yfinance as yf

        downloader = yf.download

    data = downloader(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data is None or data.empty:
        raise DataFetchError(
            "no price data returned; check the tickers and the date range, "
            "or your network connection"
        )

    close = _extract_close(data, tickers)
    close.columns = [str(column) for column in close.columns]

    missing = [ticker for ticker in tickers if ticker not in close.columns]
    if missing and len(missing) == len(list(tickers)):
        raise DataFetchError(f"no data returned for any requested ticker: {missing}")

    return close


def prepare_prices(raw_prices: pd.DataFrame) -> PreparedPrices:
    """Align, forward-fill, and prune a raw price panel.

    Mismatched trading calendars are reconciled by forward-filling internal
    gaps, then dropping the leading rows where any asset has not started
    trading yet (which truncates the range to the common overlap). Assets whose
    price never changes are excluded because they make the covariance matrix
    singular.

    Args:
        raw_prices: Raw price panel indexed by date, one column per ticker.

    Returns:
        A :class:`PreparedPrices` with the cleaned panel and any warnings.

    Raises:
        DataFetchError: If no usable assets remain after cleaning.
    """
    if raw_prices is None or raw_prices.empty:
        raise DataFetchError("no price data to prepare")

    warnings: list[str] = []
    prices = raw_prices.sort_index().ffill()
    prices = prices.dropna(axis=1, how="all")

    if prices.empty or prices.shape[1] == 0:
        raise DataFetchError("no usable assets with price data")

    cleaned = prices.dropna(how="any")
    if cleaned.empty:
        raise DataFetchError("no overlapping date range across the requested tickers")

    if len(cleaned) < len(prices):
        warnings.append(
            f"truncated to the overlap period {cleaned.index[0].date()} to "
            f"{cleaned.index[-1].date()}: some assets had shorter histories"
        )

    log_returns = compute_log_returns(cleaned)
    excluded = detect_zero_variance_assets(log_returns)
    if excluded:
        warnings.append(f"excluded zero variance assets: {excluded}")
        cleaned = cleaned.drop(columns=excluded)

    if cleaned.shape[1] == 0:
        raise DataFetchError("all assets were excluded as zero variance")

    return PreparedPrices(prices=cleaned, warnings=warnings, excluded_assets=excluded)
