"""Fundamental data with an on-disk cache.

Yahoo Finance serves company fundamentals through ``Ticker.info``, which is
one HTTP request per ticker and is rate limited. The data changes slowly, so
it is cached on disk with a time-to-live; a run that reuses the cache costs
no network traffic at all.

The cache stores the **raw** payload and normalization happens on read, so
changing a normalization rule does not invalidate the cache.

Normalization exists because Yahoo Finance has changed the units of some
fields between versions. ``trailingAnnualDividendYield`` was a percentage in
older releases and is a fraction in current ones, so a value above 1 is read
as a percentage. Missing values are kept as ``None`` rather than zero, so a
missing field never masquerades as a real measurement.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.errors import OptimizationError
from src.market import FxRates, MarketMetadata

DEFAULT_CACHE_PATH = Path("data/fundamentals_cache.json")
DEFAULT_TTL_HOURS = 24.0
_FETCH_DELAY_SECONDS = 0.15
_MAX_PLAUSIBLE_DIVIDEND_YIELD = 0.5

Fetcher = Callable[[str], Mapping[str, object]]


@dataclass(frozen=True)
class Fundamentals:
    """Normalized fundamental data for one ticker.

    Attributes:
        ticker: Ticker the data belongs to.
        available: Whether a payload was retrieved at all.
        price: Latest price.
        market_cap: Market capitalization, in the listing currency.
        currency: Currency the ticker is quoted in.
        avg_volume: Average daily share volume.
        dividend_yield: Annual dividend yield as a fraction of price.
        analyst_target: Mean analyst price target.
        analyst_count: Number of analyst opinions behind the target.
        profit_margin: Profit margin as a fraction.
        trailing_eps: Trailing earnings per share.
    """

    ticker: str
    available: bool = False
    price: float | None = None
    market_cap: float | None = None
    currency: str | None = None
    avg_volume: float | None = None
    dividend_yield: float | None = None
    analyst_target: float | None = None
    analyst_count: int | None = None
    profit_margin: float | None = None
    trailing_eps: float | None = None


def _optional_float(value: object) -> float | None:
    """Return ``value`` as a float, or ``None`` when it is missing or unusable."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def normalize_dividend_yield(value: object) -> float | None:
    """Normalize a dividend yield into a fraction of price.

    Values above 1 are read as percentages, which is how older Yahoo Finance
    releases reported them. The result is clamped to a plausible range.

    Args:
        value: The raw value from Yahoo Finance.

    Returns:
        The dividend yield as a fraction, or ``None`` when it is missing.
    """
    number = _optional_float(value)
    if number is None:
        return None
    if number < 0.0:
        return None
    if number > 1.0:
        number = number / 100.0
    return min(number, _MAX_PLAUSIBLE_DIVIDEND_YIELD)


def normalize_fundamentals(ticker: str, raw: Mapping[str, object]) -> Fundamentals:
    """Turn a raw Yahoo Finance payload into normalized fundamentals.

    Args:
        ticker: Ticker the payload belongs to.
        raw: The raw payload.

    Returns:
        The normalized :class:`Fundamentals`. An empty payload is reported as
        unavailable rather than as a ticker full of zeroes.
    """
    if not raw:
        return Fundamentals(ticker=ticker, available=False)

    currency = raw.get("currency")
    analyst_count = _optional_float(raw.get("numberOfAnalystOpinions"))
    analyst_target = _optional_float(raw.get("targetMeanPrice"))
    if analyst_target is not None and analyst_target <= 0.0:
        analyst_target = None

    market_cap = _optional_float(raw.get("marketCap"))
    if market_cap is not None and market_cap <= 0.0:
        market_cap = None

    return Fundamentals(
        ticker=ticker,
        available=True,
        price=_optional_float(
            raw.get("currentPrice")
            or raw.get("previousClose")
            or raw.get("regularMarketPrice")
        ),
        market_cap=market_cap,
        currency=str(currency).upper() if currency else None,
        avg_volume=_optional_float(raw.get("averageVolume")),
        dividend_yield=normalize_dividend_yield(raw.get("trailingAnnualDividendYield")),
        analyst_target=analyst_target,
        analyst_count=int(analyst_count) if analyst_count is not None else None,
        profit_margin=_optional_float(raw.get("profitMargins")),
        trailing_eps=_optional_float(raw.get("trailingEps")),
    )


class FundamentalsCache:
    """A JSON file holding the raw fundamentals of every fetched ticker.

    Args:
        path: Where the cache file lives.
        ttl_hours: How long the cache stays fresh.
    """

    def __init__(self, path: str | Path = DEFAULT_CACHE_PATH, ttl_hours: float = DEFAULT_TTL_HOURS) -> None:
        self.path = Path(path)
        self.ttl_hours = float(ttl_hours)

    def read(self) -> dict[str, dict]:
        """Return the cached raw payloads.

        Returns:
            The raw payload of every cached ticker. An empty mapping when the
            cache file does not exist or cannot be parsed.
        """
        if not self.path.exists():
            return {}
        try:
            with self.path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}
        tickers = payload.get("tickers") if isinstance(payload, dict) else None
        if not isinstance(tickers, dict):
            return {}
        return {str(key): dict(value) for key, value in tickers.items() if isinstance(value, dict)}

    def write(self, entries: Mapping[str, Mapping[str, object]]) -> None:
        """Store raw payloads and stamp the cache.

        Args:
            entries: Raw payload per ticker.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "tickers": {str(key): dict(value) for key, value in entries.items()},
        }
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def fetched_at(self) -> datetime | None:
        """Return when the cache was last written."""
        if not self.path.exists():
            return None
        try:
            with self.path.open(encoding="utf-8") as handle:
                stamp = json.load(handle).get("fetched_at")
        except (OSError, json.JSONDecodeError, AttributeError):
            return None
        if not stamp:
            return None
        try:
            return datetime.fromisoformat(stamp)
        except ValueError:
            return None

    def is_fresh(self) -> bool:
        """Return whether the cache is still within its time-to-live."""
        stamp = self.fetched_at()
        if stamp is None:
            return False
        age = datetime.now(timezone.utc) - stamp
        return age < timedelta(hours=self.ttl_hours)


def _yfinance_fetcher(ticker: str) -> Mapping[str, object]:
    """Fetch the raw Yahoo Finance payload for one ticker.

    The delay between calls keeps the request rate inside what Yahoo Finance
    tolerates; it belongs to the network client, not to the orchestration, so
    an injected fetcher never waits.
    """
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info
    finally:
        time.sleep(_FETCH_DELAY_SECONDS)
    return info if isinstance(info, dict) else {}


def fetch_fundamentals(
    tickers: Sequence[str],
    cache: FundamentalsCache | None = None,
    fetcher: Fetcher | None = None,
    refresh: bool = False,
) -> dict[str, Fundamentals]:
    """Fetch fundamentals for several tickers, reusing a fresh cache.

    Args:
        tickers: Tickers to fetch.
        cache: Optional on-disk cache.
        fetcher: Callable returning the raw payload for a ticker; injected in
            tests so nothing ever touches the network.
        refresh: Whether to ignore the cache and fetch everything again.

    Returns:
        Normalized fundamentals for every requested ticker. A ticker whose
        fetch failed is reported as unavailable instead of raising.
    """
    fetch = fetcher if fetcher is not None else _yfinance_fetcher
    requested = [ticker.upper() for ticker in tickers]

    raw_entries: dict[str, dict] = {}
    if cache is not None and not refresh and cache.is_fresh():
        raw_entries = {
            ticker: payload
            for ticker, payload in cache.read().items()
            if ticker in requested
        }

    missing = [ticker for ticker in requested if ticker not in raw_entries]
    fetched: dict[str, dict] = {}
    for ticker in missing:
        try:
            payload = fetch(ticker)
        except Exception:  # noqa: BLE001 - one bad ticker must not stop the run
            payload = {}
        if payload:
            fetched[ticker] = dict(payload)

    if cache is not None and fetched:
        # A failed fetch must not erase what the cache already knows.
        cache.write({**raw_entries, **fetched})

    entries = {**raw_entries, **fetched}
    return {
        ticker: normalize_fundamentals(ticker, entries.get(ticker, {}))
        for ticker in requested
    }


def market_caps_in_usd(
    fundamentals: Mapping[str, Fundamentals],
    metadata: MarketMetadata,
    fx_rates: FxRates,
) -> pd.Series:
    """Convert market capitalizations into US dollars.

    Yahoo Finance reports a market capitalization in the currency the asset
    is listed in, so capitalizations from different markets cannot be added
    up until they are converted.

    Args:
        fundamentals: Normalized fundamentals per ticker.
        metadata: Market metadata, used for the listing currency.
        fx_rates: Exchange rates used for the conversion.

    Returns:
        Market capitalization per ticker in US dollars. Tickers without a
        usable capitalization are left out.
    """
    caps: dict[str, float] = {}
    for ticker, entry in fundamentals.items():
        if not entry.available or entry.market_cap is None:
            continue
        currency = metadata.info(ticker).currency
        caps[ticker] = fx_rates.to_usd(entry.market_cap, currency)
    return pd.Series(caps, dtype=float)
