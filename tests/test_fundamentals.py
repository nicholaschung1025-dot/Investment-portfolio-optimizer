"""Tests for fundamental data fetching, caching and normalization.

The normalization rules matter more than the fetching: Yahoo Finance has
changed the units of some fields between versions, and a dividend yield of
``0.5`` means 0.5% in one version and 50% in another. Every rule here is
pinned down with a test.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.fundamentals import (
    Fundamentals,
    FundamentalsCache,
    fetch_fundamentals,
    normalize_fundamentals,
)


def _raw(**overrides) -> dict:
    base = {
        "currentPrice": 100.0,
        "marketCap": 1_000_000_000.0,
        "currency": "HKD",
        "averageVolume": 5_000_000,
        "trailingAnnualDividendYield": 0.03,
        "targetMeanPrice": 120.0,
        "numberOfAnalystOpinions": 12,
        "profitMargins": 0.25,
        "trailingEps": 4.5,
    }
    base.update(overrides)
    return base


def test_dividend_yield_as_a_fraction_is_left_alone() -> None:
    fundamentals = normalize_fundamentals("0700.HK", _raw(trailingAnnualDividendYield=0.03))

    assert fundamentals.dividend_yield == pytest.approx(0.03)


def test_dividend_yield_as_a_percentage_is_converted() -> None:
    fundamentals = normalize_fundamentals("0700.HK", _raw(trailingAnnualDividendYield=3.0))

    assert fundamentals.dividend_yield == pytest.approx(0.03)


def test_dividend_yield_is_clamped_to_a_plausible_range() -> None:
    fundamentals = normalize_fundamentals("0700.HK", _raw(trailingAnnualDividendYield=0.9))

    assert fundamentals.dividend_yield == pytest.approx(0.5)


def test_missing_fields_become_none_rather_than_zero() -> None:
    fundamentals = normalize_fundamentals("AAA", {"currentPrice": 10.0})

    assert fundamentals.available is True
    assert fundamentals.market_cap is None
    assert fundamentals.avg_volume is None
    assert fundamentals.analyst_target is None
    assert fundamentals.analyst_count is None
    assert fundamentals.dividend_yield is None


def test_an_empty_payload_marks_the_ticker_as_unavailable() -> None:
    fundamentals = normalize_fundamentals("AAA", {})

    assert fundamentals.available is False


def test_market_cap_keeps_the_listing_currency() -> None:
    fundamentals = normalize_fundamentals("0700.HK", _raw())

    assert fundamentals.market_cap == pytest.approx(1_000_000_000.0)
    assert fundamentals.currency == "HKD"


def test_analyst_target_is_none_when_missing_or_not_positive() -> None:
    assert normalize_fundamentals("AAA", _raw(targetMeanPrice=None)).analyst_target is None
    assert normalize_fundamentals("AAA", _raw(targetMeanPrice=0)).analyst_target is None
    assert normalize_fundamentals("AAA", _raw(targetMeanPrice=-5)).analyst_target is None


def test_analyst_count_is_an_integer_or_none() -> None:
    assert normalize_fundamentals("AAA", _raw(numberOfAnalystOpinions=7)).analyst_count == 7
    assert normalize_fundamentals("AAA", _raw(numberOfAnalystOpinions=None)).analyst_count is None


def test_average_volume_is_none_when_missing() -> None:
    fundamentals = normalize_fundamentals("AAA", _raw(averageVolume=None))

    assert fundamentals.avg_volume is None


def test_cache_round_trip_preserves_the_raw_entries(tmp_path) -> None:
    cache = FundamentalsCache(tmp_path / "fundamentals.json")
    cache.write({"AAA": _raw()})

    entries = cache.read()

    assert entries["AAA"]["marketCap"] == 1_000_000_000.0


def test_a_fresh_cache_is_used_without_calling_the_fetcher(tmp_path) -> None:
    cache = FundamentalsCache(tmp_path / "fundamentals.json")
    cache.write({"AAA": _raw()})
    calls: list[str] = []

    def fetcher(ticker: str) -> dict:
        calls.append(ticker)
        return _raw()

    fundamentals = fetch_fundamentals(["AAA"], cache=cache, fetcher=fetcher)

    assert calls == []
    assert fundamentals["AAA"].market_cap == pytest.approx(1_000_000_000.0)


def test_a_stale_cache_triggers_a_refetch(tmp_path) -> None:
    path = tmp_path / "fundamentals.json"
    cache = FundamentalsCache(path)
    stale = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    path.write_text(
        json.dumps({"fetched_at": stale, "tickers": {"AAA": _raw()}}), encoding="utf-8"
    )
    calls: list[str] = []

    def fetcher(ticker: str) -> dict:
        calls.append(ticker)
        return _raw(marketCap=2_000_000_000.0)

    fundamentals = fetch_fundamentals(["AAA"], cache=cache, fetcher=fetcher)

    assert calls == ["AAA"]
    assert fundamentals["AAA"].market_cap == pytest.approx(2_000_000_000.0)


def test_refresh_forces_a_refetch(tmp_path) -> None:
    cache = FundamentalsCache(tmp_path / "fundamentals.json")
    cache.write({"AAA": _raw()})
    calls: list[str] = []

    def fetcher(ticker: str) -> dict:
        calls.append(ticker)
        return _raw(marketCap=3_000_000_000.0)

    fundamentals = fetch_fundamentals(
        ["AAA"], cache=cache, fetcher=fetcher, refresh=True
    )

    assert calls == ["AAA"]
    assert fundamentals["AAA"].market_cap == pytest.approx(3_000_000_000.0)


def test_a_failing_ticker_does_not_break_the_others(tmp_path) -> None:
    def fetcher(ticker: str) -> dict:
        if ticker == "BAD":
            raise RuntimeError("boom")
        return _raw()

    fundamentals = fetch_fundamentals(["BAD", "GOOD"], cache=None, fetcher=fetcher)

    assert fundamentals["BAD"].available is False
    assert fundamentals["GOOD"].available is True


def test_every_requested_ticker_gets_an_entry(tmp_path) -> None:
    fundamentals = fetch_fundamentals(
        ["AAA", "BBB"],
        cache=FundamentalsCache(tmp_path / "f.json"),
        fetcher=lambda ticker: _raw(),
    )

    assert set(fundamentals) == {"AAA", "BBB"}
    assert all(isinstance(entry, Fundamentals) for entry in fundamentals.values())


def test_a_missing_cache_file_is_not_an_error(tmp_path) -> None:
    cache = FundamentalsCache(tmp_path / "does-not-exist.json")

    assert cache.read() == {}
    assert cache.is_fresh() is False


def test_a_failed_fetch_keeps_the_previous_cache_entries(tmp_path) -> None:
    path = tmp_path / "fundamentals.json"
    cache = FundamentalsCache(path)
    stale = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    path.write_text(
        json.dumps({"fetched_at": stale, "tickers": {"AAA": _raw()}}), encoding="utf-8"
    )

    fetch_fundamentals(["AAA"], cache=cache, fetcher=lambda ticker: {})

    assert "AAA" in cache.read()


def test_the_cache_is_written_after_a_fetch(tmp_path) -> None:
    path = tmp_path / "fundamentals.json"
    cache = FundamentalsCache(path)

    fetch_fundamentals(["AAA"], cache=cache, fetcher=lambda ticker: _raw())

    assert path.exists()
    assert "AAA" in cache.read()
