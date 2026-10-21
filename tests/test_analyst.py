"""Tests for analyst views and quality filters.

Consensus price targets are a weak, optimistic and momentum-contaminated
signal, so they are turned into a *view* with a modest confidence rather
than into an expected return. These tests pin down how that implied return
is computed, when it is thrown away, and what the quality filters exclude.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analyst import analyst_implied_return, analyst_views
from src.filters import QualityRules, apply_quality_filters
from src.fundamentals import Fundamentals
from src.market import MarketMetadata


def _fundamentals(
    ticker: str,
    target: float | None = 120.0,
    analyst_count: int | None = 20,
    market_cap: float | None = 500e9,
    avg_volume: float | None = 10_000_000,
    dividend_yield: float | None = 0.01,
    price: float | None = 100.0,
) -> Fundamentals:
    return Fundamentals(
        ticker=ticker,
        available=True,
        price=price,
        market_cap=market_cap,
        currency="USD",
        avg_volume=avg_volume,
        dividend_yield=dividend_yield,
        analyst_target=target,
        analyst_count=analyst_count,
        profit_margin=0.2,
        trailing_eps=5.0,
    )


def test_the_implied_return_adds_the_dividend_to_the_price_gap() -> None:
    implied = analyst_implied_return(120.0, 100.0, 0.02)

    assert implied == pytest.approx(0.22)


def test_the_implied_return_without_a_dividend_is_the_price_gap() -> None:
    implied = analyst_implied_return(120.0, 100.0, None)

    assert implied == pytest.approx(0.20)


def test_a_target_below_the_price_produces_a_negative_view() -> None:
    implied = analyst_implied_return(80.0, 100.0, None)

    assert implied == pytest.approx(-0.20)


def test_views_are_built_for_every_ticker_with_a_target() -> None:
    fundamentals = {
        "AAA": _fundamentals("AAA", target=120.0),
        "BBB": _fundamentals("BBB", target=90.0),
        "CCC": _fundamentals("CCC", target=None),
    }

    views = analyst_views(fundamentals, {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0})

    assert set(views) == {"AAA", "BBB"}
    assert views["AAA"] == pytest.approx(0.21)
    assert views["BBB"] == pytest.approx(-0.09)


def test_a_ticker_without_a_price_produces_no_view() -> None:
    fundamentals = {"AAA": _fundamentals("AAA", price=None)}

    views = analyst_views(fundamentals, {})

    assert views == {}


def test_an_outlier_view_is_dropped() -> None:
    fundamentals = {
        "AAA": _fundamentals("AAA", target=110.0),
        "BBB": _fundamentals("BBB", target=112.0),
        "CCC": _fundamentals("CCC", target=108.0),
        "DDD": _fundamentals("DDD", target=500.0),
    }
    prices = {"AAA": 100.0, "BBB": 100.0, "CCC": 100.0, "DDD": 100.0}
    prior = pd.Series({"AAA": 0.05, "BBB": 0.05, "CCC": 0.05, "DDD": 0.05})

    views = analyst_views(fundamentals, prices, prior=prior, outlier_sigma=3.0)

    assert "DDD" not in views
    assert set(views) == {"AAA", "BBB", "CCC"}


def test_a_small_analyst_count_is_excluded() -> None:
    universe = ["AAA", "BBB"]
    fundamentals = {
        "AAA": _fundamentals("AAA", analyst_count=20),
        "BBB": _fundamentals("BBB", analyst_count=3),
    }
    returns = pd.Series({"AAA": 0.08, "BBB": 0.08})
    volatilities = pd.Series({"AAA": 0.20, "BBB": 0.20})

    result = apply_quality_filters(
        universe, fundamentals, returns, volatilities, QualityRules()
    )

    assert result.kept == ["AAA"]
    assert "BBB" in dict(result.dropped)


def test_a_small_market_cap_is_excluded() -> None:
    universe = ["AAA", "BBB"]
    fundamentals = {
        "AAA": _fundamentals("AAA", market_cap=500e9),
        "BBB": _fundamentals("BBB", market_cap=1e9),
    }
    returns = pd.Series({"AAA": 0.08, "BBB": 0.08})
    volatilities = pd.Series({"AAA": 0.20, "BBB": 0.20})

    result = apply_quality_filters(
        universe, fundamentals, returns, volatilities, QualityRules()
    )

    assert result.kept == ["AAA"]


def test_a_hong_kong_market_cap_uses_the_hong_kong_threshold() -> None:
    universe = ["0700.HK"]
    fundamentals = {
        "0700.HK": Fundamentals(
            ticker="0700.HK",
            available=True,
            price=400.0,
            market_cap=100e9,
            currency="HKD",
            avg_volume=10_000_000,
            dividend_yield=0.01,
            analyst_target=450.0,
            analyst_count=30,
            profit_margin=0.2,
            trailing_eps=5.0,
        )
    }
    returns = pd.Series({"0700.HK": 0.08})
    volatilities = pd.Series({"0700.HK": 0.30})

    result = apply_quality_filters(
        universe, fundamentals, returns, volatilities, QualityRules()
    )

    # 100 billion HKD is above the 80 billion HKD floor, unlike the same
    # number of US dollars.
    assert result.kept == ["0700.HK"]


def test_a_high_volatility_is_excluded() -> None:
    universe = ["AAA", "BBB"]
    fundamentals = {"AAA": _fundamentals("AAA"), "BBB": _fundamentals("BBB")}
    returns = pd.Series({"AAA": 0.08, "BBB": 0.08})
    volatilities = pd.Series({"AAA": 0.20, "BBB": 0.70})

    result = apply_quality_filters(
        universe, fundamentals, returns, volatilities, QualityRules()
    )

    assert result.kept == ["AAA"]


def test_a_low_volume_is_excluded() -> None:
    universe = ["AAA", "BBB"]
    fundamentals = {
        "AAA": _fundamentals("AAA"),
        "BBB": _fundamentals("BBB", avg_volume=100_000),
    }
    returns = pd.Series({"AAA": 0.08, "BBB": 0.08})
    volatilities = pd.Series({"AAA": 0.20, "BBB": 0.20})

    result = apply_quality_filters(
        universe, fundamentals, returns, volatilities, QualityRules()
    )

    assert result.kept == ["AAA"]


def test_an_expected_return_outside_the_band_is_excluded() -> None:
    universe = ["AAA", "BBB"]
    fundamentals = {"AAA": _fundamentals("AAA"), "BBB": _fundamentals("BBB")}
    returns = pd.Series({"AAA": 0.08, "BBB": -0.40})
    volatilities = pd.Series({"AAA": 0.20, "BBB": 0.20})

    result = apply_quality_filters(
        universe, fundamentals, returns, volatilities, QualityRules()
    )

    assert result.kept == ["AAA"]


def test_an_unknown_field_does_not_exclude_by_default() -> None:
    universe = ["AAA"]
    fundamentals = {"AAA": _fundamentals("AAA", analyst_count=None, avg_volume=None)}
    returns = pd.Series({"AAA": 0.08})
    volatilities = pd.Series({"AAA": 0.20})

    result = apply_quality_filters(
        universe, fundamentals, returns, volatilities, QualityRules()
    )

    assert result.kept == ["AAA"]


def test_the_unknown_policy_can_exclude() -> None:
    universe = ["AAA"]
    fundamentals = {"AAA": _fundamentals("AAA", analyst_count=None)}
    returns = pd.Series({"AAA": 0.08})
    volatilities = pd.Series({"AAA": 0.20})
    rules = QualityRules(unknown_policy="exclude")

    result = apply_quality_filters(
        universe, fundamentals, returns, volatilities, rules
    )

    assert result.kept == []


def test_the_filter_reports_why_a_ticker_was_dropped() -> None:
    universe = ["AAA"]
    fundamentals = {"AAA": _fundamentals("AAA", analyst_count=3)}
    returns = pd.Series({"AAA": 0.08})
    volatilities = pd.Series({"AAA": 0.20})

    result = apply_quality_filters(
        universe, fundamentals, returns, volatilities, QualityRules()
    )

    reasons = dict(result.dropped)["AAA"]
    assert any("analyst" in reason.lower() for reason in reasons)


def test_the_currency_of_the_market_cap_follows_the_listing() -> None:
    universe = ["0700.HK"]
    fundamentals = {
        "0700.HK": Fundamentals(
            ticker="0700.HK",
            available=True,
            price=400.0,
            market_cap=50e9,
            currency="HKD",
            avg_volume=10_000_000,
            dividend_yield=0.01,
            analyst_target=450.0,
            analyst_count=30,
            profit_margin=0.2,
            trailing_eps=5.0,
        )
    }
    returns = pd.Series({"0700.HK": 0.08})
    volatilities = pd.Series({"0700.HK": 0.30})

    result = apply_quality_filters(
        universe,
        fundamentals,
        returns,
        volatilities,
        QualityRules(),
        metadata=MarketMetadata(),
    )

    assert result.kept == []
    assert any("market cap" in reason.lower() for reason in dict(result.dropped)["0700.HK"])
