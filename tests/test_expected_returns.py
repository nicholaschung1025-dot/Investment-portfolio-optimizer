"""Tests for the expected-return estimators.

Covers the historical-mean, CAPM and Black-Litterman estimators plus the
weighted blending helper. All inputs are synthetic, so no network access is
required and every expected value can be derived by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.errors import OptimizationError
from src.expected_returns import (
    black_litterman_returns,
    blend_estimates,
    capm_returns,
    historical_mean,
)
from src.math_engine import annualized_expected_returns


def _panel() -> pd.DataFrame:
    """Synthetic daily log-return panel with three independent assets."""
    rng = np.random.default_rng(5)
    dates = pd.date_range("2020-01-01", periods=250, freq="B")
    return pd.DataFrame(
        rng.normal(loc=0.0005, scale=0.01, size=(250, 3)),
        index=dates,
        columns=["AAA", "BBB", "CCC"],
    )


def _beta_case(beta: float) -> tuple[pd.DataFrame, pd.Series]:
    """A single asset with an exactly known beta against a market series."""
    rng = np.random.default_rng(17)
    dates = pd.date_range("2020-01-01", periods=250, freq="B")
    market = pd.Series(rng.normal(loc=0.0005, scale=0.01, size=250), index=dates)
    frame = pd.DataFrame({"AAA": beta * market.to_numpy()}, index=dates)
    return frame, market


def test_historical_mean_matches_existing_annualized_expected_returns() -> None:
    panel = _panel()

    result = historical_mean(panel)
    expected = annualized_expected_returns(panel)

    pd.testing.assert_series_equal(result, expected)


def test_capm_gives_unit_beta_the_market_excess_return() -> None:
    frame, market = _beta_case(beta=1.0)
    risk_free_rate = 0.04

    result = capm_returns(frame, market, risk_free_rate)

    market_annual = market.mean() * 252
    assert result["AAA"] == pytest.approx(market_annual, rel=1e-9)


def test_capm_scales_expected_return_with_beta() -> None:
    frame, market = _beta_case(beta=2.0)
    risk_free_rate = 0.04

    result = capm_returns(frame, market, risk_free_rate)

    market_annual = market.mean() * 252
    expected = risk_free_rate + 2.0 * (market_annual - risk_free_rate)
    assert result["AAA"] == pytest.approx(expected, rel=1e-9)


def test_capm_aligns_market_returns_to_the_asset_index() -> None:
    panel = _panel()
    market = panel["AAA"]
    extra = market.iloc[:5].copy()
    extra.index = extra.index + pd.Timedelta(days=500)
    extended_market = pd.concat([market, extra])

    result = capm_returns(panel, extended_market, 0.04)
    expected = capm_returns(panel, market, 0.04)

    pd.testing.assert_series_equal(result, expected)


def test_capm_rejects_empty_market_returns() -> None:
    panel = _panel()
    empty_market = pd.Series(dtype=float)

    with pytest.raises(OptimizationError):
        capm_returns(panel, empty_market, 0.04)


def test_capm_rejects_a_market_series_without_variance() -> None:
    panel = _panel()
    flat_market = pd.Series(0.0, index=panel.index)

    with pytest.raises(OptimizationError):
        capm_returns(panel, flat_market, 0.04)


def test_capm_rejects_duplicate_market_dates() -> None:
    panel = _panel()
    duplicated = pd.concat([panel["AAA"], panel["AAA"].iloc[:1]])

    with pytest.raises(OptimizationError):
        capm_returns(panel, duplicated, 0.04)


def test_black_litterman_without_views_returns_implied_equilibrium_returns() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )
    market_weights = {"AAA": 0.5, "BBB": 0.5}
    risk_free_rate = 0.04
    market_risk_premium = 0.05

    result = black_litterman_returns(
        covariance, market_weights, risk_free_rate, market_risk_premium=market_risk_premium
    )

    expected_prior = risk_free_rate + market_risk_premium * np.array([0.02, 0.045])
    np.testing.assert_allclose(result.to_numpy(), expected_prior, rtol=1e-9)


def test_black_litterman_with_full_confidence_moves_close_to_the_view() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )
    prior = black_litterman_returns(
        covariance, {"AAA": 0.5, "BBB": 0.5}, 0.04, market_risk_premium=0.05
    )
    view = {"AAA": 0.10}

    result = black_litterman_returns(
        covariance, {"AAA": 0.5, "BBB": 0.5}, 0.04, views=view, view_confidences={"AAA": 0.95}
    )

    distance_to_view = abs(prior["AAA"] - view["AAA"])
    assert abs(result["AAA"] - view["AAA"]) < 0.1 * distance_to_view


def test_black_litterman_with_low_confidence_stays_close_to_the_prior() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )
    prior = black_litterman_returns(
        covariance, {"AAA": 0.5, "BBB": 0.5}, 0.04, market_risk_premium=0.05
    )
    view = {"AAA": 0.10}

    result = black_litterman_returns(
        covariance, {"AAA": 0.5, "BBB": 0.5}, 0.04, views=view, view_confidences={"AAA": 0.05}
    )

    distance_to_view = abs(prior["AAA"] - view["AAA"])
    assert abs(result["AAA"] - prior["AAA"]) < 0.15 * distance_to_view


def test_black_litterman_rejects_view_on_unknown_ticker() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )

    with pytest.raises(OptimizationError):
        black_litterman_returns(
            covariance, {"AAA": 0.5, "BBB": 0.5}, 0.04, views={"ZZZ": 0.10}
        )


def test_blend_estimates_is_a_weighted_average() -> None:
    first = pd.Series({"AAA": 0.10, "BBB": 0.06})
    second = pd.Series({"AAA": 0.20, "BBB": 0.02})

    result = blend_estimates({"capm": first, "analyst": second}, {"capm": 0.7, "analyst": 0.3})

    expected = (0.7 * first + 0.3 * second).rename("expected_return")
    pd.testing.assert_series_equal(result, expected)


def test_blend_estimates_rejects_weights_that_do_not_sum_to_one() -> None:
    estimate = pd.Series({"AAA": 0.10, "BBB": 0.06})

    with pytest.raises(OptimizationError):
        blend_estimates({"capm": estimate}, {"capm": 0.5})


def test_blend_estimates_rejects_negative_weights() -> None:
    estimate = pd.Series({"AAA": 0.10, "BBB": 0.06})

    with pytest.raises(OptimizationError):
        blend_estimates({"capm": estimate, "other": estimate}, {"capm": 1.5, "other": -0.5})


def test_blend_estimates_rejects_mismatched_indices() -> None:
    first = pd.Series({"AAA": 0.10, "BBB": 0.06})
    second = pd.Series({"AAA": 0.20})

    with pytest.raises(OptimizationError):
        blend_estimates({"capm": first, "analyst": second}, {"capm": 0.5, "analyst": 0.5})
