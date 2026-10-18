"""Tests for the portfolio optimization engine.

Covers the long-only, fully-invested constraints, both objectives, the
single-asset bypass, and the failure paths. Inputs are synthetic so no network
access is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.optimizer import (
    OptimizationError,
    OptimizationResult,
    monte_carlo_frontier,
    optimize_portfolio,
)


def _inputs() -> tuple[pd.Series, pd.DataFrame]:
    expected_returns = pd.Series({"AAA": 0.12, "BBB": 0.08, "CCC": 0.05})
    covariance = pd.DataFrame(
        [
            [0.0400, 0.0060, 0.0020],
            [0.0060, 0.0225, 0.0030],
            [0.0020, 0.0030, 0.0100],
        ],
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
    )
    return expected_returns, covariance


@pytest.mark.parametrize("objective", ["max_sharpe", "min_vol"])
def test_weights_are_long_only_and_sum_to_one(objective: str) -> None:
    expected_returns, covariance = _inputs()

    result = optimize_portfolio(expected_returns, covariance, objective, risk_free_rate=0.04)

    weights = result.weights.to_numpy()
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= -1e-9).all()
    assert (weights <= 1.0 + 1e-9).all()


def test_max_sharpe_beats_equal_weight_on_sharpe() -> None:
    expected_returns, covariance = _inputs()

    best = optimize_portfolio(expected_returns, covariance, "max_sharpe", risk_free_rate=0.04)
    equal = np.full(len(expected_returns), 1.0 / len(expected_returns))
    _, _, equal_sharpe = _performance(equal, expected_returns, covariance)

    assert best.sharpe_ratio >= equal_sharpe - 1e-9


def test_min_vol_has_lowest_volatility_of_the_two_objectives() -> None:
    expected_returns, covariance = _inputs()

    min_vol = optimize_portfolio(expected_returns, covariance, "min_vol", risk_free_rate=0.04)
    max_sharpe = optimize_portfolio(expected_returns, covariance, "max_sharpe", risk_free_rate=0.04)

    assert min_vol.volatility <= max_sharpe.volatility + 1e-9


def test_single_asset_returns_full_weight() -> None:
    expected_returns = pd.Series({"SOLO": 0.10})
    covariance = pd.DataFrame([[0.04]], index=["SOLO"], columns=["SOLO"])

    result = optimize_portfolio(expected_returns, covariance, "max_sharpe", risk_free_rate=0.04)

    assert isinstance(result, OptimizationResult)
    assert result.weights["SOLO"] == pytest.approx(1.0)
    assert result.volatility == pytest.approx(0.2)


def test_empty_universe_raises_optimization_error() -> None:
    with pytest.raises(OptimizationError):
        optimize_portfolio(pd.Series(dtype=float), pd.DataFrame(), "max_sharpe", risk_free_rate=0.04)


def test_non_positive_semidefinite_covariance_raises() -> None:
    expected_returns = pd.Series({"AAA": 0.10, "BBB": 0.10})
    indefinite = pd.DataFrame(
        [[0.01, 0.05], [0.05, 0.01]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )

    with pytest.raises(OptimizationError):
        optimize_portfolio(expected_returns, indefinite, "min_vol", risk_free_rate=0.04)


def test_monte_carlo_frontier_is_long_only_and_normalized() -> None:
    expected_returns, covariance = _inputs()

    cloud = monte_carlo_frontier(expected_returns, covariance, n_portfolios=500, seed=1)

    assert len(cloud) == 500
    for column in expected_returns.index:
        assert (cloud[column] >= 0.0).all()
    assert np.allclose(cloud[list(expected_returns.index)].sum(axis=1), 1.0)
    assert {"return", "volatility", "sharpe"}.issubset(cloud.columns)


def test_monte_carlo_frontier_is_reproducible_with_seed() -> None:
    expected_returns, covariance = _inputs()

    first = monte_carlo_frontier(expected_returns, covariance, n_portfolios=200, seed=7)
    second = monte_carlo_frontier(expected_returns, covariance, n_portfolios=200, seed=7)

    pd.testing.assert_frame_equal(first, second)


def _performance(weights: np.ndarray, expected_returns: pd.Series, covariance: pd.DataFrame):
    from src.math_engine import portfolio_performance

    return portfolio_performance(weights, expected_returns, covariance, 0.04)
