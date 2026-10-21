"""Tests for weight constraints and the constrained optimizer.

Covers the constraint validation rules and the three ways a portfolio can be
constrained: a minimum weight per asset (the mechanism a mandatory holding
uses), a maximum weight per asset, and a portfolio volatility cap. All inputs
are synthetic, so no network access is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constraints import WeightConstraints
from src.errors import ConstraintError, PortfolioError
from src.optimizer import optimize_portfolio


def _two_asset_inputs() -> tuple[pd.Series, pd.DataFrame]:
    expected_returns = pd.Series({"AAA": 0.20, "BBB": 0.02})
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.04]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )
    return expected_returns, covariance


def _three_asset_inputs() -> tuple[pd.Series, pd.DataFrame]:
    expected_returns = pd.Series({"AAA": 0.20, "BBB": 0.02, "CCC": 0.03})
    covariance = pd.DataFrame(
        [[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]],
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
    )
    return expected_returns, covariance


def test_empty_constraints_are_valid() -> None:
    constraints = WeightConstraints()

    constraints.validate(["AAA", "BBB"])


def test_validate_rejects_min_weight_on_an_unknown_ticker() -> None:
    constraints = WeightConstraints(min_weights={"ZZZ": 0.1})

    with pytest.raises(ConstraintError):
        constraints.validate(["AAA", "BBB"])


def test_validate_rejects_negative_min_weight() -> None:
    constraints = WeightConstraints(min_weights={"AAA": -0.1})

    with pytest.raises(ConstraintError):
        constraints.validate(["AAA", "BBB"])


def test_validate_rejects_min_weights_that_cannot_be_invested() -> None:
    constraints = WeightConstraints(min_weights={"AAA": 0.7, "BBB": 0.7})

    with pytest.raises(ConstraintError):
        constraints.validate(["AAA", "BBB"])


def test_validate_rejects_max_weight_below_a_min_weight() -> None:
    constraints = WeightConstraints(min_weights={"AAA": 0.4}, max_weight=0.3)

    with pytest.raises(ConstraintError):
        constraints.validate(["AAA", "BBB"])


def test_validate_rejects_max_weight_that_cannot_reach_full_investment() -> None:
    constraints = WeightConstraints(max_weight=0.3)

    with pytest.raises(ConstraintError):
        constraints.validate(["AAA", "BBB"])


def test_min_weight_is_respected_by_the_optimizer() -> None:
    expected_returns, covariance = _two_asset_inputs()
    constraints = WeightConstraints(min_weights={"BBB": 0.25})

    result = optimize_portfolio(
        expected_returns, covariance, "max_sharpe", 0.04, constraints=constraints
    )

    assert result.weights["BBB"] >= 0.25 - 1e-6


def test_min_weight_binds_when_the_unconstrained_optimum_would_ignore_it() -> None:
    expected_returns, covariance = _two_asset_inputs()
    constraints = WeightConstraints(min_weights={"BBB": 0.25})

    result = optimize_portfolio(
        expected_returns, covariance, "max_sharpe", 0.04, constraints=constraints
    )

    assert result.weights["AAA"] <= 0.75 + 1e-6


def test_max_weight_is_respected_by_the_optimizer() -> None:
    expected_returns, covariance = _three_asset_inputs()
    constraints = WeightConstraints(max_weight=0.4)

    result = optimize_portfolio(
        expected_returns, covariance, "max_sharpe", 0.04, constraints=constraints
    )

    assert (result.weights <= 0.4 + 1e-6).all()
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_constrained_weights_still_sum_to_one() -> None:
    expected_returns, covariance = _three_asset_inputs()
    constraints = WeightConstraints(min_weights={"BBB": 0.1}, max_weight=0.5)

    result = optimize_portfolio(
        expected_returns, covariance, "max_sharpe", 0.04, constraints=constraints
    )

    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_max_volatility_constraint_is_respected() -> None:
    expected_returns, covariance = _two_asset_inputs()
    constraints = WeightConstraints(max_volatility=0.15)

    result = optimize_portfolio(
        expected_returns, covariance, "max_sharpe", 0.04, constraints=constraints
    )

    assert result.volatility <= 0.15 + 1e-6


def test_unconstrained_call_is_unaffected_by_the_new_argument() -> None:
    expected_returns, covariance = _two_asset_inputs()

    result = optimize_portfolio(expected_returns, covariance, "max_sharpe", 0.04)

    assert result.weights["AAA"] == pytest.approx(1.0, abs=1e-6)


def test_constraint_error_is_a_portfolio_error() -> None:
    assert issubclass(ConstraintError, PortfolioError)


def test_min_weight_vector_defaults_to_zero() -> None:
    constraints = WeightConstraints(min_weights={"AAA": 0.25})

    vector = constraints.min_weight_vector(["AAA", "BBB"])

    np.testing.assert_allclose(vector, np.array([0.25, 0.0]))
