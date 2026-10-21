"""Tests for the risk simulation.

The simulation turns a portfolio into a distribution of outcomes, which is
what turns "18% volatility" into "a 5% chance of losing 12% of the money".
The tests pin the properties that make those numbers trustworthy: the
simulation is reproducible, the tail measures are ordered, and the risk
actually responds to the inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.errors import OptimizationError
from src.risk import simulate_risk


def _inputs(
    expected_return: float = 0.10, volatility: float = 0.20
) -> tuple[pd.Series, pd.DataFrame]:
    expected_returns = pd.Series({"AAA": expected_return})
    covariance = pd.DataFrame([[volatility**2]], index=["AAA"], columns=["AAA"])
    return expected_returns, covariance


def test_the_simulation_is_reproducible_with_a_seed() -> None:
    expected_returns, covariance = _inputs()

    first = simulate_risk(
        pd.Series({"AAA": 1.0}), expected_returns, covariance, 100_000.0, seed=7
    )
    second = simulate_risk(
        pd.Series({"AAA": 1.0}), expected_returns, covariance, 100_000.0, seed=7
    )

    np.testing.assert_array_equal(first.terminal_values, second.terminal_values)


def test_the_tail_measures_are_ordered() -> None:
    expected_returns, covariance = _inputs()

    result = simulate_risk(
        pd.Series({"AAA": 1.0}), expected_returns, covariance, 100_000.0, seed=1
    )

    assert result.metrics.cvar_95 >= result.metrics.var_95


def test_the_profit_probability_is_a_probability() -> None:
    expected_returns, covariance = _inputs()

    result = simulate_risk(
        pd.Series({"AAA": 1.0}), expected_returns, covariance, 100_000.0, seed=1
    )

    assert 0.0 <= result.metrics.prob_profit <= 1.0


def test_a_riskless_portfolio_cannot_lose() -> None:
    expected_returns, covariance = _inputs(expected_return=0.10, volatility=0.0)

    result = simulate_risk(
        pd.Series({"AAA": 1.0}), expected_returns, covariance, 100_000.0, seed=1
    )

    # The whole distribution sits above the starting budget, so the loss at
    # the 95% level is measured as a gain.
    assert result.metrics.var_95 < 0.0
    assert result.metrics.cvar_95 == pytest.approx(result.metrics.var_95)
    assert result.metrics.prob_profit == pytest.approx(1.0)


def test_more_volatility_means_a_larger_loss_at_the_tail() -> None:
    calm_returns, calm_covariance = _inputs(volatility=0.10)
    wild_returns, wild_covariance = _inputs(volatility=0.40)

    calm = simulate_risk(
        pd.Series({"AAA": 1.0}), calm_returns, calm_covariance, 100_000.0, seed=3
    )
    wild = simulate_risk(
        pd.Series({"AAA": 1.0}), wild_returns, wild_covariance, 100_000.0, seed=3
    )

    assert wild.metrics.var_95 > calm.metrics.var_95
    assert wild.metrics.cvar_95 > calm.metrics.cvar_95


def test_fatter_tails_worsen_the_expected_shortfall() -> None:
    expected_returns, covariance = _inputs(volatility=0.30)

    thin = simulate_risk(
        pd.Series({"AAA": 1.0}),
        expected_returns,
        covariance,
        100_000.0,
        degrees_of_freedom=200.0,
        seed=5,
    )
    fat = simulate_risk(
        pd.Series({"AAA": 1.0}),
        expected_returns,
        covariance,
        100_000.0,
        degrees_of_freedom=3.0,
        seed=5,
    )

    assert fat.metrics.cvar_95 > thin.metrics.cvar_95


def test_holding_cash_reduces_the_risk() -> None:
    expected_returns, covariance = _inputs(volatility=0.30)

    fully_invested = simulate_risk(
        pd.Series({"AAA": 1.0}), expected_returns, covariance, 100_000.0, seed=9
    )
    half_cash = simulate_risk(
        pd.Series({"AAA": 0.5}), expected_returns, covariance, 100_000.0, cash_weight=0.5, seed=9
    )

    assert half_cash.metrics.var_95 < fully_invested.metrics.var_95


def test_the_bootstrap_method_follows_the_history() -> None:
    expected_returns = pd.Series({"AAA": 0.10})
    covariance = pd.DataFrame([[0.04]], index=["AAA"], columns=["AAA"])
    history = pd.Series([0.001] * 252)

    result = simulate_risk(
        pd.Series({"AAA": 1.0}),
        expected_returns,
        covariance,
        100_000.0,
        distribution="bootstrap",
        history=history,
        seed=1,
    )

    assert result.metrics.prob_profit == pytest.approx(1.0)


def test_an_unknown_distribution_is_refused() -> None:
    expected_returns, covariance = _inputs()

    with pytest.raises(OptimizationError):
        simulate_risk(
            pd.Series({"AAA": 1.0}),
            expected_returns,
            covariance,
            100_000.0,
            distribution="crystal-ball",
        )


def test_the_bootstrap_method_needs_history() -> None:
    expected_returns, covariance = _inputs()

    with pytest.raises(OptimizationError):
        simulate_risk(
            pd.Series({"AAA": 1.0}),
            expected_returns,
            covariance,
            100_000.0,
            distribution="bootstrap",
        )
