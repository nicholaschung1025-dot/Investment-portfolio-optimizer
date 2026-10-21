"""Tests for the walk-forward validation backtest.

The backtest re-estimates the portfolio on a rolling window and applies it to
the following period, so the reported performance is out-of-sample. These
tests check that the mechanics are right -- in particular that no future data
leaks into a training window -- rather than that the optimizer wins, which
depends on the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.errors import OptimizationError
from src.math_engine import compute_log_returns
from src.optimizer import optimize_portfolio
from src.validation import compute_performance, walk_forward_backtest


def _prices(periods: int = 750, seed: int = 7) -> pd.DataFrame:
    """Synthetic geometric random walk of daily prices."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=periods, freq="B")
    shocks = rng.normal(0.0004, 0.01, size=(periods, 3))
    paths = np.exp(np.cumsum(shocks, axis=0))
    return pd.DataFrame(
        paths * np.array([100.0, 50.0, 200.0]), index=dates, columns=["AAA", "BBB", "CCC"]
    )


def test_walk_forward_produces_metrics_for_both_portfolios() -> None:
    prices = _prices()

    result = walk_forward_backtest(prices, initial_window=300)

    assert result.optimized.period_count > 0
    assert result.benchmark.period_count == result.optimized.period_count
    assert np.isfinite(result.optimized.cagr)
    assert np.isfinite(result.benchmark.cagr)


def test_walk_forward_is_deterministic() -> None:
    prices = _prices()

    first = walk_forward_backtest(prices, initial_window=300)
    second = walk_forward_backtest(prices, initial_window=300)

    pd.testing.assert_frame_equal(first.optimized_weights, second.optimized_weights)


def test_training_window_ends_at_the_period_start() -> None:
    prices = _prices()
    windows: list[pd.DatetimeIndex] = []

    def recording_estimator(log_returns: pd.DataFrame):
        windows.append(log_returns.index.copy())
        return log_returns.mean() * 252, log_returns.cov() * 252

    result = walk_forward_backtest(
        prices, initial_window=300, estimator=recording_estimator
    )

    assert len(windows) == result.rebalance_count
    for window, period_start in zip(windows, result.period_starts):
        assert str(window[-1].date()) == period_start


def test_optimized_weights_match_an_estimate_on_the_same_window() -> None:
    prices = _prices()
    result = walk_forward_backtest(prices, initial_window=300)

    first_start = result.period_starts[0]
    first_weights = result.optimized_weights.iloc[0]

    start_position = prices.index.get_loc(pd.Timestamp(first_start))
    training = compute_log_returns(prices).iloc[:start_position]
    expected_returns = training.mean() * 252
    covariance = training.cov() * 252

    expected = optimize_portfolio(expected_returns, covariance, "max_sharpe", 0.04)

    np.testing.assert_allclose(
        first_weights.to_numpy(), expected.weights.to_numpy(), atol=1e-6
    )


def test_benchmark_metrics_match_an_equal_weight_computation() -> None:
    prices = _prices()
    result = walk_forward_backtest(prices, initial_window=300)

    period_returns = []
    for start, end in zip(result.period_starts, result.period_ends):
        start_price = prices.loc[pd.Timestamp(start)]
        end_price = prices.loc[pd.Timestamp(end)]
        period_returns.append(float(((end_price / start_price) - 1.0).mean()))

    expected = compute_performance(
        pd.Series(period_returns), 0.04, result.periods_per_year
    )

    assert result.benchmark.total_return == pytest.approx(expected.total_return, rel=1e-9)
    assert result.benchmark.max_drawdown == pytest.approx(expected.max_drawdown, rel=1e-9)


def test_max_drawdown_is_never_positive() -> None:
    prices = _prices()

    result = walk_forward_backtest(prices, initial_window=300)

    assert result.optimized.max_drawdown <= 0.0
    assert result.benchmark.max_drawdown <= 0.0


def test_rebalance_count_matches_the_requested_frequency() -> None:
    prices = _prices(periods=750)

    monthly = walk_forward_backtest(prices, initial_window=300, rebalance="monthly")
    quarterly = walk_forward_backtest(prices, initial_window=300, rebalance="quarterly")

    assert monthly.rebalance_count > quarterly.rebalance_count


def test_insufficient_history_raises_optimization_error() -> None:
    prices = _prices(periods=120)

    with pytest.raises(OptimizationError):
        walk_forward_backtest(prices, initial_window=300)


def test_unknown_rebalance_frequency_raises() -> None:
    prices = _prices()

    with pytest.raises(OptimizationError):
        walk_forward_backtest(prices, initial_window=300, rebalance="hourly")


def test_compute_performance_annualizes_a_constant_return() -> None:
    returns = pd.Series([0.01] * 12)

    metrics = compute_performance(returns, risk_free_rate=0.0, periods_per_year=12.0)

    assert metrics.total_return == pytest.approx(1.01**12 - 1)
    assert metrics.cagr == pytest.approx(1.01**12 - 1)
    assert metrics.volatility == pytest.approx(0.0)


def test_compute_performance_reports_the_maximum_drawdown() -> None:
    returns = pd.Series([0.10, -0.20, 0.05])

    metrics = compute_performance(returns, risk_free_rate=0.0, periods_per_year=1.0)

    equity = (1.0 + returns).cumprod()
    expected_drawdown = float((equity / equity.cummax() - 1.0).min())

    assert metrics.max_drawdown == pytest.approx(expected_drawdown)
