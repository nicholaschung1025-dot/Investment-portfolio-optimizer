"""Walk-forward validation of the optimizer.

An in-sample optimization result says nothing about whether the portfolio
would have worked. This module re-estimates the portfolio on a rolling
window, applies it to the period that follows, and reports the resulting
performance against an equal-weight benchmark.

The training window for a period ends at the price the portfolio is bought
at, so no future information can leak into the estimate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.constraints import WeightConstraints
from src.covariance import CovarianceMethod, shrunk_covariance
from src.errors import OptimizationError
from src.expected_returns import historical_mean
from src.math_engine import compute_log_returns
from src.optimizer import Objective, optimize_portfolio

Estimator = Callable[[pd.DataFrame], "tuple[pd.Series, pd.DataFrame]"]

_TRADING_DAYS_PER_YEAR = 252
_REBALANCE_FREQUENCIES = ("monthly", "quarterly")


@dataclass(frozen=True)
class PerformanceMetrics:
    """Performance of a series of holding-period returns.

    Attributes:
        total_return: Cumulative return over the whole backtest.
        cagr: Compound annual growth rate.
        volatility: Annualized volatility of the period returns.
        sharpe_ratio: Annualized Sharpe ratio.
        max_drawdown: Largest peak-to-trough decline, as a negative number.
        period_count: Number of holding periods.
    """

    total_return: float
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    period_count: int


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of a walk-forward validation.

    Attributes:
        optimized: Metrics of the optimized portfolio.
        benchmark: Metrics of the equal-weight benchmark.
        optimized_weights: Weights used in each period, indexed by the period
            end date.
        period_starts: First date of each holding period.
        period_ends: Last date of each holding period.
        periods_per_year: Number of holding periods in a year, used to
            annualize the metrics.
        average_turnover: Average one-way turnover per rebalance.
        rebalance_count: Number of holding periods.
    """

    optimized: PerformanceMetrics
    benchmark: PerformanceMetrics
    optimized_weights: pd.DataFrame
    period_starts: list[str]
    period_ends: list[str]
    periods_per_year: float
    average_turnover: float
    rebalance_count: int


def compute_performance(
    period_returns: pd.Series,
    risk_free_rate: float = 0.04,
    periods_per_year: float = 12.0,
) -> PerformanceMetrics:
    """Turn a series of holding-period returns into performance metrics.

    Args:
        period_returns: One return per holding period.
        risk_free_rate: Annualized risk-free rate.
        periods_per_year: Number of holding periods in a year.

    Returns:
        The :class:`PerformanceMetrics` for the series.
    """
    returns = pd.Series(period_returns, dtype=float).dropna()
    if returns.empty:
        raise OptimizationError("no period returns to evaluate")

    growth = (1.0 + returns).cumprod()
    total_return = float(growth.iloc[-1] - 1.0)

    if len(returns) > 1 and growth.iloc[-1] > 0.0:
        cagr = float(growth.iloc[-1] ** (periods_per_year / len(returns)) - 1.0)
    else:
        cagr = total_return

    volatility = (
        float(returns.std(ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0
    )
    sharpe = (cagr - risk_free_rate) / volatility if volatility > 1e-12 else 0.0
    max_drawdown = float((growth / growth.cummax() - 1.0).min())

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        period_count=int(len(returns)),
    )


def default_estimator(covariance_method: CovarianceMethod = "ledoit-wolf") -> Estimator:
    """Return the estimator used when the caller does not supply one.

    Args:
        covariance_method: How the covariance matrix is estimated.

    Returns:
        A callable that turns a log-return panel into expected returns and a
        covariance matrix.
    """

    def estimate(log_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
        expected = historical_mean(log_returns)
        covariance = shrunk_covariance(log_returns, method=covariance_method)
        return expected, covariance

    return estimate


def _period_end_positions(index: pd.DatetimeIndex, rebalance: str) -> list[int]:
    """Return the positions at which each holding period ends."""
    if rebalance == "monthly":
        keys = [(stamp.year, stamp.month) for stamp in index]
    elif rebalance == "quarterly":
        keys = [(stamp.year, (stamp.month - 1) // 3) for stamp in index]
    else:
        raise OptimizationError(
            f"unknown rebalance frequency {rebalance!r}; "
            f"expected one of {_REBALANCE_FREQUENCIES}"
        )

    ends: list[int] = []
    for position in range(1, len(keys)):
        if keys[position] != keys[position - 1]:
            ends.append(position - 1)
    if not ends or ends[-1] != len(index) - 1:
        ends.append(len(index) - 1)
    return ends


def walk_forward_backtest(
    prices: pd.DataFrame,
    objective: Objective = "max_sharpe",
    risk_free_rate: float = 0.04,
    covariance_method: CovarianceMethod = "ledoit-wolf",
    initial_window: int = 252,
    rebalance: str = "monthly",
    constraints: WeightConstraints | None = None,
    estimator: Estimator | None = None,
) -> BacktestResult:
    """Re-estimate the portfolio on a rolling window and track its performance.

    Each holding period is optimized using only the data available at the
    start of that period, and the resulting weights are then applied to the
    period's realized prices. The equal-weight portfolio is tracked over the
    same periods as a benchmark.

    Args:
        prices: Daily adjusted close prices, one column per ticker.
        objective: ``"max_sharpe"`` or ``"min_vol"``.
        risk_free_rate: Annualized risk-free rate.
        covariance_method: How the covariance matrix is estimated.
        initial_window: Minimum number of observations required before the
            first rebalance.
        rebalance: ``"monthly"`` or ``"quarterly"``.
        constraints: Optional limits applied to every rebalance.
        estimator: Optional callable producing expected returns and covariance
            from a log-return panel.

    Returns:
        The :class:`BacktestResult` for the run.

    Raises:
        OptimizationError: If the inputs are unusable.
    """
    if prices is None or prices.empty:
        raise OptimizationError("no prices supplied for the backtest")
    if initial_window < 2:
        raise OptimizationError("the initial training window must hold at least 2 observations")

    log_returns = compute_log_returns(prices)
    if len(log_returns) < initial_window:
        raise OptimizationError(
            f"only {len(log_returns)} observations available, "
            f"but the initial window needs {initial_window}"
        )

    estimate = estimator if estimator is not None else default_estimator(covariance_method)
    ends = _period_end_positions(prices.index, rebalance)

    period_starts: list[str] = []
    period_ends: list[str] = []
    optimized_returns: list[float] = []
    benchmark_returns: list[float] = []
    weight_rows: list[pd.Series] = []
    previous_weights: pd.Series | None = None
    turnovers: list[float] = []
    trading_days = 0

    for position in range(len(ends) - 1):
        start = ends[position]
        end = ends[position + 1]
        if start < initial_window:
            continue

        training = log_returns.iloc[:start]
        expected, covariance = estimate(training)
        result = optimize_portfolio(
            expected, covariance, objective, risk_free_rate, constraints=constraints
        )

        start_prices = prices.iloc[start]
        end_prices = prices.iloc[end]
        asset_returns = (end_prices / start_prices) - 1.0

        optimized_returns.append(float(result.weights.to_numpy() @ asset_returns.to_numpy()))
        benchmark_returns.append(float(asset_returns.mean()))

        period_starts.append(str(prices.index[start].date()))
        period_ends.append(str(prices.index[end].date()))
        weight_rows.append(result.weights)

        if previous_weights is not None:
            turnovers.append(
                0.5 * float((result.weights - previous_weights).abs().sum())
            )
        previous_weights = result.weights
        trading_days += end - start

    if not optimized_returns:
        raise OptimizationError(
            "no holding period had enough history to optimize; "
            "increase the data range or lower the initial window"
        )

    periods_per_year = len(optimized_returns) / (trading_days / _TRADING_DAYS_PER_YEAR)

    optimized_metrics = compute_performance(
        pd.Series(optimized_returns), risk_free_rate, periods_per_year
    )
    benchmark_metrics = compute_performance(
        pd.Series(benchmark_returns), risk_free_rate, periods_per_year
    )

    return BacktestResult(
        optimized=optimized_metrics,
        benchmark=benchmark_metrics,
        optimized_weights=pd.DataFrame(weight_rows, index=period_ends),
        period_starts=period_starts,
        period_ends=period_ends,
        periods_per_year=periods_per_year,
        average_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        rebalance_count=len(optimized_returns),
    )
