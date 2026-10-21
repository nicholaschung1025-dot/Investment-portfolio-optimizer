"""Risk simulation: what the portfolio's return distribution looks like.

A volatility number says how far returns typically travel. It says nothing
about the shape of the tail, which is where the money is actually lost. This
module turns the portfolio into a distribution of terminal values so the
losses can be measured directly: the value at risk at the 95% level, the
expected loss in the worst 5% of outcomes, and the probability of ending
above where the portfolio started.

Two distributions are available. The Student-t one keeps the fat tails that
financial returns actually have -- a normal distribution would pretend the
worst day of a hundred years is three standard deviations away, which is not
true. The bootstrap one resamples the portfolio's own history and makes no
assumption at all, at the cost of assuming the past repeats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.errors import OptimizationError

TRADING_DAYS_PER_YEAR = 252
_DISTRIBUTIONS = ("student_t", "bootstrap")


@dataclass(frozen=True)
class RiskMetrics:
    """Risk of a portfolio over the simulation horizon.

    Attributes:
        var_95: Loss at the 95% confidence level. A negative value means the
            fifth percentile is above the starting budget, so the loss is
            measured as a gain.
        cvar_95: Average loss in the worst 5% of outcomes.
        prob_profit: Probability of ending above the starting budget.
        median_value: Median terminal value.
        p05: Fifth percentile of the terminal value.
        p95: Ninety-fifth percentile of the terminal value.
        simulations: Number of simulated paths.
    """

    var_95: float
    cvar_95: float
    prob_profit: float
    median_value: float
    p05: float
    p95: float
    simulations: int


@dataclass(frozen=True)
class RiskSimulation:
    """A simulation and the metrics read off it.

    Attributes:
        metrics: The risk metrics.
        terminal_values: The simulated terminal value of every path.
    """

    metrics: RiskMetrics
    terminal_values: np.ndarray


def _portfolio_moments(
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> tuple[float, float]:
    """Return the annualized expected log return and volatility."""
    tickers = [ticker for ticker in weights.index if abs(weights[ticker]) > 1e-9]
    if not tickers:
        return 0.0, 0.0
    weights = weights[tickers]
    mu = expected_returns.reindex(tickers).to_numpy(dtype=float)
    sigma = covariance.loc[tickers, tickers].to_numpy(dtype=float)

    expected = float(weights.to_numpy() @ mu)
    variance = float(weights.to_numpy() @ sigma @ weights.to_numpy())
    return expected, float(np.sqrt(max(variance, 0.0)))


def _student_t_paths(
    expected: float,
    volatility: float,
    horizon_days: int,
    simulations: int,
    degrees_of_freedom: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate cumulative log returns with fat-tailed daily shocks."""
    if volatility <= 0.0:
        return np.full(simulations, expected * horizon_days / TRADING_DAYS_PER_YEAR)

    scale = (
        np.sqrt(degrees_of_freedom / (degrees_of_freedom - 2.0))
        if degrees_of_freedom > 2.0
        else 1.0
    )
    shocks = rng.standard_t(degrees_of_freedom, size=(horizon_days, simulations)) / scale
    daily_drift = (expected - 0.5 * volatility**2) / TRADING_DAYS_PER_YEAR
    daily_shock = volatility / np.sqrt(TRADING_DAYS_PER_YEAR) * shocks
    return np.sum(daily_drift + daily_shock, axis=0)


def _bootstrap_paths(
    history: pd.Series,
    horizon_days: int,
    simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate cumulative log returns by resampling the portfolio history."""
    values = np.asarray(history.dropna(), dtype=float)
    if values.size == 0:
        raise OptimizationError("the bootstrap method needs at least one historical return")
    draws = rng.choice(values, size=(horizon_days, simulations), replace=True)
    return np.sum(draws, axis=0)


def simulate_risk(
    weights: pd.Series,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    budget: float,
    risk_free_rate: float = 0.04,
    horizon_days: int = TRADING_DAYS_PER_YEAR,
    simulations: int = 10000,
    distribution: str = "student_t",
    degrees_of_freedom: float = 5.0,
    cash_weight: float = 0.0,
    history: pd.Series | None = None,
    seed: int | None = 42,
) -> RiskSimulation:
    """Simulate the portfolio's terminal value and measure its risk.

    Args:
        weights: Weight of each risky asset; the remainder is cash.
        expected_returns: Annualized expected return per asset.
        covariance: Annualized covariance matrix.
        budget: Starting value of the portfolio.
        risk_free_rate: Annualized risk-free rate earned by the cash part.
        horizon_days: Length of the simulated horizon in trading days.
        simulations: Number of simulated paths.
        distribution: ``"student_t"`` or ``"bootstrap"``.
        degrees_of_freedom: Degrees of freedom of the Student-t shocks.
        cash_weight: Part of the budget held in the risk-free asset.
        history: Historical portfolio log returns, required by the bootstrap
            method.
        seed: Seed of the random generator; makes the result reproducible.

    Returns:
        The :class:`RiskSimulation` for the portfolio.

    Raises:
        OptimizationError: If the inputs are unusable.
    """
    if distribution not in _DISTRIBUTIONS:
        raise OptimizationError(
            f"unknown risk distribution {distribution!r}; expected one of {_DISTRIBUTIONS}"
        )
    if budget <= 0:
        raise OptimizationError(f"the budget must be positive, got {budget}")
    if simulations < 1:
        raise OptimizationError(f"the simulation count must be positive, got {simulations}")
    if distribution == "bootstrap" and history is None:
        raise OptimizationError("the bootstrap method needs the portfolio's own history")

    rng = np.random.default_rng(seed)
    expected, volatility = _portfolio_moments(weights, expected_returns, covariance)

    if distribution == "bootstrap":
        cumulative = _bootstrap_paths(history, horizon_days, simulations, rng)
    else:
        cumulative = _student_t_paths(
            expected, volatility, horizon_days, simulations, degrees_of_freedom, rng
        )

    risky_value = budget * (1.0 - cash_weight) * np.exp(cumulative)
    cash_growth = (1.0 + risk_free_rate) ** (horizon_days / TRADING_DAYS_PER_YEAR)
    terminal_values = risky_value + budget * cash_weight * cash_growth

    p05 = float(np.percentile(terminal_values, 5))
    p95 = float(np.percentile(terminal_values, 95))
    worst = terminal_values[terminal_values <= p05]

    metrics = RiskMetrics(
        var_95=budget - p05,
        cvar_95=budget - float(worst.mean()) if worst.size else budget - p05,
        prob_profit=float(np.mean(terminal_values > budget)),
        median_value=float(np.median(terminal_values)),
        p05=p05,
        p95=p95,
        simulations=simulations,
    )
    return RiskSimulation(metrics=metrics, terminal_values=terminal_values)
