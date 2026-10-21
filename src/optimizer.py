"""Mean-variance optimization engine.

Solves for the long-only, fully-invested portfolio that either maximizes the
Sharpe ratio or minimizes volatility, using ``scipy.optimize`` sequential
least-squares programming (SLSQP). Also provides a Monte Carlo sampler used to
draw the efficient-frontier cloud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.errors import OptimizationError
from src.math_engine import portfolio_performance, sharpe_ratio

Objective = Literal["max_sharpe", "min_vol"]

_PSD_TOLERANCE = -1e-8


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of a portfolio optimization.

    Attributes:
        weights: Optimal weights indexed by ticker.
        expected_return: Annualized expected portfolio return.
        volatility: Annualized portfolio volatility.
        sharpe_ratio: Annualized Sharpe ratio.
        objective: The objective that produced these weights.
    """

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    objective: Objective


def _validate(expected_returns: pd.Series, covariance: pd.DataFrame) -> None:
    """Validate that the inputs form a solvable optimization problem."""
    if expected_returns.empty or covariance.empty:
        raise OptimizationError("expected_returns and covariance must be non-empty")
    if len(expected_returns) != covariance.shape[0] or covariance.shape[0] != covariance.shape[1]:
        raise OptimizationError("covariance must be square and match expected_returns length")
    if list(covariance.index) != list(expected_returns.index):
        raise OptimizationError("covariance index must match expected_returns index")
    if expected_returns.isna().any() or covariance.isna().any().any():
        raise OptimizationError("inputs must not contain NaN values")

    eigenvalues = np.linalg.eigvalsh(covariance.to_numpy(dtype=float))
    if float(eigenvalues.min()) < _PSD_TOLERANCE:
        raise OptimizationError(
            "covariance matrix is not positive semidefinite; "
            "exclude zero-variance or degenerate assets before optimizing"
        )


def _clean_weights(weights: np.ndarray) -> np.ndarray:
    """Clip tiny negatives to zero and renormalize so the weights sum to 1."""
    weights = np.clip(weights, 0.0, None)
    total = weights.sum()
    if total <= 0.0:
        raise OptimizationError("optimizer produced non-normalizable weights")
    return weights / total


def optimize_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    objective: Objective,
    risk_free_rate: float = 0.04,
) -> OptimizationResult:
    """Find the optimal long-only, fully-invested portfolio.

    Args:
        expected_returns: Annualized expected return per asset.
        covariance: Annualized covariance matrix.
        objective: ``"max_sharpe"`` or ``"min_vol"``.
        risk_free_rate: Annualized risk-free rate used by the Sharpe objective.

    Returns:
        The optimal :class:`OptimizationResult`.

    Raises:
        OptimizationError: If inputs are invalid or the solver fails to converge.
    """
    if objective not in ("max_sharpe", "min_vol"):
        raise OptimizationError(f"unknown objective: {objective!r}")

    _validate(expected_returns, covariance)

    tickers = list(expected_returns.index)
    n_assets = len(tickers)

    if n_assets == 1:
        weights = np.array([1.0])
    else:
        sigma = covariance.to_numpy(dtype=float)

        def negative_sharpe(w: np.ndarray) -> float:
            ret, vol, _ = portfolio_performance(w, expected_returns, covariance, risk_free_rate)
            return -sharpe_ratio(ret, vol, risk_free_rate)

        def variance(w: np.ndarray) -> float:
            return float(w @ sigma @ w)

        objective_fn = negative_sharpe if objective == "max_sharpe" else variance

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * n_assets
        initial = np.full(n_assets, 1.0 / n_assets)

        solution = minimize(
            objective_fn,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )

        if not solution.success:
            raise OptimizationError(
                f"optimizer failed to converge for objective {objective!r}: {solution.message}"
            )
        weights = _clean_weights(np.asarray(solution.x, dtype=float))

    ret, vol, sharpe = portfolio_performance(weights, expected_returns, covariance, risk_free_rate)
    return OptimizationResult(
        weights=pd.Series(weights, index=tickers),
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        objective=objective,
    )


def monte_carlo_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    n_portfolios: int = 5000,
    risk_free_rate: float = 0.04,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Sample random long-only portfolios to draw the efficient frontier.

    Weights are drawn uniformly from the probability simplex via a Dirichlet
    distribution, so every sampled portfolio is long-only and fully invested.

    Args:
        expected_returns: Annualized expected return per asset.
        covariance: Annualized covariance matrix.
        n_portfolios: Number of random portfolios to sample.
        risk_free_rate: Annualized risk-free rate used for Sharpe ratios.
        seed: Seed for the random generator; makes the output reproducible.

    Returns:
        DataFrame with one row per portfolio: the per-asset weights plus
        ``return``, ``volatility`` and ``sharpe`` columns.

    Raises:
        OptimizationError: If the inputs are invalid.
    """
    _validate(expected_returns, covariance)

    tickers = list(expected_returns.index)
    n_assets = len(tickers)
    rng = np.random.default_rng(seed)
    weight_matrix = rng.dirichlet(np.ones(n_assets), size=n_portfolios)

    mu = expected_returns.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)

    returns = weight_matrix @ mu
    variances = np.einsum("ij,jk,ik->i", weight_matrix, sigma, weight_matrix)
    volatilities = np.sqrt(np.clip(variances, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(volatilities > 0.0, (returns - risk_free_rate) / volatilities, 0.0)

    cloud = pd.DataFrame(weight_matrix, columns=tickers)
    cloud["return"] = returns
    cloud["volatility"] = volatilities
    cloud["sharpe"] = sharpes
    return cloud
