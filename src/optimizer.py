"""Mean-variance optimization engine.

Solves for the long-only, fully-invested portfolio that either maximizes the
Sharpe ratio or minimizes volatility, using ``scipy.optimize`` sequential
least-squares programming (SLSQP). Also provides a Monte Carlo sampler used to
draw the efficient-frontier cloud.

The solver accepts :class:`~src.constraints.WeightConstraints`, so a portfolio
can be forced to hold a core position, to cap single-asset concentration, or
to respect a volatility budget. Minimum weights are handled by substituting
``w_i = m_i + u_i`` with ``u_i >= 0``, which keeps the problem a clean convex
program instead of leaving the solver to work on an active bound.

Both objectives are solved with an analytic gradient: it is several times
faster than the numerical difference SLSQP would otherwise use, and it makes
the result reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.constraints import WeightConstraints
from src.errors import OptimizationError
from src.math_engine import portfolio_performance, sharpe_ratio

Objective = Literal["max_sharpe", "min_vol"]

_PSD_TOLERANCE = -1e-8
_FEASIBILITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class OptimizationResult:
    """Outcome of a portfolio optimization.

    Attributes:
        weights: Optimal weights indexed by ticker.
        expected_return: Annualized expected portfolio return.
        volatility: Annualized portfolio volatility.
        sharpe_ratio: Annualized Sharpe ratio.
        objective: The objective that produced these weights.
        bound_active: Tickers sitting on their maximum weight, when a cap
            was applied.
        feasible: Whether the returned weights satisfy every active
            constraint.
    """

    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    objective: Objective
    bound_active: list[str] = field(default_factory=list)
    feasible: bool = True


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


def _sharpe_objective(
    weights: np.ndarray, mu: np.ndarray, sigma: np.ndarray, risk_free_rate: float
) -> tuple[float, np.ndarray]:
    """Return the negative Sharpe ratio of ``weights`` and its gradient."""
    excess = float(weights @ mu) - risk_free_rate
    variance = float(weights @ sigma @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    if volatility <= 0.0:
        return 0.0, np.zeros_like(weights)
    value = -excess / volatility
    gradient = -mu / volatility + excess * (sigma @ weights) / (volatility**3)
    return value, gradient


def _variance_objective(
    weights: np.ndarray, sigma: np.ndarray
) -> tuple[float, np.ndarray]:
    """Return the portfolio variance of ``weights`` and its gradient."""
    value = float(weights @ sigma @ weights)
    gradient = 2.0 * (sigma @ weights)
    return value, gradient


def _feasible_start(residual: float, upper: np.ndarray) -> np.ndarray:
    """Spread ``residual`` across the assets without breaking their caps.

    Water-filling: every open asset receives an equal share, and assets whose
    cap is below that share are filled to their cap and dropped from the
    rotation.
    """
    allocation = np.zeros(len(upper))
    remaining = float(residual)
    open_assets = list(range(len(upper)))
    while remaining > 1e-12 and open_assets:
        share = remaining / len(open_assets)
        saturated = [index for index in open_assets if share >= upper[index] - 1e-15]
        if not saturated:
            for index in open_assets:
                allocation[index] = share
            remaining = 0.0
            break
        for index in saturated:
            allocation[index] = upper[index]
            remaining -= upper[index]
        open_assets = [index for index in open_assets if index not in saturated]
    return allocation


def _solve_unconstrained(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_free_rate: float,
    objective: Objective,
) -> np.ndarray:
    """Solve the plain long-only, fully-invested problem."""
    n_assets = len(mu)

    def objective_fn(weights: np.ndarray) -> tuple[float, np.ndarray]:
        if objective == "max_sharpe":
            return _sharpe_objective(weights, mu, sigma, risk_free_rate)
        return _variance_objective(weights, sigma)

    solver_constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_assets
    initial = np.full(n_assets, 1.0 / n_assets)

    solution = minimize(
        objective_fn,
        initial,
        method="SLSQP",
        jac=True,
        bounds=bounds,
        constraints=solver_constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if not solution.success:
        raise OptimizationError(
            f"optimizer failed to converge for objective {objective!r}: {solution.message}"
        )
    return _clean_weights(np.asarray(solution.x, dtype=float))


def _solve_constrained(
    mu: np.ndarray,
    sigma: np.ndarray,
    risk_free_rate: float,
    objective: Objective,
    constraints: WeightConstraints,
    tickers: list[str],
) -> np.ndarray:
    """Solve the problem with minimum weights, weight caps and a volatility cap.

    The minimum weights are substituted out (``w_i = m_i + u_i``) so the
    solver always works on strictly interior variables, which keeps SLSQP
    stable when a floor binds.
    """
    n_assets = len(tickers)
    if n_assets == 1:
        return np.array([1.0])

    minimum = constraints.min_weight_vector(tickers)
    upper = constraints.upper_bounds(tickers)
    residual = 1.0 - float(minimum.sum())
    if residual <= 1e-12:
        return minimum / minimum.sum()

    def expand(extra: np.ndarray) -> np.ndarray:
        return minimum + np.asarray(extra, dtype=float)

    def objective_fn(extra: np.ndarray) -> tuple[float, np.ndarray]:
        weights = expand(extra)
        if objective == "max_sharpe":
            return _sharpe_objective(weights, mu, sigma, risk_free_rate)
        return _variance_objective(weights, sigma)

    solver_constraints: list[dict[str, object]] = [
        {"type": "eq", "fun": lambda extra: float(np.sum(extra)) - residual}
    ]
    if constraints.max_volatility is not None:
        cap = float(constraints.max_volatility)

        def volatility_headroom(extra: np.ndarray) -> float:
            weights = expand(extra)
            return cap**2 - float(weights @ sigma @ weights)

        def volatility_headroom_jac(extra: np.ndarray) -> np.ndarray:
            return -2.0 * (sigma @ expand(extra))

        solver_constraints.append(
            {
                "type": "ineq",
                "fun": volatility_headroom,
                "jac": volatility_headroom_jac,
            }
        )

    upper_extra = np.maximum(upper - minimum, 0.0)
    bounds = [(0.0, float(bound)) for bound in upper_extra]
    initial = _feasible_start(residual, upper_extra)

    solution = minimize(
        objective_fn,
        initial,
        method="SLSQP",
        jac=True,
        bounds=bounds,
        constraints=solver_constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if not solution.success:
        raise OptimizationError(
            f"optimizer failed to converge for objective {objective!r}: {solution.message}"
        )

    extra = np.clip(np.asarray(solution.x, dtype=float), 0.0, None)
    return _clean_weights(expand(extra))


def _bound_active(
    weights: np.ndarray, tickers: list[str], constraints: WeightConstraints
) -> list[str]:
    """List the tickers sitting on their maximum weight."""
    if constraints.max_weight is None:
        return []
    return [
        ticker
        for ticker, weight in zip(tickers, weights)
        if weight >= constraints.max_weight - _FEASIBILITY_TOLERANCE
    ]


def _is_feasible(
    weights: np.ndarray,
    sigma: np.ndarray,
    constraints: WeightConstraints,
    tickers: list[str],
) -> bool:
    """Check the returned weights against every active constraint."""
    minimum = constraints.min_weight_vector(tickers)
    if (weights < minimum - _FEASIBILITY_TOLERANCE).any():
        return False
    if constraints.max_weight is not None and (
        weights > constraints.max_weight + _FEASIBILITY_TOLERANCE
    ).any():
        return False
    if constraints.max_volatility is not None:
        volatility = float(np.sqrt(max(float(weights @ sigma @ weights), 0.0)))
        if volatility > constraints.max_volatility + _FEASIBILITY_TOLERANCE:
            return False
    return True


def optimize_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    objective: Objective,
    risk_free_rate: float = 0.04,
    constraints: WeightConstraints | None = None,
) -> OptimizationResult:
    """Find the optimal long-only, fully-invested portfolio.

    Args:
        expected_returns: Annualized expected return per asset.
        covariance: Annualized covariance matrix.
        objective: ``"max_sharpe"`` or ``"min_vol"``.
        risk_free_rate: Annualized risk-free rate used by the Sharpe objective.
        constraints: Optional limits on the portfolio. Minimum weights force
            core holdings into the portfolio, a maximum weight caps
            concentration, and a maximum volatility caps total risk.

    Returns:
        The optimal :class:`OptimizationResult`.

    Raises:
        OptimizationError: If inputs are invalid or the solver fails to converge.
        ConstraintError: If the constraints are invalid or infeasible.
    """
    if objective not in ("max_sharpe", "min_vol"):
        raise OptimizationError(f"unknown objective: {objective!r}")

    _validate(expected_returns, covariance)

    tickers = list(expected_returns.index)
    n_assets = len(tickers)
    active = constraints if constraints is not None else WeightConstraints()
    active.validate(tickers)

    mu = expected_returns.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)

    if n_assets == 1 and not active.is_active():
        weights = np.array([1.0])
    elif active.is_active():
        weights = _solve_constrained(
            mu, sigma, risk_free_rate, objective, active, tickers
        )
    else:
        weights = _solve_unconstrained(mu, sigma, risk_free_rate, objective)

    ret, vol, sharpe = portfolio_performance(weights, expected_returns, covariance, risk_free_rate)
    return OptimizationResult(
        weights=pd.Series(weights, index=tickers),
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        objective=objective,
        bound_active=_bound_active(weights, tickers, active),
        feasible=_is_feasible(weights, sigma, active, tickers),
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
