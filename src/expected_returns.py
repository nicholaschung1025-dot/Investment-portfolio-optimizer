"""Expected-return estimators.

Three interchangeable estimators feed the optimizer, so the source of the
expected returns can be swapped without touching the optimization code:

* :func:`historical_mean` -- the plain sample mean of log returns. Cheap, but
  a very noisy estimator of a forward-looking return.
* :func:`capm_returns` -- the capital asset pricing model, using a market
  index to estimate each asset's beta.
* :func:`black_litterman_returns` -- the posterior mean implied by a market
  equilibrium prior, optionally updated with absolute analyst views. This is
  the estimator that shrinks the expected-return estimate towards something
  the market itself prices, which is what makes mean-variance outputs
  survivable.

Everything works in annualized log-return space, consistent with
:mod:`src.math_engine`. The risk-free rate is an annualized total-return
rate, matching how the Sharpe ratio is computed elsewhere in the package.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.errors import OptimizationError
from src.math_engine import annualized_expected_returns

ExpectedReturnMethod = Literal["historical_mean", "capm"]


def historical_mean(
    log_returns: pd.DataFrame, periods_per_year: int = 252
) -> pd.Series:
    """Estimate expected returns as the annualized mean of log returns.

    Args:
        log_returns: Daily log returns, one column per ticker.
        periods_per_year: Trading periods in a year, used to annualize.

    Returns:
        Annualized expected return per ticker.
    """
    return annualized_expected_returns(log_returns, periods_per_year)


def capm_returns(
    log_returns: pd.DataFrame,
    market_returns: pd.Series,
    risk_free_rate: float,
    periods_per_year: int = 252,
) -> pd.Series:
    """Estimate expected returns with the capital asset pricing model.

    Each asset's beta is the ratio of its covariance with the market to the
    market's variance, computed on daily log returns and aligned to the asset
    index.

    Args:
        log_returns: Daily log returns, one column per ticker.
        market_returns: Daily log returns of the market index.
        risk_free_rate: Annualized risk-free rate.
        periods_per_year: Trading periods in a year, used to annualize.

    Returns:
        Annualized expected return per ticker.

    Raises:
        OptimizationError: If the market series is empty or has no variance.
    """
    if log_returns is None or log_returns.empty:
        raise OptimizationError("no log returns supplied for CAPM estimation")
    if market_returns is None or market_returns.empty:
        raise OptimizationError("no market returns supplied for CAPM estimation")
    if not log_returns.index.is_unique:
        raise OptimizationError("log returns contain duplicate dates")
    if not market_returns.index.is_unique:
        raise OptimizationError("market returns contain duplicate dates")

    aligned_market = market_returns.reindex(log_returns.index)
    market_variance = float(aligned_market.var())
    if not np.isfinite(market_variance) or market_variance <= 0.0:
        raise OptimizationError(
            "market returns have no variance, so betas cannot be estimated"
        )

    betas = log_returns.apply(lambda series: float(series.cov(aligned_market)))
    betas = betas / market_variance

    market_annual = float(aligned_market.mean()) * periods_per_year
    market_excess = market_annual - risk_free_rate

    expected = risk_free_rate + betas * market_excess
    return expected.rename("expected_return")


def black_litterman_returns(
    covariance: pd.DataFrame,
    market_weights: dict[str, float] | pd.Series,
    risk_free_rate: float,
    market_risk_premium: float = 0.05,
    tau: float = 0.05,
    views: dict[str, float] | None = None,
    view_confidences: dict[str, float] | None = None,
) -> pd.Series:
    """Estimate expected returns with the Black-Litterman model.

    The prior is the market equilibrium return ``pi = rf + delta * Sigma * w``
    for market weights ``w``. Absolute views on individual assets update that
    prior. A view confidence ``theta`` in ``(0, 1)`` says how much of the
    posterior is the view rather than the prior: ``theta = 0.5`` splits the
    difference evenly, ``theta = 0.95`` moves the posterior 95% of the way to
    the view, and ``theta -> 0`` leaves the prior untouched.

    Args:
        covariance: Annualized covariance matrix indexed by ticker.
        market_weights: Market-capitalization weights, indexed by ticker.
        risk_free_rate: Annualized risk-free rate.
        market_risk_premium: Market risk premium used to scale the
            equilibrium returns (the risk-aversion term).
        tau: Uncertainty of the prior, as a fraction of the covariance.
        views: Optional absolute return views keyed by ticker.
        view_confidences: Optional per-view confidences in ``(0, 1)``. A view
            without an entry uses a confidence of ``0.5``.

    Returns:
        Annualized expected return per ticker.

    Raises:
        OptimizationError: If the market weights are unusable or a view
            refers to a ticker outside the covariance matrix.
    """
    if covariance is None or covariance.empty:
        raise OptimizationError("no covariance matrix supplied for Black-Litterman")
    tickers = list(covariance.index)

    weights = pd.Series(0.0, index=tickers)
    for ticker, weight in dict(market_weights).items():
        if ticker not in weights.index:
            raise OptimizationError(
                f"market weight refers to {ticker!r}, which is not in the covariance matrix"
            )
        weights[ticker] = float(weight)
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        raise OptimizationError("market weights must sum to a positive number")
    weights = weights / total_weight

    sigma = covariance.to_numpy(dtype=float)
    prior = risk_free_rate + market_risk_premium * (sigma @ weights.to_numpy())

    if not views:
        return pd.Series(prior, index=tickers, name="expected_return")

    view_tickers = list(views)
    pick = np.zeros((len(view_tickers), len(tickers)))
    for row, ticker in enumerate(view_tickers):
        if ticker not in tickers:
            raise OptimizationError(
                f"view refers to {ticker!r}, which is not in the covariance matrix"
            )
        pick[row, tickers.index(ticker)] = 1.0

    view_values = np.array([float(views[ticker]) for ticker in view_tickers])
    confidences = np.array(
        [float((view_confidences or {}).get(ticker, 0.5)) for ticker in view_tickers]
    )
    if not np.isfinite(confidences).all() or (confidences <= 0.0).any() or (confidences >= 1.0).any():
        raise OptimizationError("view confidences must be strictly between 0 and 1")

    # Omega is diagonal for independent absolute views: Omega_ii = tau * (P S P')_ii / C_i
    # with C_i = theta_i / (1 - theta_i), so C_i = 1 splits prior and view evenly.
    precision_weights = confidences / (1.0 - confidences)
    omega_precision = precision_weights / (tau * np.diag(pick @ sigma @ pick.T))

    prior_precision = np.linalg.pinv(tau * sigma)
    view_precision = pick.T @ (omega_precision[:, None] * pick)

    posterior_precision = prior_precision + view_precision
    posterior = np.linalg.pinv(posterior_precision) @ (
        prior_precision @ prior + pick.T @ (omega_precision * view_values)
    )

    return pd.Series(posterior, index=tickers, name="expected_return")


def blend_estimates(
    estimates: dict[str, pd.Series], weights: dict[str, float]
) -> pd.Series:
    """Blend several expected-return estimates with fixed weights.

    Args:
        estimates: Named expected-return estimates, all indexed by ticker.
        weights: Blend weight per estimate. Must be non-negative and sum to 1.

    Returns:
        The weighted average estimate, indexed by ticker.

    Raises:
        OptimizationError: If the estimates do not share one index, if the
            weights are invalid, or if they do not sum to 1.
    """
    if not estimates:
        raise OptimizationError("no estimates supplied to blend")
    if set(estimates) != set(weights):
        raise OptimizationError(
            "every estimate needs exactly one weight; "
            f"estimates {sorted(estimates)} vs weights {sorted(weights)}"
        )

    weight_values = np.array([float(weights[name]) for name in estimates])
    if not np.isfinite(weight_values).all() or (weight_values < 0.0).any():
        raise OptimizationError("blend weights must be finite and non-negative")
    if not np.isclose(weight_values.sum(), 1.0, atol=1e-9):
        raise OptimizationError(
            f"blend weights must sum to 1, got {weight_values.sum():.6f}"
        )

    names = list(estimates)
    reference_index = estimates[names[0]].index
    for name in names[1:]:
        if not estimates[name].index.equals(reference_index):
            raise OptimizationError(
                "every estimate must be indexed by the same tickers to be blended"
            )

    blended = pd.Series(0.0, index=reference_index)
    for name in names:
        blended = blended + float(weights[name]) * estimates[name]
    return blended.rename("expected_return")
