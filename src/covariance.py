"""Covariance estimation with optional shrinkage.

Mean-variance optimization is very sensitive to estimation error in the
covariance matrix: a noisy sample covariance produces extreme, unstable
weights. Shrinkage estimators pull the sample covariance towards a scaled
identity matrix, which stabilizes the optimizer without assuming any
structure in the correlation matrix.

The shrinkage estimators are fitted on **daily** returns (the frequency the
estimator expects) and the resulting covariance is annualized afterwards.
Note that the shrinkage estimators use the biased (maximum-likelihood)
empirical covariance internally, so they differ from the ``sample`` method by
a factor of ``(n - 1) / n`` -- negligible for the hundreds of daily
observations a real panel contains.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.errors import CovarianceError

CovarianceMethod = Literal["sample", "ledoit-wolf", "oas"]

_VALID_METHODS = ("sample", "ledoit-wolf", "oas")
_PSD_TOLERANCE = -1e-8


def ensure_positive_semidefinite(
    covariance: pd.DataFrame, tolerance: float = _PSD_TOLERANCE
) -> pd.DataFrame:
    """Validate that a covariance matrix is square and positive semidefinite.

    Args:
        covariance: Candidate covariance matrix.
        tolerance: Smallest acceptable minimum eigenvalue. A matrix with a
            more negative eigenvalue is rejected.

    Returns:
        The covariance matrix, unchanged, when it passes the checks.

    Raises:
        CovarianceError: If the matrix is empty, not square, contains NaN
            values, or has a negative eigenvalue beyond ``tolerance``.
    """
    if covariance is None or covariance.empty:
        raise CovarianceError("covariance matrix is empty")
    if covariance.shape[0] != covariance.shape[1]:
        raise CovarianceError(
            f"covariance matrix must be square, got shape {covariance.shape}"
        )

    values = covariance.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise CovarianceError("covariance matrix contains NaN or infinite values")

    eigenvalues = np.linalg.eigvalsh((values + values.T) / 2)
    if float(eigenvalues.min()) < tolerance:
        raise CovarianceError(
            "covariance matrix is not positive semidefinite "
            f"(minimum eigenvalue {float(eigenvalues.min()):.3e}); "
            "exclude zero-variance or degenerate assets before optimizing"
        )

    return covariance


def shrunk_covariance(
    log_returns: pd.DataFrame,
    method: CovarianceMethod = "ledoit-wolf",
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Estimate an annualized covariance matrix, optionally with shrinkage.

    Args:
        log_returns: Daily log returns, indexed by date with one column per
            ticker.
        method: ``"sample"`` for the plain unbiased sample covariance,
            ``"ledoit-wolf"`` or ``"oas"`` for shrinkage estimators.
        periods_per_year: Trading periods in a year, used to annualize.

    Returns:
        A symmetric, positive semidefinite covariance matrix indexed and
        columned by ticker.

    Raises:
        CovarianceError: If the inputs are unusable or the result is not
            positive semidefinite.
    """
    if method not in _VALID_METHODS:
        raise CovarianceError(
            f"unknown covariance method {method!r}; expected one of {_VALID_METHODS}"
        )
    if log_returns is None or log_returns.empty:
        raise CovarianceError("no log returns supplied for covariance estimation")
    if log_returns.shape[0] < 2:
        raise CovarianceError(
            f"need at least 2 observations to estimate a covariance, got {log_returns.shape[0]}"
        )

    returns = log_returns.astype(float)
    if not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise CovarianceError("log returns contain NaN or infinite values")

    if method == "sample":
        covariance = returns.cov() * periods_per_year
    else:
        from sklearn.covariance import OAS, LedoitWolf

        estimator = LedoitWolf() if method == "ledoit-wolf" else OAS()
        estimator.fit(returns.to_numpy(dtype=float))
        covariance = pd.DataFrame(
            estimator.covariance_ * periods_per_year,
            index=returns.columns,
            columns=returns.columns,
        )

    covariance = (covariance + covariance.T) / 2
    ensure_positive_semidefinite(covariance)
    return covariance
