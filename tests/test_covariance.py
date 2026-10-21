"""Tests for covariance estimation with shrinkage.

Covers the plain sample estimator (which must stay identical to the existing
``math_engine`` behaviour), the Ledoit-Wolf and OAS shrinkage estimators, and
the positive-semidefiniteness guard. All inputs are synthetic, so no network
access is required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import OAS, LedoitWolf

from src.covariance import ensure_positive_semidefinite, shrunk_covariance
from src.errors import CovarianceError, PortfolioError
from src.math_engine import annualized_covariance


def _log_returns() -> pd.DataFrame:
    """Build a synthetic daily log-return panel with a shared market factor."""
    rng = np.random.default_rng(11)
    dates = pd.date_range("2020-01-01", periods=260, freq="B")
    frame = pd.DataFrame(
        rng.normal(loc=0.0004, scale=0.01, size=(260, 4)),
        index=dates,
        columns=["AAA", "BBB", "CCC", "DDD"],
    )
    frame["AAA"] *= 1.4
    frame["BBB"] *= 0.7
    frame["CCC"] += frame["AAA"] * 0.5
    return frame


def _mean_off_diagonal(matrix: pd.DataFrame) -> float:
    """Average absolute off-diagonal element of a covariance matrix."""
    values = matrix.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, 0.0)
    return float(np.abs(values).mean())


def test_sample_method_matches_existing_annualized_covariance() -> None:
    returns = _log_returns()

    result = shrunk_covariance(returns, method="sample")
    expected = annualized_covariance(returns)

    pd.testing.assert_frame_equal(result, expected)


def test_shrunk_covariance_preserves_shape_and_index() -> None:
    returns = _log_returns()

    result = shrunk_covariance(returns, method="ledoit-wolf")

    assert result.shape == (4, 4)
    assert list(result.index) == ["AAA", "BBB", "CCC", "DDD"]
    assert list(result.columns) == ["AAA", "BBB", "CCC", "DDD"]


def test_ledoit_wolf_matches_sklearn_direct_call() -> None:
    returns = _log_returns()

    result = shrunk_covariance(returns, method="ledoit-wolf")
    expected = pd.DataFrame(
        LedoitWolf().fit(returns.to_numpy(dtype=float)).covariance_ * 252,
        index=returns.columns,
        columns=returns.columns,
    )

    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy())


def test_oas_matches_sklearn_direct_call() -> None:
    returns = _log_returns()

    result = shrunk_covariance(returns, method="oas")
    expected = pd.DataFrame(
        OAS().fit(returns.to_numpy(dtype=float)).covariance_ * 252,
        index=returns.columns,
        columns=returns.columns,
    )

    np.testing.assert_allclose(result.to_numpy(), expected.to_numpy())


def test_shrinkage_reduces_off_diagonal_magnitude() -> None:
    returns = _log_returns()

    sample = shrunk_covariance(returns, method="sample")
    shrunk = shrunk_covariance(returns, method="ledoit-wolf")

    assert _mean_off_diagonal(shrunk) < _mean_off_diagonal(sample)


def test_ledoit_wolf_covariance_is_positive_semidefinite() -> None:
    returns = _log_returns()

    result = shrunk_covariance(returns, method="ledoit-wolf")

    assert float(np.linalg.eigvalsh(result.to_numpy(dtype=float)).min()) > -1e-8


def test_empty_returns_raise_covariance_error() -> None:
    with pytest.raises(CovarianceError):
        shrunk_covariance(pd.DataFrame())


def test_returns_with_missing_values_raise_covariance_error() -> None:
    returns = _log_returns()
    returns.iloc[0, 0] = np.nan

    with pytest.raises(CovarianceError):
        shrunk_covariance(returns)


def test_unknown_method_raises_covariance_error() -> None:
    returns = _log_returns()

    with pytest.raises(CovarianceError):
        shrunk_covariance(returns, method="not-a-method")


def test_ensure_positive_semidefinite_rejects_indefinite_matrix() -> None:
    indefinite = pd.DataFrame(
        [[0.01, 0.05], [0.05, 0.01]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )

    with pytest.raises(CovarianceError):
        ensure_positive_semidefinite(indefinite)


def test_ensure_positive_semidefinite_accepts_psd_matrix() -> None:
    psd = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.02]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )

    result = ensure_positive_semidefinite(psd)

    assert result is not None


def test_covariance_error_is_a_portfolio_error() -> None:
    assert issubclass(CovarianceError, PortfolioError)
