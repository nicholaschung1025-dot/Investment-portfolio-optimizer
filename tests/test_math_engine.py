"""Tests for the quantitative math engine.

All inputs are synthetic in-memory DataFrames so the suite never touches the
network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.math_engine import (
    TRADING_DAYS_PER_YEAR,
    annualized_covariance,
    annualized_expected_returns,
    compute_log_returns,
    detect_zero_variance_assets,
    portfolio_performance,
    sharpe_ratio,
)


def _prices() -> pd.DataFrame:
    """Two-asset, three-day price frame with known log returns."""
    return pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 99.0],
            "BBB": [50.0, 55.0, 60.5],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )


def test_compute_log_returns_matches_closed_form() -> None:
    returns = compute_log_returns(_prices())

    assert returns.shape == (2, 2)
    assert returns.iloc[0]["AAA"] == pytest.approx(np.log(1.1))
    assert returns.iloc[1]["AAA"] == pytest.approx(np.log(99.0 / 110.0))
    assert returns.iloc[0]["BBB"] == pytest.approx(np.log(1.1))
    assert returns.iloc[1]["BBB"] == pytest.approx(np.log(60.5 / 55.0))


def test_compute_log_returns_drops_first_nan_row() -> None:
    returns = compute_log_returns(_prices())

    assert not returns.isna().any().any()
    assert len(returns) == len(_prices()) - 1


def test_annualized_expected_returns_scales_daily_mean_by_252() -> None:
    returns = compute_log_returns(_prices())

    expected = annualized_expected_returns(returns)

    assert expected["AAA"] == pytest.approx(returns["AAA"].mean() * TRADING_DAYS_PER_YEAR)
    assert TRADING_DAYS_PER_YEAR == 252


def test_annualized_covariance_is_symmetric_and_scaled_by_252() -> None:
    returns = compute_log_returns(_prices())

    covariance = annualized_covariance(returns)

    assert covariance.shape == (2, 2)
    assert np.allclose(covariance.to_numpy(), covariance.to_numpy().T)
    assert covariance.loc["AAA", "AAA"] == pytest.approx(
        returns["AAA"].var(ddof=1) * TRADING_DAYS_PER_YEAR
    )


def test_detect_zero_variance_assets_flags_flat_series() -> None:
    flat = pd.DataFrame(
        {
            "FLAT": [0.0, 0.0, 0.0],
            "MOVE": [0.01, -0.02, 0.03],
        }
    )

    flagged = detect_zero_variance_assets(flat)

    assert flagged == ["FLAT"]


def test_detect_zero_variance_assets_flags_undefined_variance() -> None:
    single_observation = pd.DataFrame({"ONE": [0.01]})

    flagged = detect_zero_variance_assets(single_observation)

    assert flagged == ["ONE"]


def test_sharpe_ratio_uses_excess_return_over_volatility() -> None:
    result = sharpe_ratio(0.10, 0.20, risk_free_rate=0.04)

    assert result == pytest.approx((0.10 - 0.04) / 0.20)


def test_sharpe_ratio_returns_zero_for_zero_volatility() -> None:
    assert sharpe_ratio(0.10, 0.0, risk_free_rate=0.04) == 0.0


def test_portfolio_performance_uses_quadratic_forms() -> None:
    expected_returns = pd.Series({"AAA": 0.12, "BBB": 0.06})
    covariance = pd.DataFrame(
        [[0.04, 0.006], [0.006, 0.01]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )
    weights = np.array([0.6, 0.4])

    ret, vol, sharpe = portfolio_performance(weights, expected_returns, covariance, 0.04)

    expected_var = float(weights @ covariance.to_numpy() @ weights)
    assert ret == pytest.approx(0.6 * 0.12 + 0.4 * 0.06)
    assert vol == pytest.approx(np.sqrt(expected_var))
    assert sharpe == pytest.approx((ret - 0.04) / vol)
