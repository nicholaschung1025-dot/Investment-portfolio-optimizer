"""Quantitative math engine: returns, annualized metrics, and covariance.

Transforms a price panel into the statistical inputs the optimizer needs.
All annualization uses a fixed 252-trading-day year so results are
deterministic and reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from a price panel.

    Log returns are used throughout because they are time-additive and are the
    standard input for continuous-compounding portfolio mathematics.

    Args:
        prices: DataFrame of adjusted close prices, indexed by date with one
            column per ticker.

    Returns:
        DataFrame of daily log returns with the leading all-NaN row removed.
    """
    prices = prices.astype(float)
    returns = np.log(prices / prices.shift(1))
    return returns.dropna(how="all")


def annualized_expected_returns(
    log_returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """Annualize each asset's mean daily log return.

    Args:
        log_returns: DataFrame of daily log returns, one column per ticker.
        periods_per_year: Trading periods in a year (default 252).

    Returns:
        Series of annualized expected returns indexed by ticker.
    """
    return log_returns.mean() * periods_per_year


def annualized_covariance(
    log_returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.DataFrame:
    """Annualize the covariance matrix of daily log returns.

    Args:
        log_returns: DataFrame of daily log returns, one column per ticker.
        periods_per_year: Trading periods in a year (default 252).

    Returns:
        Symmetric annualized covariance matrix indexed and columned by ticker.
    """
    covariance = log_returns.cov() * periods_per_year
    return covariance


def detect_zero_variance_assets(log_returns: pd.DataFrame) -> list[str]:
    """Find tickers whose returns never move.

    A zero-variance asset makes the covariance matrix singular and breaks the
    optimizer, so it must be excluded before optimization.

    Args:
        log_returns: DataFrame of daily log returns, one column per ticker.

    Returns:
        Sorted list of ticker names with zero or undefined variance.
    """
    variances = log_returns.var()
    unusable = (variances <= 0.0) | variances.isna()
    return sorted(variances[unusable].index.tolist())


def sharpe_ratio(expected_return: float, volatility: float, risk_free_rate: float) -> float:
    """Compute the annualized Sharpe ratio.

    Args:
        expected_return: Annualized expected portfolio return.
        volatility: Annualized portfolio volatility.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        The Sharpe ratio, or 0.0 when volatility is zero.
    """
    if volatility == 0.0:
        return 0.0
    return (expected_return - risk_free_rate) / volatility


def portfolio_performance(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
) -> tuple[float, float, float]:
    """Evaluate a weight vector's return, volatility, and Sharpe ratio.

    Args:
        weights: Portfolio weights aligned with ``expected_returns`` order.
        expected_returns: Annualized expected returns per asset.
        covariance: Annualized covariance matrix.
        risk_free_rate: Annualized risk-free rate.

    Returns:
        Tuple of ``(expected_return, volatility, sharpe)``.
    """
    weights = np.asarray(weights, dtype=float)
    expected_return = float(weights @ expected_returns.to_numpy())
    variance = float(weights @ covariance.to_numpy() @ weights)
    volatility = float(np.sqrt(max(variance, 0.0)))
    return expected_return, volatility, sharpe_ratio(expected_return, volatility, risk_free_rate)
