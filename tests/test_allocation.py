"""Tests for capital allocation to a target return.

Covers the capital allocation line: scaling the risky portfolio against cash
so the combined portfolio hits a requested return, and refusing targets that
cannot be reached.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.allocation import allocate_to_target_return
from src.errors import AllocationError, PortfolioError
from src.optimizer import OptimizationResult


def _tangency(
    expected_return: float = 0.20, volatility: float = 0.15
) -> OptimizationResult:
    weights = pd.Series({"AAA": 0.6, "BBB": 0.4})
    return OptimizationResult(
        weights=weights,
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=(expected_return - 0.04) / volatility,
        objective="max_sharpe",
    )


def test_target_below_tangency_mixes_in_cash() -> None:
    result = allocate_to_target_return(_tangency(), risk_free_rate=0.04, target_return=0.12)

    assert result.cash_weight == pytest.approx(0.5)
    assert result.expected_return == pytest.approx(0.12)


def test_target_below_tangency_scales_the_risky_weights() -> None:
    result = allocate_to_target_return(_tangency(), risk_free_rate=0.04, target_return=0.12)

    assert result.risky_weights["AAA"] == pytest.approx(0.3)
    assert result.risky_weights["BBB"] == pytest.approx(0.2)
    assert result.risky_weights.sum() == pytest.approx(0.5)


def test_target_below_tangency_scales_the_volatility() -> None:
    result = allocate_to_target_return(_tangency(), risk_free_rate=0.04, target_return=0.12)

    assert result.volatility == pytest.approx(0.075)


def test_target_equal_to_tangency_stays_fully_invested() -> None:
    result = allocate_to_target_return(_tangency(), risk_free_rate=0.04, target_return=0.20)

    assert result.cash_weight == pytest.approx(0.0)
    assert result.risky_weights.sum() == pytest.approx(1.0)


def test_target_equal_to_risk_free_rate_holds_only_cash() -> None:
    result = allocate_to_target_return(_tangency(), risk_free_rate=0.04, target_return=0.04)

    assert result.cash_weight == pytest.approx(1.0)
    assert result.risky_weights.sum() == pytest.approx(0.0)


def test_target_above_tangency_raises_allocation_error() -> None:
    with pytest.raises(AllocationError):
        allocate_to_target_return(_tangency(), risk_free_rate=0.04, target_return=0.30)


def test_target_below_risk_free_rate_raises_allocation_error() -> None:
    with pytest.raises(AllocationError):
        allocate_to_target_return(_tangency(), risk_free_rate=0.04, target_return=0.01)


def test_tangency_without_a_risk_premium_raises_allocation_error() -> None:
    tangency = _tangency(expected_return=0.04)

    with pytest.raises(AllocationError):
        allocate_to_target_return(tangency, risk_free_rate=0.04, target_return=0.05)


def test_allocation_error_is_a_portfolio_error() -> None:
    assert issubclass(AllocationError, PortfolioError)
