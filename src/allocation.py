"""Capital allocation between the risky portfolio and cash.

Mean-variance optimization answers "what is the best mix of risky assets?".
It does not answer "how much of my money should be in risky assets at all?".
That is the capital allocation line: the tangency portfolio is scaled
against a risk-free asset so the combined portfolio hits a requested return.

A target return that the tangency portfolio cannot reach is an error, not a
label. Reporting an unreachable target as if it had been solved is the kind
of quiet inaccuracy that makes an optimizer untrustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.errors import AllocationError
from src.optimizer import OptimizationResult


@dataclass(frozen=True)
class AllocationResult:
    """A risky portfolio combined with a cash position.

    Attributes:
        risky_weights: Weight of each risky asset, already scaled.
        cash_weight: Weight held in the risk-free asset.
        expected_return: Annualized expected return of the combined portfolio.
        volatility: Annualized volatility of the combined portfolio.
        sharpe_ratio: Annualized Sharpe ratio of the combined portfolio.
        target_return: The return that was requested.
    """

    risky_weights: pd.Series
    cash_weight: float
    expected_return: float
    volatility: float
    sharpe_ratio: float
    target_return: float


def allocate_to_target_return(
    tangency: OptimizationResult,
    risk_free_rate: float,
    target_return: float,
) -> AllocationResult:
    """Scale the tangency portfolio against cash to hit a target return.

    The scaling factor is the capital allocation line:
    ``w_risky = (target - rf) / (tangency_return - rf)``, with the remainder
    held in the risk-free asset.

    Args:
        tangency: The optimal risky portfolio.
        risk_free_rate: Annualized risk-free rate.
        target_return: Annualized return the combined portfolio should reach.

    Returns:
        The combined :class:`AllocationResult`.

    Raises:
        AllocationError: If the target cannot be reached, either because it
            exceeds the tangency portfolio's return or because it is below the
            risk-free rate.
    """
    tangency_return = float(tangency.expected_return)

    if tangency_return <= risk_free_rate:
        raise AllocationError(
            "the tangency portfolio does not earn a risk premium, so it cannot "
            f"be scaled to a target return (tangency {tangency_return:.2%}, "
            f"risk-free {risk_free_rate:.2%})"
        )
    if target_return > tangency_return + 1e-12:
        raise AllocationError(
            f"target return {target_return:.2%} exceeds the maximum achievable "
            f"return {tangency_return:.2%}"
        )
    if target_return < risk_free_rate - 1e-12:
        raise AllocationError(
            f"target return {target_return:.2%} is below the risk-free rate "
            f"{risk_free_rate:.2%} and cannot be reached with long-only holdings"
        )

    risky_weight = (target_return - risk_free_rate) / (tangency_return - risk_free_rate)
    risky_weight = min(max(risky_weight, 0.0), 1.0)
    cash_weight = 1.0 - risky_weight

    risky_weights = tangency.weights * risky_weight
    expected_return = risky_weight * tangency_return + cash_weight * risk_free_rate
    volatility = risky_weight * float(tangency.volatility)
    sharpe = (expected_return - risk_free_rate) / volatility if volatility > 1e-12 else 0.0

    return AllocationResult(
        risky_weights=risky_weights,
        cash_weight=cash_weight,
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=sharpe,
        target_return=target_return,
    )
