"""Portfolio selection with mandatory holdings.

Choosing which assets to hold is a search problem: a portfolio of six names
out of a universe of thirty has more combinations than there are sensible
ways to enumerate them one at a time. A beam search keeps the most promising
partial selections alive while it grows them, which finds the same subset an
exhaustive enumeration would on a small universe while staying fast on a
large one.

Mandatory holdings are not negotiable. They are part of the portfolio from
the first step and carry a minimum weight, so the optimizer cannot satisfy
the constraint by holding them in name only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from src.constraints import WeightConstraints
from src.errors import ConstraintError, OptimizationError
from src.optimizer import Objective, OptimizationResult, optimize_portfolio

DEFAULT_FIXED_WEIGHT = 0.05


@dataclass(frozen=True)
class SelectionResult:
    """The outcome of a portfolio selection.

    Attributes:
        tickers: Every ticker in the selected portfolio.
        weights: Optimized weight of each selected ticker.
        fixed_tickers: The mandatory holdings.
        selected_tickers: The tickers the search chose.
        sharpe_ratio: Sharpe ratio of the selected portfolio.
        expected_return: Annualized expected return of the selection.
        volatility: Annualized volatility of the selection.
        steps: One entry per search step, with the ticker it added and the
            score of the best portfolio at that step.
    """

    tickers: list[str]
    weights: pd.Series
    fixed_tickers: list[str]
    selected_tickers: list[str]
    sharpe_ratio: float
    expected_return: float
    volatility: float
    constraints: WeightConstraints | None = None
    steps: list[dict[str, object]] = field(default_factory=list)


def _subset_constraints(
    tickers: Sequence[str],
    constraints: WeightConstraints | None,
    relax: bool = False,
) -> WeightConstraints | None:
    """Return the constraints that apply to a subset of the universe.

    Args:
        tickers: The subset.
        constraints: The constraints of the whole portfolio.
        relax: Whether a partial selection may use a looser weight cap. A
            partial portfolio has to be feasible to be scored at all, and a
            cap of 40% leaves two assets unable to fill the portfolio even
            though four of them could. The final portfolio always keeps the
            real cap.
    """
    if constraints is None:
        return None
    allowed = set(tickers)
    max_weight = constraints.max_weight
    if relax and max_weight is not None and tickers:
        max_weight = max(max_weight, 1.0 / len(tickers))
    return WeightConstraints(
        min_weights={
            ticker: weight
            for ticker, weight in constraints.min_weights.items()
            if ticker in allowed
        },
        max_weight=max_weight,
        max_volatility=constraints.max_volatility,
    )


def _score(result: OptimizationResult, objective: Objective) -> float:
    """Score a portfolio; higher is better."""
    if objective == "max_sharpe":
        return result.sharpe_ratio
    return -result.volatility


def _optimize_subset(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    subset: Sequence[str],
    objective: Objective,
    risk_free_rate: float,
    constraints: WeightConstraints | None,
    relax: bool = False,
) -> OptimizationResult:
    """Optimize one candidate subset of the universe."""
    tickers = list(subset)
    return optimize_portfolio(
        expected_returns[tickers],
        covariance.loc[tickers, tickers],
        objective,
        risk_free_rate,
        constraints=_subset_constraints(tickers, constraints, relax=relax),
    )


def select_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    fixed_tickers: Sequence[str] = (),
    max_k: int = 0,
    objective: Objective = "max_sharpe",
    risk_free_rate: float = 0.04,
    constraints: WeightConstraints | None = None,
    beam_width: int = 5,
    default_fixed_weight: float = DEFAULT_FIXED_WEIGHT,
) -> SelectionResult:
    """Choose the assets to hold, keeping the mandatory holdings in.

    Args:
        expected_returns: Annualized expected return per asset.
        covariance: Annualized covariance matrix.
        fixed_tickers: Holdings that must be part of the portfolio.
        max_k: Maximum number of assets to hold. A value of zero or less
            keeps the whole universe.
        objective: ``"max_sharpe"`` or ``"min_vol"``.
        risk_free_rate: Annualized risk-free rate.
        constraints: Limits applied to the portfolio. Mandatory holdings get
            a minimum weight even when the constraints do not mention them.
        beam_width: How many partial selections the search keeps per step.
        default_fixed_weight: Minimum weight given to a mandatory holding
            that has no explicit minimum.

    Returns:
        The chosen :class:`SelectionResult`.

    Raises:
        ConstraintError: If a mandatory holding is not in the universe, if
            there are more mandatory holdings than slots, or if no feasible
            portfolio exists.
    """
    universe = list(expected_returns.index)
    fixed = list(dict.fromkeys(fixed_tickers))

    for ticker in fixed:
        if ticker not in universe:
            raise ConstraintError(
                f"mandatory holding {ticker!r} is not in the asset universe"
            )
    if max_k > 0 and len(fixed) > max_k:
        raise ConstraintError(
            f"{len(fixed)} mandatory holdings cannot fit into {max_k} slots"
        )
    if beam_width < 1:
        raise ConstraintError(f"the beam width must be at least 1, got {beam_width}")

    effective = WeightConstraints(
        min_weights=dict(constraints.min_weights) if constraints else {},
        max_weight=constraints.max_weight if constraints else None,
        max_volatility=constraints.max_volatility if constraints else None,
    )
    for ticker in fixed:
        effective.min_weights.setdefault(ticker, default_fixed_weight)

    keep = max_k if max_k > 0 else len(universe)
    keep = min(keep, len(universe))

    if constraints is not None and constraints.max_weight is not None:
        if keep * constraints.max_weight < 1.0 - 1e-12:
            needed = math.ceil(1.0 / constraints.max_weight)
            raise ConstraintError(
                f"a maximum weight of {constraints.max_weight:.0%} leaves {keep} slots "
                f"unable to invest fully; use at least {needed} assets or a higher "
                f"maximum weight"
            )

    if len(fixed) == keep:
        # The mandatory holdings already fill the portfolio, so there is
        # nothing to search for.
        result = _optimize_subset(
            expected_returns, covariance, fixed, objective, risk_free_rate, effective
        )
        return SelectionResult(
            tickers=list(fixed),
            weights=result.weights,
            fixed_tickers=fixed,
            selected_tickers=[],
            sharpe_ratio=result.sharpe_ratio,
            expected_return=result.expected_return,
            volatility=result.volatility,
            constraints=_subset_constraints(list(fixed), effective),
        )

    if keep == len(universe):
        result = _optimize_subset(
            expected_returns, covariance, universe, objective, risk_free_rate, effective
        )
        return SelectionResult(
            tickers=list(universe),
            weights=result.weights,
            fixed_tickers=fixed,
            selected_tickers=[ticker for ticker in universe if ticker not in fixed],
            sharpe_ratio=result.sharpe_ratio,
            expected_return=result.expected_return,
            volatility=result.volatility,
            constraints=effective,
        )

    beam: list[tuple[str, ...]] = [tuple(fixed)]
    best_result: OptimizationResult | None = None
    best_subset: tuple[str, ...] = tuple(fixed)
    steps: list[dict[str, object]] = []

    for _ in range(len(fixed), keep):
        expansions: list[tuple[float, tuple[str, ...], OptimizationResult]] = []
        for partial in beam:
            for candidate in universe:
                if candidate in partial:
                    continue
                subset = tuple(partial) + (candidate,)
                try:
                    candidate_result = _optimize_subset(
                        expected_returns,
                        covariance,
                        subset,
                        objective,
                        risk_free_rate,
                        effective,
                        relax=len(subset) < keep,
                    )
                except (ConstraintError, OptimizationError):
                    continue
                expansions.append(
                    (_score(candidate_result, objective), subset, candidate_result)
                )

        if not expansions:
            break

        expansions.sort(key=lambda item: (-item[0], item[1]))
        beam = [subset for _, subset, _ in expansions[:beam_width]]
        score, subset, candidate_result = expansions[0]
        best_result = candidate_result
        best_subset = subset
        steps.append(
            {
                "added": subset[-1],
                "score": score,
                "portfolio": list(subset),
            }
        )

    if best_result is None:
        raise ConstraintError(
            "no feasible portfolio was found for the given constraints and holdings"
        )

    return SelectionResult(
        tickers=list(best_subset),
        weights=best_result.weights,
        fixed_tickers=fixed,
        selected_tickers=[ticker for ticker in best_subset if ticker not in fixed],
        sharpe_ratio=best_result.sharpe_ratio,
        expected_return=best_result.expected_return,
        volatility=best_result.volatility,
        constraints=_subset_constraints(list(best_subset), effective),
        steps=steps,
    )
