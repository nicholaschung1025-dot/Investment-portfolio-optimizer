"""Tests for portfolio selection with mandatory holdings.

A mandatory holding has to survive the search, and the search itself has to
find the subset a brute-force enumeration would find -- that correctness
property is what the beam search is checked against here.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from src.constraints import WeightConstraints
from src.errors import ConstraintError
from src.optimizer import optimize_portfolio
from src.selection import select_portfolio


def _inputs() -> tuple[pd.Series, pd.DataFrame]:
    expected_returns = pd.Series(
        {"FIXED": 0.05, "AAA": 0.20, "BBB": 0.12, "CCC": 0.09, "DDD": 0.07, "EEE": 0.04}
    )
    covariance = pd.DataFrame(
        [
            [0.0400, 0.0060, 0.0040, 0.0030, 0.0020, 0.0010],
            [0.0060, 0.0900, 0.0100, 0.0080, 0.0050, 0.0030],
            [0.0040, 0.0100, 0.0625, 0.0120, 0.0060, 0.0040],
            [0.0030, 0.0080, 0.0120, 0.0300, 0.0090, 0.0050],
            [0.0020, 0.0050, 0.0060, 0.0090, 0.0200, 0.0060],
            [0.0010, 0.0030, 0.0040, 0.0050, 0.0060, 0.0225],
        ],
        index=list(expected_returns.index),
        columns=list(expected_returns.index),
    )
    return expected_returns, covariance


def _constraints() -> WeightConstraints:
    return WeightConstraints(min_weights={"FIXED": 0.1})


def test_fixed_holdings_are_always_in_the_selection() -> None:
    expected_returns, covariance = _inputs()

    result = select_portfolio(
        expected_returns, covariance, ["FIXED"], max_k=3, constraints=_constraints()
    )

    assert "FIXED" in result.tickers


def test_the_selection_never_exceeds_max_k() -> None:
    expected_returns, covariance = _inputs()

    result = select_portfolio(
        expected_returns, covariance, ["FIXED"], max_k=3, constraints=_constraints()
    )

    assert len(result.tickers) == 3


def test_the_weights_of_the_selection_sum_to_one() -> None:
    expected_returns, covariance = _inputs()

    result = select_portfolio(
        expected_returns, covariance, ["FIXED"], max_k=3, constraints=_constraints()
    )

    assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)


def test_fixed_holdings_receive_a_minimum_weight() -> None:
    expected_returns, covariance = _inputs()

    result = select_portfolio(
        expected_returns, covariance, ["FIXED"], max_k=3, constraints=_constraints()
    )

    assert result.weights["FIXED"] >= 0.1 - 1e-6


def test_a_fixed_holding_gets_a_default_floor_when_none_is_given() -> None:
    expected_returns, covariance = _inputs()

    result = select_portfolio(expected_returns, covariance, ["FIXED"], max_k=3)

    assert result.weights["FIXED"] >= 0.05 - 1e-6


def test_the_search_finds_what_an_exhaustive_enumeration_finds() -> None:
    expected_returns, covariance = _inputs()
    constraints = _constraints()

    result = select_portfolio(
        expected_returns, covariance, ["FIXED"], max_k=3, constraints=constraints
    )

    best_sharpe = -np.inf
    optional = [ticker for ticker in expected_returns.index if ticker != "FIXED"]
    for combination in itertools.combinations(optional, 2):
        subset = ["FIXED", *combination]
        candidate = optimize_portfolio(
            expected_returns[subset],
            covariance.loc[subset, subset],
            "max_sharpe",
            0.04,
            constraints=WeightConstraints(min_weights={"FIXED": 0.1}),
        )
        best_sharpe = max(best_sharpe, candidate.sharpe_ratio)

    assert result.sharpe_ratio == pytest.approx(best_sharpe, abs=1e-6)


def test_the_selection_reports_which_tickers_were_chosen() -> None:
    expected_returns, covariance = _inputs()

    result = select_portfolio(
        expected_returns, covariance, ["FIXED"], max_k=3, constraints=_constraints()
    )

    assert result.fixed_tickers == ["FIXED"]
    assert set(result.selected_tickers).isdisjoint({"FIXED"})
    assert set(result.tickers) == {"FIXED", *result.selected_tickers}


def test_a_fixed_holding_outside_the_universe_is_refused() -> None:
    expected_returns, covariance = _inputs()

    with pytest.raises(ConstraintError):
        select_portfolio(expected_returns, covariance, ["NOPE"], max_k=3)


def test_more_fixed_holdings_than_max_k_is_refused() -> None:
    expected_returns, covariance = _inputs()

    with pytest.raises(ConstraintError):
        select_portfolio(
            expected_returns, covariance, ["FIXED", "AAA", "BBB"], max_k=2
        )


def test_mandatory_holdings_that_already_fill_the_portfolio_are_kept() -> None:
    expected_returns, covariance = _inputs()
    constraints = WeightConstraints(min_weights={"FIXED": 0.3, "AAA": 0.2})

    result = select_portfolio(
        expected_returns, covariance, ["FIXED", "AAA"], max_k=2, constraints=constraints
    )

    assert result.tickers == ["FIXED", "AAA"]
    assert result.weights["FIXED"] >= 0.3 - 1e-6
    assert result.weights["AAA"] >= 0.2 - 1e-6


def test_a_maximum_weight_that_needs_more_slots_still_selects() -> None:
    expected_returns, covariance = _inputs()

    result = select_portfolio(
        expected_returns,
        covariance,
        ["FIXED"],
        max_k=4,
        constraints=WeightConstraints(min_weights={"FIXED": 0.1}, max_weight=0.4),
    )

    assert len(result.tickers) == 4
    assert (result.weights <= 0.4 + 1e-6).all()
    assert result.weights["FIXED"] >= 0.1 - 1e-6


def test_a_maximum_weight_that_cannot_fill_the_slots_is_refused() -> None:
    expected_returns, covariance = _inputs()

    with pytest.raises(ConstraintError):
        select_portfolio(
            expected_returns,
            covariance,
            ["FIXED"],
            max_k=2,
            constraints=WeightConstraints(max_weight=0.4),
        )


def test_a_max_k_larger_than_the_universe_keeps_everything() -> None:
    expected_returns, covariance = _inputs()

    result = select_portfolio(expected_returns, covariance, ["FIXED"], max_k=99)

    assert len(result.tickers) == len(expected_returns)
