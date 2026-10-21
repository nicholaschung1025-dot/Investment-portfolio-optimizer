"""Tests for position sizing.

Whole shares and board lots mean a target weight cannot be hit exactly, so
the interesting question is what happens to the leftover. These tests pin
down the share counts, the currency conversion, and the rule that spends the
leftover cash instead of leaving it idle.
"""

from __future__ import annotations

import pytest

from src.market import FxRates, MarketInfo, MarketMetadata
from src.sizing import size_positions


def _prices() -> dict[str, float]:
    return {"AAPL": 101.0, "MSFT": 100.0, "0700.HK": 400.0}


def test_us_positions_are_whole_shares() -> None:
    plan = size_positions(
        {"AAPL": 0.5, "MSFT": 0.5}, _prices(), budget=10_000.0
    )

    for position in plan.positions:
        assert position.shares == int(position.shares)


def test_hong_kong_positions_are_rounded_to_a_board_lot() -> None:
    plan = size_positions({"0700.HK": 1.0}, _prices(), budget=1_000_000.0)

    position = plan.positions[0]
    assert position.shares % position.board_lot == 0
    assert position.board_lot == 100


def test_the_position_value_follows_the_shares_and_the_price() -> None:
    plan = size_positions({"MSFT": 0.5}, _prices(), budget=10_000.0)

    position = plan.positions[0]
    assert position.shares == 50
    assert position.value_usd == pytest.approx(5_000.0)


def test_hong_kong_values_are_converted_into_us_dollars() -> None:
    plan = size_positions({"0700.HK": 1.0}, _prices(), budget=1_000_000.0)

    position = plan.positions[0]
    assert position.currency == "HKD"
    assert position.value_usd == pytest.approx(position.shares * 400.0 / 7.82)


def test_the_leftover_cash_is_reported() -> None:
    plan = size_positions({"AAPL": 1.0}, _prices(), budget=10_000.0)

    assert plan.cash_value == pytest.approx(10_000.0 - 99 * 101.0)
    assert plan.invested_value + plan.cash_value == pytest.approx(10_000.0)


def test_leftover_cash_is_spent_when_a_lot_fits() -> None:
    # AAPL's rounding residual is too small for its own lot, but BB is still
    # short of its target by more than half a lot, so one cheap lot is bought.
    prices = {"AAPL": 101.0, "BB": 0.30}
    plan = size_positions({"AAPL": 0.39998, "BB": 0.60002}, prices, budget=10_000.0)

    idle_without_recycling = 10_000.0 - plan.invested_value
    assert idle_without_recycling > 0.0
    assert plan.cash_value < idle_without_recycling


def test_no_lot_is_bought_when_it_would_move_a_position_away() -> None:
    prices = {"AAPL": 101.0, "BB": 0.30}
    plan = size_positions({"AAPL": 0.4, "BB": 0.6}, prices, budget=10_000.0)

    # AAPL leaves a residual its own lot cannot use and BB sits exactly on
    # its target, so nothing is bought.
    assert plan.cash_value == pytest.approx(10_000.0 - plan.invested_value)


def test_a_position_too_small_for_one_board_lot_is_reported() -> None:
    plan = size_positions({"0700.HK": 0.0343}, {"0700.HK": 393.80}, budget=100_000.0)

    assert plan.positions[0].shares == 0
    assert any("0700.HK" in warning for warning in plan.warnings)
    assert "board lot" in plan.warnings[0]


def test_cash_the_caller_never_allocated_is_not_recycled() -> None:
    plan = size_positions({"MSFT": 0.5}, _prices(), budget=10_000.0)

    assert plan.positions[0].shares == 50
    assert plan.cash_value == pytest.approx(5_000.0)


def test_a_ticker_without_a_price_is_skipped_and_reported() -> None:
    plan = size_positions({"AAPL": 0.5, "NOPE": 0.5}, _prices(), budget=10_000.0)

    assert "NOPE" in plan.skipped
    assert [position.ticker for position in plan.positions] == ["AAPL"]


def test_the_estimated_cost_follows_the_fee() -> None:
    plan = size_positions(
        {"MSFT": 1.0}, _prices(), budget=10_000.0, fee_bps=10.0
    )

    assert plan.estimated_cost == pytest.approx(plan.invested_value * 0.001)


def test_the_realized_weights_stay_close_to_the_targets() -> None:
    plan = size_positions(
        {"AAPL": 0.5, "MSFT": 0.5}, _prices(), budget=10_000.0
    )

    for position in plan.positions:
        assert position.realized_weight == pytest.approx(
            position.target_weight, abs=0.01
        )


def test_a_custom_market_metadata_changes_the_board_lot() -> None:
    metadata = MarketMetadata({"0700.HK": MarketInfo("HKD", 200, "HK")})
    plan = size_positions(
        {"0700.HK": 1.0},
        _prices(),
        budget=1_000_000.0,
        metadata=metadata,
        fx_rates=FxRates.default(),
    )

    assert plan.positions[0].board_lot == 200
    assert plan.positions[0].shares % 200 == 0
