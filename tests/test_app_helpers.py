"""Tests for the Streamlit front end helpers.

The app is a thin layer over the optimizer, so everything worth testing
lives in ``src.app_helpers``. These tests cover the parsing of the sidebar
inputs and the tables the app renders.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.app_helpers import (
    comparison_frame,
    history_frame,
    parse_tickers,
    parse_weight_assignments,
    positions_frame,
    weights_frame,
)
from src.sizing import Position, PositionPlan


def test_tickers_are_split_on_commas_spaces_and_newlines() -> None:
    assert parse_tickers("AAPL, MSFT\nGOOGL  AMZN") == ["AAPL", "MSFT", "GOOGL", "AMZN"]


def test_tickers_are_upper_cased_and_deduplicated() -> None:
    assert parse_tickers("aapl aapl msft") == ["AAPL", "MSFT"]


def test_empty_ticker_text_gives_an_empty_list() -> None:
    assert parse_tickers("   \n ") == []


def test_weight_assignments_are_parsed() -> None:
    weights = parse_weight_assignments("AAPL=0.25, MSFT=0.15\nGOOGL=0.10")

    assert weights == {"AAPL": 0.25, "MSFT": 0.15, "GOOGL": 0.10}


def test_a_malformed_weight_assignment_is_refused() -> None:
    with pytest.raises(ValueError):
        parse_weight_assignments("AAPL")


def test_an_out_of_range_weight_is_refused() -> None:
    with pytest.raises(ValueError):
        parse_weight_assignments("AAPL=1.5")


def test_the_weights_table_is_ordered_by_weight() -> None:
    frame = weights_frame(pd.Series({"AAPL": 0.3, "MSFT": 0.7}))

    assert list(frame["Ticker"]) == ["MSFT", "AAPL"]


def test_the_weights_table_can_include_cash() -> None:
    frame = weights_frame(pd.Series({"AAPL": 1.0}), cash_weight=0.25)

    assert "CASH (risk-free)" in list(frame["Ticker"])


def test_the_positions_table_ends_with_cash() -> None:
    plan = PositionPlan(
        positions=[
            Position(
                ticker="AAPL",
                target_weight=0.5,
                shares=10,
                board_lot=1,
                price=100.0,
                currency="USD",
                value_usd=1_000.0,
                realized_weight=0.5,
                drift=0.0,
            )
        ],
        invested_value=1_000.0,
        cash_value=20.0,
        total_value=1_020.0,
        estimated_cost=0.5,
        skipped=[],
    )

    frame = positions_frame(plan)

    assert list(frame["Ticker"]) == ["AAPL", "CASH"]
    assert frame.iloc[-1]["Value (USD)"] == 20.0


def test_the_history_table_reads_the_stored_run() -> None:
    runs = [
        {
            "Run_ID": 3,
            "Created_At": "2026-10-22 10:00:00",
            "tickers": ["AAPL", "MSFT"],
            "objective_used": "max_sharpe",
            "Expected_Return": 0.11,
            "Expected_Volatility": 0.18,
            "Sharpe_Ratio": 0.39,
            "cash_weight": 0.1,
            "expected_return_method": "black_litterman",
            "var_95": 8_200.0,
            "cvar_95": 12_900.0,
            "weights": {"AAPL": 0.6, "MSFT": 0.4},
        }
    ]

    frame = history_frame(runs)

    assert frame.iloc[0]["Run"] == 3
    assert frame.iloc[0]["Top holding"] == "AAPL"
    assert frame.iloc[0]["Top weight"] == pytest.approx(0.6)


def test_the_comparison_table_gets_readable_column_names() -> None:
    comparison = pd.DataFrame(
        [
            {
                "method": "historical_mean",
                "optimized_cagr": 0.28,
                "optimized_sharpe": 1.27,
                "optimized_max_drawdown": -0.12,
                "benchmark_cagr": 0.40,
                "benchmark_sharpe": 2.25,
                "turnover": 0.1,
            }
        ]
    )

    frame = comparison_frame(comparison)

    assert "Method" in frame.columns
    assert "Benchmark Sharpe" in frame.columns


def test_an_empty_comparison_table_is_returned_unchanged() -> None:
    empty = pd.DataFrame()

    assert comparison_frame(empty).empty
