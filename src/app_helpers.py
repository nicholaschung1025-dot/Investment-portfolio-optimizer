"""Helpers for the Streamlit front end.

The Streamlit app itself is a thin layer over the optimizer: it collects the
inputs, calls :func:`main.run_analysis` and renders the result. Everything
that can be tested without a browser lives here, so the app stays readable
and the logic stays verifiable.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.sizing import PositionPlan


def parse_tickers(text: str) -> list[str]:
    """Split a block of text into ticker symbols.

    Tickers may be separated by commas, spaces or newlines. The order the
    user wrote them in is preserved, and duplicates are removed.

    Args:
        text: The raw text from the input box.

    Returns:
        The tickers, upper-cased, without duplicates.
    """
    cleaned = text.replace(",", " ").replace("\n", " ")
    tickers: list[str] = []
    for chunk in cleaned.split():
        ticker = chunk.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def parse_weight_assignments(text: str) -> dict[str, float]:
    """Parse ``TICKER=WEIGHT`` assignments.

    Entries may be separated by commas, semicolons or newlines.

    Args:
        text: The raw text from the input box.

    Returns:
        The weight per ticker.

    Raises:
        ValueError: If an entry is malformed or its weight is out of range.
    """
    weights: dict[str, float] = {}
    for chunk in text.replace(";", ",").replace("\n", ",").split(","):
        entry = chunk.strip()
        if not entry:
            continue
        ticker, separator, value = entry.partition("=")
        if not separator:
            raise ValueError(f"expected TICKER=WEIGHT, got {entry!r}")
        ticker = ticker.strip().upper()
        try:
            weight = float(value)
        except ValueError as error:
            raise ValueError(f"{entry!r} does not contain a numeric weight") from error
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"the weight for {ticker} must be between 0 and 1")
        weights[ticker] = weight
    return weights


def weights_frame(
    target: pd.Series, cash_weight: float = 0.0
) -> pd.DataFrame:
    """Build the table of target and realized weights.

    Args:
        target: The target weight per ticker.
        cash_weight: The share held in the risk-free asset.

    Returns:
        One row per ticker plus a cash row, ordered by target weight.
    """
    rows = [
        {
            "Ticker": ticker,
            "Target weight": float(weight),
        }
        for ticker, weight in target.items()
    ]
    if cash_weight:
        rows.append(
            {
                "Ticker": "CASH (risk-free)",
                "Target weight": float(cash_weight),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("Target weight", ascending=False, ignore_index=True)


def positions_frame(plan: PositionPlan) -> pd.DataFrame:
    """Build the table of tradable positions.

    Args:
        plan: The position plan of a run.

    Returns:
        One row per position, with the cash as the last row.
    """
    rows = [
        {
            "Ticker": position.ticker,
            "Target weight": position.target_weight,
            "Realized weight": position.realized_weight,
            "Drift": position.drift,
            "Shares": position.shares,
            "Board lot": position.board_lot,
            "Price": position.price,
            "Currency": position.currency,
            "Value (USD)": position.value_usd,
        }
        for position in plan.positions
    ]
    rows.append(
        {
            "Ticker": "CASH",
            "Target weight": 0.0,
            "Realized weight": 0.0,
            "Drift": 0.0,
            "Shares": 0,
            "Board lot": 0,
            "Price": 0.0,
            "Currency": "USD",
            "Value (USD)": plan.cash_value,
        }
    )
    return pd.DataFrame(rows)


def history_frame(runs: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Build the table of past runs.

    Args:
        runs: The run records read from the database.

    Returns:
        One row per run, most recent first.
    """
    rows = []
    for run in runs:
        weights = run.get("weights") or {}
        rows.append(
            {
                "Run": run.get("Run_ID"),
                "Created": run.get("Created_At"),
                "Tickers": ", ".join(run.get("tickers") or []),
                "Objective": run.get("objective_used"),
                "Expected return": run.get("Expected_Return"),
                "Volatility": run.get("Expected_Volatility"),
                "Sharpe": run.get("Sharpe_Ratio"),
                "Cash weight": run.get("cash_weight"),
                "Estimation": run.get("expected_return_method"),
                "VAR 95": run.get("var_95"),
                "CVaR 95": run.get("cvar_95"),
                "Top holding": (
                    max(weights, key=lambda ticker: weights[ticker]) if weights else ""
                ),
                "Top weight": max(weights.values()) if weights else None,
            }
        )
    return pd.DataFrame(rows)


def comparison_frame(comparison: pd.DataFrame) -> pd.DataFrame:
    """Format the estimator comparison for display.

    Args:
        comparison: The comparison table produced by the validation module.

    Returns:
        The same rows with readable column names.
    """
    if comparison.empty:
        return comparison
    return comparison.rename(
        columns={
            "method": "Method",
            "optimized_cagr": "CAGR",
            "optimized_sharpe": "Sharpe",
            "optimized_max_drawdown": "Max drawdown",
            "benchmark_cagr": "Benchmark CAGR",
            "benchmark_sharpe": "Benchmark Sharpe",
            "turnover": "Turnover",
        }
    )
