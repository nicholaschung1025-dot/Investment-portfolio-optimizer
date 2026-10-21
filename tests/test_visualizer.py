"""Tests for the Plotly HTML chart generator.

These assert on the produced file and the embedded chart structure, so a broken
export is caught without a browser.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.allocation import AllocationResult
from src.optimizer import OptimizationResult
from src.risk import RiskSimulation
from src.visualizer import build_frontier_figure, build_risk_figure, generate_efficient_frontier


def _inputs():
    expected_returns = pd.Series({"AAA": 0.12, "BBB": 0.08, "CCC": 0.05})
    covariance = pd.DataFrame(
        [[0.04, 0.006, 0.002], [0.006, 0.0225, 0.003], [0.002, 0.003, 0.01]],
        index=["AAA", "BBB", "CCC"],
        columns=["AAA", "BBB", "CCC"],
    )
    cloud = pd.DataFrame(
        {
            "AAA": [0.5, 0.2],
            "BBB": [0.3, 0.5],
            "CCC": [0.2, 0.3],
            "return": [0.09, 0.07],
            "volatility": [0.15, 0.12],
            "sharpe": [0.33, 0.25],
        }
    )
    max_sharpe = OptimizationResult(
        weights=pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}),
        expected_return=0.09,
        volatility=0.15,
        sharpe_ratio=0.33,
        objective="max_sharpe",
    )
    min_vol = OptimizationResult(
        weights=pd.Series({"AAA": 0.2, "BBB": 0.5, "CCC": 0.3}),
        expected_return=0.07,
        volatility=0.12,
        sharpe_ratio=0.25,
        objective="min_vol",
    )
    return expected_returns, covariance, cloud, max_sharpe, min_vol


def test_generate_writes_self_contained_html(tmp_path) -> None:
    expected_returns, covariance, cloud, max_sharpe, min_vol = _inputs()
    output_path = tmp_path / "frontier.html"

    written = generate_efficient_frontier(
        output_path, cloud, max_sharpe, min_vol, expected_returns, covariance
    )

    assert written == output_path
    html = output_path.read_text(encoding="utf-8")
    assert "Efficient Frontier" in html
    assert "plotly" in html.lower()
    assert "Max Sharpe" in html
    assert "Min Volatility" in html


def test_generate_includes_one_trace_per_asset_plus_cloud_and_markers(tmp_path) -> None:
    expected_returns, covariance, cloud, max_sharpe, min_vol = _inputs()
    output_path = tmp_path / "frontier.html"

    generate_efficient_frontier(
        output_path, cloud, max_sharpe, min_vol, expected_returns, covariance
    )

    html = output_path.read_text(encoding="utf-8")
    for ticker in expected_returns.index:
        assert ticker in html


def test_generate_creates_parent_directory(tmp_path) -> None:
    expected_returns, covariance, cloud, max_sharpe, min_vol = _inputs()
    output_path = tmp_path / "nested" / "deeper" / "frontier.html"

    generate_efficient_frontier(
        output_path, cloud, max_sharpe, min_vol, expected_returns, covariance
    )

    assert output_path.exists()


def test_build_frontier_figure_returns_a_figure_with_the_expected_traces() -> None:
    expected_returns, covariance, cloud, max_sharpe, min_vol = _inputs()

    figure = build_frontier_figure(cloud, max_sharpe, min_vol, expected_returns, covariance)

    names = {trace.name for trace in figure.data}
    assert "Random Portfolios" in names
    assert "Individual Assets" in names
    assert "Max Sharpe" in names
    assert "Min Volatility" in names


def test_build_risk_figure_returns_a_figure_with_the_tail_markers() -> None:
    risk = RiskSimulation(
        metrics=SimpleNamespace(
            var_95=10_000.0,
            cvar_95=15_000.0,
            prob_profit=0.7,
            median_value=100_000.0,
            p05=90_000.0,
            p95=112_000.0,
            simulations=500,
        ),
        terminal_values=np.linspace(80_000.0, 120_000.0, 500),
    )

    figure = build_risk_figure(risk)

    names = {trace.name for trace in figure.data}
    assert "Terminal values" in names
    assert "Value at Risk (95%)" in names


def test_the_two_panels_of_the_combined_figure_use_different_axes() -> None:
    expected_returns, covariance, cloud, max_sharpe, min_vol = _inputs()
    risk = RiskSimulation(
        metrics=SimpleNamespace(
            var_95=10_000.0,
            cvar_95=15_000.0,
            prob_profit=0.7,
            median_value=100_000.0,
            p05=90_000.0,
            p95=112_000.0,
            simulations=500,
        ),
        terminal_values=np.linspace(80_000.0, 120_000.0, 500),
    )

    figure = build_frontier_figure(
        cloud, max_sharpe, min_vol, expected_returns, covariance, risk=risk
    )

    panels: dict[str, set[str]] = {}
    for trace in figure.data:
        panels.setdefault(trace.name, set()).add(trace.xaxis)
    # The frontier and the risk distribution must not be drawn on the same
    # axes, or the two panels overlap on top of each other.
    assert panels["Max Sharpe"] == {"x"}
    assert panels["Terminal values"] == {"x2"}


def test_generate_draws_the_capital_allocation_line(tmp_path) -> None:
    expected_returns, covariance, cloud, max_sharpe, min_vol = _inputs()
    output_path = tmp_path / "frontier.html"
    allocation = AllocationResult(
        risky_weights=pd.Series({"AAA": 0.25, "BBB": 0.15, "CCC": 0.10}),
        cash_weight=0.5,
        expected_return=0.065,
        volatility=0.075,
        sharpe_ratio=0.33,
        target_return=0.065,
    )

    generate_efficient_frontier(
        output_path,
        cloud,
        max_sharpe,
        min_vol,
        expected_returns,
        covariance,
        allocation=allocation,
    )

    html = output_path.read_text(encoding="utf-8")
    assert "Capital Allocation Line" in html


def test_generate_draws_the_risk_distribution(tmp_path) -> None:
    expected_returns, covariance, cloud, max_sharpe, min_vol = _inputs()
    output_path = tmp_path / "frontier.html"
    terminal_values = np.linspace(80_000.0, 120_000.0, 500)
    risk = RiskSimulation(
        metrics=SimpleNamespace(
            var_95=10_000.0,
            cvar_95=15_000.0,
            prob_profit=0.7,
            median_value=100_000.0,
            p05=90_000.0,
            p95=112_000.0,
            simulations=500,
        ),
        terminal_values=terminal_values,
    )

    generate_efficient_frontier(
        output_path,
        cloud,
        max_sharpe,
        min_vol,
        expected_returns,
        covariance,
        risk=risk,
    )

    html = output_path.read_text(encoding="utf-8")
    assert "Terminal value" in html
    assert "Value at Risk" in html
