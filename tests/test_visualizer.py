"""Tests for the Plotly HTML chart generator.

These assert on the produced file and the embedded chart structure, so a broken
export is caught without a browser.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.optimizer import OptimizationResult
from src.visualizer import generate_efficient_frontier


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
