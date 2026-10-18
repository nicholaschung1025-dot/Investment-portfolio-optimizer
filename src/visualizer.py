"""Plotly-based chart generation for the efficient frontier.

The Plotly JavaScript bundle is inlined into the output so the HTML file is
fully self-contained: the user can double-click it and open it in a native
Windows browser with no web server and no internet connection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from src.optimizer import OptimizationResult


def generate_efficient_frontier(
    output_path: str | Path,
    cloud: pd.DataFrame,
    max_sharpe: OptimizationResult,
    min_vol: OptimizationResult,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> Path:
    """Render the efficient frontier to a standalone interactive HTML file.

    Draws the Monte Carlo portfolio cloud (colored by Sharpe ratio), each asset
    as a labelled point, and the two optimized portfolios as highlighted stars.

    Args:
        output_path: Destination ``.html`` file.
        cloud: Monte Carlo portfolios with ``return``, ``volatility`` and
            ``sharpe`` columns.
        max_sharpe: The maximum-Sharpe portfolio.
        min_vol: The minimum-volatility portfolio.
        expected_returns: Annualized expected return per asset.
        covariance: Annualized covariance matrix.

    Returns:
        The path written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=cloud["volatility"],
            y=cloud["return"],
            mode="markers",
            name="Random Portfolios",
            marker=dict(
                size=4,
                color=cloud["sharpe"],
                colorscale="Viridis",
                colorbar=dict(title="Sharpe"),
                opacity=0.5,
            ),
            hovertemplate="Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    asset_vols = [float(covariance.loc[ticker, ticker]) ** 0.5 for ticker in expected_returns.index]
    figure.add_trace(
        go.Scatter(
            x=asset_vols,
            y=expected_returns.to_numpy(),
            mode="markers+text",
            name="Individual Assets",
            text=list(expected_returns.index),
            textposition="top center",
            marker=dict(size=11, color="black", symbol="circle"),
            hovertemplate="%{text}<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[max_sharpe.volatility],
            y=[max_sharpe.expected_return],
            mode="markers",
            name="Max Sharpe",
            marker=dict(size=18, color="red", symbol="star"),
            hovertemplate="Max Sharpe<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[min_vol.volatility],
            y=[min_vol.expected_return],
            mode="markers",
            name="Min Volatility",
            marker=dict(size=18, color="blue", symbol="star"),
            hovertemplate="Min Volatility<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    figure.update_layout(
        title="Efficient Frontier (Mean-Variance Optimization)",
        xaxis_title="Annualized Volatility (Risk)",
        yaxis_title="Annualized Expected Return",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    figure.update_xaxes(tickformat=".1%")
    figure.update_yaxes(tickformat=".1%")

    figure.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    return output_path
