"""Plotly-based chart generation for the efficient frontier.

The Plotly JavaScript bundle is inlined into the output so the HTML file is
fully self-contained: the user can double-click it and open it in a native
Windows browser with no web server and no internet connection.

The chart carries three things when they are available: the efficient
frontier itself, the capital allocation line when the portfolio is scaled
against cash, and the distribution of terminal values when the portfolio has
been simulated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.allocation import AllocationResult
from src.optimizer import OptimizationResult
from src.risk import RiskSimulation


def _add_frontier(
    figure: go.Figure,
    cloud: pd.DataFrame,
    max_sharpe: OptimizationResult,
    min_vol: OptimizationResult,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
) -> None:
    """Draw the portfolio cloud, the assets and the two optimal portfolios."""
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


def _add_capital_allocation_line(
    figure: go.Figure,
    max_sharpe: OptimizationResult,
    allocation: AllocationResult,
    risk_free_rate: float,
) -> None:
    """Draw the capital allocation line and the portfolio that sits on it."""
    if max_sharpe.volatility <= 0.0:
        return

    slope = (max_sharpe.expected_return - risk_free_rate) / max_sharpe.volatility
    end_volatility = max(max_sharpe.volatility * 2.0, allocation.volatility * 1.2)
    volatilities = np.linspace(0.0, end_volatility, 50)
    returns = risk_free_rate + slope * volatilities

    figure.add_trace(
        go.Scatter(
            x=volatilities,
            y=returns,
            mode="lines",
            name="Capital Allocation Line",
            line=dict(color="grey", dash="dash", width=2),
            hovertemplate="Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[allocation.volatility],
            y=[allocation.expected_return],
            mode="markers",
            name="Target Return",
            marker=dict(size=16, color="orange", symbol="diamond"),
            hovertemplate=(
                "Target Return<br>Volatility: %{x:.2%}<br>Return: %{y:.2%}"
                f"<br>Cash: {allocation.cash_weight:.2%}<extra></extra>"
            ),
        )
    )


def _add_risk_distribution(figure: go.Figure, risk: RiskSimulation) -> None:
    """Draw the distribution of terminal values with its tail markers."""
    values = np.asarray(risk.terminal_values, dtype=float)
    metrics = risk.metrics
    budget = metrics.p05 + metrics.var_95
    cvar_level = budget - metrics.cvar_95

    figure.add_trace(
        go.Histogram(
            x=values,
            name="Terminal values",
            marker=dict(color="steelblue", opacity=0.75),
            nbinsx=60,
            hovertemplate="Terminal value: $%{x:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[budget, budget],
            y=[0, 1],
            mode="lines",
            name="Starting budget",
            line=dict(color="black", width=2),
            hovertemplate="Starting budget: $%{x:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[metrics.p05, metrics.p05],
            y=[0, 1],
            mode="lines",
            name="Value at Risk (95%)",
            line=dict(color="crimson", width=2, dash="dot"),
            hovertemplate="Value at Risk: $%{x:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[cvar_level, cvar_level],
            y=[0, 1],
            mode="lines",
            name="Expected Shortfall (95%)",
            line=dict(color="darkorange", width=2, dash="dot"),
            hovertemplate="Expected Shortfall: $%{x:,.0f}<extra></extra>",
        )
    )
    figure.update_yaxes(title_text="Paths", row=2, col=1)
    figure.update_xaxes(title_text="Terminal value (USD)", row=2, col=1)
    figure.update_xaxes(tickformat="$,.0f", row=2, col=1)


def generate_efficient_frontier(
    output_path: str | Path,
    cloud: pd.DataFrame,
    max_sharpe: OptimizationResult,
    min_vol: OptimizationResult,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    allocation: AllocationResult | None = None,
    risk: RiskSimulation | None = None,
    risk_free_rate: float = 0.04,
) -> Path:
    """Render the efficient frontier to a standalone interactive HTML file.

    Draws the Monte Carlo portfolio cloud (colored by Sharpe ratio), each asset
    as a labelled point, and the two optimized portfolios as highlighted stars.
    When a capital allocation is given, the capital allocation line and the
    resulting portfolio are added; when a risk simulation is given, the
    distribution of terminal values is added underneath.

    Args:
        output_path: Destination ``.html`` file.
        cloud: Monte Carlo portfolios with ``return``, ``volatility`` and
            ``sharpe`` columns.
        max_sharpe: The maximum-Sharpe portfolio.
        min_vol: The minimum-volatility portfolio.
        expected_returns: Annualized expected return per asset.
        covariance: Annualized covariance matrix.
        allocation: The cash allocation that reaches a target return.
        risk: The simulated distribution of terminal values.
        risk_free_rate: Annualized risk-free rate, used for the allocation line.

    Returns:
        The path written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if risk is not None:
        figure = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.68, 0.32],
            vertical_spacing=0.08,
            subplot_titles=("Efficient frontier", "Risk distribution"),
        )
    else:
        figure = go.Figure()

    row = 1
    _add_frontier(figure, cloud, max_sharpe, min_vol, expected_returns, covariance)

    if allocation is not None:
        _add_capital_allocation_line(
            figure, max_sharpe, allocation, risk_free_rate
        )

    if risk is not None:
        _add_risk_distribution(figure, risk)
        row = 2

    figure.update_layout(
        title="Efficient Frontier (Mean-Variance Optimization)",
        xaxis_title="Annualized Volatility (Risk)",
        yaxis_title="Annualized Expected Return",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if row == 1:
        figure.update_xaxes(tickformat=".1%")
        figure.update_yaxes(tickformat=".1%")
    else:
        figure.update_xaxes(tickformat=".1%", row=1, col=1)
        figure.update_yaxes(tickformat=".1%", row=1, col=1)

    figure.write_html(str(output_path), include_plotlyjs=True, full_html=True)
    return output_path
