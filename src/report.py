"""A self-contained HTML report for one portfolio run.

The report is a single file with everything inlined: the styles, the charts
and the numbers. It opens by double-click, works with no network, and can be
attached to an email or dropped into a folder without dragging anything else
along.

The design is deliberately plain: a header with the run's provenance, a strip
of metric cards, the charts, and the tables. Numbers are aligned and
tabular, colours mean something (green for a gain, red for a loss, amber for
a warning), and the page reads in both light and dark mode.
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

from src.visualizer import build_frontier_figure, build_risk_figure

if TYPE_CHECKING:
    from main import AnalysisResult

_STYLE = """
:root {
  color-scheme: light dark;
  --bg: #f4f5f7;
  --surface: #ffffff;
  --surface-2: #f9fafb;
  --border: #e4e6ea;
  --text: #16181d;
  --muted: #6b7280;
  --accent: #2563eb;
  --accent-soft: #eff4ff;
  --positive: #047857;
  --negative: #b42318;
  --warning: #b54708;
  --shadow: 0 1px 2px rgba(16, 24, 40, .05), 0 8px 24px rgba(16, 24, 40, .05);
  --radius: 14px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0e1014;
    --surface: #16191f;
    --surface-2: #1c2027;
    --border: #272c35;
    --text: #e8eaed;
    --muted: #9aa3af;
    --accent: #6f9bff;
    --accent-soft: #1b2333;
    --positive: #4ade80;
    --negative: #f87171;
    --warning: #fbbf24;
    --shadow: 0 1px 2px rgba(0, 0, 0, .4), 0 8px 24px rgba(0, 0, 0, .3);
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 0 0 64px;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 0 24px; }
header.top {
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  margin-bottom: 28px;
}
header.top .wrap { padding-top: 30px; padding-bottom: 26px; }
h1 { font-size: 27px; line-height: 1.2; margin: 0 0 6px; letter-spacing: -.02em; }
.sub { color: var(--muted); font-size: 14px; margin: 0; }
.badges { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
.badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--accent-soft); color: var(--accent);
  border: 1px solid transparent; border-radius: 999px;
  padding: 4px 11px; font-size: 12.5px; font-weight: 600; white-space: nowrap;
}
.badge.grey { background: var(--surface-2); color: var(--muted); }
h2 {
  font-size: 13px; text-transform: uppercase; letter-spacing: .09em;
  color: var(--muted); margin: 34px 0 12px; font-weight: 700;
}
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); gap: 14px; }
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 16px 18px; box-shadow: var(--shadow);
}
.card .label { color: var(--muted); font-size: 12.5px; margin-bottom: 7px; }
.card .value { font-size: 25px; font-weight: 650; letter-spacing: -.02em; font-variant-numeric: tabular-nums; }
.card .sub { font-size: 12.5px; margin-top: 5px; }
.panel {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 6px 10px 10px; box-shadow: var(--shadow);
  margin-bottom: 18px; overflow: hidden;
}
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
@media (max-width: 900px) { .grid2 { grid-template-columns: 1fr; } }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { padding: 10px 12px; text-align: right; border-bottom: 1px solid var(--border); }
th:first-child, td:first-child { text-align: left; }
th {
  color: var(--muted); font-size: 12px; font-weight: 650;
  text-transform: uppercase; letter-spacing: .05em; white-space: nowrap;
}
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: var(--surface-2); }
td.pos { color: var(--positive); }
td.neg { color: var(--negative); }
.bar { position: relative; height: 6px; border-radius: 999px; background: var(--surface-2); min-width: 90px; }
.bar > span { position: absolute; inset: 0 auto 0 0; border-radius: 999px; background: var(--accent); }
.callout {
  border: 1px solid var(--border); border-left: 3px solid var(--warning);
  background: var(--surface); border-radius: 10px; padding: 11px 14px;
  margin-bottom: 10px; font-size: 14px; color: var(--muted);
}
.callout strong { color: var(--text); }
footer {
  margin-top: 44px; padding-top: 20px; border-top: 1px solid var(--border);
  color: var(--muted); font-size: 12.5px;
}
footer p { margin: 0 0 6px; }
"""


_PLOTLY_BUNDLE: str | None = None


def _plotly_bundle() -> str:
    """Return the inlined Plotly bundle, fetched from the package once.

    Asking Plotly to inline the bundle itself re-encodes several megabytes on
    every call, which is slow enough to be noticeable. The bundle is read
    from the installed package instead and inserted by hand.
    """
    global _PLOTLY_BUNDLE
    if _PLOTLY_BUNDLE is None:
        from plotly.offline import get_plotlyjs

        _PLOTLY_BUNDLE = get_plotlyjs()
    return _PLOTLY_BUNDLE


def _escape(value: object) -> str:
    """Escape a value for inclusion in the HTML."""
    return html.escape(str(value))


def _percent(value: float | None, digits: int = 2) -> str:
    """Format a number as a percentage."""
    if value is None:
        return "&mdash;"
    return f"{value * 100:.{digits}f}%"


def _money(value: float | None) -> str:
    """Format a number as an amount."""
    if value is None:
        return "&mdash;"
    return f"${value:,.0f}"


def _number(value: float | None, digits: int = 2) -> str:
    """Format a plain number."""
    if value is None:
        return "&mdash;"
    return f"{value:,.{digits}f}"


def _card(label: str, value: str, sub: str = "", tone: str = "") -> str:
    """Build one metric card."""
    classes = f"card {tone}".strip()
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="{classes}"><div class="label">{_escape(label)}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>'
    )


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a table from headers and pre-rendered cells."""
    head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _weights_table(weights: pd.Series, cash_weight: float) -> str:
    """Build the target-weight table with a bar for each position."""
    rows = []
    items = [(ticker, float(weight)) for ticker, weight in weights.items()]
    if cash_weight:
        items.append(("CASH (risk-free)", cash_weight))
    largest = max((weight for _, weight in items), default=1.0) or 1.0
    for ticker, weight in sorted(items, key=lambda item: item[1], reverse=True):
        share = max(0.0, min(1.0, weight / largest))
        rows.append(
            [
                _escape(ticker),
                _percent(weight),
                f'<div class="bar"><span style="width:{share * 100:.1f}%"></span></div>',
            ]
        )
    return _table(["Ticker", "Target weight", ""], rows)


def _positions_table(plan) -> str:
    """Build the table of tradable positions."""
    rows = []
    for position in plan.positions:
        drift = position.drift
        tone = "pos" if drift > 1e-6 else "neg" if drift < -1e-6 else ""
        rows.append(
            [
                _escape(position.ticker),
                _percent(position.target_weight),
                _percent(position.realized_weight),
                f'<span class="{tone}">{_percent(drift)}</span>',
                f"{position.shares:,}",
                f"{position.board_lot:,}",
                f"{position.price:,.2f}",
                _escape(position.currency),
                f"${position.value_usd:,.2f}",
            ]
        )
    rows.append(
        [
            "<strong>CASH</strong>",
            "&mdash;",
            "&mdash;",
            "&mdash;",
            "&mdash;",
            "&mdash;",
            "&mdash;",
            "USD",
            f"${plan.cash_value:,.2f}",
        ]
    )
    return _table(
        [
            "Ticker",
            "Target",
            "Realized",
            "Drift",
            "Shares",
            "Lot",
            "Price",
            "Currency",
            "Value (USD)",
        ],
        rows,
    )


def _validation_table(validation) -> str:
    """Build the out-of-sample comparison table."""
    rows = []
    for label, metrics in (
        ("Optimized", validation.optimized),
        ("Equal weight", validation.benchmark),
    ):
        rows.append(
            [
                _escape(label),
                _percent(metrics.cagr),
                _percent(metrics.volatility),
                f"{metrics.sharpe_ratio:.3f}",
                _percent(metrics.max_drawdown),
            ]
        )
    return _table(
        ["Portfolio", "CAGR", "Volatility", "Sharpe", "Max drawdown"], rows
    )


def _comparison_table(comparison: pd.DataFrame) -> str:
    """Build the estimator comparison table."""
    rows = []
    for _, row in comparison.iterrows():
        rows.append(
            [
                _escape(row["method"]),
                _percent(row["optimized_cagr"]),
                f"{row['optimized_sharpe']:.3f}",
                _percent(row["optimized_max_drawdown"]),
                _percent(row["benchmark_cagr"]),
                f"{row['benchmark_sharpe']:.3f}",
                _percent(row["turnover"]),
            ]
        )
    return _table(
        [
            "Estimator",
            "CAGR",
            "Sharpe",
            "Max drawdown",
            "Benchmark CAGR",
            "Benchmark Sharpe",
            "Turnover",
        ],
        rows,
    )


def _restyle(figure, height: int = 460) -> object:
    """Make a Plotly figure blend into the page.

    The page follows the reader's colour scheme, but a chart cannot react to
    it without scripting, so the chart uses a neutral middle tone for its
    text, gridlines and asset markers: readable on a light page and on a
    dark one.
    """
    figure.update_layout(
        height=height,
        title=None,
        margin=dict(l=58, r=18, t=26, b=48),
        font=dict(
            family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            size=13,
            color="#7b8494",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(123,132,148,0.22)", zerolinecolor="rgba(123,132,148,0.3)"),
        yaxis=dict(gridcolor="rgba(123,132,148,0.22)", zerolinecolor="rgba(123,132,148,0.3)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        hoverlabel=dict(font_size=13),
    )
    for trace in figure.data:
        if trace.name == "Individual Assets":
            trace.marker.color = "#7b8494"
    return figure


def render_report(result: "AnalysisResult", title: str = "Portfolio Report") -> str:
    """Render a run as a self-contained HTML document.

    Args:
        result: The completed analysis to report on.
        title: The heading of the report.

    Returns:
        The HTML document, with the styles and the charts inlined.
    """
    headline = result.headline
    allocation = result.allocation
    cash_weight = allocation.cash_weight if allocation else 0.0

    badges = [
        f'<span class="badge">{_escape(result.expected_return_method)}</span>',
        f'<span class="badge">{_escape(result.covariance_method)}</span>',
        f'<span class="badge grey">{_escape(headline.objective)}</span>',
    ]
    if result.selection is not None:
        badges.append(
            f'<span class="badge grey">{len(result.selection.tickers)} assets selected</span>'
        )
    if result.validation is not None:
        badges.append(
            f'<span class="badge grey">{result.validation.rebalance_count} rebalances tested</span>'
        )

    cards = [
        _card("Expected return", _percent(headline.expected_return)),
        _card("Volatility", _percent(headline.volatility)),
        _card("Sharpe ratio", f"{headline.sharpe_ratio:.3f}"),
        _card("Cash", _percent(cash_weight)),
    ]
    if result.risk is not None:
        metrics = result.risk.metrics
        cards.append(
            _card(
                "Value at Risk (95%)",
                _money(metrics.var_95),
                f"Expected shortfall {_money(metrics.cvar_95)}",
            )
        )
        cards.append(
            _card(
                "Probability of profit",
                f"{metrics.prob_profit:.0%}",
                f"Median value {_money(metrics.median_value)}",
            )
        )

    callouts = "".join(
        f'<div class="callout"><strong>{_escape(ticker)}</strong> '
        f"{_escape(reason)}</div>"
        for ticker, reason in getattr(result, "dropped", [])
    ) or ""
    callouts += "".join(
        f'<div class="callout">{_escape(warning)}</div>' for warning in result.warnings
    )
    if result.positions is not None:
        callouts += "".join(
            f'<div class="callout">{_escape(warning)}</div>'
            for warning in result.positions.warnings
        )

    sections: list[str] = []

    frontier = _restyle(
        build_frontier_figure(
            result.cloud,
            result.max_sharpe,
            result.min_vol,
            result.expected_returns,
            result.covariance,
            allocation=result.allocation,
        )
    )
    sections.append(
        '<h2>Efficient frontier</h2><div class="panel">'
        + frontier.to_html(full_html=False, include_plotlyjs=False)
        + "</div>"
    )

    if result.risk is not None:
        risk_figure = _restyle(build_risk_figure(result.risk), height=380)
        sections.append(
            '<h2>Risk over one year</h2><div class="panel">'
            + risk_figure.to_html(full_html=False, include_plotlyjs=False)
            + "</div>"
        )

    sections.append(
        "<h2>Target weights</h2><div class=\"panel\">"
        + _weights_table(headline.weights, cash_weight)
        + "</div>"
    )

    if result.positions is not None:
        sections.append(
            '<h2>Positions</h2><div class="panel">'
            + _positions_table(result.positions)
            + "</div>"
        )
        sections.append(
            '<div class="cards">'
            + _card("Invested", _money(result.positions.invested_value))
            + _card("Cash", _money(result.positions.cash_value))
            + _card("Estimated entry cost", _money(result.positions.estimated_cost))
            + "</div>"
        )

    if result.validation is not None:
        sections.append(
            "<h2>Out of sample</h2><div class=\"panel\">"
            + _validation_table(result.validation)
            + "</div>"
        )

    if result.comparison is not None and not result.comparison.empty:
        sections.append(
            "<h2>Estimator comparison</h2><div class=\"panel\">"
            + _comparison_table(result.comparison)
            + "</div>"
        )

    stamp = datetime.now().strftime("%d %B %Y, %H:%M")
    universe = ", ".join(result.expected_returns.index)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>{_STYLE}</style>
</head>
<body>
<header class="top">
  <div class="wrap">
    <h1>{_escape(title)}</h1>
    <p class="sub">{_escape(universe)} &middot; generated {stamp}</p>
    <div class="badges">{"".join(badges)}</div>
  </div>
</header>
<main class="wrap">
  <div class="cards">{"".join(cards)}</div>
  {callouts}
  <script>{_plotly_bundle()}</script>
  {"".join(sections)}
  <footer>
    <p>Generated locally by the mean-variance portfolio optimizer. No data left this machine.</p>
    <p>For research and education only. Past performance does not guarantee future results, and nothing here is investment advice.</p>
  </footer>
</main>
</body>
</html>"""


def write_report(path: str | "Path", result: "AnalysisResult", title: str = "Portfolio Report") -> "Path":
    """Write a run's report to a file.

    Args:
        path: Where to write the report.
        result: The completed analysis to report on.
        title: The heading of the report.

    Returns:
        The path written to.
    """
    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_report(result, title), encoding="utf-8")
    return destination
