"""Streamlit front end for the local mean-variance portfolio optimizer.

Run it with::

    .\\.venv\\Scripts\\python.exe -m streamlit run app.py

The app is a thin layer over the optimizer: the sidebar collects the inputs,
``main.run_analysis`` does the work, and the tabs render the result. Every run
is also appended to the local SQLite database, so the history tab shows what
has been calculated before.

Nothing here reaches into the optimizer's internals. The app calls the same
pipeline the command line does, so the two cannot drift apart.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from main import DEFAULT_DB_PATH, AnalysisResult, run_analysis
from src.app_helpers import (
    comparison_frame,
    history_frame,
    parse_tickers,
    parse_weight_assignments,
    positions_frame,
    weights_frame,
)
from src.database import fetch_positions, fetch_runs
from src.filters import QualityRules
from src.fundamentals import DEFAULT_CACHE_PATH
from src.market import DEFAULT_USD_HKD, MarketMetadata
from src.report import render_report
from src.visualizer import build_frontier_figure, build_risk_figure

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")

DEFAULT_TICKER_TEXT = "AAPL MSFT GOOGL AMZN NVDA"
DEFAULT_START = date(2020, 1, 1)


@st.cache_data(ttl=3600, show_spinner="Running the optimization...")
def _run_cached(params: dict) -> AnalysisResult:
    """Run the optimizer with the parameters collected in the sidebar.

    Only plain values are passed in, so the cache key describes the run
    completely. The objects the pipeline needs are built here, inside the
    cached call.
    """
    call = dict(params)
    call["quality_rules"] = QualityRules(
        min_analyst_count=call.pop("min_analyst_count"),
        min_market_cap_usd=call.pop("min_market_cap_usd"),
        min_market_cap_hkd=call.pop("min_market_cap_hkd"),
        max_annual_volatility=call.pop("filter_max_volatility"),
        min_avg_volume=call.pop("min_avg_volume"),
    )
    market_meta = call.pop("market_meta")
    call["market_metadata"] = (
        MarketMetadata.from_csv(market_meta) if market_meta else MarketMetadata()
    )
    return run_analysis(**call)


def _sidebar() -> dict:
    """Read every input from the sidebar and return the run parameters."""
    with st.sidebar:
        st.header("Universe")
        ticker_text = st.text_area(
            "Tickers",
            DEFAULT_TICKER_TEXT,
            help="Separate by commas, spaces or newlines. Hong Kong listings use "
            "the Yahoo form, for example 0700.HK.",
        )
        date_range = st.date_input(
            "Date range",
            value=(DEFAULT_START, date.today()),
            max_value=date.today(),
        )
        market_index = st.text_input("Market index", "^GSPC")

        st.header("Objective")
        objective = st.selectbox("Objective", ["max_sharpe", "min_vol"])
        risk_free_rate = st.number_input(
            "Risk-free rate", 0.0, 0.20, 0.04, 0.005, format="%.3f"
        )
        use_target = st.checkbox("Reach a target return")
        target_return = (
            st.number_input("Target return", 0.0, 1.0, 0.12, 0.005, format="%.3f")
            if use_target
            else None
        )

        st.header("Estimation")
        method = st.selectbox(
            "Expected returns",
            ["auto", "historical_mean", "capm", "black_litterman"],
            help="auto uses Black-Litterman when market capitalizations are "
            "available and the historical mean otherwise.",
        )
        covariance_method = st.selectbox(
            "Covariance", ["ledoit-wolf", "oas", "sample"]
        )
        market_risk_premium = st.number_input(
            "Market risk premium", 0.0, 0.20, 0.05, 0.005, format="%.3f"
        )
        tau = st.number_input("Prior uncertainty (tau)", 0.005, 0.5, 0.05, 0.005, format="%.3f")
        use_analyst_views = st.checkbox(
            "Use analyst targets as views",
            help="Off by default: consensus targets are optimistic and "
            "momentum-driven, and measured out of sample they did not help.",
        )
        analyst_confidence = st.slider("Analyst confidence", 0.0, 1.0, 0.15, 0.05)
        refresh_fundamentals = st.checkbox("Refresh fundamentals cache")

        st.header("Constraints")
        fixed_text = st.text_area(
            "Mandatory holdings",
            "",
            help="One ticker per line. Each one gets a minimum weight and is "
            "always part of the portfolio.",
        )
        min_weight_text = st.text_area(
            "Minimum weights",
            "",
            help="TICKER=WEIGHT, separated by commas. A mandatory holding "
            "without an entry here gets 5%.",
        )
        max_weight = st.number_input(
            "Maximum weight per asset", 0.05, 1.0, 0.35, 0.05, format="%.2f"
        )
        use_max_volatility = st.checkbox("Cap the portfolio volatility")
        max_volatility = (
            st.number_input("Maximum volatility", 0.01, 1.0, 0.20, 0.01, format="%.2f")
            if use_max_volatility
            else None
        )
        use_max_k = st.checkbox("Choose how many assets to hold")
        max_k = st.number_input("Maximum number of assets", 1, 30, 6, 1) if use_max_k else 0
        selection_beam = st.number_input("Selection search width", 1, 20, 5, 1)

        st.header("Risk and sizing")
        use_budget = st.checkbox("Size a portfolio")
        budget = st.number_input("Budget (USD)", 1_000.0, 1e9, 100_000.0, 10_000.0)
        simulations = st.number_input("Simulated paths", 1_000, 200_000, 10_000, 1_000)
        risk_distribution = st.selectbox("Return distribution", ["student_t", "bootstrap"])
        student_t_df = st.number_input("Student-t degrees of freedom", 2.1, 100.0, 5.0, 0.1)
        fee_bps = st.number_input("Trading cost (bps)", 0.0, 100.0, 5.0, 1.0)

        st.header("Filters and validation")
        use_quality_filters = st.checkbox("Apply quality filters")
        min_analyst_count = st.number_input("Minimum analyst coverage", 0, 100, 10, 1)
        min_market_cap_usd = st.number_input(
            "Minimum market cap (USD, billions)", 0.0, 1000.0, 10.0, 1.0
        )
        min_market_cap_hkd = st.number_input(
            "Minimum market cap (HKD, billions)", 0.0, 2000.0, 80.0, 1.0
        )
        filter_max_volatility = st.number_input(
            "Maximum volatility of a candidate", 0.05, 3.0, 0.50, 0.05, format="%.2f"
        )
        min_avg_volume = st.number_input(
            "Minimum average volume (millions)", 0.0, 1000.0, 5.0, 0.5
        )
        validate = st.checkbox("Validate out of sample", help="Runs a walk-forward test. Slow on long histories.")
        validate_rebalance = st.selectbox("Rebalance frequency", ["monthly", "quarterly"])
        validate_window = st.number_input("Validation window (days)", 60, 1000, 252, 10)
        compare_estimators = st.checkbox(
            "Compare estimators", help="Walks every estimator forward. Slow."
        )
        portfolios = st.number_input("Monte Carlo portfolios", 500, 50_000, 5_000, 500)

    start, end = date_range if len(date_range) == 2 else (date_range[0], date_range[0])
    return {
        "tickers": parse_tickers(ticker_text),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "objective": objective,
        "risk_free_rate": risk_free_rate,
        "db_path": DEFAULT_DB_PATH,
        "output_dir": None,
        "n_portfolios": portfolios,
        "expected_return_method": method,
        "covariance_method": covariance_method,
        "market_index": market_index,
        "market_risk_premium": market_risk_premium,
        "tau": tau,
        "fundamentals_cache": DEFAULT_CACHE_PATH,
        "refresh_fundamentals": refresh_fundamentals,
        "market_meta": None,
        "usd_hkd": DEFAULT_USD_HKD,
        "min_weights": parse_weight_assignments(min_weight_text),
        "max_weight": max_weight,
        "max_volatility": max_volatility,
        "fixed_tickers": parse_tickers(fixed_text),
        "max_k": max_k,
        "selection_beam": selection_beam,
        "quality_filters": use_quality_filters,
        "min_analyst_count": min_analyst_count,
        "min_market_cap_usd": min_market_cap_usd * 1e9,
        "min_market_cap_hkd": min_market_cap_hkd * 1e9,
        "filter_max_volatility": filter_max_volatility,
        "min_avg_volume": min_avg_volume * 1e6,
        "analyst_views": use_analyst_views,
        "analyst_confidence": analyst_confidence,
        "target_return": target_return,
        "budget": budget if use_budget else None,
        "simulations": simulations,
        "risk_distribution": risk_distribution,
        "student_t_df": student_t_df,
        "fee_bps": fee_bps,
        "validate": validate,
        "validate_window": validate_window,
        "validate_rebalance": validate_rebalance,
        "compare_estimators": compare_estimators,
    }


def _render_portfolio(result: AnalysisResult) -> None:
    """Render the portfolio tab."""
    headline = result.headline
    allocation = result.allocation
    cash = allocation.cash_weight if allocation else 0.0

    columns = st.columns(4)
    columns[0].metric("Expected return", f"{headline.expected_return:.2%}")
    columns[1].metric("Volatility", f"{headline.volatility:.2%}")
    columns[2].metric("Sharpe ratio", f"{headline.sharpe_ratio:.3f}")
    columns[3].metric("Cash", f"{cash:.1%}")

    st.subheader("Weights")
    st.dataframe(
        weights_frame(headline.weights, cash),
        hide_index=True,
        width="stretch",
        column_config={
            "Target weight": st.column_config.NumberColumn(format="percent"),
        },
    )

    if result.selection is not None:
        selection = result.selection
        st.caption(
            f"Selected {len(selection.tickers)} assets. "
            f"Mandatory: {', '.join(selection.fixed_tickers) or 'none'}."
        )

    st.subheader("Efficient frontier")
    figure = build_frontier_figure(
        result.cloud,
        result.max_sharpe,
        result.min_vol,
        result.expected_returns,
        result.covariance,
        allocation=result.allocation,
    )
    st.plotly_chart(figure, width="stretch")

    st.download_button(
        "Download the report (HTML)",
        data=render_report(result),
        file_name="portfolio_report.html",
        mime="text/html",
        help="A single self-contained file with the numbers, the charts and the "
        "positions. It opens without a network connection.",
    )


def _render_risk(result: AnalysisResult) -> None:
    """Render the risk tab."""
    if result.risk is None:
        st.info("Set a budget in the sidebar to simulate the risk of the portfolio.")
        return

    metrics = result.risk.metrics
    columns = st.columns(4)
    columns[0].metric("Value at Risk (95%)", f"${metrics.var_95:,.0f}")
    columns[1].metric("Expected shortfall (95%)", f"${metrics.cvar_95:,.0f}")
    columns[2].metric("Probability of profit", f"{metrics.prob_profit:.0%}")
    columns[3].metric("Median value", f"${metrics.median_value:,.0f}")

    st.plotly_chart(build_risk_figure(result.risk), width="stretch")
    st.caption(
        f"Simulated {metrics.simulations:,} paths over one year with "
        f"{result.risk_distribution} shocks. The 5th percentile is "
        f"${metrics.p05:,.0f} and the 95th is ${metrics.p95:,.0f}."
    )


def _render_positions(result: AnalysisResult) -> None:
    """Render the positions tab."""
    if result.positions is None:
        st.info("Set a budget in the sidebar to size the portfolio into shares.")
        return

    plan = result.positions
    st.dataframe(
        positions_frame(plan),
        hide_index=True,
        width="stretch",
        column_config={
            "Target weight": st.column_config.NumberColumn(format="percent"),
            "Realized weight": st.column_config.NumberColumn(format="percent"),
            "Drift": st.column_config.NumberColumn(format="%+.2f%%"),
            "Price": st.column_config.NumberColumn(format="$%.2f"),
            "Value (USD)": st.column_config.NumberColumn(format="$%.2f"),
        },
    )
    columns = st.columns(3)
    columns[0].metric("Invested", f"${plan.invested_value:,.0f}")
    columns[1].metric("Cash", f"${plan.cash_value:,.0f}")
    columns[2].metric("Estimated entry cost", f"${plan.estimated_cost:,.0f}")

    for warning in plan.warnings:
        st.warning(warning)
    for ticker in plan.skipped:
        st.warning(f"{ticker}: no price available, so the position was skipped.")


def _render_validation(result: AnalysisResult) -> None:
    """Render the validation tab."""
    if result.validation is None and result.comparison is None:
        st.info(
            "Turn on **Validate out of sample** or **Compare estimators** in the "
            "sidebar to see how the portfolio would have performed."
        )
        return

    if result.validation is not None:
        validation = result.validation
        st.subheader("Walk-forward test")
        st.write(
            f"{validation.rebalance_count} rebalances, "
            f"{validation.period_starts[0]} to {validation.period_ends[-1]}."
        )
        rows = []
        for label, metrics in (
            ("Optimized", validation.optimized),
            ("Equal weight", validation.benchmark),
        ):
            rows.append(
                {
                    "Portfolio": label,
                    "CAGR": metrics.cagr,
                    "Volatility": metrics.volatility,
                    "Sharpe": metrics.sharpe_ratio,
                    "Max drawdown": metrics.max_drawdown,
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "CAGR": st.column_config.NumberColumn(format="percent"),
                "Volatility": st.column_config.NumberColumn(format="percent"),
                "Max drawdown": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.caption(f"Average turnover per rebalance: {validation.average_turnover:.2%}")

    if result.comparison is not None:
        st.subheader("Estimator comparison")
        st.dataframe(
            comparison_frame(result.comparison),
            hide_index=True,
            width="stretch",
            column_config={
                "CAGR": st.column_config.NumberColumn(format="percent"),
                "Max drawdown": st.column_config.NumberColumn(format="percent"),
                "Benchmark CAGR": st.column_config.NumberColumn(format="percent"),
            },
        )
        st.caption(
            "The Black-Litterman row uses today's market capitalizations for every "
            "historical period, so its backtest is mildly optimistic."
        )


def _render_history() -> None:
    """Render the history tab."""
    runs = fetch_runs(DEFAULT_DB_PATH, limit=100)
    if not runs:
        st.info("No runs have been saved yet. Press **Run optimization** to make one.")
        return

    st.dataframe(
        history_frame(runs),
        hide_index=True,
        width="stretch",
    )

    run_ids = [int(run["Run_ID"]) for run in runs]
    chosen = st.selectbox("Show the positions of run", run_ids)
    stored = fetch_positions(DEFAULT_DB_PATH, chosen)
    if stored:
        st.dataframe(
            pd.DataFrame(stored),
            hide_index=True,
            width="stretch",
        )


def main() -> None:
    """Render the app."""
    st.title("Mean-Variance Portfolio Optimizer")
    st.caption(
        "Local Markowitz optimization with robust estimation, portfolio "
        "constraints, tail risk and tradable positions."
    )

    params = _sidebar()

    if st.button("Run optimization", type="primary"):
        if not params["tickers"]:
            st.error("Name at least one ticker.")
        else:
            try:
                st.session_state["result"] = _run_cached(params)
            except Exception as error:  # noqa: BLE001 - the app must not die
                st.session_state.pop("result", None)
                st.error(f"{type(error).__name__}: {error}")

    result = st.session_state.get("result")

    if result is not None:
        for warning in result.warnings:
            st.warning(warning)
        st.caption(
            f"Expected returns: {result.expected_return_method} | "
            f"Covariance: {result.covariance_method} | "
            f"Run #{result.db_run_id} saved to the local database."
        )

    portfolio_tab, risk_tab, positions_tab, validation_tab, history_tab = st.tabs(
        ["Portfolio", "Risk", "Positions", "Validation", "History"]
    )

    with portfolio_tab:
        if result is None:
            st.info("Press **Run optimization** to see the portfolio.")
        else:
            _render_portfolio(result)
    with risk_tab:
        if result is None:
            st.info("Press **Run optimization** to simulate the risk.")
        else:
            _render_risk(result)
    with positions_tab:
        if result is None:
            st.info("Press **Run optimization** to size the portfolio.")
        else:
            _render_positions(result)
    with validation_tab:
        if result is None:
            st.info("Press **Run optimization** to validate the portfolio.")
        else:
            _render_validation(result)
    with history_tab:
        _render_history()


main()
