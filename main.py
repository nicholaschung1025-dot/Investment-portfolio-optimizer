"""Command-line entry point for the local mean-variance portfolio optimizer.

Runs the full pipeline in order: fetch -> clean -> math -> optimize -> persist
-> visualize. Everything runs natively on Windows with no server, no cloud
service, and no API keys.

Example:
    python main.py --tickers AAPL MSFT GOOGL --start 2020-01-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.data_fetcher import Downloader, PreparedPrices, fetch_close_prices, prepare_prices
from src.database import RunRecord, save_run
from src.math_engine import (
    annualized_covariance,
    annualized_expected_returns,
    compute_log_returns,
)
from src.optimizer import (
    Objective,
    OptimizationResult,
    monte_carlo_frontier,
    optimize_portfolio,
)
from src.visualizer import generate_efficient_frontier

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
DEFAULT_START = "2020-01-01"
DEFAULT_RISK_FREE_RATE = 0.04
DEFAULT_DB_PATH = Path("data/portfolio_history.db")
DEFAULT_OUTPUT_DIR = Path("outputs")


@dataclass
class AnalysisResult:
    """Everything produced by one optimization run.

    Attributes:
        prepared: Cleaned prices and any cleaning warnings.
        expected_returns: Annualized expected return per asset.
        covariance: Annualized covariance matrix.
        max_sharpe: Maximum-Sharpe portfolio.
        min_vol: Minimum-volatility portfolio.
        headline: The portfolio matching the requested objective.
        cloud: Monte Carlo portfolio cloud used for the chart.
        db_run_id: ``Run_ID`` of the persisted row.
        html_path: Path of the generated HTML chart.
        warnings: Combined cleaning warnings.
    """

    prepared: PreparedPrices
    expected_returns: pd.Series
    covariance: pd.DataFrame
    max_sharpe: OptimizationResult
    min_vol: OptimizationResult
    headline: OptimizationResult
    cloud: pd.DataFrame
    db_run_id: int
    html_path: Path
    warnings: list[str] = field(default_factory=list)


def run_analysis(
    tickers: list[str],
    start: str,
    end: str,
    objective: Objective,
    risk_free_rate: float,
    db_path: str | Path,
    output_dir: str | Path,
    downloader: Downloader | None = None,
    n_portfolios: int = 5000,
    seed: int | None = 42,
) -> AnalysisResult:
    """Run the full mean-variance optimization pipeline.

    Args:
        tickers: Ticker symbols to optimize over.
        start: Inclusive start date (``YYYY-MM-DD``).
        end: Inclusive end date (``YYYY-MM-DD``).
        objective: ``"max_sharpe"`` or ``"min_vol"``; selects the headline result.
        risk_free_rate: Annualized risk-free rate.
        db_path: Path to the local SQLite file.
        output_dir: Directory for the generated HTML chart.
        downloader: Optional price downloader injected for tests.
        n_portfolios: Number of Monte Carlo portfolios to sample.
        seed: Seed for the Monte Carlo sampler.

    Returns:
        The completed :class:`AnalysisResult`.
    """
    raw_prices = fetch_close_prices(tickers, start, end, downloader)
    prepared = prepare_prices(raw_prices)

    log_returns = compute_log_returns(prepared.prices)
    expected_returns = annualized_expected_returns(log_returns)
    covariance = annualized_covariance(log_returns)

    max_sharpe = optimize_portfolio(expected_returns, covariance, "max_sharpe", risk_free_rate)
    min_vol = optimize_portfolio(expected_returns, covariance, "min_vol", risk_free_rate)
    headline = max_sharpe if objective == "max_sharpe" else min_vol

    cloud = monte_carlo_frontier(
        expected_returns,
        covariance,
        n_portfolios=n_portfolios,
        risk_free_rate=risk_free_rate,
        seed=seed,
    )

    db_run_id = save_run(
        db_path,
        RunRecord(
            tickers=list(expected_returns.index),
            start=start,
            end=end,
            objective=objective,
            weights={key: float(value) for key, value in headline.weights.items()},
            expected_return=headline.expected_return,
            expected_volatility=headline.volatility,
            sharpe_ratio=headline.sharpe_ratio,
            risk_free_rate=risk_free_rate,
        ),
    )

    html_path = generate_efficient_frontier(
        Path(output_dir) / f"efficient_frontier_{date.today():%Y%m%d}.html",
        cloud,
        max_sharpe,
        min_vol,
        expected_returns,
        covariance,
    )

    return AnalysisResult(
        prepared=prepared,
        expected_returns=expected_returns,
        covariance=covariance,
        max_sharpe=max_sharpe,
        min_vol=min_vol,
        headline=headline,
        cloud=cloud,
        db_run_id=db_run_id,
        html_path=html_path,
        warnings=list(prepared.warnings),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="mean-variance-optimizer",
        description=(
            "Local, deterministic mean-variance (Markowitz) portfolio optimizer. "
            "Fetches free Yahoo Finance data, then maximizes the Sharpe ratio or "
            "minimizes volatility, all offline on your machine."
        ),
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help=f"Ticker symbols to optimize (default: {' '.join(DEFAULT_TICKERS)})",
    )
    parser.add_argument("--start", default=DEFAULT_START, help=f"Start date (default: {DEFAULT_START})")
    parser.add_argument(
        "--end",
        default=date.today().isoformat(),
        help="End date (default: today)",
    )
    parser.add_argument(
        "--objective",
        choices=["max_sharpe", "min_vol"],
        default="max_sharpe",
        help="Optimization objective (default: max_sharpe)",
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=DEFAULT_RISK_FREE_RATE,
        help=f"Annualized risk-free rate (default: {DEFAULT_RISK_FREE_RATE})",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the HTML chart (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--portfolios",
        type=int,
        default=5000,
        help="Number of Monte Carlo portfolios to sample (default: 5000)",
    )
    return parser


def _print_report(result: AnalysisResult, objective: Objective, risk_free_rate: float) -> None:
    """Print a human-readable summary of a completed run."""
    line = "-" * 62
    print(line)
    print("Mean-Variance Portfolio Optimizer".center(62))
    print(line)

    for warning in result.warnings:
        print(f"WARNING: {warning}")

    print(f"\nAssets      : {', '.join(result.expected_returns.index)}")
    print(
        f"Price range : {result.prepared.prices.index[0].date()} to "
        f"{result.prepared.prices.index[-1].date()} "
        f"({len(result.prepared.prices)} observations)"
    )
    print(f"Risk-free   : {risk_free_rate:.2%}")

    print("\nPer-asset annualized metrics")
    for ticker in result.expected_returns.index:
        ret = result.expected_returns[ticker]
        vol = result.covariance.loc[ticker, ticker] ** 0.5
        print(f"  {ticker:<8} return {ret:>8.2%}   volatility {vol:>8.2%}")

    print(f"\nObjective   : {objective}")
    print("\nOptimal weights")
    for ticker, weight in result.headline.weights.items():
        print(f"  {ticker:<8} {weight:>7.2%}")

    print("\nPortfolio metrics")
    print(f"  Expected return     : {result.headline.expected_return:.2%}")
    print(f"  Expected volatility : {result.headline.volatility:.2%}")
    print(f"  Sharpe ratio        : {result.headline.sharpe_ratio:.3f}")

    print("\nReference portfolios")
    print(
        f"  Max Sharpe : return {result.max_sharpe.expected_return:.2%}, "
        f"vol {result.max_sharpe.volatility:.2%}, "
        f"Sharpe {result.max_sharpe.sharpe_ratio:.3f}"
    )
    print(
        f"  Min Vol    : return {result.min_vol.expected_return:.2%}, "
        f"vol {result.min_vol.volatility:.2%}, "
        f"Sharpe {result.min_vol.sharpe_ratio:.3f}"
    )

    print(f"\nSaved run #{result.db_run_id} to the local SQLite database.")
    print(f"Chart written to: {result.html_path}")
    print("Open that HTML file in your browser (double-click) to explore the plot.")
    print(line)


def main(argv: Sequence[str] | None = None, downloader: Downloader | None = None) -> int:
    """Run the optimizer CLI.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.
        downloader: Optional price downloader injected for tests.

    Returns:
        Process exit code (0 on success).
    """
    args = build_parser().parse_args(argv)

    tickers = [ticker.upper() for ticker in args.tickers]
    objective: Objective = args.objective

    result = run_analysis(
        tickers=tickers,
        start=args.start,
        end=args.end,
        objective=objective,
        risk_free_rate=args.risk_free_rate,
        db_path=args.db_path,
        output_dir=args.output_dir,
        downloader=downloader,
        n_portfolios=args.portfolios,
    )

    _print_report(result, objective, args.risk_free_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
