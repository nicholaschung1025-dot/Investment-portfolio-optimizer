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

from src.allocation import AllocationResult, allocate_to_target_return
from src.constraints import WeightConstraints
from src.covariance import CovarianceMethod, shrunk_covariance
from src.data_fetcher import Downloader, PreparedPrices, fetch_close_prices, prepare_prices
from src.database import RunRecord, save_run
from src.errors import OptimizationError
from src.expected_returns import (
    ExpectedReturnMethod,
    black_litterman_returns,
    capm_returns,
    historical_mean,
)
from src.fundamentals import (
    DEFAULT_CACHE_PATH,
    Fetcher,
    FundamentalsCache,
    fetch_fundamentals,
    market_caps_in_usd,
)
from src.market import (
    DEFAULT_USD_HKD,
    FxRates,
    MarketMetadata,
    capitalization_weights,
)
from src.math_engine import compute_log_returns
from src.optimizer import (
    Objective,
    OptimizationResult,
    monte_carlo_frontier,
    optimize_portfolio,
)
from src.selection import DEFAULT_FIXED_WEIGHT, SelectionResult, select_portfolio
from src.validation import BacktestResult, compare_expected_return_methods, walk_forward_backtest
from src.visualizer import generate_efficient_frontier

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
DEFAULT_START = "2020-01-01"
DEFAULT_RISK_FREE_RATE = 0.04
DEFAULT_DB_PATH = Path("data/portfolio_history.db")
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_MARKET_INDEX = "^GSPC"
DEFAULT_COVARIANCE_METHOD = "ledoit-wolf"
DEFAULT_EXPECTED_RETURN_METHOD = "auto"
DEFAULT_MARKET_RISK_PREMIUM = 0.05
DEFAULT_TAU = 0.05


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
        expected_return_method: Estimator used for the expected returns.
        covariance_method: Estimator used for the covariance matrix.
        constraints: Limits applied to the portfolio, if any.
        allocation: Cash allocation that reaches the requested target return.
        validation: Out-of-sample walk-forward results, when requested.
        comparison: Out-of-sample comparison of the expected-return methods,
            when requested.
        selection: The chosen portfolio, when a selection was requested.
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
    expected_return_method: str = DEFAULT_EXPECTED_RETURN_METHOD
    covariance_method: str = DEFAULT_COVARIANCE_METHOD
    constraints: WeightConstraints | None = None
    allocation: AllocationResult | None = None
    validation: BacktestResult | None = None
    comparison: pd.DataFrame | None = None
    selection: SelectionResult | None = None


def parse_min_weights(values: Sequence[str] | None) -> dict[str, float]:
    """Parse ``TICKER=WEIGHT`` arguments into a mapping.

    Args:
        values: Raw argument values such as ``["AAPL=0.25"]``.

    Returns:
        Minimum weight per ticker.

    Raises:
        ValueError: If an entry is malformed or its weight is out of range.
    """
    weights: dict[str, float] = {}
    for item in values or []:
        ticker, separator, weight_text = str(item).partition("=")
        if not separator:
            raise ValueError(f"expected TICKER=WEIGHT, got {item!r}")
        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError(f"expected TICKER=WEIGHT, got {item!r}")
        try:
            weight = float(weight_text)
        except ValueError as error:
            raise ValueError(f"{item!r} does not contain a numeric weight") from error
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"the weight for {ticker} must be between 0 and 1, got {weight}")
        weights[ticker] = weight
    return weights


def _market_log_returns(
    market_index: str,
    start: str,
    end: str,
    downloader: Downloader | None,
) -> pd.Series:
    """Download the market index and return its daily log returns."""
    market_prices = fetch_close_prices([market_index], start, end, downloader)
    market_log_returns = compute_log_returns(market_prices)
    return market_log_returns.iloc[:, 0]


def _estimate_expected_returns(
    method: ExpectedReturnMethod,
    log_returns: pd.DataFrame,
    risk_free_rate: float,
    market_index: str,
    start: str,
    end: str,
    downloader: Downloader | None,
) -> pd.Series:
    """Estimate the expected returns with the requested method."""
    if method == "historical_mean":
        return historical_mean(log_returns)
    if method == "capm":
        market_returns = _market_log_returns(market_index, start, end, downloader)
        return capm_returns(log_returns, market_returns, risk_free_rate)
    raise OptimizationError(f"unknown expected return method: {method!r}")


def _black_litterman_expected_returns(
    covariance: pd.DataFrame,
    risk_free_rate: float,
    market_risk_premium: float,
    tau: float,
    tickers: list[str],
    fundamentals_fetcher: Fetcher | None,
    fundamentals_cache: str | Path,
    refresh_fundamentals: bool,
    market_metadata: MarketMetadata,
    fx_rates: FxRates,
) -> pd.Series:
    """Estimate expected returns with Black-Litterman.

    The market portfolio is approximated by the capitalization-weighted
    candidate universe, which needs the market capitalization of every
    ticker.
    """
    fundamentals = fetch_fundamentals(
        tickers,
        cache=FundamentalsCache(fundamentals_cache),
        fetcher=fundamentals_fetcher,
        refresh=refresh_fundamentals,
    )
    caps = market_caps_in_usd(fundamentals, market_metadata, fx_rates)
    if caps.empty:
        raise OptimizationError(
            "no market capitalizations are available, so the Black-Litterman market "
            "portfolio cannot be built; refresh the fundamentals cache or check the tickers"
        )
    market_weights = capitalization_weights(caps)
    return black_litterman_returns(
        covariance,
        market_weights,
        risk_free_rate,
        market_risk_premium=market_risk_premium,
        tau=tau,
    )


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
    expected_return_method: ExpectedReturnMethod = DEFAULT_EXPECTED_RETURN_METHOD,
    covariance_method: CovarianceMethod = DEFAULT_COVARIANCE_METHOD,
    market_index: str = DEFAULT_MARKET_INDEX,
    market_risk_premium: float = DEFAULT_MARKET_RISK_PREMIUM,
    tau: float = DEFAULT_TAU,
    fundamentals_cache: str | Path = DEFAULT_CACHE_PATH,
    refresh_fundamentals: bool = False,
    fundamentals_fetcher: Fetcher | None = None,
    market_metadata: MarketMetadata | None = None,
    usd_hkd: float = DEFAULT_USD_HKD,
    min_weights: dict[str, float] | None = None,
    max_weight: float | None = None,
    max_volatility: float | None = None,
    fixed_tickers: Sequence[str] | None = None,
    max_k: int = 0,
    selection_beam: int = 5,
    target_return: float | None = None,
    validate: bool = False,
    validate_window: int = 252,
    validate_rebalance: str = "monthly",
    compare_estimators: bool = False,
    compare_methods: Sequence[str] | None = None,
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
        expected_return_method: How the expected returns are estimated.
        covariance_method: How the covariance matrix is estimated.
        market_index: Market index used by the CAPM estimator.
        market_risk_premium: Market risk premium used by the Black-Litterman
            equilibrium returns.
        tau: Uncertainty of the Black-Litterman prior.
        fundamentals_cache: Where the raw fundamentals are cached.
        refresh_fundamentals: Whether to ignore the fundamentals cache.
        fundamentals_fetcher: Callable returning a raw fundamentals payload;
            injected in tests so nothing ever touches the network.
        market_metadata: Market metadata, used for currencies and board lots.
        usd_hkd: Hong Kong dollars per US dollar.
        min_weights: Minimum weight per ticker, for mandatory holdings.
        max_weight: Maximum weight for any single ticker.
        max_volatility: Maximum annualized portfolio volatility.
        fixed_tickers: Holdings that must be part of the portfolio.
        max_k: Maximum number of assets to hold; zero keeps the whole
            universe.
        selection_beam: How many partial selections the search keeps per
            step.
        target_return: Return the portfolio should reach; the remainder is
            held in the risk-free asset.
        validate: Whether to run the out-of-sample walk-forward backtest.
        validate_window: Observations required before the first rebalance of
            the validation run.
        validate_rebalance: ``"monthly"`` or ``"quarterly"``.
        compare_estimators: Whether to walk-forward every expected-return
            method and compare them.
        compare_methods: Which methods to compare; every available method by
            default.

    Returns:
        The completed :class:`AnalysisResult`.
    """
    raw_prices = fetch_close_prices(tickers, start, end, downloader)
    prepared = prepare_prices(raw_prices)

    log_returns = compute_log_returns(prepared.prices)
    covariance = shrunk_covariance(log_returns, method=covariance_method)

    warnings = list(prepared.warnings)
    effective_method = expected_return_method

    if expected_return_method in ("black_litterman", "auto"):
        try:
            expected_returns = _black_litterman_expected_returns(
                covariance=covariance,
                risk_free_rate=risk_free_rate,
                market_risk_premium=market_risk_premium,
                tau=tau,
                tickers=list(log_returns.columns),
                fundamentals_fetcher=fundamentals_fetcher,
                fundamentals_cache=fundamentals_cache,
                refresh_fundamentals=refresh_fundamentals,
                market_metadata=market_metadata or MarketMetadata(),
                fx_rates=FxRates.default(usd_hkd),
            )
            effective_method = "black_litterman"
        except OptimizationError:
            if expected_return_method == "black_litterman":
                raise
            expected_returns = historical_mean(log_returns)
            effective_method = "historical_mean"
            warnings.append(
                "no market capitalizations were available, so the expected returns "
                "fell back to the historical mean"
            )
    else:
        expected_returns = _estimate_expected_returns(
            expected_return_method,
            log_returns,
            risk_free_rate,
            market_index,
            start,
            end,
            downloader,
        )

    fixed = list(dict.fromkeys(fixed_tickers or []))
    effective_min_weights = dict(min_weights or {})
    for ticker in fixed:
        effective_min_weights.setdefault(ticker, DEFAULT_FIXED_WEIGHT)
    constraints = WeightConstraints(
        min_weights=effective_min_weights,
        max_weight=max_weight,
        max_volatility=max_volatility,
    )

    selection = None
    if max_k and max_k < len(expected_returns.index):
        selection = select_portfolio(
            expected_returns,
            covariance,
            fixed,
            max_k,
            objective=objective,
            risk_free_rate=risk_free_rate,
            constraints=constraints,
            beam_width=selection_beam,
        )
        expected_returns = expected_returns[selection.tickers]
        covariance = covariance.loc[selection.tickers, selection.tickers]
        if selection.constraints is not None:
            constraints = selection.constraints

    max_sharpe = optimize_portfolio(
        expected_returns, covariance, "max_sharpe", risk_free_rate, constraints=constraints
    )
    min_vol = optimize_portfolio(
        expected_returns, covariance, "min_vol", risk_free_rate, constraints=constraints
    )
    headline = max_sharpe if objective == "max_sharpe" else min_vol

    allocation = None
    if target_return is not None:
        allocation = allocate_to_target_return(headline, risk_free_rate, target_return)

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

    validation = None
    if validate:
        validation = walk_forward_backtest(
            prepared.prices,
            objective=objective,
            risk_free_rate=risk_free_rate,
            covariance_method=covariance_method,
            initial_window=validate_window,
            rebalance=validate_rebalance,
            constraints=constraints,
        )

    comparison = None
    if compare_estimators:
        methods = list(compare_methods) if compare_methods else [
            "historical_mean",
            "capm",
            "black_litterman",
        ]
        metadata = market_metadata or MarketMetadata()
        fx_rates = FxRates.default(usd_hkd)

        market_returns = (
            _market_log_returns(market_index, start, end, downloader)
            if "capm" in methods
            else None
        )
        market_weights = None
        if "black_litterman" in methods:
            fundamentals = fetch_fundamentals(
                list(log_returns.columns),
                cache=FundamentalsCache(fundamentals_cache),
                fetcher=fundamentals_fetcher,
                refresh=refresh_fundamentals,
            )
            caps = market_caps_in_usd(fundamentals, metadata, fx_rates)
            market_weights = capitalization_weights(caps) if not caps.empty else None

        def estimator_factory(method: str):
            def estimate(log_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
                covariance = shrunk_covariance(log_returns, method=covariance_method)
                if method == "historical_mean":
                    expected = historical_mean(log_returns)
                elif method == "capm":
                    if market_returns is None:
                        raise OptimizationError("the CAPM estimator needs a market index")
                    expected = capm_returns(log_returns, market_returns, risk_free_rate)
                elif method == "black_litterman":
                    if market_weights is None:
                        raise OptimizationError(
                            "the Black-Litterman estimator needs market capitalizations"
                        )
                    expected = black_litterman_returns(
                        covariance,
                        market_weights,
                        risk_free_rate,
                        market_risk_premium=market_risk_premium,
                        tau=tau,
                    )
                else:
                    raise OptimizationError(f"unknown expected return method: {method!r}")
                return expected, covariance

            return estimate

        comparison = compare_expected_return_methods(
            prepared.prices,
            methods,
            estimator_factory,
            objective=objective,
            risk_free_rate=risk_free_rate,
            covariance_method=covariance_method,
            initial_window=validate_window,
            rebalance=validate_rebalance,
            constraints=constraints,
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
        warnings=warnings,
        expected_return_method=effective_method,
        covariance_method=covariance_method,
        constraints=constraints if constraints.is_active() else None,
        allocation=allocation,
        validation=validation,
        comparison=comparison,
        selection=selection,
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
    parser.add_argument(
        "--expected-return-method",
        choices=["auto", "historical_mean", "capm", "black_litterman"],
        default=DEFAULT_EXPECTED_RETURN_METHOD,
        help=(
            "How expected returns are estimated: auto picks Black-Litterman when "
            "market capitalizations are available and falls back to the sample "
            "mean otherwise, historical_mean uses the sample mean, capm uses the "
            "capital asset pricing model, and black_litterman requires market "
            f"capitalizations (default: {DEFAULT_EXPECTED_RETURN_METHOD})"
        ),
    )
    parser.add_argument(
        "--covariance-method",
        choices=["sample", "ledoit-wolf", "oas"],
        default=DEFAULT_COVARIANCE_METHOD,
        help=(
            "How the covariance matrix is estimated. Shrinkage estimators pull the "
            "noisy sample covariance towards a scaled identity matrix, which "
            f"stabilizes the optimizer (default: {DEFAULT_COVARIANCE_METHOD})"
        ),
    )
    parser.add_argument(
        "--market-index",
        default=DEFAULT_MARKET_INDEX,
        help=f"Market index used by the CAPM estimator (default: {DEFAULT_MARKET_INDEX})",
    )
    parser.add_argument(
        "--market-risk-premium",
        type=float,
        default=DEFAULT_MARKET_RISK_PREMIUM,
        help=(
            "Market risk premium used to scale the Black-Litterman equilibrium "
            f"returns (default: {DEFAULT_MARKET_RISK_PREMIUM})"
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=DEFAULT_TAU,
        help=f"Uncertainty of the Black-Litterman prior (default: {DEFAULT_TAU})",
    )
    parser.add_argument(
        "--fundamentals-cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Where raw fundamentals are cached (default: {DEFAULT_CACHE_PATH})",
    )
    parser.add_argument(
        "--refresh-fundamentals",
        action="store_true",
        help="Ignore the fundamentals cache and fetch everything again",
    )
    parser.add_argument(
        "--market-meta",
        type=Path,
        default=None,
        help=(
            "CSV file with ticker, currency, board_lot and market columns that "
            "overrides the built-in market metadata"
        ),
    )
    parser.add_argument(
        "--usd-hkd",
        type=float,
        default=DEFAULT_USD_HKD,
        help=f"Hong Kong dollars per US dollar (default: {DEFAULT_USD_HKD})",
    )
    parser.add_argument(
        "--fixed",
        action="append",
        default=[],
        metavar="TICKER",
        help=(
            "A holding that must be part of the portfolio. Repeat the flag for "
            "several mandatory holdings; each gets a minimum weight."
        ),
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=0,
        help=(
            "Maximum number of assets to hold. The optimizer searches for the "
            "best subset; zero keeps the whole universe (default: 0)"
        ),
    )
    parser.add_argument(
        "--selection-beam",
        type=int,
        default=5,
        help="How many partial selections the search keeps per step (default: 5)",
    )
    parser.add_argument(
        "--min-weight",
        action="append",
        default=[],
        metavar="TICKER=WEIGHT",
        help=(
            "Minimum weight for a ticker, e.g. AAPL=0.25. Repeat the flag for "
            "several mandatory holdings."
        ),
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=None,
        help="Maximum weight for any single ticker, e.g. 0.35 (default: no cap)",
    )
    parser.add_argument(
        "--max-volatility",
        type=float,
        default=None,
        help="Maximum annualized portfolio volatility, e.g. 0.15 (default: no cap)",
    )
    parser.add_argument(
        "--target-return",
        type=float,
        default=None,
        help=(
            "Annualized return the portfolio should reach. The remainder is held "
            "in the risk-free asset (default: fully invested)"
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the out-of-sample walk-forward backtest and report it",
    )
    parser.add_argument(
        "--validate-window",
        type=int,
        default=252,
        help="Observations required before the first rebalance of the validation run (default: 252)",
    )
    parser.add_argument(
        "--validate-rebalance",
        choices=["monthly", "quarterly"],
        default="monthly",
        help="Rebalance frequency of the validation run (default: monthly)",
    )
    parser.add_argument(
        "--compare-estimators",
        action="store_true",
        help=(
            "Walk-forward every expected-return method and compare the results "
            "out of sample, so the choice is measured rather than assumed"
        ),
    )
    return parser


def _print_report(
    result: AnalysisResult,
    objective: Objective,
    risk_free_rate: float,
) -> None:
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
    print(
        f"Estimation  : expected returns = {result.expected_return_method}, "
        f"covariance = {result.covariance_method}"
    )

    if result.constraints is not None:
        constraints = result.constraints
        parts = []
        if constraints.min_weights:
            floors = ", ".join(
                f"{ticker} {weight:.0%}"
                for ticker, weight in sorted(constraints.min_weights.items())
            )
            parts.append(f"minimum weights {floors}")
        if constraints.max_weight is not None:
            parts.append(f"maximum weight {constraints.max_weight:.0%}")
        if constraints.max_volatility is not None:
            parts.append(f"maximum volatility {constraints.max_volatility:.0%}")
        print(f"Constraints : {'; '.join(parts)}")

    if result.selection is not None:
        selection = result.selection
        print("\nSelection")
        if selection.fixed_tickers:
            print(f"  Mandatory holdings : {', '.join(selection.fixed_tickers)}")
        else:
            print("  Mandatory holdings : none")
        print(f"  Chosen             : {', '.join(selection.tickers)}")
        for step in selection.steps:
            print(
                f"    step {step['added']:<10} added, "
                f"portfolio {', '.join(step['portfolio'])}"
            )

    print("\nPer-asset annualized metrics")
    for ticker in result.expected_returns.index:
        ret = result.expected_returns[ticker]
        vol = result.covariance.loc[ticker, ticker] ** 0.5
        print(f"  {ticker:<8} return {ret:>8.2%}   volatility {vol:>8.2%}")

    print(f"\nObjective   : {objective}")
    print("\nOptimal weights")
    recommended = (
        result.allocation.risky_weights if result.allocation else result.headline.weights
    )
    for ticker, weight in recommended.items():
        print(f"  {ticker:<8} {weight:>7.2%}")
    if result.allocation is not None:
        print(f"  {'CASH':<8} {result.allocation.cash_weight:>7.2%}  (risk-free asset)")

    print("\nPortfolio metrics")
    if result.allocation is not None:
        print(f"  Expected return     : {result.allocation.expected_return:.2%}")
        print(f"  Expected volatility : {result.allocation.volatility:.2%}")
        print(f"  Sharpe ratio        : {result.allocation.sharpe_ratio:.3f}")
    else:
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

    if result.comparison is not None:
        print("\nEstimator comparison (walk-forward, out of sample)")
        print(
            f"  {'method':<18} {'CAGR':>8} {'Sharpe':>8} {'max drawdown':>13} "
            f"{'bench CAGR':>11} {'bench Sharpe':>13}"
        )
        for _, row in result.comparison.iterrows():
            print(
                f"  {row['method']:<18} {row['optimized_cagr']:>7.2%} "
                f"{row['optimized_sharpe']:>8.3f} {row['optimized_max_drawdown']:>12.2%} "
                f"{row['benchmark_cagr']:>10.2%} {row['benchmark_sharpe']:>13.3f}"
            )
        print(
            "  Note: the Black-Litterman run uses today's market capitalizations, "
            "so its backtest is mildly optimistic."
        )

    if result.validation is not None:
        validation = result.validation
        print(f"\nOut-of-sample validation ({validation.rebalance_count} rebalances)")
        print(
            f"  Optimized : CAGR {validation.optimized.cagr:>7.2%}   "
            f"vol {validation.optimized.volatility:>7.2%}   "
            f"Sharpe {validation.optimized.sharpe_ratio:>6.3f}   "
            f"max drawdown {validation.optimized.max_drawdown:>7.2%}"
        )
        print(
            f"  Benchmark  : CAGR {validation.benchmark.cagr:>7.2%}   "
            f"vol {validation.benchmark.volatility:>7.2%}   "
            f"Sharpe {validation.benchmark.sharpe_ratio:>6.3f}   "
            f"max drawdown {validation.benchmark.max_drawdown:>7.2%}"
        )
        print(f"  Turnover   : {validation.average_turnover:.2%} per rebalance")

    print(f"\nSaved run #{result.db_run_id} to the local SQLite database.")
    print(f"Chart written to: {result.html_path}")
    print("Open that HTML file in your browser (double-click) to explore the plot.")
    print(line)


def main(
    argv: Sequence[str] | None = None,
    downloader: Downloader | None = None,
    fundamentals_fetcher: Fetcher | None = None,
) -> int:
    """Run the optimizer CLI.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.
        downloader: Optional price downloader injected for tests.
        fundamentals_fetcher: Optional fundamentals fetcher injected for
            tests.

    Returns:
        Process exit code (0 on success).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    tickers = [ticker.upper() for ticker in args.tickers]
    objective: Objective = args.objective

    try:
        min_weights = parse_min_weights(args.min_weight)
    except ValueError as error:
        parser.error(str(error))

    market_metadata = (
        MarketMetadata.from_csv(args.market_meta) if args.market_meta else MarketMetadata()
    )

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
        expected_return_method=args.expected_return_method,
        covariance_method=args.covariance_method,
        market_index=args.market_index,
        market_risk_premium=args.market_risk_premium,
        tau=args.tau,
        fundamentals_cache=args.fundamentals_cache,
        refresh_fundamentals=args.refresh_fundamentals,
        fundamentals_fetcher=fundamentals_fetcher,
        market_metadata=market_metadata,
        usd_hkd=args.usd_hkd,
        min_weights=min_weights,
        max_weight=args.max_weight,
        max_volatility=args.max_volatility,
        fixed_tickers=args.fixed,
        max_k=args.max_k,
        selection_beam=args.selection_beam,
        target_return=args.target_return,
        validate=args.validate,
        validate_window=args.validate_window,
        validate_rebalance=args.validate_rebalance,
        compare_estimators=args.compare_estimators,
    )

    _print_report(result, objective, args.risk_free_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
