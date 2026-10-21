"""End-to-end tests for the CLI entry point.

The yfinance download is injected, so the full pipeline (fetch -> clean -> math
-> optimize -> persist -> visualize) runs offline in a temp directory.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.database import fetch_runs
from src.errors import AllocationError
from main import AnalysisResult, build_parser, main, parse_min_weights, run_analysis


def _fake_downloader(tickers, start=None, end=None, **kwargs):
    rng = np.random.default_rng(0)
    periods = 300
    dates = pd.bdate_range("2021-01-01", periods=periods)
    frames = {
        ticker: 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, periods)))
        for ticker in tickers
    }
    columns = pd.MultiIndex.from_tuples([("Close", ticker) for ticker in tickers])
    data = np.column_stack([frames[ticker] for ticker in tickers])
    return pd.DataFrame(data, index=dates, columns=columns)


def _run(tmp_path, tickers, objective="max_sharpe"):
    return run_analysis(
        tickers=tickers,
        start="2021-01-01",
        end="2022-06-01",
        objective=objective,
        risk_free_rate=0.04,
        db_path=tmp_path / "portfolio_history.db",
        output_dir=tmp_path / "outputs",
        downloader=_fake_downloader,
        n_portfolios=500,
        seed=1,
    )


def test_run_analysis_returns_weights_summing_to_one(tmp_path) -> None:
    result = _run(tmp_path, ["AAA", "BBB", "CCC"])

    assert isinstance(result, AnalysisResult)
    assert sum(result.headline.weights.values) == pytest.approx(1.0, abs=1e-6)


def test_run_analysis_persists_a_run(tmp_path) -> None:
    result = _run(tmp_path, ["AAA", "BBB", "CCC"])

    runs = fetch_runs(tmp_path / "portfolio_history.db")
    assert len(runs) == 1
    assert result.db_run_id == 1
    assert runs[0]["objective_used"] == "max_sharpe"


def test_run_analysis_writes_dated_html(tmp_path) -> None:
    result = _run(tmp_path, ["AAA", "BBB", "CCC"], objective="min_vol")

    expected_name = f"efficient_frontier_{date.today():%Y%m%d}.html"
    assert result.html_path.name == expected_name
    assert result.html_path.exists()


def test_run_analysis_handles_single_ticker(tmp_path) -> None:
    result = _run(tmp_path, ["AAA"])

    assert result.headline.weights["AAA"] == pytest.approx(1.0)
    assert result.html_path.exists()


def test_parser_defaults_match_documented_values() -> None:
    args = build_parser().parse_args([])

    assert args.objective == "max_sharpe"
    assert args.risk_free_rate == pytest.approx(0.04)
    assert args.portfolios == 5000
    assert args.tickers


def test_cli_end_to_end_exits_zero(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--tickers",
            "AAA",
            "BBB",
            "CCC",
            "--start",
            "2021-01-01",
            "--end",
            "2022-06-01",
            "--objective",
            "min_vol",
            "--db-path",
            str(tmp_path / "portfolio_history.db"),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--portfolios",
            "300",
        ],
        downloader=_fake_downloader,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Optimal weights" in captured.out
    assert (tmp_path / "outputs").exists()


def test_parser_defaults_match_the_documented_estimation_flags() -> None:
    args = build_parser().parse_args([])

    assert args.expected_return_method == "historical_mean"
    assert args.covariance_method == "ledoit-wolf"
    assert args.target_return is None
    assert args.max_weight is None
    assert args.min_weight == []
    assert args.validate is False


def test_parse_min_weights_reads_ticker_and_weight() -> None:
    assert parse_min_weights(["aaa=0.25", "BBB=0.1"]) == {"AAA": 0.25, "BBB": 0.1}


def test_parse_min_weights_rejects_a_malformed_value() -> None:
    with pytest.raises(ValueError):
        parse_min_weights(["AAA"])


def test_parse_min_weights_rejects_an_out_of_range_weight() -> None:
    with pytest.raises(ValueError):
        parse_min_weights(["AAA=1.5"])


def test_run_analysis_reports_the_methods_it_used(tmp_path) -> None:
    result = run_analysis(
        tickers=["AAA", "BBB", "CCC"],
        start="2021-01-01",
        end="2022-06-01",
        objective="max_sharpe",
        risk_free_rate=0.04,
        db_path=tmp_path / "portfolio_history.db",
        output_dir=tmp_path / "outputs",
        downloader=_fake_downloader,
        n_portfolios=200,
        seed=1,
        covariance_method="sample",
        expected_return_method="historical_mean",
    )

    assert result.covariance_method == "sample"
    assert result.expected_return_method == "historical_mean"


def test_capm_expected_return_method_uses_the_market_index(tmp_path) -> None:
    result = run_analysis(
        tickers=["AAA", "BBB", "CCC"],
        start="2021-01-01",
        end="2022-06-01",
        objective="max_sharpe",
        risk_free_rate=0.04,
        db_path=tmp_path / "portfolio_history.db",
        output_dir=tmp_path / "outputs",
        downloader=_fake_downloader,
        n_portfolios=200,
        seed=1,
        expected_return_method="capm",
        market_index="^GSPC",
    )

    assert result.expected_return_method == "capm"
    assert result.expected_returns.notna().all()
    assert set(result.expected_returns.index) == {"AAA", "BBB", "CCC"}


def test_min_weight_flag_is_applied(tmp_path) -> None:
    result = run_analysis(
        tickers=["AAA", "BBB", "CCC"],
        start="2021-01-01",
        end="2022-06-01",
        objective="max_sharpe",
        risk_free_rate=0.04,
        db_path=tmp_path / "portfolio_history.db",
        output_dir=tmp_path / "outputs",
        downloader=_fake_downloader,
        n_portfolios=200,
        seed=1,
        min_weights={"BBB": 0.2},
    )

    assert result.headline.weights["BBB"] >= 0.2 - 1e-6


def test_max_weight_flag_is_applied(tmp_path) -> None:
    result = run_analysis(
        tickers=["AAA", "BBB", "CCC"],
        start="2021-01-01",
        end="2022-06-01",
        objective="max_sharpe",
        risk_free_rate=0.04,
        db_path=tmp_path / "portfolio_history.db",
        output_dir=tmp_path / "outputs",
        downloader=_fake_downloader,
        n_portfolios=200,
        seed=1,
        max_weight=0.4,
    )

    assert (result.headline.weights <= 0.4 + 1e-6).all()


def test_target_return_holds_part_of_the_portfolio_in_cash(tmp_path) -> None:
    base = _run(tmp_path, ["AAA", "BBB", "CCC"])
    target = base.headline.expected_return - 0.02

    result = run_analysis(
        tickers=["AAA", "BBB", "CCC"],
        start="2021-01-01",
        end="2022-06-01",
        objective="max_sharpe",
        risk_free_rate=0.04,
        db_path=tmp_path / "portfolio_history.db",
        output_dir=tmp_path / "outputs",
        downloader=_fake_downloader,
        n_portfolios=200,
        seed=1,
        target_return=target,
    )

    assert result.allocation is not None
    assert result.allocation.cash_weight > 0.0
    assert result.allocation.risky_weights.sum() == pytest.approx(
        1.0 - result.allocation.cash_weight
    )


def test_target_above_the_tangency_return_is_refused(tmp_path) -> None:
    base = _run(tmp_path, ["AAA", "BBB", "CCC"])

    with pytest.raises(AllocationError):
        run_analysis(
            tickers=["AAA", "BBB", "CCC"],
            start="2021-01-01",
            end="2022-06-01",
            objective="max_sharpe",
            risk_free_rate=0.04,
            db_path=tmp_path / "portfolio_history.db",
            output_dir=tmp_path / "outputs",
            downloader=_fake_downloader,
            n_portfolios=200,
            seed=1,
            target_return=base.headline.expected_return + 0.05,
        )


def test_validation_reports_out_of_sample_metrics(tmp_path) -> None:
    result = run_analysis(
        tickers=["AAA", "BBB", "CCC"],
        start="2021-01-01",
        end="2022-06-01",
        objective="max_sharpe",
        risk_free_rate=0.04,
        db_path=tmp_path / "portfolio_history.db",
        output_dir=tmp_path / "outputs",
        downloader=_fake_downloader,
        n_portfolios=200,
        seed=1,
        validate=True,
        validate_window=100,
    )

    assert result.validation is not None
    assert result.validation.rebalance_count > 0
    assert result.validation.optimized.period_count == result.validation.benchmark.period_count
