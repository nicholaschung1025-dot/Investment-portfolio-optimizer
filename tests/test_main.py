"""End-to-end tests for the CLI entry point.

The yfinance download is injected, so the full pipeline (fetch -> clean -> math
-> optimize -> persist -> visualize) runs offline in a temp directory.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.database import fetch_positions, fetch_runs
from src.errors import AllocationError, ConstraintError, OptimizationError
from src.expected_returns import black_litterman_returns
from main import AnalysisResult, build_parser, main, parse_min_weights, run_analysis


def _fundamentals_payload(ticker: str) -> dict:
    """Raw Yahoo Finance style payload with a market capitalization."""
    caps = {"AAA": 1_000_000_000.0, "BBB": 3_000_000_000.0, "CCC": 2_000_000_000.0}
    return {
        "currentPrice": 100.0,
        "marketCap": caps.get(ticker, 1_000_000_000.0),
        "currency": "USD",
        "averageVolume": 5_000_000,
        "trailingAnnualDividendYield": 0.01,
    }


def _no_fundamentals(ticker: str) -> dict:
    """Fundamentals fetcher that reports nothing, so no network is touched."""
    return {}


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
        fundamentals_fetcher=_no_fundamentals,
        fundamentals_cache=tmp_path / "fundamentals.json",
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


def test_no_output_directory_means_no_chart_file(tmp_path) -> None:
    result = run_analysis(
        tickers=["AAA", "BBB", "CCC"],
        start="2021-01-01",
        end="2022-06-01",
        objective="max_sharpe",
        risk_free_rate=0.04,
        db_path=tmp_path / "portfolio_history.db",
        output_dir=None,
        downloader=_fake_downloader,
        n_portfolios=200,
        seed=1,
        fundamentals_fetcher=_no_fundamentals,
        fundamentals_cache=tmp_path / "fundamentals.json",
    )

    assert result.html_path is None


def test_parser_defaults_match_the_documented_estimation_flags() -> None:
    args = build_parser().parse_args([])

    assert args.expected_return_method == "auto"
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


def test_auto_uses_black_litterman_when_market_caps_are_available(tmp_path) -> None:
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
        expected_return_method="auto",
        fundamentals_fetcher=_fundamentals_payload,
        fundamentals_cache=tmp_path / "fundamentals.json",
    )

    assert result.expected_return_method == "black_litterman"


def test_auto_falls_back_to_the_historical_mean_without_market_caps(tmp_path) -> None:
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
        expected_return_method="auto",
        fundamentals_fetcher=lambda ticker: {"currentPrice": 100.0},
        fundamentals_cache=tmp_path / "fundamentals.json",
    )

    assert result.expected_return_method == "historical_mean"
    assert any("capitalization" in warning for warning in result.warnings)


def test_a_budget_produces_a_risk_simulation_and_a_position_plan(tmp_path) -> None:
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
        fundamentals_fetcher=_fundamentals_payload,
        fundamentals_cache=tmp_path / "fundamentals.json",
        budget=100_000.0,
    )

    assert result.risk is not None
    assert result.positions is not None
    assert result.positions.total_value == pytest.approx(100_000.0, abs=1.0)


def test_the_positions_are_stored_with_the_run(tmp_path) -> None:
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
        fundamentals_fetcher=_fundamentals_payload,
        fundamentals_cache=tmp_path / "fundamentals.json",
        budget=100_000.0,
    )

    stored = fetch_positions(tmp_path / "portfolio_history.db", result.db_run_id)

    assert len(stored) == len(result.positions.positions)
    assert {row["Ticker"] for row in stored} == {
        position.ticker for position in result.positions.positions
    }


def test_the_run_records_the_risk_metrics(tmp_path) -> None:
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
        fundamentals_fetcher=_fundamentals_payload,
        fundamentals_cache=tmp_path / "fundamentals.json",
        budget=100_000.0,
    )

    runs = fetch_runs(tmp_path / "portfolio_history.db")

    assert runs[0]["var_95"] == pytest.approx(result.risk.metrics.var_95)
    assert runs[0]["prob_profit"] == pytest.approx(result.risk.metrics.prob_profit)


def test_the_report_shows_the_risk_metrics_and_the_positions(tmp_path, capsys) -> None:
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
            "--db-path",
            str(tmp_path / "portfolio_history.db"),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--portfolios",
            "200",
            "--budget",
            "100000",
            "--fundamentals-cache",
            str(tmp_path / "fundamentals.json"),
        ],
        downloader=_fake_downloader,
        fundamentals_fetcher=_fundamentals_payload,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Value at Risk" in captured.out
    assert "Expected Shortfall" in captured.out
    assert "Positions" in captured.out


def test_quality_filters_exclude_a_small_company(tmp_path) -> None:
    def payloads(ticker: str) -> dict:
        caps = {"AAA": 500e9, "BBB": 1e9, "CCC": 200e9}
        return {
            "currentPrice": 100.0,
            "marketCap": caps[ticker],
            "currency": "USD",
            "averageVolume": 20_000_000,
            "numberOfAnalystOpinions": 20,
            "trailingAnnualDividendYield": 0.01,
        }

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
        fundamentals_fetcher=payloads,
        fundamentals_cache=tmp_path / "fundamentals.json",
        quality_filters=True,
    )

    assert "BBB" not in result.expected_returns.index
    assert any("BBB" in warning for warning in result.warnings)


def test_the_analyst_view_moves_the_expected_return_towards_the_target(tmp_path) -> None:
    def payloads(ticker: str) -> dict:
        return {
            "currentPrice": 100.0,
            "marketCap": 100e9,
            "currency": "USD",
            "averageVolume": 20_000_000,
            "numberOfAnalystOpinions": 20,
            "targetMeanPrice": 200.0,
            "trailingAnnualDividendYield": 0.0,
        }

    without_views = run_analysis(
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
        fundamentals_fetcher=payloads,
        fundamentals_cache=tmp_path / "fundamentals.json",
    )
    with_views = run_analysis(
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
        fundamentals_fetcher=payloads,
        fundamentals_cache=tmp_path / "fundamentals.json",
        analyst_views=True,
    )

    prior = without_views.expected_returns["AAA"]
    with_view = with_views.expected_returns["AAA"]
    assert with_view > prior
    assert with_view < 1.01


def test_the_analyst_views_are_off_by_default(tmp_path) -> None:
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
        fundamentals_fetcher=_fundamentals_payload,
        fundamentals_cache=tmp_path / "fundamentals.json",
    )

    expected = black_litterman_returns(
        result.covariance,
        {"AAA": 0.2, "BBB": 0.6, "CCC": 0.4},
        0.04,
        market_risk_premium=0.05,
    )
    np.testing.assert_allclose(
        result.expected_returns.to_numpy(), expected.to_numpy(), rtol=1e-9
    )


def test_fixed_holdings_are_applied_and_reported(tmp_path) -> None:
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
        fundamentals_fetcher=_no_fundamentals,
        fundamentals_cache=tmp_path / "fundamentals.json",
        fixed_tickers=["BBB"],
        max_k=2,
    )

    assert result.selection is not None
    assert "BBB" in result.selection.tickers
    assert result.selection.weights["BBB"] >= 0.05 - 1e-6


def test_max_k_limits_the_size_of_the_portfolio(tmp_path) -> None:
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
        fundamentals_fetcher=_no_fundamentals,
        fundamentals_cache=tmp_path / "fundamentals.json",
        max_k=2,
    )

    assert result.selection is not None
    assert len(result.selection.tickers) == 2
    assert len(result.headline.weights) == 2


def test_a_fixed_holding_outside_the_universe_is_refused(tmp_path) -> None:
    with pytest.raises(ConstraintError):
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
            fundamentals_fetcher=_no_fundamentals,
            fundamentals_cache=tmp_path / "fundamentals.json",
            fixed_tickers=["NOPE"],
            max_k=2,
        )


def test_duplicate_tickers_are_collapsed(tmp_path) -> None:
    result = run_analysis(
        tickers=["AAA", "AAA", "BBB"],
        start="2021-01-01",
        end="2022-06-01",
        objective="max_sharpe",
        risk_free_rate=0.04,
        db_path=tmp_path / "portfolio_history.db",
        output_dir=tmp_path / "outputs",
        downloader=_fake_downloader,
        n_portfolios=200,
        seed=1,
        fundamentals_fetcher=_no_fundamentals,
        fundamentals_cache=tmp_path / "fundamentals.json",
    )

    assert list(result.expected_returns.index) == ["AAA", "BBB"]


def test_a_ticker_without_a_market_capitalization_does_not_break_the_estimate(tmp_path) -> None:
    def only_one_has_a_cap(ticker: str) -> dict:
        payload = _fundamentals_payload(ticker)
        if ticker != "AAA":
            payload.pop("marketCap", None)
        return payload

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
        fundamentals_fetcher=only_one_has_a_cap,
        fundamentals_cache=tmp_path / "fundamentals.json",
        expected_return_method="black_litterman",
    )

    assert result.expected_return_method == "black_litterman"
    assert result.expected_returns.notna().all()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--fee-bps", "-1"),
        ("--max-k", "-1"),
        ("--portfolios", "0"),
        ("--simulations", "0"),
        ("--student-t-df", "1"),
        ("--budget", "0"),
    ],
)
def test_the_cli_refuses_values_that_cannot_mean_anything(flag, value, tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--tickers",
                "AAA",
                "--start",
                "2021-01-01",
                "--end",
                "2022-06-01",
                "--db-path",
                str(tmp_path / "portfolio_history.db"),
                "--output-dir",
                str(tmp_path / "outputs"),
                flag,
                value,
            ],
            downloader=_fake_downloader,
        )


def test_a_refused_request_prints_a_clean_error_instead_of_a_traceback(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--tickers",
            "AAA",
            "BBB",
            "--start",
            "2021-01-01",
            "--end",
            "2022-06-01",
            "--min-weight",
            "AAA=0.9",
            "--min-weight",
            "BBB=0.9",
            "--db-path",
            str(tmp_path / "portfolio_history.db"),
            "--output-dir",
            str(tmp_path / "outputs"),
        ],
        downloader=_fake_downloader,
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error" in captured.err.lower()
    assert "Traceback" not in captured.err


def test_a_minimum_weight_makes_the_stock_a_mandatory_holding(tmp_path) -> None:
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
        fundamentals_fetcher=_no_fundamentals,
        fundamentals_cache=tmp_path / "fundamentals.json",
        max_k=2,
        min_weights={"BBB": 0.3},
    )

    assert result.selection is not None
    assert "BBB" in result.selection.tickers
    assert result.headline.weights["BBB"] >= 0.3 - 1e-6


def test_a_minimum_weight_stock_rejected_by_the_quality_filters_is_refused(tmp_path) -> None:
    def small_company(ticker: str) -> dict:
        payload = _fundamentals_payload(ticker)
        if ticker == "BBB":
            payload["marketCap"] = 1_000_000_000.0
        return payload

    with pytest.raises(ConstraintError):
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
            fundamentals_fetcher=small_company,
            fundamentals_cache=tmp_path / "fundamentals.json",
            quality_filters=True,
            min_weights={"BBB": 0.2},
        )


def test_the_parser_accepts_the_selection_flags() -> None:
    args = build_parser().parse_args(["--fixed", "AAA", "--fixed", "BBB", "--max-k", "4"])

    assert args.fixed == ["AAA", "BBB"]
    assert args.max_k == 4


def test_black_litterman_uses_market_capitalization_weights(tmp_path) -> None:
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
        expected_return_method="black_litterman",
        fundamentals_fetcher=_fundamentals_payload,
        fundamentals_cache=tmp_path / "fundamentals.json",
    )

    assert result.expected_return_method == "black_litterman"
    expected = black_litterman_returns(
        result.covariance,
        {"AAA": 0.2, "BBB": 0.6, "CCC": 0.4},
        0.04,
        market_risk_premium=0.05,
        tau=0.05,
    )
    np.testing.assert_allclose(
        result.expected_returns.to_numpy(), expected.to_numpy(), rtol=1e-9
    )


def test_hong_kong_market_caps_are_converted_before_weighting(tmp_path) -> None:
    caps = {"AAA": 1_000_000_000.0, "0700.HK": 7_820_000_000.0}

    def payloads(ticker: str) -> dict:
        return {
            "currentPrice": 100.0,
            "marketCap": caps[ticker],
            "currency": "HKD" if ticker.endswith(".HK") else "USD",
        }

    result = run_analysis(
        tickers=["AAA", "0700.HK"],
        start="2021-01-01",
        end="2022-06-01",
        objective="max_sharpe",
        risk_free_rate=0.04,
        db_path=tmp_path / "portfolio_history.db",
        output_dir=tmp_path / "outputs",
        downloader=_fake_downloader,
        n_portfolios=200,
        seed=1,
        expected_return_method="black_litterman",
        fundamentals_fetcher=payloads,
        fundamentals_cache=tmp_path / "fundamentals.json",
    )

    expected = black_litterman_returns(
        result.covariance, {"AAA": 0.5, "0700.HK": 0.5}, 0.04, market_risk_premium=0.05
    )
    np.testing.assert_allclose(
        result.expected_returns.to_numpy(), expected.to_numpy(), rtol=1e-9
    )


def test_black_litterman_without_market_caps_fails_clearly(tmp_path) -> None:
    with pytest.raises(OptimizationError):
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
            expected_return_method="black_litterman",
            fundamentals_fetcher=lambda ticker: {"currentPrice": 100.0},
            fundamentals_cache=tmp_path / "fundamentals.json",
        )


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
        fundamentals_fetcher=_no_fundamentals,
        fundamentals_cache=tmp_path / "fundamentals.json",
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
        fundamentals_fetcher=_no_fundamentals,
        fundamentals_cache=tmp_path / "fundamentals.json",
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
        fundamentals_fetcher=_no_fundamentals,
        fundamentals_cache=tmp_path / "fundamentals.json",
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
            fundamentals_fetcher=_no_fundamentals,
            fundamentals_cache=tmp_path / "fundamentals.json",
            target_return=base.headline.expected_return + 0.05,
        )


def test_compare_estimators_prints_a_comparison(tmp_path, capsys) -> None:
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
            "--db-path",
            str(tmp_path / "portfolio_history.db"),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--portfolios",
            "200",
            "--compare-estimators",
            "--validate-window",
            "100",
            "--fundamentals-cache",
            str(tmp_path / "fundamentals.json"),
        ],
        downloader=_fake_downloader,
        fundamentals_fetcher=_fundamentals_payload,
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "historical_mean" in captured.out
    assert "black_litterman" in captured.out
    assert "bench CAGR" in captured.out


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
        fundamentals_fetcher=_no_fundamentals,
        fundamentals_cache=tmp_path / "fundamentals.json",
        validate=True,
        validate_window=100,
    )

    assert result.validation is not None
    assert result.validation.rebalance_count > 0
    assert result.validation.optimized.period_count == result.validation.benchmark.period_count
