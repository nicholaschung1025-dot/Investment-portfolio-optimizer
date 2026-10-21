"""Tests for the self-contained HTML report.

The report has to be readable on its own: the styles, the charts and the
numbers in one file, with nothing fetched from the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from main import run_analysis
from src.report import render_report, write_report


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


def _fundamentals(ticker: str) -> dict:
    return {
        "currentPrice": 100.0,
        "marketCap": 500_000_000_000.0,
        "currency": "USD",
        "averageVolume": 20_000_000,
        "numberOfAnalystOpinions": 20,
        "trailingAnnualDividendYield": 0.01,
    }


def _run(tmp_path, **overrides):
    params = dict(
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
        fundamentals_fetcher=_fundamentals,
        fundamentals_cache=tmp_path / "fundamentals.json",
    )
    params.update(overrides)
    return run_analysis(**params)


def test_the_report_names_the_tickers_and_the_run(tmp_path) -> None:
    result = _run(tmp_path)

    html = render_report(result)

    assert "Portfolio Report" in html
    assert "AAA" in html and "BBB" in html and "CCC" in html
    assert "black_litterman" in html


def test_the_report_shows_the_portfolio_metrics(tmp_path) -> None:
    result = _run(tmp_path)

    html = render_report(result)

    assert "Expected return" in html
    assert "Volatility" in html
    assert "Sharpe ratio" in html
    assert f"{result.headline.expected_return * 100:.2f}%" in html


def test_the_report_is_self_contained(tmp_path) -> None:
    result = _run(tmp_path)

    html = render_report(result)

    # The chart library is inlined, so the file works with no network.
    assert "<script src=" not in html
    assert '<link rel="stylesheet"' not in html
    assert "plotly" in html.lower()


def test_the_report_omits_the_sections_that_have_no_data(tmp_path) -> None:
    result = _run(tmp_path)

    html = render_report(result)

    assert "<h2>Positions</h2>" not in html
    assert "<h2>Risk over one year</h2>" not in html
    assert "<h2>Out of sample</h2>" not in html


def test_the_report_includes_the_risk_and_the_positions(tmp_path) -> None:
    result = _run(tmp_path, budget=100_000.0)

    html = render_report(result)

    assert "Value at Risk (95%)" in html
    assert "<h2>Risk over one year</h2>" in html
    assert "<h2>Positions</h2>" in html
    assert "Estimated entry cost" in html


def test_the_report_includes_the_out_of_sample_result(tmp_path) -> None:
    result = _run(tmp_path, validate=True, validate_window=100)

    html = render_report(result)

    assert "<h2>Out of sample</h2>" in html
    assert "Equal weight" in html


def test_the_report_includes_the_estimator_comparison(tmp_path) -> None:
    result = _run(tmp_path, compare_estimators=True, validate_window=100)

    html = render_report(result)

    assert "<h2>Estimator comparison</h2>" in html
    assert "black_litterman" in html


def test_the_report_lists_the_warnings(tmp_path) -> None:
    def small_company(ticker: str) -> dict:
        payload = _fundamentals(ticker)
        if ticker == "CCC":
            payload["marketCap"] = 1_000_000_000.0
            payload["numberOfAnalystOpinions"] = 2
        return payload

    result = _run(tmp_path, quality_filters=True, fundamentals_fetcher=small_company)

    html = render_report(result)

    assert "excluded by the quality filters" in html
    assert "CCC" in html


def test_write_report_puts_a_single_file_on_disk(tmp_path) -> None:
    result = _run(tmp_path)
    destination = tmp_path / "reports" / "report.html"

    written = write_report(destination, result)

    assert written == destination
    assert destination.exists()
    text = destination.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert "AAA" in text
