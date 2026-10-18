"""Tests for the data ingestion and cleaning engine.

The network boundary is injected as a ``downloader`` callable, so these tests
run fully offline and assert on real cleaning behaviour rather than on mocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_fetcher import (
    DataFetchError,
    PreparedPrices,
    fetch_close_prices,
    prepare_prices,
)


def _raw() -> pd.DataFrame:
    """Two tickers on mismatched calendars with an internal and a leading gap."""
    dates = pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )
    return pd.DataFrame(
        {
            "AAA": [np.nan, 10.0, np.nan, 11.0, 11.5],
            "BBB": [np.nan, np.nan, 20.0, 20.5, 21.0],
        },
        index=dates,
    )


def test_prepare_prices_drops_leading_rows_and_reports_overlap() -> None:
    prepared = prepare_prices(_raw())

    assert isinstance(prepared, PreparedPrices)
    assert prepared.prices.index[0] == pd.Timestamp("2024-01-03")
    assert prepared.prices.index[-1] == pd.Timestamp("2024-01-05")
    assert any("overlap" in warning.lower() for warning in prepared.warnings)


def test_prepare_prices_forward_fills_internal_gaps() -> None:
    prepared = prepare_prices(_raw())

    assert not prepared.prices.isna().any().any()
    assert prepared.prices.loc["2024-01-03", "AAA"] == pytest.approx(10.0)
    assert prepared.prices.loc["2024-01-04", "BBB"] == pytest.approx(20.5)


def test_prepare_prices_excludes_zero_variance_assets() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    raw = pd.DataFrame(
        {"FLAT": [5.0, 5.0, 5.0], "MOVE": [10.0, 11.0, 12.0]}, index=dates
    )

    prepared = prepare_prices(raw)

    assert prepared.excluded_assets == ["FLAT"]
    assert list(prepared.prices.columns) == ["MOVE"]
    assert any("zero variance" in warning.lower() for warning in prepared.warnings)


def test_prepare_prices_raises_when_no_assets_survive() -> None:
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    raw = pd.DataFrame({"FLAT": [5.0, 5.0]}, index=dates)

    with pytest.raises(DataFetchError):
        prepare_prices(raw)


def test_prepare_prices_raises_on_empty_input() -> None:
    with pytest.raises(DataFetchError):
        prepare_prices(pd.DataFrame())


def test_fetch_close_prices_extracts_close_from_multiindex() -> None:
    def fake_downloader(tickers, **kwargs):
        index = pd.to_datetime(["2024-01-01", "2024-01-02"])
        columns = pd.MultiIndex.from_tuples(
            [("Close", "AAA"), ("Close", "BBB"), ("Open", "AAA"), ("Open", "BBB")]
        )
        data = np.array(
            [[10.0, 20.0, 9.5, 19.5], [11.0, 21.0, 10.5, 20.5]], dtype=float
        )
        return pd.DataFrame(data, index=index, columns=columns)

    prices = fetch_close_prices(["AAA", "BBB"], "2024-01-01", "2024-01-02", fake_downloader)

    assert list(prices.columns) == ["AAA", "BBB"]
    assert prices.loc[pd.Timestamp("2024-01-02"), "AAA"] == pytest.approx(11.0)


def test_fetch_close_prices_raises_on_empty_response() -> None:
    def empty_downloader(tickers, **kwargs):
        return pd.DataFrame()

    with pytest.raises(DataFetchError):
        fetch_close_prices(["AAA"], "2024-01-01", "2024-01-02", empty_downloader)


def test_fetch_close_prices_raises_when_all_tickers_missing() -> None:
    def missing_downloader(tickers, **kwargs):
        index = pd.to_datetime(["2024-01-01"])
        return pd.DataFrame(
            [[1.0]], index=index, columns=pd.MultiIndex.from_tuples([("Close", "ZZZ")])
        )

    with pytest.raises(DataFetchError):
        fetch_close_prices(["AAA", "BBB"], "2024-01-01", "2024-01-02", missing_downloader)
