"""Tests for market metadata, currency conversion and market weights.

Covers the built-in ticker table, the fallback rules for unknown tickers,
CSV overrides, USD conversion, and the capitalization weights that
Black-Litterman needs as its market portfolio.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.errors import OptimizationError
from src.market import (
    FxRates,
    MarketInfo,
    MarketMetadata,
    capitalization_weights,
)


def test_us_ticker_defaults_to_usd_and_a_single_share() -> None:
    info = MarketMetadata().info("AAPL")

    assert info.currency == "USD"
    assert info.board_lot == 1
    assert info.market == "US"


def test_unknown_hk_ticker_falls_back_to_hkd_and_a_default_board_lot() -> None:
    info = MarketMetadata().info("1234.HK")

    assert info.currency == "HKD"
    assert info.board_lot == 100
    assert info.market == "HK"


def test_known_hk_ticker_uses_its_board_lot() -> None:
    info = MarketMetadata().info("0005.HK")

    assert info.currency == "HKD"
    assert info.board_lot == 500


def test_lookup_is_case_insensitive() -> None:
    info = MarketMetadata().info("aapl")

    assert info.currency == "USD"


def test_overrides_take_precedence_over_the_built_in_table() -> None:
    metadata = MarketMetadata({"AAPL": MarketInfo("EUR", 10, "US")})

    info = metadata.info("AAPL")

    assert info.currency == "EUR"
    assert info.board_lot == 10


def test_market_metadata_from_csv_overrides_the_built_in_table(tmp_path) -> None:
    path = tmp_path / "market_meta.csv"
    path.write_text(
        "ticker,currency,board_lot,market\n0700.HK,HKD,200,HK\nACME,EUR,5,US\n",
        encoding="utf-8",
    )

    metadata = MarketMetadata.from_csv(path)

    assert metadata.info("0700.HK").board_lot == 200
    assert metadata.info("ACME").currency == "EUR"
    assert metadata.info("ACME").board_lot == 5


def test_fx_rates_convert_hong_kong_dollars_to_us_dollars() -> None:
    rates = FxRates.default(usd_hkd=7.82)

    assert rates.to_usd(782.0, "HKD") == pytest.approx(100.0)


def test_fx_rates_convert_us_dollars_to_us_dollars() -> None:
    rates = FxRates.default()

    assert rates.to_usd(100.0, "USD") == pytest.approx(100.0)


def test_fx_rates_reject_an_unknown_currency() -> None:
    rates = FxRates.default()

    with pytest.raises(OptimizationError):
        rates.to_usd(100.0, "JPY")


def test_capitalization_weights_sum_to_one() -> None:
    caps = pd.Series({"AAA": 1000.0, "BBB": 3000.0})

    weights = capitalization_weights(caps)

    assert weights.sum() == pytest.approx(1.0)
    assert weights["AAA"] == pytest.approx(0.25)
    assert weights["BBB"] == pytest.approx(0.75)


def test_capitalization_weights_ignore_non_positive_caps() -> None:
    caps = pd.Series({"AAA": 1000.0, "BBB": 0.0, "CCC": float("nan")})

    weights = capitalization_weights(caps)

    assert list(weights.index) == ["AAA"]
    assert weights["AAA"] == pytest.approx(1.0)


def test_capitalization_weights_reject_an_empty_set() -> None:
    with pytest.raises(OptimizationError):
        capitalization_weights(pd.Series(dtype=float))
