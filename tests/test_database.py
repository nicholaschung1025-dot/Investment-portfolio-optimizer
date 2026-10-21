"""Tests for the local SQLite persistence layer.

Each test uses a temporary database file, so nothing is written outside the
pytest temporary directory.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from src.database import RunRecord, fetch_runs, init_db, save_run
from src.sizing import Position


@pytest.fixture()
def db_path(tmp_path):
    return tmp_path / "portfolio_history.db"


def _record(objective: str = "max_sharpe") -> RunRecord:
    return RunRecord(
        tickers=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-12-31",
        objective=objective,
        weights={"AAA": 0.7, "BBB": 0.3},
        expected_return=0.11,
        expected_volatility=0.18,
        sharpe_ratio=0.39,
        risk_free_rate=0.04,
    )


def test_init_db_creates_run_table_with_required_columns(db_path) -> None:
    init_db(db_path)

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}

    assert {
        "Run_ID",
        "Tickers",
        "Date_Range",
        "Objective_Used",
        "Optimal_Weights",
        "Expected_Return",
        "Expected_Volatility",
        "Sharpe_Ratio",
    }.issubset(columns)


def test_save_run_returns_incrementing_ids(db_path) -> None:
    init_db(db_path)

    first = save_run(db_path, _record())
    second = save_run(db_path, _record("min_vol"))

    assert first == 1
    assert second == 2


def test_weights_round_trip_as_json(db_path) -> None:
    init_db(db_path)
    save_run(db_path, _record())

    with sqlite3.connect(db_path) as connection:
        raw = connection.execute("SELECT Optimal_Weights FROM runs").fetchone()[0]

    assert json.loads(raw) == {"AAA": 0.7, "BBB": 0.3}


def test_fetch_runs_returns_most_recent_first(db_path) -> None:
    init_db(db_path)
    save_run(db_path, _record("max_sharpe"))
    save_run(db_path, _record("min_vol"))

    runs = fetch_runs(db_path)

    assert len(runs) == 2
    assert runs[0]["objective_used"] == "min_vol"
    assert runs[0]["weights"] == {"AAA": 0.7, "BBB": 0.3}


def test_init_db_is_idempotent(db_path) -> None:
    init_db(db_path)
    init_db(db_path)

    assert fetch_runs(db_path) == []


def test_init_db_adds_the_new_run_columns(db_path) -> None:
    init_db(db_path)

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}

    assert {
        "Cash_Weight",
        "Target_Return",
        "Expected_Return_Method",
        "Covariance_Method",
        "VAR_95",
        "CVAR_95",
        "Prob_Profit",
    }.issubset(columns)


def test_init_db_creates_the_positions_table(db_path) -> None:
    init_db(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }

    assert "positions" in tables


def test_a_run_records_the_cash_weight_and_the_risk_metrics(db_path) -> None:
    record = RunRecord(
        tickers=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-12-31",
        objective="max_sharpe",
        weights={"AAA": 0.7, "BBB": 0.3},
        expected_return=0.11,
        expected_volatility=0.18,
        sharpe_ratio=0.39,
        risk_free_rate=0.04,
        cash_weight=0.25,
        target_return=0.09,
        expected_return_method="black_litterman",
        covariance_method="ledoit-wolf",
        var_95=8_200.0,
        cvar_95=12_900.0,
        prob_profit=0.68,
    )

    save_run(db_path, record)
    runs = fetch_runs(db_path)

    assert runs[0]["cash_weight"] == pytest.approx(0.25)
    assert runs[0]["expected_return_method"] == "black_litterman"
    assert runs[0]["var_95"] == pytest.approx(8_200.0)
    assert runs[0]["prob_profit"] == pytest.approx(0.68)


def test_positions_are_stored_and_read_back(db_path) -> None:
    record = RunRecord(
        tickers=["AAA"],
        start="2024-01-01",
        end="2024-12-31",
        objective="max_sharpe",
        weights={"AAA": 1.0},
        expected_return=0.11,
        expected_volatility=0.18,
        sharpe_ratio=0.39,
        risk_free_rate=0.04,
        positions=[
            Position(
                ticker="AAA",
                target_weight=1.0,
                shares=10,
                board_lot=1,
                price=100.0,
                currency="USD",
                value_usd=1_000.0,
                realized_weight=1.0,
                drift=0.0,
            )
        ],
    )

    run_id = save_run(db_path, record)

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = list(connection.execute("SELECT * FROM positions WHERE Run_ID = ?", (run_id,)))

    assert len(rows) == 1
    assert rows[0]["Ticker"] == "AAA"
    assert rows[0]["Shares"] == 10
    assert rows[0]["Currency"] == "USD"


def test_an_existing_database_is_migrated_in_place(db_path) -> None:
    # A database written by an earlier version has the original columns only.
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE runs (
                Run_ID INTEGER PRIMARY KEY AUTOINCREMENT,
                Tickers TEXT NOT NULL,
                Date_Range TEXT NOT NULL,
                Objective_Used TEXT NOT NULL,
                Optimal_Weights TEXT NOT NULL,
                Expected_Return REAL NOT NULL,
                Expected_Volatility REAL NOT NULL,
                Sharpe_Ratio REAL NOT NULL,
                Risk_Free_Rate REAL NOT NULL,
                Created_At TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )

    init_db(db_path)
    save_run(db_path, _record())

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}

    assert "Cash_Weight" in columns
    assert fetch_runs(db_path)[0]["tickers"] == ["AAA", "BBB"]
