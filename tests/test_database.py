"""Tests for the local SQLite persistence layer.

Each test uses a temporary database file, so nothing is written outside the
pytest temporary directory.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from src.database import RunRecord, fetch_runs, init_db, save_run


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
