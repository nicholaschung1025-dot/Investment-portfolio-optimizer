"""Local, file-based persistence using Python's built-in ``sqlite3``.

SQLite is a single local ``.db`` file, so it satisfies the project's
"no database server, no VM, no cloud" constraint while still giving durable,
queryable history of every optimization run.

The schema is versioned and migrated in place: a database written by an
earlier release gains the new columns instead of being abandoned, so the
history stays readable across upgrades.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.sizing import Position

SCHEMA_VERSION = 2

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    Run_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Tickers TEXT NOT NULL,
    Date_Range TEXT NOT NULL,
    Objective_Used TEXT NOT NULL,
    Optimal_Weights TEXT NOT NULL,
    Expected_Return REAL NOT NULL,
    Expected_Volatility REAL NOT NULL,
    Sharpe_Ratio REAL NOT NULL,
    Risk_Free_Rate REAL NOT NULL,
    Cash_Weight REAL,
    Target_Return REAL,
    Expected_Return_Method TEXT,
    Covariance_Method TEXT,
    VAR_95 REAL,
    CVAR_95 REAL,
    Prob_Profit REAL,
    Estimated_Cost REAL,
    Created_At TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_CREATE_POSITIONS_TABLE = """
CREATE TABLE IF NOT EXISTS positions (
    Run_ID INTEGER NOT NULL,
    Ticker TEXT NOT NULL,
    Target_Weight REAL NOT NULL,
    Shares INTEGER NOT NULL,
    Board_Lot INTEGER NOT NULL,
    Price REAL NOT NULL,
    Currency TEXT NOT NULL,
    Value_USD REAL NOT NULL,
    Realized_Weight REAL NOT NULL,
    PRIMARY KEY (Run_ID, Ticker)
)
"""

# Columns added after the first release. They are applied to an existing
# database with ALTER TABLE, which SQLite cannot do inside a CREATE TABLE
# statement, so a database that already exists keeps its history.
_ADDED_COLUMNS: dict[str, str] = {
    "Cash_Weight": "REAL",
    "Target_Return": "REAL",
    "Expected_Return_Method": "TEXT",
    "Covariance_Method": "TEXT",
    "VAR_95": "REAL",
    "CVAR_95": "REAL",
    "Prob_Profit": "REAL",
    "Estimated_Cost": "REAL",
}


@dataclass
class RunRecord:
    """A single optimization run to persist.

    Attributes:
        tickers: Tickers in the optimized universe.
        start: Requested start date (``YYYY-MM-DD``).
        end: Requested end date (``YYYY-MM-DD``).
        objective: Objective used, ``"max_sharpe"`` or ``"min_vol"``.
        weights: Optimal weight per ticker.
        expected_return: Annualized expected portfolio return.
        expected_volatility: Annualized portfolio volatility.
        sharpe_ratio: Annualized Sharpe ratio.
        risk_free_rate: Annualized risk-free rate used.
        cash_weight: Share of the portfolio held in the risk-free asset.
        target_return: Return the portfolio was asked to reach.
        expected_return_method: Estimator used for the expected returns.
        covariance_method: Estimator used for the covariance matrix.
        var_95: Value at risk at the 95% level.
        cvar_95: Expected loss in the worst 5% of outcomes.
        prob_profit: Probability of ending above the starting budget.
        estimated_cost: Estimated cost of putting the portfolio into effect.
        positions: The tradable positions, when the run produced any.
    """

    tickers: list[str]
    start: str
    end: str
    objective: str
    weights: dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    risk_free_rate: float
    cash_weight: float | None = None
    target_return: float | None = None
    expected_return_method: str | None = None
    covariance_method: str | None = None
    var_95: float | None = None
    cvar_95: float | None = None
    prob_profit: float | None = None
    estimated_cost: float | None = None
    positions: list["Position"] = field(default_factory=list)


def _migrate(connection: sqlite3.Connection) -> None:
    """Bring an existing database up to the current schema."""
    existing = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
    for column, column_type in _ADDED_COLUMNS.items():
        if column not in existing:
            connection.execute(f"ALTER TABLE runs ADD COLUMN {column} {column_type}")
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def init_db(db_path: str | Path) -> None:
    """Create the database file and its tables if they do not exist.

    Args:
        db_path: Path to the local SQLite file.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(_CREATE_TABLE)
        connection.execute(_CREATE_POSITIONS_TABLE)
        _migrate(connection)


def save_run(db_path: str | Path, record: RunRecord) -> int:
    """Persist one optimization run.

    Args:
        db_path: Path to the local SQLite file.
        record: The run to store.

    Returns:
        The autoincremented ``Run_ID`` assigned to the row.
    """
    init_db(db_path)
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO runs (
                Tickers, Date_Range, Objective_Used, Optimal_Weights,
                Expected_Return, Expected_Volatility, Sharpe_Ratio, Risk_Free_Rate,
                Cash_Weight, Target_Return, Expected_Return_Method,
                Covariance_Method, VAR_95, CVAR_95, Prob_Profit, Estimated_Cost
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                json.dumps(record.tickers),
                f"{record.start} to {record.end}",
                record.objective,
                json.dumps(record.weights),
                record.expected_return,
                record.expected_volatility,
                record.sharpe_ratio,
                record.risk_free_rate,
                record.cash_weight,
                record.target_return,
                record.expected_return_method,
                record.covariance_method,
                record.var_95,
                record.cvar_95,
                record.prob_profit,
                record.estimated_cost,
            ),
        )
        run_id = int(cursor.lastrowid)

        connection.executemany(
            """
            INSERT OR REPLACE INTO positions (
                Run_ID, Ticker, Target_Weight, Shares, Board_Lot,
                Price, Currency, Value_USD, Realized_Weight
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    position.ticker,
                    position.target_weight,
                    position.shares,
                    position.board_lot,
                    position.price,
                    position.currency,
                    position.value_usd,
                    position.realized_weight,
                )
                for position in record.positions
            ],
        )
        return run_id


def fetch_runs(db_path: str | Path, limit: int = 20) -> list[dict[str, object]]:
    """Read recent runs, most recent first.

    Args:
        db_path: Path to the local SQLite file.
        limit: Maximum number of rows to return.

    Returns:
        List of run dictionaries with parsed JSON fields.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY Run_ID DESC LIMIT ?", (limit,)
            ).fetchall()
        except sqlite3.OperationalError:
            return []

    runs: list[dict[str, object]] = []
    for row in rows:
        record = dict(row)
        record["tickers"] = json.loads(record["Tickers"])  # type: ignore[arg-type]
        record["weights"] = json.loads(record["Optimal_Weights"])  # type: ignore[arg-type]
        record["objective_used"] = record["Objective_Used"]
        record["cash_weight"] = record.get("Cash_Weight")
        record["target_return"] = record.get("Target_Return")
        record["expected_return_method"] = record.get("Expected_Return_Method")
        record["covariance_method"] = record.get("Covariance_Method")
        record["var_95"] = record.get("VAR_95")
        record["cvar_95"] = record.get("CVAR_95")
        record["prob_profit"] = record.get("Prob_Profit")
        record["estimated_cost"] = record.get("Estimated_Cost")
        runs.append(record)
    return runs


def fetch_positions(db_path: str | Path, run_id: int) -> list[dict[str, object]]:
    """Read the positions of one run.

    Args:
        db_path: Path to the local SQLite file.
        run_id: Identifier of the run.

    Returns:
        One dictionary per position.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT * FROM positions WHERE Run_ID = ? ORDER BY Ticker", (run_id,)
            ).fetchall()
        except sqlite3.OperationalError:
            # A database written before the positions table existed has no
            # positions to show; the table is created on the next run.
            return []
    return [dict(row) for row in rows]
