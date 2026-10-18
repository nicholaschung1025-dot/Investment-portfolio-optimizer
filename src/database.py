"""Local, file-based persistence using Python's built-in ``sqlite3``.

SQLite is a single local ``.db`` file, so it satisfies the project's
"no database server, no VM, no cloud" constraint while still giving durable,
queryable history of every optimization run.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

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
    Created_At TEXT NOT NULL DEFAULT (datetime('now'))
)
"""


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


def init_db(db_path: str | Path) -> None:
    """Create the database file and ``runs`` table if they do not exist.

    Args:
        db_path: Path to the local SQLite file.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(_CREATE_TABLE)


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
                Expected_Return, Expected_Volatility, Sharpe_Ratio, Risk_Free_Rate
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        return int(cursor.lastrowid)


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
        rows = connection.execute(
            "SELECT * FROM runs ORDER BY Run_ID DESC LIMIT ?", (limit,)
        ).fetchall()

    runs: list[dict[str, object]] = []
    for row in rows:
        record = dict(row)
        record["tickers"] = json.loads(record["Tickers"])  # type: ignore[arg-type]
        record["weights"] = json.loads(record["Optimal_Weights"])  # type: ignore[arg-type]
        record["objective_used"] = record["Objective_Used"]
        runs.append(record)
    return runs
