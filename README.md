# mean-variance-optimizer

A fully local, deterministic Python implementation of Harry Markowitz's
Modern Portfolio Theory. Give it a list of tickers and a date range, and it
fetches free historical prices, computes annualized risk/return statistics, and
solves for the mathematically optimal long-only asset allocation that maximizes
the Sharpe ratio or minimizes volatility.

Everything runs natively on Windows: no cloud services, no API keys, no Docker,
no WSL, and no database server. The data source (`yfinance`) is free and
key-less, the store is a single local SQLite file, and the chart is a
self-contained HTML file you open by double-clicking.

![Efficient Frontier](docs/efficient_frontier.png)

*Monte Carlo portfolio cloud coloured by Sharpe ratio, with each asset plotted
individually and the optimal Max-Sharpe (red star) and Min-Volatility (blue
star) portfolios highlighted.*

## Features

- **Deterministic Markowitz optimization** — classical convex optimization via
  `scipy.optimize` (SLSQP), with a fully invested constraint (`sum(w) = 1`) and
  long-only bounds (`0 <= w <= 1`). No machine learning, no heuristics.
- **Two objectives** — maximize the Sharpe ratio or minimize portfolio variance;
  both are computed on every run and the chosen one is reported and persisted.
- **Efficient frontier visualization** — ~5,000 random long-only portfolios
  sampled from the probability simplex, exported as a standalone interactive
  Plotly HTML file with the Plotly JS bundle inlined (works offline).
- **Robust data handling** — mismatched trading calendars are aligned,
  internal gaps forward-filled, and leading gaps truncated to the common
  overlap with a warning.
- **Real-world edge cases handled** — zero-variance assets are detected and
  excluded, single-asset portfolios bypass the solver, and optimization
  failures raise a clear, custom error.
- **Durable local history** — every run is appended to
  `data/portfolio_history.db` (SQLite), including tickers, date range,
  objective, JSON weights, expected return, volatility, and Sharpe ratio.
- **Verified** — a `pytest` suite covers the math, the optimizer constraints,
  data cleaning, persistence, and the full CLI pipeline offline.

## Tech Stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.14 (standard library `sqlite3`, `argparse`) |
| Data | `pandas`, `numpy` |
| Data source | `yfinance` (free, no API key) |
| Optimization | `scipy.optimize` (SLSQP) |
| Visualization | `plotly` (self-contained HTML export) |
| Persistence | SQLite (single local `.db` file) |
| Testing | `pytest` |

## Project Structure

```text
mean_variance_optimizer/
├── data/                     # Local SQLite database (gitignored)
│   └── portfolio_history.db
├── outputs/                  # Generated HTML charts (gitignored)
│   └── efficient_frontier_YYYYMMDD.html
├── docs/                     # README screenshot
│   └── efficient_frontier.png
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py       # Fetch + clean prices, align calendars
│   ├── math_engine.py        # Log returns, annualized metrics, covariance
│   ├── optimizer.py          # scipy SLSQP optimization + Monte Carlo sampler
│   ├── database.py           # Local SQLite persistence
│   └── visualizer.py         # Plotly HTML chart generation
├── tests/
│   ├── test_math_engine.py
│   ├── test_optimizer.py
│   ├── test_data_fetcher.py
│   ├── test_database.py
│   ├── test_visualizer.py
│   └── test_main.py
├── main.py                   # CLI entry point / pipeline orchestration
├── requirements.txt
├── pytest.ini
└── README.md
```

## Installation (native Windows)

```powershell
git clone <repo>
cd mean_variance_optimizer
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

Run the optimizer with a list of tickers and a date range:

```powershell
.\.venv\Scripts\python.exe main.py --tickers AAPL MSFT GOOGL --start 2020-01-01 --end 2024-12-31
```

Run with the minimum-volatility objective:

```powershell
.\.venv\Scripts\python.exe main.py --tickers AAPL MSFT GOOGL AMZN NVDA --objective min_vol
```

Or just run it with no arguments for a sensible default universe:

```powershell
.\.venv\Scripts\python.exe main.py
```

| Flag | Default | Description |
| --- | --- | --- |
| `--tickers` | `AAPL MSFT GOOGL AMZN NVDA` | Ticker symbols to optimize |
| `--start` | `2020-01-01` | Inclusive start date (`YYYY-MM-DD`) |
| `--end` | today | Inclusive end date (`YYYY-MM-DD`) |
| `--objective` | `max_sharpe` | `max_sharpe` or `min_vol` |
| `--risk-free-rate` | `0.04` | Annualized risk-free rate |
| `--db-path` | `data/portfolio_history.db` | SQLite database path |
| `--output-dir` | `outputs` | Directory for the HTML chart |
| `--portfolios` | `5000` | Monte Carlo portfolios to sample |

After a run, double-click the generated
`outputs/efficient_frontier_YYYYMMDD.html` to explore the frontier in your
browser. No web server is required.

## How It Works

1. **Data ingestion** — daily adjusted close prices are downloaded per ticker
   (`auto_adjust=True`, so `Close` is the adjusted series). Trading calendars
   are aligned across assets.
2. **Cleaning** — internal gaps are forward-filled; leading rows where any
   asset has not started trading are dropped, which truncates the range to the
   common overlap and emits a warning.
3. **Financial math** — daily **log returns** `ln(P_t / P_{t-1})` are computed,
   then annualized over a fixed **252-trading-day** year:
   - expected return = mean daily log return x 252
   - covariance = daily covariance x 252
4. **Optimization** — SLSQP minimizes `-(Sharpe)` or portfolio variance subject
   to `sum(w) = 1` and `0 <= w <= 1`, where
   `Sharpe = (E[r] - r_f) / sigma` and `sigma = sqrt(wᵀ Σ w)`.
5. **Persistence** — the run is appended to the local SQLite database.
6. **Visualization** — 5,000 random long-only portfolios (Dirichlet-sampled,
   so weights always sum to 1) are plotted as a cloud coloured by Sharpe ratio,
   alongside the assets and both optimal portfolios.

## Outputs

- `data/portfolio_history.db` — one row per run (`Run_ID`, `Tickers`,
  `Date_Range`, `Objective_Used`, `Optimal_Weights` as JSON, `Expected_Return`,
  `Expected_Volatility`, `Sharpe_Ratio`, `Risk_Free_Rate`, `Created_At`).
- `outputs/efficient_frontier_YYYYMMDD.html` — self-contained interactive chart.

Inspect stored history at any time:

```powershell
.\.venv\Scripts\python.exe -c "from src.database import fetch_runs; [print(r['Run_ID'], r['Objective_Used'], r['weights']) for r in fetch_runs('data/portfolio_history.db')]"
```

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The suite runs entirely offline: the yfinance network boundary is injected, so
no test ever downloads data.

## Robustness & Edge Cases

| Case | Behaviour |
| --- | --- |
| Zero-variance asset (price never moves) | Detected and excluded before optimization, with a warning |
| Asset with insufficient history | Range truncated to the common overlap, with a warning |
| Single-asset portfolio | Solver bypassed; returns weight `1.0` |
| Empty / failed download | Raises `DataFetchError` with a clear message |
| All assets excluded / unusable | Raises `DataFetchError` |
| Singular or indefinite covariance | Raises `OptimizationError` |
| Solver fails to converge | Raises `OptimizationError` |

## Limitations

- Local research tool only: no live trading, broker integration, or order
  execution.
- Unconstrained mean-variance is sensitive to expected-return estimation error,
  so the Max-Sharpe solution can concentrate in a single asset (as it does with
  NVDA in the screenshot). Per-asset weight caps are deliberately out of scope;
  the spec only requires `0 <= w <= 1`.
- Expected returns are estimated from historical means, which is a noisy
  estimator; treat the output as an illustration of the method, not investment
  advice.
- Long-only, fully invested portfolios only (no short selling, no leverage,
  no risk-free allocation).

## Notes

- `data/` and `outputs/` are gitignored; only source, tests, and the README
  screenshot are committed.
- `pytest.ini` sets `pythonpath = .` so `tests/` can import both `src.*` and
  the root `main.py`.
