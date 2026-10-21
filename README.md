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
  long-only bounds (`0 <= w <= 1`). Both objectives are solved with an analytic
  gradient, which is faster and more reproducible than numerical differences.
  No machine learning, no heuristics.
- **Robust covariance estimation** — Ledoit-Wolf and OAS shrinkage estimators
  pull the noisy sample covariance towards a scaled identity matrix. This is
  the single biggest defence against the estimation error that makes plain
  mean-variance allocate everything to one asset.
- **Interchangeable expected-return estimators** — the sample mean of log
  returns (default) or the capital asset pricing model with betas estimated
  against a market index. A Black-Litterman posterior is available in the
  library for when market-capitalization data is at hand.
- **Portfolio constraints** — minimum weights (how a mandatory core holding is
  expressed), a maximum weight per asset, and a maximum portfolio volatility.
  Infeasible requests are refused with a clear message instead of an obscure
  solver error.
- **Capital allocation to a target return** — scale the risky portfolio against
  the risk-free asset so the combined portfolio reaches a requested return. A
  target the tangency portfolio cannot reach is an error, not a label.
- **Out-of-sample validation** — a walk-forward backtest re-estimates the
  portfolio on a rolling window and reports its performance next to an
  equal-weight benchmark over the same periods.
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
| Optimization | `scipy.optimize` (SLSQP, analytic gradients) |
| Covariance | `scikit-learn` (Ledoit-Wolf, OAS shrinkage) |
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
│   ├── errors.py             # Shared exception hierarchy
│   ├── data_fetcher.py       # Fetch + clean prices, align calendars
│   ├── math_engine.py        # Log returns, annualized metrics, covariance
│   ├── covariance.py         # Sample / Ledoit-Wolf / OAS covariance
│   ├── expected_returns.py   # Historical mean, CAPM, Black-Litterman
│   ├── constraints.py        # Minimum weights, weight and volatility caps
│   ├── optimizer.py          # scipy SLSQP optimization + Monte Carlo sampler
│   ├── allocation.py         # Capital allocation line to a target return
│   ├── validation.py         # Walk-forward backtest vs equal weight
│   ├── database.py           # Local SQLite persistence
│   └── visualizer.py         # Plotly HTML chart generation
├── tests/
│   ├── test_math_engine.py
│   ├── test_optimizer.py
│   ├── test_covariance.py
│   ├── test_expected_returns.py
│   ├── test_constraints.py
│   ├── test_allocation.py
│   ├── test_validation.py
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

Force a core holding into the portfolio and cap single-asset concentration:

```powershell
.\.venv\Scripts\python.exe main.py --min-weight AAPL=0.25 --min-weight MSFT=0.15 --max-weight 0.30
```

Target a return and hold the rest in the risk-free asset:

```powershell
.\.venv\Scripts\python.exe main.py --target-return 0.12
```

Check the result out of sample against an equal-weight benchmark:

```powershell
.\.venv\Scripts\python.exe main.py --validate --validate-rebalance monthly
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
| `--expected-return-method` | `historical_mean` | `historical_mean` or `capm` |
| `--covariance-method` | `ledoit-wolf` | `sample`, `ledoit-wolf` or `oas` |
| `--market-index` | `^GSPC` | Market index used by the CAPM estimator |
| `--min-weight` | – | Minimum weight per ticker, e.g. `AAPL=0.25`; repeatable |
| `--max-weight` | – | Maximum weight for any single ticker, e.g. `0.35` |
| `--max-volatility` | – | Maximum annualized portfolio volatility, e.g. `0.15` |
| `--target-return` | – | Annualized return to reach; the rest is held in cash |
| `--validate` | off | Run the out-of-sample walk-forward backtest |
| `--validate-window` | `252` | Observations required before the first rebalance |
| `--validate-rebalance` | `monthly` | `monthly` or `quarterly` |

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
   - covariance = shrinkage estimator applied to the daily covariance, x 252
4. **Optimization** — SLSQP minimizes `-(Sharpe)` or portfolio variance subject
   to `sum(w) = 1` and `0 <= w <= 1`, where
   `Sharpe = (E[r] - r_f) / sigma` and `sigma = sqrt(wᵀ Σ w)`. Minimum weights
   are substituted out (`w_i = m_i + u_i`) so the solver always works on
   interior variables, and a volatility cap is passed as a nonlinear
   constraint.
5. **Capital allocation** — when a target return is requested, the tangency
   portfolio is scaled against the risk-free asset along the capital
   allocation line.
6. **Persistence** — the run is appended to the local SQLite database.
7. **Visualization** — 5,000 random long-only portfolios (Dirichlet-sampled,
   so weights always sum to 1) are plotted as a cloud coloured by Sharpe ratio,
   alongside the assets and both optimal portfolios.
8. **Validation** (optional) — the portfolio is re-estimated on a rolling
   window and applied to the period that follows, then compared with an
   equal-weight benchmark over the same periods.

### Expected-Return Estimators

```
Historical mean : mu_i = mean(daily log return) x 252

CAPM            : mu_i = r_f + beta_i x (E[R_m] - r_f)
                  beta_i = cov(r_i, R_m) / var(R_m)

Black-Litterman : prior  pi = r_f + delta x Sigma x w_market
                  posterior = [(tau Sigma)^-1 + P' Omega^-1 P]^-1
                              [(tau Sigma)^-1 pi + P' Omega^-1 Q]
```

A view confidence `theta` in `(0, 1)` controls how far the posterior moves
towards the view: `0.5` splits the difference evenly, `0.95` moves it 95% of
the way, and `theta -> 0` leaves the prior untouched.

### Covariance Estimation

```python
# Ledoit-Wolf shrinkage for numerical stability
Sigma = shrunk_covariance(log_returns, method="ledoit-wolf")  # x 252

# Positive-definiteness guard
ensure_positive_semidefinite(Sigma)
```

The shrinkage estimators are fitted on **daily** returns, which is the
frequency they expect, and the result is annualized afterwards.

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
| Minimum weights that cannot be invested | Raises `ConstraintError` |
| Maximum weight too small to invest fully | Raises `ConstraintError` |
| Volatility cap that cannot be met | Raises `OptimizationError` |
| Target return above the tangency return | Raises `AllocationError` |
| Target return below the risk-free rate | Raises `AllocationError` |
| Not enough history for a validation window | Raises `OptimizationError` |

## Limitations

- Local research tool only: no live trading, broker integration, or order
  execution.
- Unconstrained mean-variance is still sensitive to expected-return estimation
  error. Shrinkage, weight caps and a risk-free sleeve reduce the damage, but
  the out-of-sample comparison against an equal-weight benchmark is the only
  honest way to judge a given universe — and the optimized portfolio does not
  win that comparison on every dataset.
- Expected returns are estimated from historical means, which is a noisy
  estimator; treat the output as an illustration of the method, not investment
  advice.
- Long-only portfolios. Leverage, short selling and derivative overlays are out
  of scope.
- The Black-Litterman estimator is available in the library
  (`src.expected_returns.black_litterman_returns`) but is not yet exposed on
  the command line: it needs market-capitalization weights, which are not
  fetched yet.
- The persisted run record does not yet store the cash weight, the estimation
  methods or the validation result.
- Prices come from a free, unofficial data source and can be wrong.

## Notes

- `data/` and `outputs/` are gitignored; only source, tests, and the README
  screenshot are committed.
- `pytest.ini` sets `pythonpath = .` so `tests/` can import both `src.*` and
  the root `main.py`.
- The default covariance estimator is `ledoit-wolf`, which changes the
  weights compared with an earlier plain sample-covariance release. Pass
  `--covariance-method sample` to reproduce the old behaviour.
