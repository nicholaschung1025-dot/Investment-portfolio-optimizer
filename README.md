# Investment-portfolio-optimizer

A portfolio optimization engine that combines modern portfolio theory with practical trading constraints. This tool selects optimal stock allocations while **forcing inclusion of core holdings**, applying quality filters, and providing comprehensive risk analytics.

---

## 🚀 Key Features

### 🔒 Fixed Stocks Support
- **Mandatory Holdings**: Specify core stocks that must be included in every portfolio (e.g., `AAPL`, `MSFT`, `0700.HK`)
- **Flexible Allocation**: Optimizer selects remaining slots from the universe up to `MAX_K` total stocks
- **Validation**: Automatic checks ensure fixed stocks exist in universe and don't exceed capacity

### 📊 Advanced Optimization Engine
- **Tangency Portfolio**: Maximizes Sharpe ratio using convex optimization (CVXPY + OSQP solver)
- **Forward Stepwise Selection**: Greedy algorithm to select top-K stocks with fixed holdings pre-included
- **Ledoit-Wolf Covariance**: Robust shrinkage estimator for stable covariance matrices
- **Student-t Monte Carlo**: Heavy-tailed simulations for realistic risk metrics (VaR, CVaR)

### 🎯 Intelligent Stock Filtering
```python
# Trend Filters (lenient for long-term value)
MIN_PRICE_VS_MA50 = -0.15      # Price can be 15% below 50-day MA
MIN_MOMENTUM_3M = -0.50        # Allow significant short-term drawdowns

# Quality Filters (exclude small/unstable companies)
MIN_ANALYST_COUNT = 10         # Minimum analyst coverage
MIN_MARKET_CAP_USD = 10e9      # $10B minimum for US stocks
MAX_ANNUAL_VOLATILITY = 0.50   # Exclude extremely volatile stocks
```

### 🌍 Multi-Market Support
- **US & HK Stocks**: Seamless handling of USD and HKD denominated securities
- **Board Lot Compliance**: HK stock quantities rounded to exchange-mandated lot sizes
- **Currency Conversion**: Automatic USD/HKD conversion using configurable exchange rate

### ⚡ Performance Optimizations
- **Single API Call per Stock**: Fetches all metadata (price, dividend, analyst targets) in one yfinance request
- **Cached Data Reuse**: Dividend yields, analyst targets, and prices extracted from initial fetch
- **Memory Management**: Explicit garbage collection for large datasets

---

## 📦 Installation

### Requirements
```bash
pip install yfinance pandas numpy cvxpy scikit-learn scipy
```

### Optional: Virtual Environment
```bash
python -m venv portfolio-env
source portfolio-env/bin/activate  # Windows: portfolio-env\Scripts\activate
pip install -r requirements.txt
```

---

## ⚙️ Configuration

### Fixed Stocks Setup
```python
class FixedStocksConfig:
    FIXED_STOCKS = [
        "AAPL",      # Apple (Core holding)
        "MSFT",      # Microsoft (Core holding)
        "0700.HK",   # Tencent (HK core holding)
    ]
    MAX_K = 6  # Total portfolio size
```

### Risk & Return Parameters
```python
risk_free_rate = 0.04      # 4% annual risk-free rate
budget = 100000            # $100,000 USD portfolio
target_return = 0.12       # 12% target annual return (optional)
```

### Filter Thresholds (Edit `QualityFilterConfig` & `TrendFilterConfig`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_ANALYST_COUNT` | 10 | Minimum analyst opinions |
| `MIN_MARKET_CAP_USD` | $10B | Minimum US market cap |
| `MIN_MARKET_CAP_HKD` | $80B | Minimum HK market cap |
| `MAX_ANNUAL_VOLATILITY` | 50% | Max acceptable volatility |
| `MIN_AVG_VOLUME` | 5M | Minimum daily volume |

---

## ▶️ Usage

### Basic Execution
```bash
python portfolio_optimizer.py
```

### Custom Configuration (Edit `main()`)
```python
def main():
    # Modify fixed stocks
    FixedStocksConfig.FIXED_STOCKS = ["NVDA", "0700.HK"]
    FixedStocksConfig.MAX_K = 8
    
    # Adjust budget/targets
    budget = 250000
    target_return = 0.15
    
    # Run optimization
    main()  # Call as normal
```

### Output Example
```
======================================================================
FINAL PORTFOLIO
======================================================================
Strategy: CML (23% RF)
Return: 12.00% | Vol: 18.42% | Sharpe: 0.4347
Risk-Free: 23% | Active Stocks: 5
----------------------------------------------------------------------
Stock           Weight     Shares      Lot   Beta          Price     Value(USD)    Fixed
----------------------------------------------------------------------
AAPL             18.50%         45        1    1.24      $178.32 USD    $18,456.32      Yes
MSFT             22.10%         52        1    0.98      $412.15 USD    $22,089.40      Yes
0700.HK          15.30%        300      100    1.15   $385.20 HKD    $14,712.18      Yes
NVDA             24.80%         28        1    1.67      $862.45 USD    $24,148.60       No
JPM              19.30%        125        1    1.05      $151.23 USD    $18,903.75       No
----------------------------------------------------------------------
Total Stocks: $98,310.25 USD | Cash (RF): $1,689.75 USD

======================================================================
RISK METRICS
======================================================================
Median Value: $112,450.30 | Profit Prob: 68%
VaR (95%): $8,234.50 | CVaR (95%): $12,891.20
```

---

## 🧠 Technical Details

### Expected Returns Model
```
Blended Return = 0.7 × CAPM Return + 0.3 × Analyst Implied Return

CAPM Return = Rf + β × (Market Risk Premium)
Analyst Return = (Target Price / Current Price) - 1

+ After-tax dividend yield adjustment (10% HK / 15% US withholding)
```

### Optimization Workflow
1. **Data Fetch**: Single yfinance call per stock for all metadata
2. **Quality Filtering**: Apply analyst coverage, market cap, volatility filters
3. **CAPM Estimation**: Calculate betas vs. S&P 500 (^GSPC)
4. **Forward Selection**: Iteratively add stocks maximizing Sharpe ratio
5. **Tangency Solve**: Convex optimization for optimal risky portfolio
6. **Capital Allocation**: Blend with risk-free asset to hit target return
7. **Board Lot Adjustment**: Round HK positions to exchange lot sizes
8. **Risk Simulation**: 10,000 Student-t Monte Carlo paths for VaR/CVaR

### Covariance Estimation
```python
# Ledoit-Wolf shrinkage for numerical stability
Sigma = LedoitWolf().fit(returns[filtered_stocks].dropna().values).covariance_ * 252

# Eigenvalue correction for positive definiteness
min_eig = np.min(np.linalg.eigvalsh(Sigma))
if min_eig < 1e-8:
    Sigma = Sigma + (1e-6 - min_eig) * np.eye(n)
```

---

## 📈 Output Metrics Explained

| Metric | Description |
|--------|-------------|
| **Weight** | Portfolio allocation percentage (excludes cash) |
| **Shares** | Integer share count (HK stocks rounded to board lots) |
| **Beta** | Systematic risk vs. S&P 500 |
| **Value(USD)** | Position value converted to USD |
| **Fixed** | Whether stock was mandated in configuration |
| **Sharpe Ratio** | (Return - Risk-Free) / Volatility |
| **VaR (95%)** | Maximum expected loss at 95% confidence |
| **CVaR (95%)** | Expected loss in worst 5% of scenarios |
| **Profit Prob** | Probability portfolio exceeds initial budget |

---

## 🔧 Troubleshooting

### Common Issues
```
⚠️  [STOCK]: No analyst target (count=0)
→ Normal for small-cap stocks; falls back to CAPM-only returns

⚠️  Optimization status: unbounded, using equal weights
→ Check for NaN/inf in covariance matrix; Ledoit-Wolf should prevent this

❌  Fixed stock 'XXX' not in stock universe
→ Verify ticker symbol matches yfinance format (e.g., '0700.HK' not '0700')
```

### API Rate Limiting
The script includes `time.sleep(0.15)` between yfinance calls. If you encounter rate limits:
```python
# Increase delay in fetch_all_stock_info()
time.sleep(0.30)  # Conservative for bulk downloads
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 conventions
- Add docstrings to new functions
- Include type hints for function signatures
- Test with both US and HK stock universes

---

## ⚠️ Disclaimer

> This tool is for **educational and research purposes only**. 
> - Past performance does not guarantee future results
> - Optimization outputs are not financial advice
> - Always conduct independent due diligence before investing
> - Author assumes no liability for investment decisions based on this software

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024 | Initial release with fixed stocks support |
| 1.1.0 | 2024 | Added Student-t Monte Carlo, board lot handling |
| 1.2.0 | 2024 | Optimized API calls, enhanced quality filters |

---

*Built with ❤️ for quantitative investors who value both theory and practical constraints.* 🎯📊
