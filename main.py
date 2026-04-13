#!/usr/bin/env python3
"""
Portfolio Optimization - Production Version (WITH FIXED STOCKS)
================================================================
"""
import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import gc
import time
from typing import Dict, Optional, List, Tuple
from sklearn.covariance import LedoitWolf
from scipy.stats import t as t_dist

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
HK_BOARD_LOTS = {
    '0700.HK': 100, '9988.HK': 100, '0005.HK': 500, '0941.HK': 500,
    '1299.HK': 500, '0001.HK': 500, '0388.HK': 100, '0002.HK': 500,
    '0003.HK': 500, '0011.HK': 500, '0016.HK': 500, '0027.HK': 500,
    '1810.HK': 200, '3690.HK': 100, '9618.HK': 50, '9888.HK': 100,
    '9999.HK': 100,
}
DEFAULT_HK_BOARD_LOT = 100
USD_HKD_RATE = 7.82

# ============================================================================
#  FIXED STOCKS CONFIGURATION
# ============================================================================
class FixedStocksConfig:
    """Configuration for fixed stocks that must be included"""
    # List of stocks that MUST be in the portfolio
    FIXED_STOCKS = [
        "AAPL",      # Apple (Core holding)
        "MSFT",      # Microsoft (Core holding)
        "0700.HK",   # Tencent (HK core holding)
    ]
    
    # Maximum number of stocks in portfolio
    MAX_K = 6
    
    # Validation: Fixed stocks count must be <= MAX_K
    @classmethod
    def validate(cls):
        if len(cls.FIXED_STOCKS) > cls.MAX_K:
            raise ValueError(
                f"Fixed stocks count ({len(cls.FIXED_STOCKS)}) cannot exceed MAX_K ({cls.MAX_K})"
            )
        return True


# ============================================================================
#  TREND FILTER CONFIGURATION
# ============================================================================
class TrendFilterConfig:
    """Lenient Trend Filters for Long-Term Value & Optimization"""
    MIN_PRICE_VS_MA50 = -0.15
    MIN_PRICE_VS_MA200 = -0.20
    MIN_MOMENTUM_3M = -0.50
    MIN_MOMENTUM_6M = -0.50
    MIN_RSI_14 = 0
    MAX_RSI_14 = 85


# ============================================================================
#  QUALITY FILTER CONFIGURATION
# ============================================================================
class QualityFilterConfig:
    """Quality filters to exclude small/unstable companies"""
    MIN_ANALYST_COUNT = 10
    MAX_CAPM_DEVIATION = 0.30
    MIN_MARKET_CAP_USD = 10e9
    MIN_MARKET_CAP_HKD = 80e9
    MIN_STOCK_PRICE = 10.0
    MAX_ANNUAL_VOLATILITY = 0.50
    MIN_AVG_VOLUME = 5000000
    MIN_STOCK_RETURN = -0.20
    MAX_STOCK_RETURN = 0.50


# ============================================================================
#  OPTIMIZED: SINGLE API CALL PER STOCK
# ============================================================================
def fetch_all_stock_info(stocks: List[str]) -> Dict[str, Dict]:
    """Fetch ALL stock info in ONE pass per stock"""
    print(f"\n📊 Fetching stock information (optimized - 1 call per stock)...")
    stock_info = {}
    success_count = 0
    
    for i, stock in enumerate(stocks):
        try:
            ticker = yf.Ticker(stock)
            info = ticker.info
            
            stock_info[stock] = {
                'current_price': info.get('currentPrice') or info.get('previousClose') or info.get('regularMarketPrice', 0),
                'dividend_yield': info.get('trailingAnnualDividendYield', 0),
                'dividend_rate': info.get('trailingAnnualDividendRate', 0),
                'analyst_target': info.get('targetMeanPrice', None),
                'analyst_count': info.get('numberOfAnalystOpinions', 0),
                'market_cap': info.get('marketCap', 0),
                'avg_volume': info.get('averageVolume', 0),
                'profit_margin': info.get('profitMargins', 0),
                'earnings_eps': info.get('trailingEps', 0),
                'raw_info': info
            }
            
            if stock_info[stock]['dividend_yield'] == 0 or stock_info[stock]['dividend_yield'] is None:
                if stock_info[stock]['dividend_rate'] > 0 and stock_info[stock]['current_price'] > 0:
                    stock_info[stock]['dividend_yield'] = stock_info[stock]['dividend_rate'] / stock_info[stock]['current_price']
            
            success_count += 1
            
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(stocks)} stocks...")
            
            time.sleep(0.15)
            
        except Exception as e:
            print(f"  ⚠️  {stock}: {type(e).__name__}: {e}")
            stock_info[stock] = {
                'current_price': 0, 'dividend_yield': 0, 'dividend_rate': 0,
                'analyst_target': None, 'analyst_count': 0, 'market_cap': 0,
                'avg_volume': 0, 'profit_margin': 0, 'earnings_eps': 0, 'raw_info': {}
            }
    
    print(f"\n✓ Stock info fetched: {success_count}/{len(stocks)} stocks")
    print(f"  API calls reduced: {len(stocks) * 3} → {len(stocks)} ({100 - 100/3:.0%} reduction)")
    return stock_info


def get_dividend_yield_from_cache(stock_info: Dict[str, Dict], stocks: List[str]) -> Dict[str, float]:
    """Get dividend yields from cached info (NO API calls)"""
    print(f"\n📊 Extracting dividend yields from cache...")
    dividend_yields = {}
    for stock in stocks:
        info = stock_info.get(stock, {})
        dividend_yields[stock] = info.get('dividend_yield', 0.0) or 0.0
    print(f"  ✓ {len(dividend_yields)} dividend yields extracted")
    return dividend_yields


def fetch_analyst_targets_from_cache(stock_info: Dict[str, Dict], stocks: List[str]) -> Dict[str, Optional[float]]:
    """Get analyst targets from cached info (NO API calls)"""
    print(f"\n📊 Extracting analyst targets from cache...")
    analyst_returns = {}
    missing_count = 0
    
    for stock in stocks:
        info = stock_info.get(stock, {})
        analyst_target = info.get('analyst_target', None)
        current_price = info.get('current_price', 0)
        analyst_count = info.get('analyst_count', 0)
        
        if analyst_target and current_price and current_price > 0:
            implied_return = (analyst_target / current_price) - 1
            analyst_returns[stock] = implied_return
            print(f"  ✓ {stock}: Target=${analyst_target:.2f}, Implied={implied_return:.0%}")
        else:
            analyst_returns[stock] = None
            missing_count += 1
            print(f"  ⚠️  {stock}: No analyst target (count={analyst_count})")
    
    print(f"\n✓ Analyst targets: {len(stocks) - missing_count}/{len(stocks)} stocks")
    return analyst_returns


def get_current_prices_from_cache(stock_info: Dict[str, Dict], stocks: List[str]) -> Dict[str, float]:
    """Get current prices from cached info (NO API calls)"""
    print(f"\n📊 Extracting current prices from cache...")
    prices = {}
    for stock in stocks:
        info = stock_info.get(stock, {})
        price = info.get('current_price', 0)
        if price and price > 0:
            prices[stock] = float(price)
        else:
            try:
                data = yf.download(stock, period='5d', progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    close_col = 'Close' if 'Close' in data.columns.get_level_values(0) else data.columns[0][0]
                    price = data[close_col].iloc[-1]
                else:
                    close_col = 'Close' if 'Close' in data.columns else data.columns[0]
                    price = data[close_col].iloc[-1]
                prices[stock] = float(price)
                print(f"  ✓ {stock}: ${price:.2f} (fallback)")
            except:
                prices[stock] = 0.0
                print(f"  ✗ {stock}: Price unavailable")
    print(f"  ✓ {len(prices)} prices extracted")
    return prices


# ============================================================================
#  FIXED STOCKS VALIDATION
# ============================================================================
def validate_fixed_stocks(fixed_stocks: List[str], all_stocks: List[str], max_k: int) -> Tuple[bool, List[str]]:
    """
    Validate fixed stocks configuration
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Check count
    if len(fixed_stocks) > max_k:
        errors.append(f"❌ Fixed stocks count ({len(fixed_stocks)}) exceeds MAX_K ({max_k})")
    
    # Check if fixed stocks exist in universe
    for stock in fixed_stocks:
        if stock not in all_stocks:
            errors.append(f"❌ Fixed stock '{stock}' not in stock universe")
    
    # Check for duplicates
    if len(fixed_stocks) != len(set(fixed_stocks)):
        errors.append(f"❌ Duplicate stocks in fixed stocks list")
    
    is_valid = len(errors) == 0
    
    if is_valid:
        print(f"\n✅ Fixed Stocks Validation: PASSED")
        print(f"   Fixed Stocks: {len(fixed_stocks)}/{max_k} slots used")
        print(f"   Remaining slots for optimization: {max_k - len(fixed_stocks)}")
    else:
        print(f"\n❌ Fixed Stocks Validation: FAILED")
        for error in errors:
            print(f"   {error}")
    
    return is_valid, errors


# ============================================================================
#  OPTIMIZATION WITH FIXED STOCKS
# ============================================================================
def solve_tangency_sharpe_with_fixed(mu: np.ndarray, Sigma: np.ndarray, risk_free: float,
                                      fixed_indices: List[int]) -> Tuple[np.ndarray, float, float, float]:
    """
    Solve Tangency Portfolio WITH Fixed Stocks
    
    Fixed stocks are强制 included, optimizer selects remaining stocks
    """
    n = len(mu)
    
    if Sigma.shape != (n, n):
        Sigma = np.eye(n) * 0.04
    
    if np.any(np.isnan(Sigma)) or np.any(np.isinf(Sigma)):
        Sigma = np.nan_to_num(Sigma, nan=0.04, posinf=0.04, neginf=0.04)
    Sigma = (Sigma + Sigma.T) / 2
    min_eig = np.min(np.linalg.eigvalsh(Sigma))
    if min_eig < 1e-8:
        Sigma = Sigma + (1e-6 - min_eig) * np.eye(n)
    
    x = cp.Variable(n)
    k = cp.Variable()
    
    # Base constraints
    constraints = [
        (mu - risk_free) @ x == 1,
        cp.sum(x) == k,
        x >= 0,
        k >= 0
    ]
    
    #  FIXED STOCKS CONSTRAINT: Fixed stocks must have non-zero weight
    for idx in fixed_indices:
        constraints.append(x[idx] >= 0.001 * k)  # At least 0.1% weight
    
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, Sigma)), constraints)
    
    try:
        prob.solve(solver=cp.OSQP, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate'] and k.value is not None and k.value > 1e-10:
            weights = x.value / k.value
        else:
            print(f"  ⚠️  Optimization status: {prob.status}, using equal weights")
            weights = np.ones(n) / n
    except Exception as e:
        print(f"  ⚠️  Optimization error: {e}, using equal weights")
        weights = np.ones(n) / n
    
    # Post-processing
    weights = np.maximum(weights, 0)
    weights[weights < 0.001] = 0
    weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n) / n
    
    port_ret = float(mu @ weights)
    port_vol = float(np.sqrt(weights @ Sigma @ weights))
    sharpe = (port_ret - risk_free) / port_vol if port_vol > 1e-10 else 0
    
    return weights, sharpe, port_ret, port_vol


def select_top_k_with_fixed_stocks(mu: np.ndarray, Sigma: np.ndarray, risk_free: float,
                                    stocks: list, fixed_stocks: List[str],
                                    max_k: int) -> Tuple[list, np.ndarray, np.ndarray, list]:
    """
     Forward Stepwise Selection WITH Fixed Stocks
    
    Fixed stocks are automatically included, optimizer selects remaining (max_k - len(fixed)) stocks
    """
    n = len(mu)
    num_fixed = len(fixed_stocks)
    num_to_select = max_k - num_fixed
    
    print(f"\n📊 FORWARD STEPWISE SELECTION (Top {max_k} with {num_fixed} Fixed Stocks)")
    print(f"  Fixed Stocks ({num_fixed}): {', '.join(fixed_stocks)}")
    print(f"  Slots for Optimization: {num_to_select}")
    print(f"  {'='*70}")
    
    # Get indices of fixed stocks
    fixed_indices = []
    for stock in fixed_stocks:
        if stock in stocks:
            fixed_indices.append(stocks.index(stock))
        else:
            print(f"  ⚠️  Warning: Fixed stock '{stock}' not in stock list")
    
    # Start with fixed stocks
    selected_indices = fixed_indices.copy()
    remaining_indices = [i for i in range(n) if i not in fixed_indices]
    
    # Select remaining stocks
    for step in range(1, num_to_select + 1):
        if len(remaining_indices) == 0:
            print(f"  ⚠️  No more stocks to select at step {step}")
            break
        
        best_sharpe = -np.inf
        best_candidate = None
        
        for candidate_idx in remaining_indices:
            test_indices = selected_indices + [candidate_idx]
            test_mu = mu[test_indices]
            test_Sigma = Sigma[np.ix_(test_indices, test_indices)]
            _, test_sharpe, _, _ = solve_tangency_sharpe_with_fixed(test_mu, test_Sigma, risk_free, 
                                                                     [i for i in range(len(test_indices)) if i < len(fixed_indices)])
            
            if test_sharpe > best_sharpe:
                best_sharpe = test_sharpe
                best_candidate = candidate_idx
        
        if best_candidate is None:
            break
        
        selected_indices.append(best_candidate)
        remaining_indices.remove(best_candidate)
        print(f"  Step {step}: Added {stocks[best_candidate]} (Sharpe: {best_sharpe:.4f})")
    
    final_mu = mu[selected_indices]
    final_Sigma = Sigma[np.ix_(selected_indices, selected_indices)]
    final_stocks = [stocks[i] for i in selected_indices]
    
    print(f"\n✓ Selected {len(final_stocks)} stocks: {', '.join(final_stocks)}")
    print(f"  Fixed: {[stocks[i] for i in fixed_indices if i in selected_indices]}")
    print(f"  Optimized: {[stocks[i] for i in selected_indices if i not in fixed_indices]}")
    print(f"{'='*70}")
    
    return final_stocks, final_mu, final_Sigma, selected_indices


# ============================================================================
# QUALITY FILTERS
# ============================================================================
def apply_quality_filters(stocks: List[str], 
                          stock_info: Dict[str, Dict],
                          returns: pd.DataFrame,
                          capm_returns: Dict[str, float]) -> List[str]:
    """Apply Quality Filters before optimization"""
    print(f"\n📊 Applying Quality Filters...")
    print(f"  Filters:")
    print(f"      Analyst Coverage: MIN {QualityFilterConfig.MIN_ANALYST_COUNT} analysts")
    print(f"      CAPM Deviation: MAX {QualityFilterConfig.MAX_CAPM_DEVIATION:.0%}")
    print(f"    - Min Market Cap: ${QualityFilterConfig.MIN_MARKET_CAP_USD/1e9:.1f}B USD")
    print(f"    - Min Stock Price: ${QualityFilterConfig.MIN_STOCK_PRICE}")
    print(f"    - Max Volatility: {QualityFilterConfig.MAX_ANNUAL_VOLATILITY:.0%}")
    print(f"    - Min Avg Volume: {QualityFilterConfig.MIN_AVG_VOLUME/1e6:.1f}M shares/day")
    print(f"    - Return Caps: [{QualityFilterConfig.MIN_STOCK_RETURN:.0%}, {QualityFilterConfig.MAX_STOCK_RETURN:.0%}]")
    print(f"  {'='*70}")
    
    filtered_stocks = []
    excluded_stocks = []
    
    for stock in stocks:
        info = stock_info.get(stock, {})
        reasons = []
        
        analyst_count = info.get('analyst_count', 0)
        if analyst_count < QualityFilterConfig.MIN_ANALYST_COUNT:
            reasons.append(f"Analyst Count {analyst_count} < {QualityFilterConfig.MIN_ANALYST_COUNT}")
        
        analyst_target = info.get('analyst_target', None)
        current_price = info.get('current_price', 0)
        if analyst_target and current_price and current_price > 0:
            implied_return = (analyst_target / current_price) - 1
            capm_return = capm_returns.get(stock, 0.04)
            deviation = abs(implied_return - capm_return)
            if deviation > QualityFilterConfig.MAX_CAPM_DEVIATION:
                reasons.append(f"CAPM Deviation {deviation:.0%} > {QualityFilterConfig.MAX_CAPM_DEVIATION:.0%}")
        
        market_cap = info.get('market_cap', 0)
        min_cap = QualityFilterConfig.MIN_MARKET_CAP_HKD if stock.endswith('.HK') else QualityFilterConfig.MIN_MARKET_CAP_USD
        if market_cap < min_cap:
            reasons.append(f"Market Cap ${market_cap/1e9:.1f}B < ${min_cap/1e9:.1f}B")
        
        price = info.get('current_price', 0)
        if price < QualityFilterConfig.MIN_STOCK_PRICE:
            reasons.append(f"Price ${price:.2f} < ${QualityFilterConfig.MIN_STOCK_PRICE}")
        
        if stock in returns.columns:
            stock_vol = returns[stock].std() * np.sqrt(252)
            if stock_vol > QualityFilterConfig.MAX_ANNUAL_VOLATILITY:
                reasons.append(f"Volatility {stock_vol:.0%} > {QualityFilterConfig.MAX_ANNUAL_VOLATILITY:.0%}")
        
        avg_volume = info.get('avg_volume', 0)
        if avg_volume < QualityFilterConfig.MIN_AVG_VOLUME:
            reasons.append(f"Avg Volume {avg_volume/1e6:.1f}M < {QualityFilterConfig.MIN_AVG_VOLUME/1e6:.1f}M")
        
        if stock in capm_returns:
            ret = capm_returns[stock]
            if ret < QualityFilterConfig.MIN_STOCK_RETURN or ret > QualityFilterConfig.MAX_STOCK_RETURN:
                reasons.append(f"Return {ret:.0%} outside [{QualityFilterConfig.MIN_STOCK_RETURN:.0%}, {QualityFilterConfig.MAX_STOCK_RETURN:.0%}]")
        
        if len(reasons) == 0:
            filtered_stocks.append(stock)
        else:
            excluded_stocks.append((stock, reasons))
    
    print(f"\n Passed Quality Filters: {len(filtered_stocks)}/{len(stocks)} stocks")
    print(f"❌ Excluded by Quality: {len(excluded_stocks)}/{len(stocks)} stocks")
    
    if excluded_stocks:
        print(f"\n📋 Sample Excluded Stocks:")
        for stock, reasons in excluded_stocks[:10]:
            print(f"  {stock}: {', '.join(reasons[:2])}")
        if len(excluded_stocks) > 10:
            print(f"  ... and {len(excluded_stocks) - 10} more")
    
    print(f"{'='*70}")
    return filtered_stocks


# ============================================================================
# REMAINING FUNCTIONS
# ============================================================================
def get_returns(stocks: list, period: str = '2y') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download historical data and return BOTH prices and returns"""
    print(f"📥 Downloading historical data for {len(stocks)} stocks...")
    data = yf.download(stocks, period=period, threads=True, progress=False, auto_adjust=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        close_col = 'Adj Close' if 'Adj Close' in data.columns.get_level_values(0) else 'Close'
        prices = data[close_col]
    else:
        close_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        prices = data[close_col]
    
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    
    prices = prices.ffill().dropna(axis=1, thresh=len(prices) * 0.6)
    returns = prices.pct_change().dropna()
    
    print(f"✓ Data: {len(returns)} days, {len(returns.columns)} stocks")
    return prices, returns


def get_capm_returns_with_beta(stocks: list, returns: pd.DataFrame, 
                                risk_free: float = 0.04,
                                dividend_yields: Dict[str, float] = None,
                                analyst_targets: Dict[str, Optional[float]] = None,
                                market_ticker: str = '^GSPC', period: str = '2y') -> Tuple[pd.Series, Dict[str, float]]:
    """CAPM Expected Returns with Beta + Blended Analyst Targets"""
    print(f"\n📥 Downloading market data ({market_ticker})...")
    market_data = yf.download(market_ticker, period=period, threads=False, progress=False, auto_adjust=False)
    
    if isinstance(market_data.columns, pd.MultiIndex):
        market_close = market_data['Adj Close'] if 'Adj Close' in market_data.columns.get_level_values(0) else market_data['Close']
    else:
        market_close = market_data['Adj Close'] if 'Adj Close' in market_data.columns else market_data['Close']
    
    if isinstance(market_close, pd.DataFrame):
        market_close = market_close.iloc[:, 0]
    
    market_close = market_close.ffill().dropna()
    market_returns = market_close.pct_change().dropna()
    market_returns = market_returns.reindex(returns.index).ffill().dropna()
    returns_aligned = returns.reindex(market_returns.index).dropna()
    
    market_mean = float(market_returns.mean() * 252)
    market_excess = market_mean - risk_free
    
    capm_mu = {}
    beta_dict = {}
    
    print(f"\n📊 EXPECTED RETURNS (70% CAPM + 30% Analyst):")
    print(f"{'Stock':<15} {'Beta':>8} {'CAPM':>10} {'Analyst':>10} {'Blended':>10} {'Div':>8}")
    print("-" * 65)
    
    for stock in stocks:
        if stock in returns_aligned.columns and len(market_returns) > 0:
            cov = float(returns_aligned[stock].cov(market_returns) * 252)
            var = float(market_returns.var() * 252)
            beta = cov / var if var > 1e-10 else 1.0
        else:
            beta = 1.0
        
        beta_dict[stock] = beta
        capm_price_return = risk_free + beta * market_excess
        
        div_yield = dividend_yields.get(stock, 0.0) if dividend_yields else 0.0
        tax_rate = 0.10 if stock.endswith('.HK') else 0.15
        after_tax_div = div_yield * (1 - tax_rate)
        capm_total = capm_price_return + after_tax_div
        
        analyst_return = analyst_targets.get(stock, None) if analyst_targets else None
        
        if analyst_return is not None:
            analyst_total = analyst_return + after_tax_div
            blended = 0.7 * capm_total + 0.3 * analyst_total
        else:
            blended = capm_total
            analyst_return = capm_total
        
        blended = max(QualityFilterConfig.MIN_STOCK_RETURN, 
                     min(QualityFilterConfig.MAX_STOCK_RETURN, blended))
        
        capm_mu[stock] = blended
        print(f"{stock:<15} {beta:>7.2f} {capm_total:>9.2%} {analyst_return:>9.2%} {blended:>9.2%} {div_yield:>7.2%}")
    
    print("-" * 65)
    return pd.Series(capm_mu), beta_dict


def solve_portfolio(mu: np.ndarray, Sigma: np.ndarray, risk_free: float, budget: float, 
                    stocks: list, target_return: float = None,
                    original_stocks: list = None, original_indices: list = None) -> Dict:
    """Solve portfolio"""
    n = len(mu)
    
    tangency_weights, tangency_sharpe, tangency_return, tangency_vol = solve_tangency_sharpe_with_fixed(
        mu, Sigma, risk_free, []
    )
    
    weight_rf = 0.0
    final_weights = tangency_weights.copy()
    final_return = tangency_return
    final_vol = tangency_vol
    final_sharpe = tangency_sharpe
    strategy = "Tangency (Max Sharpe)"
    
    if target_return is not None:
        if target_return < tangency_return - 1e-6:
            if tangency_return - risk_free != 0:
                w_risky = max(0, min(1, (target_return - risk_free) / (tangency_return - risk_free)))
            else:
                w_risky = 0
            weight_rf = 1.0 - w_risky
            final_weights = w_risky * tangency_weights
            final_return = target_return
            final_vol = w_risky * tangency_vol
            strategy = f"CML ({weight_rf:.0%} RF)"
        elif abs(target_return - tangency_return) < 0.001:
            strategy = "Tangency (Max Sharpe)"
        else:
            print(f"\n📈 Target > Tangency, solving Efficient Frontier...")
            final_return = target_return
            strategy = f"Efficient Frontier ({target_return:.0%})"
    
    if original_stocks is not None and original_indices is not None:
        full_weights = np.zeros(len(original_stocks))
        for i, idx in enumerate(original_indices):
            full_weights[idx] = final_weights[i]
        final_weights = full_weights
    
    return {
        'tangency_weights': tangency_weights,
        'final_weights': final_weights,
        'weight_rf': weight_rf,
        'strategy': strategy,
        'tangency_return': tangency_return,
        'tangency_vol': tangency_vol,
        'tangency_sharpe': tangency_sharpe,
        'final_return': final_return,
        'final_vol': final_vol,
        'final_sharpe': final_sharpe,
        'active_stocks': int(np.sum(final_weights > 0.001))
    }


def calculate_shares_with_board_lots(weights: np.ndarray, budget: float,
                                      prices_dict: Dict[str, float], stocks: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate shares with board lots"""
    shares = np.zeros(len(weights), dtype=int)
    actual_values = np.zeros(len(weights))
    board_lots = np.array([HK_BOARD_LOTS.get(s, DEFAULT_HK_BOARD_LOT) if s.endswith('.HK') else 1 for s in stocks])
    stock_budgets = weights * budget
    
    for i, stock in enumerate(stocks):
        if weights[i] < 0.001:
            continue
        price = prices_dict.get(stock, 0)
        if price <= 0:
            continue
        price_usd = price / USD_HKD_RATE if stock.endswith('.HK') else price
        raw_shares = int(stock_budgets[i] / price_usd) if price_usd > 1e-6 else 0
        if stock.endswith('.HK'):
            lot_size = board_lots[i]
            shares[i] = (raw_shares // lot_size) * lot_size
        else:
            shares[i] = raw_shares
        if stock.endswith('.HK'):
            actual_values[i] = (shares[i] * price) / USD_HKD_RATE
        else:
            actual_values[i] = shares[i] * price
    
    total_value = np.sum(actual_values)
    return shares, actual_values, board_lots


def run_monte_carlo_student_t(weights: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                               budget: float, risk_free: float, weight_rf: float,
                               days: int = 252, simulations: int = 10000, df: float = 5.0) -> Dict:
    """Monte Carlo with Student-t distribution (vectorized)"""
    dt = 1.0 / days
    nonzero_mask = np.abs(weights) > 1e-6
    if np.sum(nonzero_mask) > 0:
        risky_weights = weights[nonzero_mask]
        risky_weights = risky_weights / np.sum(risky_weights)
        port_mu = float(np.dot(risky_weights, mu[nonzero_mask]))
        port_var = float(np.dot(risky_weights, Sigma[np.ix_(nonzero_mask, nonzero_mask)] @ risky_weights))
        port_vol = np.sqrt(max(0, port_var))
    else:
        port_mu, port_vol = risk_free, 0
    
    if port_vol > 0:
        scale_factor = np.sqrt(df / (df - 2)) if df > 2 else 1.0
        random_shocks = t_dist.rvs(df, size=(days, simulations)) / scale_factor
        daily_returns = (port_mu - 0.5 * port_var) * dt + port_vol * random_shocks * np.sqrt(dt)
        risky_cumulative = np.prod(1 + daily_returns, axis=0)
    else:
        risky_cumulative = np.exp(risk_free)
    
    final_values = budget * (weight_rf * np.exp(risk_free) + (1 - weight_rf) * risky_cumulative)
    
    var_95 = budget - np.percentile(final_values, 5)
    worst_5pct = final_values[final_values <= np.percentile(final_values, 5)]
    cvar_95 = budget - np.mean(worst_5pct) if len(worst_5pct) > 0 else var_95
    
    return {
        'median': np.percentile(final_values, 50),
        'var_95': var_95,
        'cvar_95': cvar_95,
        'prob_profit': np.sum(final_values > budget) / simulations
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    # ============================================
    # CONFIGURATION: FIXED STOCKS
    # ============================================
    FixedStocksConfig.FIXED_STOCKS = [
        "AAPL",      # Apple (Core holding)
        "MSFT",      # Microsoft (Core holding)
        "0700.HK",   # Tencent (HK core holding)
    ]
    FixedStocksConfig.MAX_K = 6
    
    # Validate configuration
    FixedStocksConfig.validate()
    
    all_stocks = sorted([
        # US (20)
        "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA",
        "JPM", "V", "MA", "JNJ", "KO", "PG", "WMT", "HD",
        "BRK.B", "XOM", "UNH", "LLY", "INTC",
        # HK (10)
        "0700.HK", "0941.HK", "0005.HK", "9988.HK", "1299.HK",
        "0388.HK", "2318.HK", "0883.HK", "0016.HK", "0002.HK"
    ])
    
    risk_free_rate = 0.04
    budget = 100000
    target_return = 0.12
    max_k = FixedStocksConfig.MAX_K
    market_ticker = '^GSPC'
    fixed_stocks = FixedStocksConfig.FIXED_STOCKS
    
    target_display = f"{target_return:.0%}" if target_return is not None else 'None'
    
    print(f"\n{'='*70}")
    print(f"PORTFOLIO OPTIMIZATION (WITH FIXED STOCKS)")
    print(f"{'='*70}")
    print(f"Total Stock Universe: {len(all_stocks)} stocks (US + HK)")
    print(f"Budget: ${budget:,.0f} USD | Target: {target_display} | Rf: {risk_free_rate:.0%}")
    print(f"Max Stocks (K): {max_k}")
    print(f"Fixed Stocks (i): {len(fixed_stocks)}")
    print(f"  → {', '.join(fixed_stocks)}")
    print(f"Slots for Optimization: {max_k - len(fixed_stocks)}")
    print(f"{'='*70}\n")
    
    try:
        # 1. Historical Returns
        prices_df, returns = get_returns(all_stocks)
        stocks = list(returns.columns)
        
        # 2. Validate Fixed Stocks
        is_valid, errors = validate_fixed_stocks(fixed_stocks, stocks, max_k)
        if not is_valid:
            print(f"\n❌ ERROR: Fixed stocks validation failed")
            return
        
        # 3. OPTIMIZED: Single API call per stock
        stock_info = fetch_all_stock_info(stocks)
        
        # 4. Extract from cache
        dividend_yields = get_dividend_yield_from_cache(stock_info, stocks)
        analyst_targets = fetch_analyst_targets_from_cache(stock_info, stocks)
        prices_dict = get_current_prices_from_cache(stock_info, stocks)
        
        # 5. CAPM Returns with Beta
        capm_mu_series, beta_dict = get_capm_returns_with_beta(
            stocks, returns, risk_free_rate, dividend_yields, analyst_targets, market_ticker
        )
        capm_returns_dict = capm_mu_series.to_dict()
        mu = capm_mu_series.values
        
        # 6. Apply Quality Filters
        filtered_stocks = apply_quality_filters(stocks, stock_info, returns, capm_returns_dict)
        
        if len(filtered_stocks) < max_k:
            print(f"\n❌ ERROR: Only {len(filtered_stocks)} stocks passed filters, need at least {max_k}")
            return
        
        # Filter mu and Sigma
        stock_indices = [stocks.index(s) for s in filtered_stocks]
        mu = mu[stock_indices]
        Sigma = LedoitWolf().fit(returns[filtered_stocks].dropna().values).covariance_ * 252
        stocks = filtered_stocks
        
        # 7. Stock Selection WITH Fixed Stocks
        print(f"\n🎯 Selecting top {max_k} stocks (with {len(fixed_stocks)} fixed)...")
        sel_stocks, sel_mu, sel_Sigma, sel_idx = select_top_k_with_fixed_stocks(
            mu, Sigma, risk_free_rate, stocks, fixed_stocks, max_k
        )
        
        # 8. Portfolio Optimization
        portfolio = solve_portfolio(sel_mu, sel_Sigma, risk_free_rate, budget, sel_stocks, 
                                     target_return, stocks, sel_idx)
        
        # 9. Shares
        weights = portfolio['final_weights']
        shares, actual_values, board_lots = calculate_shares_with_board_lots(weights, budget, prices_dict, stocks)
        
        # 10. Monte Carlo
        mc = run_monte_carlo_student_t(weights, mu, Sigma, budget, risk_free_rate, portfolio['weight_rf'])
        
        # Cleanup
        print(f"\n🧹 Cleaning up...")
        del returns, prices_df, stock_info, capm_mu_series
        gc.collect()
        
        # Output
        print(f"\n{'='*70}")
        print(f"TANGENCY PORTFOLIO")
        print(f"{'='*70}")
        print(f"Return: {portfolio['tangency_return']:.2%} | Vol: {portfolio['tangency_vol']:.2%} | Sharpe: {portfolio['tangency_sharpe']:.4f}")
        
        print(f"\n{'='*70}")
        print(f"FINAL PORTFOLIO")
        print(f"{'='*70}")
        print(f"Strategy: {portfolio['strategy']}")
        print(f"Return: {portfolio['final_return']:.2%} | Vol: {portfolio['final_vol']:.2%} | Sharpe: {portfolio['final_sharpe']:.4f}")
        print(f"Risk-Free: {portfolio['weight_rf']:.0%} | Active Stocks: {portfolio['active_stocks']}")
        print(f"{'-'*70}")
        print(f"{'Stock':<15} {'Weight':>10} {'Shares':>10} {'Lot':>8} {'Beta':>8} {'Price':>14} {'Value(USD)':>14} {'Fixed':>8}")
        print(f"{'-'*70}")
        
        total_value = 0
        for i, stock in enumerate(stocks):
            if weights[i] > 0.001 or shares[i] > 0:
                value = actual_values[i]
                total_value += value
                currency = 'HKD' if stock.endswith('.HK') else 'USD'
                lot = board_lots[i]
                beta = beta_dict.get(stock, 1.0)
                price = prices_dict.get(stock, 0)
                price_display = f"${price:.2f} {currency}"
                is_fixed = "Yes" if stock in fixed_stocks else "No"
                print(f"{stock:<15} {weights[i]:>9.2%} {shares[i]:>10} {lot:>8} {beta:>7.2f} {price_display:>14} ${value:>12,.2f} {is_fixed:>8}")
        
        print(f"{'-'*70}")
        print(f"Total Stocks: ${total_value:,.2f} USD | Cash (RF): ${budget - total_value:,.2f} USD")
        
        print(f"\n{'='*70}")
        print(f"RISK METRICS")
        print(f"{'='*70}")
        print(f"Median Value: ${mc['median']:,.2f} | Profit Prob: {mc['prob_profit']:.0%}")
        print(f"VaR (95%): ${mc['var_95']:,.2f} | CVaR (95%): ${mc['cvar_95']:,.2f}")
        
        print(f"\n{'='*70}")
        print(f"Optimization Complete (WITH FIXED STOCKS)")
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        try:
            del returns
            del stock_info
        except:
            pass
        gc.collect()
        raise


if __name__ == "__main__":
    main()
