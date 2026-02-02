"""Asset Features page - Explore asset behavior metrics and technical indicators."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import sys

# Add data-layer to path for imports
_data_layer_path = Path(__file__).parent.parent.parent / "data-layer"
if str(_data_layer_path) not in sys.path:
    sys.path.insert(0, str(_data_layer_path))

from data import DataInjestor
from components.layouts import render_page_header
from components.theme import THEME, CHART_WIDTH, CHART_HEIGHT, configure_altair_theme
import altair as alt

# Page header
render_page_header(
    "Asset Features",
    "Explore asset behavior metrics, market relationships, and technical indicators. "
    "Understand how to measure return, risk, and market sensitivity.",
)

# Feature categories
FEATURE_CATEGORIES = {
    "Behavior Metrics (Annualized)": [
        "Annual Return",
        "Annual Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Calmar Ratio",
        "Sortino Ratio",
    ],
    "Market Relationships": [
        "Beta",
        "Correlation",
        "Alpha",
        "R-Squared",
        "Tracking Error",
        "Information Ratio",
    ],
    "Risk Metrics": [
        "VaR (95%)",
        "CVaR (95%)",
        "Downside Deviation",
        "Skewness",
        "Kurtosis",
        "Tail Ratio",
    ],
    "Technical - Momentum": [
        "RSI",
        "ROC",
        "Momentum",
        "MACD",
    ],
    "Technical - Trend": [
        "SMA",
        "EMA",
        "Price vs SMA",
    ],
    "Technical - Volatility": [
        "Bollinger Bands",
        "ATR",
        "Rolling Volatility",
    ],
}

# Educational content for each feature
FEATURE_INFO = {
    # Behavior Metrics
    "Annual Return": {
        "formula": "R_annual = (∏(1 + r_t))^(252/n) - 1",
        "why": "Converts daily returns into an annualized figure for standardized comparison across assets and time periods. Represents the compound annual growth rate (CAGR).",
        "when": [
            "Comparing performance of different assets over varying time periods",
            "Setting return expectations and benchmarks",
            "Evaluating manager or strategy performance",
            "Financial planning and projections",
        ],
        "how": "Compound all daily returns to get total growth, then scale to annual basis using the 252 trading days convention.",
        "interpretation": "A 10% annual return means $100 invested would grow to $110 over one year, assuming returns compound at that rate.",
        "pitfalls": [
            "Past returns don't guarantee future performance",
            "Doesn't account for risk taken to achieve the return",
            "Can be misleading for short time periods",
        ],
    },
    "Annual Volatility": {
        "formula": "σ_annual = σ_daily × √252",
        "why": "Measures the dispersion of returns, serving as the standard proxy for investment risk. Higher volatility means larger price swings.",
        "when": [
            "Assessing and comparing asset risk levels",
            "Portfolio construction and risk budgeting",
            "Options pricing and derivatives valuation",
            "Setting stop-loss levels and position sizes",
        ],
        "how": "Calculate the standard deviation of daily returns and multiply by √252 to annualize (square root of time scaling).",
        "interpretation": "20% annual volatility means returns typically fluctuate within ±20% of the mean in a given year (1 standard deviation).",
        "pitfalls": [
            "Assumes returns are normally distributed (they have fat tails)",
            "Treats upside and downside volatility equally",
            "Historical volatility may not predict future volatility",
        ],
    },
    "Sharpe Ratio": {
        "formula": "Sharpe = (R_annual - R_f) / σ_annual",
        "why": "Measures risk-adjusted return - how much excess return you earn per unit of risk. The most widely used performance metric in finance.",
        "when": [
            "Comparing strategies or managers with different risk levels",
            "Evaluating if higher returns justify higher risk",
            "Portfolio optimization and allocation decisions",
            "Performance attribution and ranking",
        ],
        "how": "Subtract the risk-free rate from the return, then divide by volatility. Higher is better.",
        "interpretation": "Sharpe > 1 is good, > 2 is very good, > 3 is excellent. Negative Sharpe means returns below risk-free rate.",
        "pitfalls": [
            "Assumes returns are normally distributed",
            "Penalizes upside volatility (which investors like)",
            "Can be manipulated by smoothing returns or using leverage",
        ],
    },
    "Max Drawdown": {
        "formula": "DD_max = min_t((P_t / max(P_s for s≤t)) - 1)",
        "why": "Measures the worst peak-to-trough decline, capturing the maximum pain an investor would have experienced.",
        "when": [
            "Assessing worst-case scenario risk",
            "Setting risk limits and stop-losses",
            "Evaluating strategy robustness",
            "Comparing defensive vs aggressive investments",
        ],
        "how": "Track the running maximum price, calculate the percentage decline from each peak to subsequent trough, find the worst one.",
        "interpretation": "-30% max drawdown means at worst, an investor lost 30% from a previous high before recovering.",
        "pitfalls": [
            "Only captures one extreme event",
            "Doesn't indicate how long recovery took",
            "Historical max drawdown may be exceeded in future",
        ],
    },
    "Calmar Ratio": {
        "formula": "Calmar = R_annual / |Max Drawdown|",
        "why": "Measures return relative to worst-case loss, useful for evaluating strategies where drawdown management is critical.",
        "when": [
            "Evaluating hedge funds and CTAs",
            "Comparing strategies with drawdown constraints",
            "Risk-conscious portfolio allocation",
        ],
        "how": "Divide annualized return by the absolute value of maximum drawdown.",
        "interpretation": "Calmar of 1 means annual return equals max drawdown. Higher is better; > 1 is generally considered good.",
        "pitfalls": [
            "Single extreme drawdown can dominate the metric",
            "Doesn't capture frequency of drawdowns",
        ],
    },
    "Sortino Ratio": {
        "formula": "Sortino = (R_annual - R_f) / σ_downside",
        "why": "Like Sharpe but only penalizes downside volatility, recognizing that investors don't mind upside volatility.",
        "when": [
            "Evaluating strategies with asymmetric return profiles",
            "When upside volatility is desirable",
            "Comparing strategies with different return distributions",
        ],
        "how": "Calculate standard deviation using only negative returns (below target), then compute risk-adjusted return.",
        "interpretation": "Higher is better. A strategy with high upside volatility but controlled downside will have better Sortino than Sharpe.",
        "pitfalls": [
            "Requires more data for stable estimation",
            "Choice of target return affects calculation",
        ],
    },
    # Market Relationships
    "Beta": {
        "formula": "β = Cov(r_asset, r_market) / Var(r_market)",
        "why": "Measures systematic risk - how much the asset moves relative to the market. Essential for CAPM and portfolio construction.",
        "when": [
            "Understanding market sensitivity",
            "Building market-neutral portfolios",
            "Hedging market exposure",
            "Estimating expected returns (CAPM)",
        ],
        "how": "Regress asset returns against market returns; beta is the slope coefficient.",
        "interpretation": "β=1: moves with market. β>1: amplifies market moves. β<1: dampens market moves. β<0: moves opposite to market.",
        "pitfalls": [
            "Beta changes over time (not stable)",
            "Assumes linear relationship with market",
            "Doesn't capture non-market risks",
        ],
    },
    "Correlation": {
        "formula": "ρ = Cov(r_asset, r_market) / (σ_asset × σ_market)",
        "why": "Measures the direction and strength of linear relationship between asset and market. Key input for diversification analysis.",
        "when": [
            "Identifying diversification opportunities (low correlation)",
            "Building multi-asset portfolios",
            "Understanding co-movement patterns",
            "Stress testing (correlations increase in crises)",
        ],
        "how": "Pearson correlation coefficient between asset and market returns.",
        "interpretation": "+1: perfect positive correlation. 0: no linear relationship. -1: perfect negative correlation.",
        "pitfalls": [
            "Correlations are unstable and increase during crises",
            "Only captures linear relationships",
            "Historical correlation may not persist",
        ],
    },
    "Alpha": {
        "formula": "α = R_asset - [R_f + β × (R_market - R_f)]",
        "why": "Measures excess return after adjusting for market risk. Positive alpha indicates genuine outperformance.",
        "when": [
            "Evaluating active manager skill",
            "Identifying true outperformance vs beta exposure",
            "Performance attribution",
            "Strategy evaluation",
        ],
        "how": "Subtract the expected return (based on beta and market return) from actual return.",
        "interpretation": "α > 0: outperformed risk-adjusted benchmark. α < 0: underperformed. α = 0: returns explained by market exposure.",
        "pitfalls": [
            "Depends on choice of benchmark",
            "May be due to luck over short periods",
            "Assumes beta is constant and accurate",
        ],
    },
    "R-Squared": {
        "formula": "R² = 1 - (SS_res / SS_tot)",
        "why": "Measures how much of the asset's variance is explained by market movements. Indicates diversification from market.",
        "when": [
            "Assessing market dependency",
            "Evaluating hedge fund market exposure",
            "Portfolio diversification analysis",
        ],
        "how": "Square of correlation, or proportion of variance explained by the market regression.",
        "interpretation": "R²=0.8 means 80% of asset's variance explained by market. Lower R² suggests more idiosyncratic risk.",
        "pitfalls": [
            "Can be artificially high with leverage",
            "Doesn't indicate direction of relationship",
        ],
    },
    "Tracking Error": {
        "formula": "TE = σ(r_asset - r_benchmark)",
        "why": "Measures how closely a portfolio follows its benchmark. Low tracking error means returns closely match the index.",
        "when": [
            "Evaluating index fund replication quality",
            "Setting active risk budgets",
            "Monitoring portfolio drift",
        ],
        "how": "Standard deviation of the difference between portfolio and benchmark returns.",
        "interpretation": "TE of 2% means portfolio returns typically deviate ±2% from benchmark annually.",
        "pitfalls": [
            "Low tracking error doesn't mean good performance",
            "Can be gamed by matching benchmark composition",
        ],
    },
    "Information Ratio": {
        "formula": "IR = (R_asset - R_benchmark) / TE",
        "why": "Measures active return per unit of active risk. The key metric for evaluating active managers.",
        "when": [
            "Comparing active managers",
            "Evaluating skill vs luck in outperformance",
            "Setting performance expectations",
        ],
        "how": "Divide excess return over benchmark by tracking error.",
        "interpretation": "IR > 0.5 is good, > 1 is excellent. Shows how consistently manager generates alpha.",
        "pitfalls": [
            "Requires meaningful benchmark selection",
            "Short periods can give misleading results",
        ],
    },
    # Risk Metrics
    "VaR (95%)": {
        "formula": "VaR_95% = Percentile(returns, 5%)",
        "why": "Estimates the maximum expected loss at a given confidence level. Widely used regulatory risk metric.",
        "when": [
            "Risk limit setting",
            "Regulatory capital requirements",
            "Portfolio risk monitoring",
            "Stress testing",
        ],
        "how": "Find the 5th percentile of the return distribution (for 95% VaR).",
        "interpretation": "VaR of -2% at 95% means: on 95% of days, losses won't exceed 2%.",
        "pitfalls": [
            "Doesn't say how bad losses can be beyond VaR",
            "Not sub-additive (portfolio VaR can exceed sum of parts)",
            "Assumes historical distribution continues",
        ],
    },
    "CVaR (95%)": {
        "formula": "CVaR_95% = E[r | r ≤ VaR_95%]",
        "why": "Expected loss given that VaR is breached. More informative than VaR for tail risk.",
        "when": [
            "Understanding tail risk",
            "When worst-case scenarios matter",
            "Risk budgeting for extreme events",
        ],
        "how": "Average of all returns worse than VaR threshold.",
        "interpretation": "CVaR of -3% means: when losses exceed VaR, they average -3%.",
        "pitfalls": [
            "Requires sufficient extreme observations",
            "Still based on historical data",
        ],
    },
    "Downside Deviation": {
        "formula": "σ_down = √(E[min(r - target, 0)²])",
        "why": "Measures volatility of negative returns only. Used in Sortino ratio calculation.",
        "when": [
            "When downside risk is primary concern",
            "Evaluating asymmetric strategies",
            "Risk-averse portfolio construction",
        ],
        "how": "Standard deviation calculated using only returns below the target (usually 0).",
        "interpretation": "Lower downside deviation means less severe negative returns.",
        "pitfalls": [
            "Requires enough negative returns for stable estimate",
        ],
    },
    "Skewness": {
        "formula": "Skew = E[(r - μ)³] / σ³",
        "why": "Measures asymmetry of the return distribution. Negative skew means more frequent small gains but rare large losses.",
        "when": [
            "Evaluating tail risk",
            "Understanding return distribution shape",
            "Strategy due diligence",
        ],
        "how": "Third standardized moment of the return distribution.",
        "interpretation": "Skew > 0: right tail (occasional large gains). Skew < 0: left tail (occasional large losses).",
        "pitfalls": [
            "Sensitive to outliers",
            "May not be stable over time",
        ],
    },
    "Kurtosis": {
        "formula": "Kurt = E[(r - μ)⁴] / σ⁴ - 3",
        "why": "Measures tail heaviness. Higher kurtosis means more extreme events than normal distribution.",
        "when": [
            "Assessing fat tail risk",
            "Model risk evaluation",
            "Derivatives pricing",
        ],
        "how": "Fourth standardized moment minus 3 (excess kurtosis).",
        "interpretation": "Kurt > 0: fatter tails than normal (more extremes). Kurt = 0: normal distribution.",
        "pitfalls": [
            "Very sensitive to outliers",
            "Requires large sample for stable estimate",
        ],
    },
    "Tail Ratio": {
        "formula": "Tail Ratio = |Percentile(95%)| / |Percentile(5%)|",
        "why": "Compares right tail gains to left tail losses. Higher ratio means better upside vs downside.",
        "when": [
            "Evaluating return asymmetry",
            "Comparing strategies with different profiles",
        ],
        "how": "Ratio of 95th percentile (gains) to 5th percentile (losses) in absolute terms.",
        "interpretation": "Ratio > 1: larger potential gains than losses. Ratio < 1: larger potential losses.",
        "pitfalls": [
            "Only uses two percentiles",
        ],
    },
    # Technical Indicators
    "RSI": {
        "formula": "RSI = 100 - 100/(1 + RS), where RS = Avg Gain / Avg Loss",
        "why": "Measures momentum and overbought/oversold conditions. One of the most popular technical indicators.",
        "when": [
            "Identifying overbought (>70) or oversold (<30) conditions",
            "Spotting potential reversals",
            "Confirming trend strength",
        ],
        "how": "Calculate average gains and losses over N periods, compute relative strength, scale to 0-100.",
        "interpretation": "RSI > 70: potentially overbought. RSI < 30: potentially oversold.",
        "pitfalls": [
            "Can stay overbought/oversold for extended periods in strong trends",
            "Many false signals in ranging markets",
        ],
    },
    "ROC": {
        "formula": "ROC = (P_t - P_{t-n}) / P_{t-n} × 100",
        "why": "Rate of Change measures percentage price change over N periods. Simple momentum indicator.",
        "when": [
            "Measuring momentum strength",
            "Identifying acceleration/deceleration",
            "Divergence analysis",
        ],
        "how": "Calculate percentage change from N periods ago.",
        "interpretation": "Positive ROC: upward momentum. Negative: downward. Zero crossings signal direction changes.",
        "pitfalls": [
            "Lagging indicator",
            "Sensitive to outliers",
        ],
    },
    "Momentum": {
        "formula": "MOM = P_t - P_{t-n}",
        "why": "Raw price change over N periods. Simplest momentum measure.",
        "when": [
            "Trend following strategies",
            "Momentum factor construction",
        ],
        "how": "Subtract price N periods ago from current price.",
        "interpretation": "Positive: price rising. Negative: price falling. Magnitude shows strength.",
        "pitfalls": [
            "Scale depends on price level",
        ],
    },
    "MACD": {
        "formula": "MACD = EMA(12) - EMA(26), Signal = EMA(MACD, 9)",
        "why": "Moving Average Convergence Divergence shows relationship between two moving averages. Trend and momentum indicator.",
        "when": [
            "Identifying trend direction and strength",
            "Spotting potential reversals (crossovers)",
            "Measuring momentum divergence",
        ],
        "how": "Difference between fast and slow EMAs, with signal line as EMA of MACD.",
        "interpretation": "MACD > Signal: bullish. MACD < Signal: bearish. Histogram shows momentum.",
        "pitfalls": [
            "Lagging indicator",
            "Many false signals in sideways markets",
        ],
    },
    "SMA": {
        "formula": "SMA = (P_1 + P_2 + ... + P_n) / n",
        "why": "Simple Moving Average smooths price data to identify trends. Foundation of many strategies.",
        "when": [
            "Trend identification",
            "Support/resistance levels",
            "Crossover strategies",
        ],
        "how": "Average of last N closing prices.",
        "interpretation": "Price > SMA: uptrend. Price < SMA: downtrend. Golden/death crosses when SMAs cross.",
        "pitfalls": [
            "Lagging indicator",
            "Equal weight to all observations",
        ],
    },
    "EMA": {
        "formula": "EMA_t = α × P_t + (1-α) × EMA_{t-1}, α = 2/(n+1)",
        "why": "Exponential Moving Average gives more weight to recent prices. More responsive than SMA.",
        "when": [
            "Faster trend detection",
            "Short-term trading signals",
            "When recent price action is more relevant",
        ],
        "how": "Weighted average where recent prices have exponentially higher weight.",
        "interpretation": "Similar to SMA but reacts faster to price changes.",
        "pitfalls": [
            "More prone to whipsaws than SMA",
        ],
    },
    "Price vs SMA": {
        "formula": "Price/SMA = P_t / SMA_t",
        "why": "Shows current price relative to moving average. Normalized trend indicator.",
        "when": [
            "Mean reversion signals",
            "Comparing trend strength across assets",
            "Overbought/oversold conditions",
        ],
        "how": "Divide current price by SMA.",
        "interpretation": ">1: above average (bullish). <1: below average (bearish).",
        "pitfalls": [
            "Doesn't account for volatility",
        ],
    },
    "Bollinger Bands": {
        "formula": "Upper = SMA + 2σ, Lower = SMA - 2σ",
        "why": "Dynamic support/resistance based on volatility. Bands widen in volatile periods, narrow in calm periods.",
        "when": [
            "Identifying volatility breakouts",
            "Mean reversion trading",
            "Setting dynamic stop-losses",
        ],
        "how": "SMA ± k standard deviations of price (typically k=2).",
        "interpretation": "Price near upper band: potentially overbought. Near lower: potentially oversold. Band squeeze: expect breakout.",
        "pitfalls": [
            "Price can ride bands in strong trends",
            "Works better in ranging markets",
        ],
    },
    "ATR": {
        "formula": "ATR = EMA(True Range, n), TR = max(H-L, |H-C_prev|, |L-C_prev|)",
        "why": "Average True Range measures volatility including gaps. Used for position sizing and stop-losses.",
        "when": [
            "Setting stop-loss distances",
            "Position sizing based on volatility",
            "Identifying volatility regime changes",
        ],
        "how": "Moving average of True Range (accounts for gaps from previous close).",
        "interpretation": "Higher ATR: more volatile, wider stops needed. Lower ATR: calmer market.",
        "pitfalls": [
            "In dollar terms, not percentage",
            "Requires adjustment for different price levels",
        ],
    },
    "Rolling Volatility": {
        "formula": "σ_rolling = std(returns, window) × √252",
        "why": "Time-varying volatility estimate. Shows how risk changes over time.",
        "when": [
            "Regime detection",
            "Dynamic position sizing",
            "Volatility targeting strategies",
        ],
        "how": "Standard deviation of returns over rolling window, annualized.",
        "interpretation": "Rising volatility: increasing risk. Falling: calmer period. Volatility clusters.",
        "pitfalls": [
            "Backward-looking",
            "Window choice affects responsiveness",
        ],
    },
}

# Category-level educational content
CATEGORY_INFO = {
    "Behavior Metrics (Annualized)": {
        "description": "Core metrics that describe an asset's standalone return and risk characteristics over time.",
        "key_principle": "Convert daily data to annualized figures for standardized comparison",
        "best_for": "Performance evaluation, manager comparison, setting expectations",
        "rule_of_thumb": "Always consider return AND risk together (use Sharpe). Check drawdown for worst-case.",
    },
    "Market Relationships": {
        "description": "Metrics that describe how an asset moves relative to the overall market benchmark.",
        "key_principle": "Decompose returns into market-driven (beta) and idiosyncratic (alpha) components",
        "best_for": "Portfolio construction, hedging, active management evaluation",
        "rule_of_thumb": "High beta = market amplifier. Low correlation = diversifier. Positive alpha = true skill.",
    },
    "Risk Metrics": {
        "description": "Advanced risk measures that go beyond simple volatility to capture tail risk and distribution shape.",
        "key_principle": "Volatility alone doesn't capture extreme events or asymmetry",
        "best_for": "Tail risk assessment, regulatory requirements, stress testing",
        "rule_of_thumb": "Use VaR for regulatory, CVaR for tail risk, Sortino when downside matters most.",
    },
    "Technical - Momentum": {
        "description": "Indicators that measure the speed and strength of price movements.",
        "key_principle": "Momentum tends to persist in the short term (trend following)",
        "best_for": "Short-term trading signals, trend confirmation",
        "rule_of_thumb": "Use RSI for overbought/oversold, MACD for trend direction and momentum.",
    },
    "Technical - Trend": {
        "description": "Indicators that identify the direction and strength of price trends.",
        "key_principle": "The trend is your friend - price tends to continue in its direction",
        "best_for": "Trend identification, support/resistance, crossover signals",
        "rule_of_thumb": "Longer periods = stronger trend signal but more lag. Use multiple timeframes.",
    },
    "Technical - Volatility": {
        "description": "Indicators that measure price variability and potential for large moves.",
        "key_principle": "Volatility clusters - high vol follows high vol, low follows low",
        "best_for": "Position sizing, stop-loss placement, breakout detection",
        "rule_of_thumb": "Wide Bollinger Bands = high volatility. Band squeeze often precedes big move.",
    },
}


# Calculation functions
def annual_return(returns: pd.Series) -> float:
    """Calculate annualized return from daily returns."""
    if len(returns) == 0:
        return 0.0
    total_return = (1 + returns).prod()
    n_days = len(returns)
    return total_return ** (252 / n_days) - 1


def annual_volatility(returns: pd.Series) -> float:
    """Calculate annualized volatility from daily returns."""
    return returns.std() * np.sqrt(252)


def sharpe_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Calculate Sharpe ratio."""
    ann_ret = annual_return(returns)
    ann_vol = annual_volatility(returns)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - rf) / ann_vol


def max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown from price series."""
    peak = prices.cummax()
    drawdown = (prices / peak) - 1
    return drawdown.min()


def calmar_ratio(returns: pd.Series, prices: pd.Series) -> float:
    """Calculate Calmar ratio."""
    ann_ret = annual_return(returns)
    mdd = abs(max_drawdown(prices))
    if mdd == 0:
        return 0.0
    return ann_ret / mdd


def sortino_ratio(returns: pd.Series, rf: float = 0.0) -> float:
    """Calculate Sortino ratio."""
    ann_ret = annual_return(returns)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return 0.0
    downside_std = downside_returns.std() * np.sqrt(252)
    if downside_std == 0:
        return 0.0
    return (ann_ret - rf) / downside_std


def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculate beta relative to market."""
    cov = np.cov(asset_returns, market_returns)[0, 1]
    var = market_returns.var()
    if var == 0:
        return 0.0
    return cov / var


def alpha(asset_returns: pd.Series, market_returns: pd.Series, rf: float = 0.0) -> float:
    """Calculate alpha (CAPM)."""
    b = beta(asset_returns, market_returns)
    asset_ann = annual_return(asset_returns)
    market_ann = annual_return(market_returns)
    return asset_ann - (rf + b * (market_ann - rf))


def var_95(returns: pd.Series) -> float:
    """Calculate 95% VaR."""
    return returns.quantile(0.05)


def cvar_95(returns: pd.Series) -> float:
    """Calculate 95% CVaR (Expected Shortfall)."""
    var = var_95(returns)
    return returns[returns <= var].mean()


def compute_all_metrics(prices: pd.Series, returns: pd.Series,
                        market_prices: pd.Series = None, market_returns: pd.Series = None) -> dict:
    """Compute all metrics for a single asset."""
    metrics = {}

    # Behavior metrics
    metrics["Annual Return"] = annual_return(returns)
    metrics["Annual Volatility"] = annual_volatility(returns)
    metrics["Sharpe Ratio"] = sharpe_ratio(returns)
    metrics["Max Drawdown"] = max_drawdown(prices)
    metrics["Calmar Ratio"] = calmar_ratio(returns, prices)
    metrics["Sortino Ratio"] = sortino_ratio(returns)

    # Market relationships (if market data provided)
    if market_returns is not None and len(market_returns) > 0:
        # Align data
        common_idx = returns.index.intersection(market_returns.index)
        asset_r = returns.loc[common_idx]
        market_r = market_returns.loc[common_idx]

        metrics["Beta"] = beta(asset_r, market_r)
        metrics["Correlation"] = asset_r.corr(market_r)
        metrics["Alpha"] = alpha(asset_r, market_r)
        metrics["R-Squared"] = asset_r.corr(market_r) ** 2

        # Tracking error and IR
        excess = asset_r - market_r
        metrics["Tracking Error"] = excess.std() * np.sqrt(252)
        if metrics["Tracking Error"] > 0:
            metrics["Information Ratio"] = (annual_return(asset_r) - annual_return(market_r)) / metrics["Tracking Error"]
        else:
            metrics["Information Ratio"] = 0.0

    # Risk metrics
    metrics["VaR (95%)"] = var_95(returns)
    metrics["CVaR (95%)"] = cvar_95(returns)
    metrics["Downside Deviation"] = returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
    metrics["Skewness"] = returns.skew()
    metrics["Kurtosis"] = returns.kurtosis()

    p95 = returns.quantile(0.95)
    p5 = returns.quantile(0.05)
    metrics["Tail Ratio"] = abs(p95 / p5) if p5 != 0 else 0

    return metrics


# Sidebar controls
with st.sidebar:
    st.header("Feature Options")
    ticker = st.text_input("Ticker Symbol", value="AAPL")
    benchmark = st.text_input("Market Benchmark", value="^GSPC")
    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=365 * 2),
    )
    end_date = st.date_input("End Date", value=date.today())

    st.subheader("Feature Selection")
    feature_category = st.selectbox(
        "Category",
        list(FEATURE_CATEGORIES.keys()),
    )
    available_features = FEATURE_CATEGORIES[feature_category]

# Fetch data
df = DataInjestor.get(ticker, str(start_date), str(end_date))
market_df = DataInjestor.get(benchmark, str(start_date), str(end_date))

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Metrics Summary", "Time Series Analysis", "Distributions", "Learning Center"])

with tab1:
    if df.empty:
        st.warning(f"No data found for {ticker}. Please check the ticker symbol.")
    else:
        prices = df["Close"].dropna()
        returns = prices.pct_change().dropna()

        market_prices = market_df["Close"].dropna() if not market_df.empty else None
        market_returns = market_prices.pct_change().dropna() if market_prices is not None else None

        # Compute all metrics
        metrics = compute_all_metrics(prices, returns, market_prices, market_returns)

        # Show contextual help for selected category
        cat_info = CATEGORY_INFO.get(feature_category, {})
        with st.expander(f"About {feature_category}", expanded=False):
            st.markdown(f"**Description:** {cat_info.get('description', '')}")
            st.markdown(f"**Key Principle:** {cat_info.get('key_principle', '')}")
            st.markdown(f"**Best For:** {cat_info.get('best_for', '')}")
            st.info(f"**Rule of Thumb:** {cat_info.get('rule_of_thumb', '')}")

        st.subheader(f"{ticker} - {feature_category}")

        # Display metrics for selected category
        col1, col2 = st.columns(2)

        for i, feature in enumerate(available_features):
            if feature in metrics:
                value = metrics[feature]
                col = col1 if i % 2 == 0 else col2

                # Format based on metric type
                if "Return" in feature or "Alpha" in feature or "Drawdown" in feature:
                    formatted = f"{value:.2%}"
                elif "Ratio" in feature or "Beta" in feature or "Correlation" in feature or "R-Squared" in feature:
                    formatted = f"{value:.3f}"
                elif "VaR" in feature or "CVaR" in feature or "Deviation" in feature:
                    formatted = f"{value:.4f}"
                else:
                    formatted = f"{value:.4f}"

                with col:
                    st.metric(feature, formatted)

                    # Show feature info
                    info = FEATURE_INFO.get(feature, {})
                    if info:
                        with st.expander(f"About {feature}"):
                            st.code(info.get('formula', ''), language=None)
                            st.markdown(f"**Why:** {info.get('why', '')}")
                            st.markdown(f"**Interpretation:** {info.get('interpretation', '')}")

        st.divider()

        # Full summary table
        st.subheader("Complete Metrics Summary")

        summary_df = pd.DataFrame([
            {"Category": "Behavior", "Metric": "Annual Return", "Value": f"{metrics['Annual Return']:.2%}"},
            {"Category": "Behavior", "Metric": "Annual Volatility", "Value": f"{metrics['Annual Volatility']:.2%}"},
            {"Category": "Behavior", "Metric": "Sharpe Ratio", "Value": f"{metrics['Sharpe Ratio']:.3f}"},
            {"Category": "Behavior", "Metric": "Max Drawdown", "Value": f"{metrics['Max Drawdown']:.2%}"},
            {"Category": "Behavior", "Metric": "Calmar Ratio", "Value": f"{metrics['Calmar Ratio']:.3f}"},
            {"Category": "Behavior", "Metric": "Sortino Ratio", "Value": f"{metrics['Sortino Ratio']:.3f}"},
            {"Category": "Risk", "Metric": "VaR (95%)", "Value": f"{metrics['VaR (95%)']:.4f}"},
            {"Category": "Risk", "Metric": "CVaR (95%)", "Value": f"{metrics['CVaR (95%)']:.4f}"},
            {"Category": "Risk", "Metric": "Skewness", "Value": f"{metrics['Skewness']:.4f}"},
            {"Category": "Risk", "Metric": "Kurtosis", "Value": f"{metrics['Kurtosis']:.4f}"},
        ])

        if market_returns is not None:
            summary_df = pd.concat([summary_df, pd.DataFrame([
                {"Category": "Market", "Metric": "Beta", "Value": f"{metrics['Beta']:.3f}"},
                {"Category": "Market", "Metric": "Correlation", "Value": f"{metrics['Correlation']:.3f}"},
                {"Category": "Market", "Metric": "Alpha", "Value": f"{metrics['Alpha']:.2%}"},
                {"Category": "Market", "Metric": "R-Squared", "Value": f"{metrics['R-Squared']:.3f}"},
            ])], ignore_index=True)

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

with tab2:
    if df.empty:
        st.warning(f"No data found for {ticker}. Please check the ticker symbol.")
    else:
        prices = df["Close"].dropna()
        returns = prices.pct_change().dropna()

        st.subheader("Price and Returns Over Time")

        col1, col2 = st.columns(2)

        with col1:
            # Normalized price chart
            norm_prices = prices / prices.iloc[0] * 100
            price_df = pd.DataFrame({
                "Day": range(len(norm_prices)),
                "Value": norm_prices.values
            })

            price_chart = (
                alt.Chart(price_df)
                .mark_line(color=THEME["primary"])
                .encode(
                    x=alt.X("Day:Q", title="Trading Day"),
                    y=alt.Y("Value:Q", title="Normalized Price (Start=100)"),
                    tooltip=["Day", "Value"]
                )
                .properties(width=CHART_WIDTH // 2, height=CHART_HEIGHT // 2,
                           title=f"{ticker} Normalized Price")
            )
            st.altair_chart(configure_altair_theme(price_chart), use_container_width=True)

        with col2:
            # Cumulative returns
            cum_returns = (1 + returns).cumprod() - 1
            cum_df = pd.DataFrame({
                "Day": range(len(cum_returns)),
                "Return": cum_returns.values
            })

            cum_chart = (
                alt.Chart(cum_df)
                .mark_area(opacity=0.6, color=THEME["primary"])
                .encode(
                    x=alt.X("Day:Q", title="Trading Day"),
                    y=alt.Y("Return:Q", title="Cumulative Return", axis=alt.Axis(format='.0%')),
                    tooltip=["Day", alt.Tooltip("Return:Q", format=".2%")]
                )
                .properties(width=CHART_WIDTH // 2, height=CHART_HEIGHT // 2,
                           title=f"{ticker} Cumulative Returns")
            )
            st.altair_chart(configure_altair_theme(cum_chart), use_container_width=True)

        # Rolling metrics
        st.subheader("Rolling Metrics")
        window = st.slider("Rolling Window (days)", min_value=20, max_value=252, value=63)

        col1, col2 = st.columns(2)

        with col1:
            # Rolling volatility
            roll_vol = returns.rolling(window=window).std() * np.sqrt(252)
            roll_vol = roll_vol.dropna()

            vol_df = pd.DataFrame({
                "Day": range(len(roll_vol)),
                "Volatility": roll_vol.values
            })

            vol_chart = (
                alt.Chart(vol_df)
                .mark_area(opacity=0.6, color=THEME["secondary"])
                .encode(
                    x=alt.X("Day:Q", title="Trading Day"),
                    y=alt.Y("Volatility:Q", title="Annualized Volatility", axis=alt.Axis(format='.0%')),
                    tooltip=["Day", alt.Tooltip("Volatility:Q", format=".2%")]
                )
                .properties(width=CHART_WIDTH // 2, height=CHART_HEIGHT // 2,
                           title=f"Rolling {window}-day Volatility")
            )
            st.altair_chart(configure_altair_theme(vol_chart), use_container_width=True)

        with col2:
            # Rolling Sharpe
            roll_ret = returns.rolling(window=window).mean() * 252
            roll_vol_sharpe = returns.rolling(window=window).std() * np.sqrt(252)
            roll_sharpe = (roll_ret / roll_vol_sharpe).dropna()

            sharpe_df = pd.DataFrame({
                "Day": range(len(roll_sharpe)),
                "Sharpe": roll_sharpe.values
            })

            sharpe_base = alt.Chart(sharpe_df).encode(x=alt.X("Day:Q", title="Trading Day"))

            sharpe_line = sharpe_base.mark_line(color=THEME["primary"]).encode(
                y=alt.Y("Sharpe:Q", title="Rolling Sharpe Ratio"),
                tooltip=["Day", alt.Tooltip("Sharpe:Q", format=".2f")]
            )

            zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
                strokeDash=[3, 3], color="gray"
            ).encode(y="y:Q")

            sharpe_chart = (sharpe_line + zero_line).properties(
                width=CHART_WIDTH // 2, height=CHART_HEIGHT // 2,
                title=f"Rolling {window}-day Sharpe Ratio"
            )
            st.altair_chart(configure_altair_theme(sharpe_chart), use_container_width=True)

        # Drawdown chart
        st.subheader("Drawdown Analysis")

        peak = prices.cummax()
        drawdown = (prices / peak) - 1

        dd_df = pd.DataFrame({
            "Day": range(len(drawdown)),
            "Drawdown": drawdown.values
        })

        dd_chart = (
            alt.Chart(dd_df)
            .mark_area(opacity=0.6, color="red")
            .encode(
                x=alt.X("Day:Q", title="Trading Day"),
                y=alt.Y("Drawdown:Q", title="Drawdown", axis=alt.Axis(format='.0%')),
                tooltip=["Day", alt.Tooltip("Drawdown:Q", format=".2%")]
            )
            .properties(width=CHART_WIDTH, height=CHART_HEIGHT // 2,
                       title=f"{ticker} Drawdown from Peak")
        )
        st.altair_chart(configure_altair_theme(dd_chart), use_container_width=True)

with tab3:
    if df.empty:
        st.warning(f"No data found for {ticker}. Please check the ticker symbol.")
    else:
        prices = df["Close"].dropna()
        returns = prices.pct_change().dropna()

        st.subheader("Return Distribution Analysis")

        # Histogram
        hist_df = pd.DataFrame({"Return": returns.values})

        hist_chart = (
            alt.Chart(hist_df)
            .mark_bar(opacity=0.7, color=THEME["primary"])
            .encode(
                x=alt.X("Return:Q", bin=alt.Bin(maxbins=50), title="Daily Return"),
                y=alt.Y("count()", title="Frequency"),
                tooltip=["count()"]
            )
            .properties(width=CHART_WIDTH, height=CHART_HEIGHT // 2,
                       title=f"{ticker} Daily Return Distribution")
        )
        st.altair_chart(configure_altair_theme(hist_chart), use_container_width=True)

        # Distribution statistics
        st.subheader("Distribution Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Mean (Daily)", f"{returns.mean():.4f}")
            st.metric("Mean (Annual)", f"{returns.mean() * 252:.2%}")

        with col2:
            st.metric("Std Dev (Daily)", f"{returns.std():.4f}")
            st.metric("Std Dev (Annual)", f"{returns.std() * np.sqrt(252):.2%}")

        with col3:
            st.metric("Skewness", f"{returns.skew():.4f}")
            skew_interp = "Left tail (losses)" if returns.skew() < 0 else "Right tail (gains)"
            st.caption(skew_interp)

        with col4:
            st.metric("Excess Kurtosis", f"{returns.kurtosis():.4f}")
            kurt_interp = "Fat tails" if returns.kurtosis() > 0 else "Thin tails"
            st.caption(kurt_interp)

        # VaR/CVaR visualization
        st.subheader("Value at Risk Analysis")

        var_5 = returns.quantile(0.05)
        var_1 = returns.quantile(0.01)
        cvar_5 = returns[returns <= var_5].mean()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("VaR (95%)", f"{var_5:.4f}", help="5th percentile of returns")
        with col2:
            st.metric("VaR (99%)", f"{var_1:.4f}", help="1st percentile of returns")
        with col3:
            st.metric("CVaR (95%)", f"{cvar_5:.4f}", help="Average of returns worse than VaR")

with tab4:
    st.subheader("Asset Features: A Complete Guide")
    st.caption("Learn how to measure and interpret return, risk, and market relationships")

    # Quick reference
    st.markdown("### Quick Reference Guide")
    st.markdown("""
    | **Question** | **Key Metric** | **Interpretation** |
    |-------------|---------------|-------------------|
    | How much did it return? | Annual Return | CAGR - compound annual growth |
    | How risky is it? | Annual Volatility | Higher = more price swings |
    | Is the return worth the risk? | Sharpe Ratio | >1 good, >2 very good |
    | What's the worst case? | Max Drawdown | Largest peak-to-trough loss |
    | How does it move with market? | Beta | >1 amplifies, <1 dampens |
    | Did it beat the market? | Alpha | Positive = outperformance |
    | Is it a diversifier? | Correlation | Low/negative = good for diversification |
    | How bad can tail events be? | CVaR | Expected loss in worst scenarios |
    """)

    st.divider()

    # Category deep dives
    st.markdown("### Feature Categories")

    for category, features in FEATURE_CATEGORIES.items():
        cat_info = CATEGORY_INFO.get(category, {})
        with st.expander(f"**{category}**", expanded=False):
            st.markdown(f"**Description:** {cat_info.get('description', '')}")
            st.markdown(f"**Key Principle:** {cat_info.get('key_principle', '')}")
            st.markdown(f"**Best For:** {cat_info.get('best_for', '')}")
            st.info(f"**Rule of Thumb:** {cat_info.get('rule_of_thumb', '')}")

            st.markdown("---")
            st.markdown("**Features in this category:**")

            for feature in features:
                info = FEATURE_INFO.get(feature, {})
                if info:
                    st.markdown(f"#### {feature}")
                    st.code(info.get('formula', ''), language=None)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Why?**")
                        st.write(info.get('why', ''))
                        st.markdown("**When to use:**")
                        for use in info.get('when', []):
                            st.markdown(f"- {use}")

                    with col2:
                        st.markdown("**How?**")
                        st.write(info.get('how', ''))
                        st.markdown("**Interpretation:**")
                        st.write(info.get('interpretation', ''))
                        st.markdown("**Pitfalls:**")
                        for pitfall in info.get('pitfalls', []):
                            st.markdown(f"- {pitfall}")

                    st.markdown("---")

    st.divider()

    # Practical examples
    st.markdown("### Practical Examples")

    with st.expander("Example 1: Evaluating a Stock Investment"):
        st.markdown("""
        **Scenario:** You're considering investing in a tech stock and want to understand its risk/return profile.

        **Key metrics to check:**
        1. **Annual Return** - Is the growth rate attractive?
        2. **Annual Volatility** - Can you handle the price swings?
        3. **Sharpe Ratio** - Is return adequate for the risk?
        4. **Max Drawdown** - Could you stomach a 40% drop?
        5. **Beta** - Will it amplify market crashes?

        **Example interpretation:**
        - Annual Return: 25% → Strong growth
        - Volatility: 35% → High risk, expect ±35% swings
        - Sharpe: 0.71 → Decent but not great risk-adjusted return
        - Max Drawdown: -45% → Painful worst case
        - Beta: 1.3 → Amplifies market by 30%

        **Conclusion:** High return comes with high risk. Only suitable for investors who can tolerate significant drawdowns.
        """)

    with st.expander("Example 2: Building a Diversified Portfolio"):
        st.markdown("""
        **Scenario:** You want to add assets that reduce overall portfolio risk.

        **Key metrics to check:**
        1. **Correlation** - Low correlation = better diversification
        2. **Beta** - Different betas provide different market exposures
        3. **Max Drawdown** - Ideally not all assets draw down together

        **Example:**
        | Asset | Correlation to S&P 500 | Beta | Role |
        |-------|----------------------|------|------|
        | Tech Stock | 0.85 | 1.3 | Growth driver |
        | Utility Stock | 0.45 | 0.5 | Defensive equity |
        | Gold | 0.05 | 0.1 | Crisis hedge |
        | Bonds | -0.20 | -0.1 | Flight to safety |

        **Conclusion:** Combining assets with different correlation and beta profiles reduces overall portfolio volatility.
        """)

    with st.expander("Example 3: Comparing Fund Managers"):
        st.markdown("""
        **Scenario:** You're choosing between two active managers.

        **Key metrics:**
        1. **Alpha** - Did they generate excess return?
        2. **Information Ratio** - Consistency of outperformance
        3. **Max Drawdown** - Risk management quality
        4. **Sharpe Ratio** - Overall risk-adjusted return

        **Example comparison:**
        | Metric | Manager A | Manager B |
        |--------|----------|----------|
        | Alpha | 3% | 5% |
        | Information Ratio | 0.8 | 0.4 |
        | Max Drawdown | -15% | -35% |
        | Sharpe | 1.2 | 0.9 |

        **Conclusion:** Manager A has lower alpha but much better risk management and consistency. May be preferable for risk-averse investors.
        """)

    st.divider()

    # Common mistakes
    st.markdown("### Common Mistakes to Avoid")

    st.error("""
    **1. Ignoring Risk for Return**
    High returns often come with high volatility and drawdowns.
    **Fix:** Always check Sharpe ratio and max drawdown alongside returns.
    """)

    st.error("""
    **2. Assuming Stable Correlations**
    Correlations increase during market crises - exactly when you need diversification.
    **Fix:** Test portfolio behavior in stress scenarios, not just normal markets.
    """)

    st.error("""
    **3. Overfitting to Historical Data**
    Past performance doesn't guarantee future results.
    **Fix:** Use out-of-sample testing and be skeptical of unusually high Sharpe ratios.
    """)

    st.error("""
    **4. Misinterpreting Beta**
    High beta isn't good or bad - it depends on your market view.
    **Fix:** If bullish, high beta amplifies gains. If cautious, low beta protects.
    """)

    st.error("""
    **5. Ignoring Tail Risk**
    Volatility doesn't capture rare extreme events.
    **Fix:** Check VaR, CVaR, skewness, and kurtosis for tail risk assessment.
    """)
