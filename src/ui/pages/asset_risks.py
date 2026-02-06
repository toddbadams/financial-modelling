"""Asset Risks page - Analyze volatility, drawdowns, VaR/CVaR, and risk metrics."""

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
    "Asset Risks",
    "Analyze volatility, drawdowns, and tail risk metrics for selected assets. "
    "Learn how to measure and interpret different types of financial risk, from "
    "traditional volatility to Value at Risk (VaR) and Conditional VaR.",
)

# Educational content for risk types
RISK_TYPE_INFO = {
    "Systematic Risk (Market Risk)": {
        "description": "Risk that affects the entire market or economy and cannot be diversified away.",
        "examples": "Recessions, interest rate hikes, financial crises, pandemics, inflation shocks",
        "key_point": "Even a perfectly diversified portfolio is still exposed to systematic risk.",
        "measurement": "Measured by Beta relative to market benchmark",
    },
    "Unsystematic Risk (Specific Risk)": {
        "description": "Risk unique to a company, sector, or asset that can be reduced through diversification.",
        "examples": "Poor earnings reports, management scandals, product failures, negative sentiment news",
        "key_point": "Diversification can eliminate unsystematic risk by holding multiple uncorrelated assets.",
        "measurement": "Measured by residual volatility after removing market component",
    },
}

# Risk metric educational content
RISK_METRIC_INFO = {
    "Annual Volatility": {
        "formula": "σ_annual = σ_daily × √252",
        "what": "Standard deviation of returns scaled to annual basis, measuring how much an asset's returns typically fluctuate over a year.",
        "why": "The standard proxy for financial risk. Higher volatility means larger price swings in both directions.",
        "interpretation": "20% annual volatility means returns typically fluctuate within ±20% of the mean in a given year (1 standard deviation).",
        "pitfalls": [
            "Assumes returns are normally distributed (they have fat tails)",
            "Treats upside and downside volatility equally",
            "Historical volatility may not predict future volatility",
        ],
    },
    "Rolling Volatility": {
        "formula": "σ_rolling(t) = std(returns[t-window:t]) × √252",
        "what": "Time-varying volatility calculated over a moving window, showing how risk changes over time.",
        "why": "Volatility clusters - high volatility tends to follow high volatility. Rolling measures capture regime changes.",
        "interpretation": "Rising rolling volatility indicates increasing uncertainty; falling volatility suggests calmer markets.",
        "pitfalls": [
            "Window size affects responsiveness vs stability trade-off",
            "Backward-looking - can't predict sudden regime changes",
        ],
    },
    "Maximum Drawdown": {
        "formula": "DD_max = min_t((P_t / max(P_s for s≤t)) - 1)",
        "what": "The worst peak-to-trough decline in an asset's price history, measuring maximum pain an investor would have experienced.",
        "why": "Captures tail risk that volatility misses. A -50% drawdown requires +100% gain to recover.",
        "interpretation": "-30% max drawdown means at worst, an investor lost 30% from a previous high before any recovery.",
        "pitfalls": [
            "Only captures one extreme event",
            "Doesn't indicate how long recovery took",
            "Historical max drawdown may be exceeded in future",
        ],
    },
    "Value at Risk (VaR)": {
        "formula": "VaR_α = Percentile(returns, α)",
        "what": "The loss threshold that is only exceeded in the worst α% of days. A 'bad-day cutoff'.",
        "why": "Widely used regulatory risk metric that quantifies potential daily losses at a given confidence level.",
        "interpretation": "VaR of -2% at 95% means: on 95% of days, losses won't exceed 2%. But it doesn't say how bad the worst 5% can be.",
        "pitfalls": [
            "Doesn't say how bad losses can be beyond VaR",
            "Not sub-additive (portfolio VaR can exceed sum of parts)",
            "Assumes historical distribution continues",
        ],
    },
    "Conditional VaR (CVaR)": {
        "formula": "CVaR_α = E[r | r ≤ VaR_α]",
        "what": "The expected (average) loss when VaR is breached. Also called Expected Shortfall.",
        "why": "More informative than VaR for tail risk - tells you how bad the truly bad days are.",
        "interpretation": "CVaR of -3% means: when losses exceed VaR, they average -3%. Shows the cost of tail events.",
        "pitfalls": [
            "Requires sufficient extreme observations for stable estimate",
            "Still based on historical data",
        ],
    },
    "Downside Deviation": {
        "formula": "σ_down = √(E[min(r - target, 0)²]) × √252",
        "what": "Standard deviation calculated using only negative returns (or returns below a target).",
        "why": "Recognizes that investors don't mind upside volatility - only downside matters for pain.",
        "interpretation": "Lower downside deviation means less severe negative returns. Used in Sortino ratio calculation.",
        "pitfalls": [
            "Requires enough negative returns for stable estimate",
            "Choice of target return affects calculation",
        ],
    },
    "Skewness": {
        "formula": "Skew = E[(r - μ)³] / σ³",
        "what": "Measures asymmetry of the return distribution. Negative skew means more frequent small gains but rare large losses.",
        "why": "Fat left tails are dangerous - occasional extreme losses can devastate portfolios.",
        "interpretation": "Skew > 0: right tail (occasional large gains). Skew < 0: left tail (occasional large losses).",
        "pitfalls": [
            "Sensitive to outliers",
            "May not be stable over time",
        ],
    },
    "Kurtosis": {
        "formula": "Kurt = E[(r - μ)⁴] / σ⁴ - 3 (excess kurtosis)",
        "what": "Measures tail heaviness compared to normal distribution. Higher kurtosis means more extreme events.",
        "why": "Normal distribution underestimates extremes. Kurtosis quantifies how much fatter the tails are.",
        "interpretation": "Kurt > 0: fatter tails than normal (more extremes). Kurt = 0: normal distribution. Most assets have positive excess kurtosis.",
        "pitfalls": [
            "Very sensitive to outliers",
            "Requires large sample for stable estimate",
        ],
    },
}

# Drawdown educational content
DRAWDOWN_INFO = {
    "what": "A drawdown measures the peak-to-trough decline during a specific period of an investment. It's the percentage loss from the highest point to the lowest point before a new high is reached.",
    "why_matters": [
        "Psychological impact: A -50% loss requires +100% gain to recover",
        "Time impact: Deep drawdowns often take years to recover",
        "Survival: Excessive drawdowns can force liquidation at the worst time",
    ],
    "interpretation": {
        "0% to -10%": "Normal market fluctuation, expected regularly",
        "-10% to -20%": "Correction territory, occurs every 1-2 years on average",
        "-20% to -30%": "Bear market, significant pain but recoverable",
        "-30% to -50%": "Severe bear market, extended recovery period",
        "Below -50%": "Crisis level, may take 5+ years to recover",
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


def max_drawdown(prices: pd.Series) -> float:
    """Calculate maximum drawdown from price series."""
    peak = prices.cummax()
    drawdown = (prices / peak) - 1
    return drawdown.min()


def calculate_drawdown_series(prices: pd.Series) -> pd.Series:
    """Calculate drawdown series from price series."""
    peak = prices.cummax()
    return (prices / peak) - 1


def var_at_level(returns: pd.Series, alpha: float = 0.05) -> float:
    """Calculate Value at Risk at given confidence level."""
    return returns.quantile(alpha)


def cvar_at_level(returns: pd.Series, alpha: float = 0.05) -> float:
    """Calculate Conditional VaR (Expected Shortfall) at given level."""
    var = var_at_level(returns, alpha)
    return returns[returns <= var].mean()


def downside_deviation(returns: pd.Series, target: float = 0.0) -> float:
    """Calculate annualized downside deviation."""
    downside_returns = np.minimum(returns - target, 0)
    return np.sqrt((downside_returns ** 2).mean()) * np.sqrt(252)


def calculate_rolling_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """Calculate rolling annualized volatility."""
    return returns.rolling(window=window).std() * np.sqrt(252)


def calculate_rolling_sharpe(returns: pd.Series, window: int = 63, rf: float = 0.0) -> pd.Series:
    """Calculate rolling Sharpe ratio."""
    roll_ret = returns.rolling(window=window).mean() * 252
    roll_vol = returns.rolling(window=window).std() * np.sqrt(252)
    return (roll_ret - rf) / roll_vol


def calculate_rolling_var(returns: pd.Series, window: int = 63, alpha: float = 0.05) -> pd.Series:
    """Calculate rolling VaR."""
    return returns.rolling(window=window).quantile(alpha)


def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculate beta relative to market."""
    cov = np.cov(asset_returns, market_returns)[0, 1]
    var = market_returns.var()
    if var == 0:
        return 0.0
    return cov / var


# Sidebar controls
with st.sidebar:
    st.header("Risk Analysis Options")

    ticker = st.text_input("Ticker Symbol", value="AAPL")
    benchmark = st.text_input("Market Benchmark", value="^GSPC",
                             help="Used for beta calculation and market comparison")

    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=365 * 3),
        help="Longer history provides more reliable risk estimates",
    )
    end_date = st.date_input("End Date", value=date.today())

    st.subheader("Rolling Window Settings")
    volatility_window = st.slider(
        "Volatility Window (days)",
        min_value=5,
        max_value=126,
        value=21,
        help="21 days ≈ 1 month, 63 days ≈ 3 months",
    )

    var_window = st.slider(
        "VaR Window (days)",
        min_value=30,
        max_value=252,
        value=63,
        help="Window for calculating rolling VaR",
    )

    st.subheader("Risk Settings")
    confidence_level = st.slider(
        "VaR Confidence Level (%)",
        min_value=90,
        max_value=99,
        value=95,
        help="95% means looking at worst 5% of days",
    )

    risk_free_rate = st.number_input(
        "Risk-Free Rate (%)",
        value=4.0,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
    )

# Fetch data
df = DataInjestor.get(ticker, str(start_date), str(end_date))
market_df = DataInjestor.get(benchmark, str(start_date), str(end_date))

if df.empty:
    st.error(f"No data found for {ticker}. Please check the ticker symbol and date range.")
    st.stop()

# Calculate returns and prices
prices = df["Close"].dropna()
returns = prices.pct_change().dropna()

market_prices = market_df["Close"].dropna() if not market_df.empty else None
market_returns = market_prices.pct_change().dropna() if market_prices is not None else None

# Calculate alpha for VaR
alpha = (100 - confidence_level) / 100
rf = risk_free_rate / 100

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Volatility Analysis",
    "Drawdown Chart",
    "VaR/CVaR",
    "Learning Center"
])

with tab1:
    st.subheader(f"Volatility Analysis: {ticker}")

    with st.expander("Understanding Volatility", expanded=False):
        st.markdown("""
        **Volatility** measures the dispersion of returns - how much prices fluctuate over time.
        It's the standard measure of financial risk.

        **Key concepts:**
        - **Annualized volatility** scales daily fluctuations to yearly basis (×√252)
        - **Rolling volatility** shows how risk changes over time
        - **Volatility clusters** - high volatility tends to follow high volatility

        **Interpretation:**
        - 15% volatility = "low risk" (bonds, utilities)
        - 20-25% volatility = "moderate risk" (diversified equities)
        - 30%+ volatility = "high risk" (tech stocks, emerging markets)
        """)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    ann_vol = annual_volatility(returns)
    ann_ret = annual_return(returns)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    down_dev = downside_deviation(returns)

    with col1:
        st.metric("Annual Volatility", f"{ann_vol:.2%}")
        st.caption("Risk measure (std dev)")

    with col2:
        st.metric("Annual Return", f"{ann_ret:.2%}")
        st.caption("CAGR")

    with col3:
        st.metric("Sharpe Ratio", f"{sharpe:.3f}")
        st.caption("Risk-adjusted return")

    with col4:
        st.metric("Downside Deviation", f"{down_dev:.2%}")
        st.caption("Only negative returns")

    # Distribution statistics
    st.subheader("Return Distribution")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean (Daily)", f"{returns.mean():.4f}")
        st.metric("Mean (Annual)", f"{returns.mean() * 252:.2%}")

    with col2:
        st.metric("Std Dev (Daily)", f"{returns.std():.4f}")
        st.metric("Std Dev (Annual)", f"{ann_vol:.2%}")

    with col3:
        skew = returns.skew()
        st.metric("Skewness", f"{skew:.4f}")
        skew_interp = "Left tail (losses)" if skew < 0 else "Right tail (gains)"
        st.caption(skew_interp)

    with col4:
        kurt = returns.kurtosis()
        st.metric("Excess Kurtosis", f"{kurt:.4f}")
        kurt_interp = "Fat tails" if kurt > 0 else "Thin tails"
        st.caption(kurt_interp)

    # Return distribution histogram
    st.subheader("Daily Return Distribution")

    hist_df = pd.DataFrame({"Return": returns.values})

    hist_chart = (
        alt.Chart(hist_df)
        .mark_bar(opacity=0.7, color=THEME["primary"])
        .encode(
            x=alt.X("Return:Q", bin=alt.Bin(maxbins=50), title="Daily Return"),
            y=alt.Y("count()", title="Frequency"),
            tooltip=["count()"]
        )
        .properties(width=CHART_WIDTH, height=300,
                   title=f"{ticker} Daily Return Distribution")
    )

    # Add normal distribution reference
    st.altair_chart(configure_altair_theme(hist_chart), use_container_width=True)

    with st.expander("Distribution Analysis", expanded=False):
        st.markdown(f"""
        **{ticker} Return Distribution Analysis:**

        - **Skewness = {skew:.4f}**: {"Negative skew indicates occasional large losses (left tail risk)" if skew < 0 else "Positive skew indicates occasional large gains"}
        - **Excess Kurtosis = {kurt:.4f}**: {"Fat tails - extreme events occur more often than normal distribution predicts" if kurt > 0 else "Thin tails - extreme events are rare"}

        **Implication:** {"Standard volatility underestimates tail risk. Consider VaR/CVaR for better risk assessment." if kurt > 0 else "Distribution is close to normal."}
        """)

    # Rolling volatility chart
    st.subheader(f"Rolling {volatility_window}-Day Volatility")

    roll_vol = calculate_rolling_volatility(returns, window=volatility_window).dropna()

    vol_df = pd.DataFrame({
        "Date": roll_vol.index,
        "Volatility": roll_vol.values
    })

    vol_chart = (
        alt.Chart(vol_df)
        .mark_area(opacity=0.6, color=THEME["secondary"])
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Volatility:Q", title="Annualized Volatility",
                   axis=alt.Axis(format=".0%")),
            tooltip=["Date:T", alt.Tooltip("Volatility:Q", format=".2%")]
        )
        .properties(width=CHART_WIDTH, height=300,
                   title=f"Rolling {volatility_window}-Day Annualized Volatility")
    )

    # Add average line
    avg_vol = roll_vol.mean()
    avg_line = (
        alt.Chart(pd.DataFrame({"y": [avg_vol]}))
        .mark_rule(strokeDash=[5, 5], color="white")
        .encode(y="y:Q")
    )

    st.altair_chart(configure_altair_theme(vol_chart + avg_line), use_container_width=True)

    st.caption(f"Average rolling volatility: {avg_vol:.2%}")

    # Rolling Sharpe
    st.subheader(f"Rolling {var_window}-Day Sharpe Ratio")

    roll_sharpe = calculate_rolling_sharpe(returns, window=var_window, rf=rf).dropna()

    sharpe_df = pd.DataFrame({
        "Date": roll_sharpe.index,
        "Sharpe": roll_sharpe.values
    })

    sharpe_line = (
        alt.Chart(sharpe_df)
        .mark_line(color=THEME["primary"])
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Sharpe:Q", title="Rolling Sharpe Ratio"),
            tooltip=["Date:T", alt.Tooltip("Sharpe:Q", format=".2f")]
        )
    )

    zero_line = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(strokeDash=[3, 3], color="gray")
        .encode(y="y:Q")
    )

    sharpe_chart = (sharpe_line + zero_line).properties(
        width=CHART_WIDTH, height=300,
        title=f"Rolling {var_window}-Day Sharpe Ratio"
    )

    st.altair_chart(configure_altair_theme(sharpe_chart), use_container_width=True)

    # Market comparison
    if market_returns is not None:
        st.subheader("Market Relationship")

        # Align data
        common_idx = returns.index.intersection(market_returns.index)
        asset_r = returns.loc[common_idx]
        market_r = market_returns.loc[common_idx]

        beta = calculate_beta(asset_r, market_r)
        corr = asset_r.corr(market_r)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Beta vs Market", f"{beta:.3f}")
            if beta > 1:
                st.caption("Amplifies market moves")
            elif beta < 1:
                st.caption("Dampens market moves")
            else:
                st.caption("Moves with market")

        with col2:
            st.metric("Correlation", f"{corr:.3f}")
            if corr > 0.7:
                st.caption("High correlation")
            elif corr > 0.3:
                st.caption("Moderate correlation")
            else:
                st.caption("Low correlation - diversifier")

        with col3:
            # Calculate alpha
            market_ann_ret = annual_return(market_r)
            alpha_val = ann_ret - (rf + beta * (market_ann_ret - rf))
            st.metric("Alpha", f"{alpha_val:.2%}")
            if alpha_val > 0:
                st.caption("Outperforming risk-adjusted")
            else:
                st.caption("Underperforming risk-adjusted")


with tab2:
    st.subheader(f"Drawdown Analysis: {ticker}")

    with st.expander("Understanding Drawdowns", expanded=False):
        st.markdown("""
        **Drawdown** measures the peak-to-trough decline during a specific period.
        It answers: "How much would I have lost if I bought at the worst possible time?"

        **Why drawdowns matter:**
        - **Psychological impact:** A -50% loss requires +100% gain to recover
        - **Time impact:** Deep drawdowns often take years to recover
        - **Survival:** Excessive drawdowns can force liquidation at the worst time

        **Interpretation:**
        | Drawdown Level | Interpretation |
        |---------------|----------------|
        | 0% to -10% | Normal fluctuation |
        | -10% to -20% | Correction territory |
        | -20% to -30% | Bear market |
        | -30% to -50% | Severe bear market |
        | Below -50% | Crisis level |
        """)

    # Calculate drawdown series
    drawdown = calculate_drawdown_series(prices)
    max_dd = max_drawdown(prices)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Maximum Drawdown", f"{max_dd:.2%}")
        st.caption("Worst peak-to-trough")

    with col2:
        current_dd = drawdown.iloc[-1]
        st.metric("Current Drawdown", f"{current_dd:.2%}")
        st.caption("From recent peak")

    with col3:
        avg_dd = drawdown.mean()
        st.metric("Average Drawdown", f"{avg_dd:.2%}")
        st.caption("Mean drawdown level")

    with col4:
        # Time underwater
        underwater_pct = (drawdown < -0.05).mean()
        st.metric("Time > 5% Below Peak", f"{underwater_pct:.1%}")
        st.caption("Percentage of time")

    # Drawdown chart
    st.subheader("Drawdown Over Time")

    dd_df = pd.DataFrame({
        "Date": drawdown.index,
        "Drawdown": drawdown.values
    })

    dd_chart = (
        alt.Chart(dd_df)
        .mark_area(opacity=0.6, color="red")
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Drawdown:Q", title="Drawdown",
                   axis=alt.Axis(format=".0%")),
            tooltip=["Date:T", alt.Tooltip("Drawdown:Q", format=".2%")]
        )
        .properties(width=CHART_WIDTH, height=350,
                   title=f"{ticker} Drawdown from Peak")
    )

    # Add max drawdown line
    max_dd_line = (
        alt.Chart(pd.DataFrame({"y": [max_dd]}))
        .mark_rule(strokeDash=[5, 5], color="white")
        .encode(y="y:Q")
    )

    st.altair_chart(configure_altair_theme(dd_chart + max_dd_line), use_container_width=True)

    st.caption(f"Dashed line shows maximum drawdown: {max_dd:.2%}")

    # Drawdown histogram
    st.subheader("Drawdown Distribution")

    dd_hist_df = pd.DataFrame({"Drawdown": drawdown.values})

    dd_hist = (
        alt.Chart(dd_hist_df)
        .mark_bar(opacity=0.7, color="red")
        .encode(
            x=alt.X("Drawdown:Q", bin=alt.Bin(maxbins=30), title="Drawdown Level"),
            y=alt.Y("count()", title="Frequency"),
            tooltip=["count()"]
        )
        .properties(width=CHART_WIDTH, height=250,
                   title="Distribution of Drawdown Levels")
    )

    st.altair_chart(configure_altair_theme(dd_hist), use_container_width=True)

    # Largest drawdowns table
    st.subheader("Worst Drawdown Periods")

    # Find drawdown periods
    def find_drawdown_periods(drawdown_series, prices, n_worst=5):
        """Find the worst drawdown periods."""
        periods = []
        in_drawdown = False
        start_idx = None
        trough_idx = None
        trough_val = 0

        for i, (date, dd) in enumerate(drawdown_series.items()):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i - 1 if i > 0 else i
                trough_val = dd
                trough_idx = i
            elif dd < 0 and in_drawdown:
                if dd < trough_val:
                    trough_val = dd
                    trough_idx = i
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                periods.append({
                    "start": drawdown_series.index[start_idx],
                    "trough": drawdown_series.index[trough_idx],
                    "end": drawdown_series.index[i],
                    "max_dd": trough_val,
                    "duration": i - start_idx
                })

        # Sort by max drawdown
        periods = sorted(periods, key=lambda x: x["max_dd"])[:n_worst]
        return periods

    try:
        dd_periods = find_drawdown_periods(drawdown, prices, n_worst=5)
        if dd_periods:
            periods_df = pd.DataFrame(dd_periods)
            periods_df["max_dd"] = periods_df["max_dd"].apply(lambda x: f"{x:.2%}")
            periods_df.columns = ["Peak Date", "Trough Date", "Recovery Date", "Max Drawdown", "Duration (days)"]
            st.dataframe(periods_df, use_container_width=True, hide_index=True)
        else:
            st.info("No complete drawdown periods found (price may still be below peak).")
    except Exception:
        st.info("Unable to calculate drawdown periods.")

    # Price chart with peak overlay
    st.subheader("Price vs Running Peak")

    peak = prices.cummax()

    price_df = pd.DataFrame({
        "Date": prices.index,
        "Price": prices.values,
        "Peak": peak.values
    }).melt(id_vars="Date", var_name="Series", value_name="Value")

    price_chart = (
        alt.Chart(price_df)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Value:Q", title="Price"),
            color=alt.Color("Series:N", scale=alt.Scale(
                domain=["Price", "Peak"],
                range=[THEME["primary"], THEME["secondary"]]
            )),
            strokeDash=alt.condition(
                alt.datum.Series == "Peak",
                alt.value([5, 5]),
                alt.value([0])
            ),
            tooltip=["Date:T", "Series:N", alt.Tooltip("Value:Q", format=",.2f")]
        )
        .properties(width=CHART_WIDTH, height=300,
                   title=f"{ticker} Price vs Running Peak")
    )

    st.altair_chart(configure_altair_theme(price_chart), use_container_width=True)


with tab3:
    st.subheader(f"Value at Risk Analysis: {ticker}")

    with st.expander("Understanding VaR and CVaR", expanded=False):
        st.markdown(f"""
        **Value at Risk (VaR)** and **Conditional VaR (CVaR)** focus on extreme downside outcomes.

        **VaR at {confidence_level}%** answers: "What is the worst daily loss I can expect {confidence_level}% of the time?"

        **CVaR at {confidence_level}%** answers: "When losses exceed VaR, how bad are they on average?"

        **Example interpretation:**
        - VaR = -2%: On {confidence_level}% of days, you won't lose more than 2%
        - CVaR = -3%: When you do have a bad day (worst {100-confidence_level}%), the average loss is 3%

        **Why CVaR is better than VaR:**
        - VaR only tells you the threshold, not what happens beyond it
        - CVaR captures tail risk more completely
        - CVaR is sub-additive (portfolio CVaR ≤ sum of individual CVaRs)
        """)

    # Calculate VaR and CVaR
    var = var_at_level(returns, alpha)
    cvar = cvar_at_level(returns, alpha)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(f"VaR ({confidence_level}%)", f"{var:.4f}")
        st.caption(f"Worst {100-confidence_level}% threshold")

    with col2:
        st.metric(f"CVaR ({confidence_level}%)", f"{cvar:.4f}")
        st.caption(f"Average of worst {100-confidence_level}%")

    with col3:
        # Annualized VaR (approximate)
        ann_var = var * np.sqrt(252)
        st.metric("Annualized VaR", f"{ann_var:.2%}")
        st.caption("Scaled to yearly")

    with col4:
        # Number of VaR breaches
        breaches = (returns <= var).sum()
        breach_pct = breaches / len(returns)
        st.metric("VaR Breaches", f"{breaches} ({breach_pct:.1%})")
        st.caption(f"Expected: {100-confidence_level}%")

    # VaR visualization on histogram
    st.subheader("Return Distribution with VaR/CVaR")

    hist_df = pd.DataFrame({"Return": returns.values})

    # Histogram
    hist = (
        alt.Chart(hist_df)
        .mark_bar(opacity=0.7, color=THEME["primary"])
        .encode(
            x=alt.X("Return:Q", bin=alt.Bin(maxbins=50), title="Daily Return"),
            y=alt.Y("count()", title="Frequency"),
        )
    )

    # VaR line
    var_rule = (
        alt.Chart(pd.DataFrame({"x": [var]}))
        .mark_rule(color="red", strokeWidth=2)
        .encode(x="x:Q")
    )

    # CVaR line
    cvar_rule = (
        alt.Chart(pd.DataFrame({"x": [cvar]}))
        .mark_rule(color="orange", strokeWidth=2, strokeDash=[5, 5])
        .encode(x="x:Q")
    )

    var_chart = (hist + var_rule + cvar_rule).properties(
        width=CHART_WIDTH, height=300,
        title=f"Daily Returns with VaR (red) and CVaR (orange) at {confidence_level}%"
    )

    st.altair_chart(configure_altair_theme(var_chart), use_container_width=True)

    st.caption(f"Red line: VaR = {var:.4f} | Orange dashed: CVaR = {cvar:.4f}")

    # Rolling VaR
    st.subheader(f"Rolling {var_window}-Day VaR")

    roll_var = calculate_rolling_var(returns, window=var_window, alpha=alpha).dropna()

    var_df = pd.DataFrame({
        "Date": roll_var.index,
        "VaR": roll_var.values
    })

    roll_var_chart = (
        alt.Chart(var_df)
        .mark_area(opacity=0.6, color="red")
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("VaR:Q", title=f"VaR ({confidence_level}%)",
                   axis=alt.Axis(format=".2%")),
            tooltip=["Date:T", alt.Tooltip("VaR:Q", format=".4f")]
        )
        .properties(width=CHART_WIDTH, height=300,
                   title=f"Rolling {var_window}-Day VaR at {confidence_level}% Confidence")
    )

    st.altair_chart(configure_altair_theme(roll_var_chart), use_container_width=True)

    # VaR at different confidence levels
    st.subheader("VaR at Different Confidence Levels")

    var_levels = [90, 95, 99]
    var_table = []

    for level in var_levels:
        a = (100 - level) / 100
        v = var_at_level(returns, a)
        cv = cvar_at_level(returns, a)
        var_table.append({
            "Confidence Level": f"{level}%",
            "VaR": f"{v:.4f}",
            "CVaR": f"{cv:.4f}",
            "Annual VaR (approx)": f"{v * np.sqrt(252):.2%}",
            "Expected Breaches": f"{(100-level)/100:.1%}"
        })

    var_table_df = pd.DataFrame(var_table)
    st.dataframe(var_table_df, use_container_width=True, hide_index=True)

    # Tail analysis
    st.subheader("Tail Risk Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Worst Daily Returns**")

        worst_returns = returns.nsmallest(10)
        worst_df = pd.DataFrame({
            "Date": worst_returns.index.strftime("%Y-%m-%d"),
            "Return": worst_returns.values
        })
        worst_df["Return"] = worst_df["Return"].apply(lambda x: f"{x:.4f}")
        st.dataframe(worst_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Best Daily Returns**")

        best_returns = returns.nlargest(10)
        best_df = pd.DataFrame({
            "Date": best_returns.index.strftime("%Y-%m-%d"),
            "Return": best_returns.values
        })
        best_df["Return"] = best_df["Return"].apply(lambda x: f"{x:.4f}")
        st.dataframe(best_df, use_container_width=True, hide_index=True)

    # Tail ratio
    p95 = returns.quantile(0.95)
    p5 = returns.quantile(0.05)
    tail_ratio = abs(p95 / p5) if p5 != 0 else 0

    st.metric("Tail Ratio (95th/5th percentile)", f"{tail_ratio:.3f}",
             help="Ratio > 1 means larger potential gains than losses")


with tab4:
    st.subheader("Asset Risks: A Complete Guide")
    st.caption("Learn how to measure and interpret financial risk")

    # Quick reference
    st.markdown("### Quick Reference Guide")
    st.markdown("""
    | **Question** | **Use This Metric** | **Interpretation** |
    |-------------|--------------------|--------------------|
    | How much does price fluctuate? | Annual Volatility | Higher = more risk |
    | What's the worst historical loss? | Max Drawdown | More negative = worse |
    | How bad can daily losses get? | VaR | Threshold for bad days |
    | How bad are the really bad days? | CVaR | Average of tail losses |
    | Does it amplify market moves? | Beta | >1 amplifies, <1 dampens |
    | Is downside worse than upside? | Skewness | Negative = left tail risk |
    | Are extremes more common than expected? | Kurtosis | Positive = fat tails |
    """)

    st.divider()

    # Types of risk
    st.markdown("### Types of Financial Risk")

    for risk_type, info in RISK_TYPE_INFO.items():
        with st.expander(f"**{risk_type}**", expanded=False):
            st.markdown(f"**Description:** {info['description']}")
            st.markdown(f"**Examples:** {info['examples']}")
            st.warning(f"**Key Point:** {info['key_point']}")
            st.markdown(f"**Measurement:** {info['measurement']}")

    st.divider()

    # Risk metrics deep dive
    st.markdown("### Risk Metrics Explained")

    for metric, info in RISK_METRIC_INFO.items():
        with st.expander(f"**{metric}**", expanded=False):
            st.code(info['formula'], language=None)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**What is it?**")
                st.write(info['what'])
                st.markdown("**Why use it?**")
                st.write(info['why'])
            with col2:
                st.markdown("**Interpretation:**")
                st.write(info['interpretation'])
                st.markdown("**Pitfalls:**")
                for pitfall in info['pitfalls']:
                    st.markdown(f"- {pitfall}")

    st.divider()

    # Practical examples
    st.markdown("### Practical Examples")

    with st.expander("Example 1: Comparing Two Investments"):
        st.markdown("""
        **Scenario:** You're choosing between a tech stock and a utility stock.

        | Metric | Tech Stock | Utility Stock |
        |--------|-----------|---------------|
        | Annual Return | 25% | 8% |
        | Annual Volatility | 35% | 15% |
        | Max Drawdown | -45% | -20% |
        | Sharpe Ratio | 0.71 | 0.53 |
        | VaR (95%) | -3.2% | -1.4% |

        **Analysis:**
        - Tech has higher return but also higher risk across all metrics
        - Sharpe ratio favors tech (better risk-adjusted return)
        - But max drawdown is critical: -45% requires +82% to recover
        - VaR shows daily risk is 2x higher for tech

        **Conclusion:** Choice depends on risk tolerance and time horizon.
        """)

    with st.expander("Example 2: Understanding a -30% Drawdown"):
        st.markdown("""
        **Scenario:** Your portfolio just experienced a -30% drawdown.

        **What this means:**
        - You need +42.9% gain to get back to even
        - If you average 10% per year, recovery takes ~4 years
        - If you need the money soon, you're locked in at a loss

        **Recovery math:**
        | Drawdown | Required Gain to Recover |
        |----------|-------------------------|
        | -10% | +11.1% |
        | -20% | +25.0% |
        | -30% | +42.9% |
        | -40% | +66.7% |
        | -50% | +100.0% |

        **Lesson:** Avoiding large drawdowns is more important than maximizing returns.
        """)

    with st.expander("Example 3: Using VaR for Position Sizing"):
        st.markdown("""
        **Scenario:** You have $100,000 and want to limit daily losses to $2,000.

        **If an asset has VaR (95%) = -3%:**
        - Maximum position = $2,000 / 3% = $66,667
        - This limits expected daily loss to $2,000 on 95% of days

        **But remember:**
        - On the worst 5% of days, losses will exceed $2,000
        - CVaR tells you the average loss on those bad days
        - If CVaR = -4.5%, average bad-day loss = $66,667 × 4.5% = $3,000

        **Best practice:** Size positions using CVaR for more conservative risk management.
        """)

    st.divider()

    # Common mistakes
    st.markdown("### Common Mistakes to Avoid")

    st.error("""
    **1. Confusing Volatility with Risk**
    Volatility measures price fluctuation, not the chance of permanent loss.
    **Fix:** Also consider max drawdown, VaR, and fundamental risk factors.
    """)

    st.error("""
    **2. Assuming Normal Distribution**
    Real returns have fat tails - extreme events happen more than expected.
    **Fix:** Check kurtosis and use CVaR instead of just VaR.
    """)

    st.error("""
    **3. Relying on Historical Metrics**
    Past volatility doesn't predict future volatility, especially in crises.
    **Fix:** Stress test with higher volatility assumptions (e.g., 2x historical).
    """)

    st.error("""
    **4. Ignoring Correlation Changes**
    Correlations spike during crises, reducing diversification when you need it most.
    **Fix:** Test portfolio risk with correlation = 0.9 for stress scenarios.
    """)

    st.error("""
    **5. Looking Only at VaR**
    VaR tells you the threshold but not how bad losses can get beyond it.
    **Fix:** Always report CVaR alongside VaR for tail risk context.
    """)
