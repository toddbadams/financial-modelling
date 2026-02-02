"""Asset Normalization page - Explore normalization techniques for financial time series."""

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
from components.charts import create_time_series, create_density_plot
from components.theme import THEME, CHART_WIDTH, CHART_HEIGHT, configure_altair_theme
import altair as alt

# Page header
render_page_header(
    "Asset Normalization",
    "Explore different normalization techniques for financial time series data. "
    "Normalization helps compare assets with different price scales and volatilities.",
)

# Normalization method categories
NORMALIZATION_METHODS = {
    "Level-Based Scaling": ["Min-Max [0,1]", "Max-Abs Scaling", "Unit Norm (L2)"],
    "Distribution-Based": ["Z-Score", "Median/MAD (Robust)", "Rank Percentile"],
    "Time-Series": ["Rolling Z-Score", "Demeaning", "Volatility Scaling"],
    "Ratio-Based": ["Log Transform", "Log Returns", "Percent Change"],
}

# Educational content for each method
METHOD_INFO = {
    "Min-Max [0,1]": {
        "category": "Level-Based Scaling",
        "formula": "x_norm = (x - min) / (max - min)",
        "why": "Maps all values to a fixed [0,1] interval, making different assets directly comparable regardless of their absolute price levels. Preserves the relative distances between data points.",
        "when": [
            "Building composite indices or factor scores that combine multiple metrics",
            "Preparing features for machine learning models that require bounded inputs",
            "Comparing assets with vastly different price scales (e.g., $10 stock vs $3000 stock)",
            "Creating visual comparisons where you want values on the same scale",
        ],
        "how": "Subtract the minimum value and divide by the range. The result is always between 0 and 1, where 0 represents the historical minimum and 1 represents the maximum.",
        "pitfalls": [
            "Sensitive to outliers - a single extreme value can compress all other data",
            "Not suitable for out-of-sample data that may exceed historical min/max",
            "Loses information about the original magnitude of changes",
        ],
        "use_case": "Factor scoring, neural network inputs, dashboard visualizations",
    },
    "Max-Abs Scaling": {
        "category": "Level-Based Scaling",
        "formula": "x_norm = x / |max(x)|",
        "why": "Preserves the sign of the data (positive/negative) while scaling to [-1, 1]. Unlike Min-Max, it doesn't shift the data, only rescales it.",
        "when": [
            "Working with data that has meaningful positive and negative values (returns, flows)",
            "When zero should remain zero after normalization",
            "Comparing magnitudes while preserving directional information",
        ],
        "how": "Divide each value by the absolute maximum value in the series. Positive values stay positive, negative stay negative, and the largest absolute value becomes ±1.",
        "pitfalls": [
            "Asymmetric if data is skewed (e.g., if max is 100 but min is -10)",
            "A single outlier determines the scaling for all data",
        ],
        "use_case": "Cash flow analysis, net positioning data, symmetric signal generation",
    },
    "Unit Norm (L2)": {
        "category": "Level-Based Scaling",
        "formula": "x_norm = x / √(Σx²)",
        "why": "Normalizes the vector to have unit length (L2 norm = 1). Useful when the direction matters more than the magnitude.",
        "when": [
            "Computing portfolio weights that should sum to a specific exposure",
            "Calculating cosine similarity between asset return profiles",
            "Document/text similarity applications in NLP for finance",
        ],
        "how": "Divide each element by the Euclidean norm (square root of sum of squares). The resulting vector has a length of 1.",
        "pitfalls": [
            "All values become very small for long time series",
            "Interpretation of individual values is less intuitive",
        ],
        "use_case": "Portfolio weight normalization, similarity calculations, factor exposure vectors",
    },
    "Z-Score": {
        "category": "Distribution-Based",
        "formula": "z = (x - μ) / σ",
        "why": "Centers data at zero and scales by standard deviation, making variables statistically comparable. The most widely used normalization in quantitative finance.",
        "when": [
            "Cross-sectional factor models (comparing metrics across stocks)",
            "Combining different indicators into a single score",
            "Identifying outliers (values beyond ±2-3 standard deviations)",
            "Any time you need statistically comparable variables",
        ],
        "how": "Subtract the mean and divide by standard deviation. A z-score of 2 means the value is 2 standard deviations above the mean.",
        "pitfalls": [
            "Assumes normally distributed data (not ideal for fat-tailed returns)",
            "Sensitive to outliers which affect both mean and standard deviation",
            "Full-sample z-score introduces look-ahead bias in backtests",
        ],
        "use_case": "Factor investing, cross-sectional stock ranking, anomaly detection",
    },
    "Median/MAD (Robust)": {
        "category": "Distribution-Based",
        "formula": "z_robust = (x - median) / (MAD × 1.4826)",
        "why": "A robust alternative to z-score that uses median and MAD instead of mean and standard deviation. Much less sensitive to outliers and fat tails.",
        "when": [
            "Data has significant outliers or fat tails (common in finance)",
            "Working with fundamentals that may have extreme values",
            "When you suspect data quality issues or errors",
            "Hedge fund returns or other data prone to extreme observations",
        ],
        "how": "Subtract the median, then divide by the Median Absolute Deviation (scaled by 1.4826 for consistency with normal distribution).",
        "pitfalls": [
            "Less efficient than z-score when data is truly normal",
            "The 1.4826 scaling factor assumes underlying normality",
            "Less widely used, so may need explanation to stakeholders",
        ],
        "use_case": "Fundamental factor models, outlier-resistant scoring, data cleaning",
    },
    "Rank Percentile": {
        "category": "Distribution-Based",
        "formula": "percentile = rank(x) / n",
        "why": "Replaces values with their percentile rank, eliminating the impact of outliers entirely. All information except ordering is discarded.",
        "when": [
            "Cross-sectional equity factor models (industry standard)",
            "When the ranking matters more than the magnitude",
            "Data with extreme outliers that would dominate other methods",
            "Combining factors with very different distributions",
        ],
        "how": "Rank all values and convert to percentiles (0 to 1). The highest value gets rank 1.0, lowest gets close to 0.",
        "pitfalls": [
            "Loses all information about the magnitude of differences",
            "A stock at the 90th percentile might be barely above the 89th",
            "Time-series application loses temporal information",
        ],
        "use_case": "Equity factor models, quintile/decile portfolio construction, non-parametric analysis",
    },
    "Rolling Z-Score": {
        "category": "Time-Series",
        "formula": "z_t = (x_t - μ_rolling) / σ_rolling",
        "why": "Standardizes relative to recent history rather than full sample, adapting to changing market regimes. Avoids look-ahead bias.",
        "when": [
            "Building trading signals that adapt to market conditions",
            "Detecting regime changes or mean reversion opportunities",
            "Real-time signal generation without look-ahead bias",
            "When recent history is more relevant than distant past",
        ],
        "how": "Calculate rolling mean and standard deviation over a lookback window, then compute z-score. Window size controls adaptiveness.",
        "pitfalls": [
            "Requires choosing an appropriate window length",
            "Early observations have fewer data points for calculation",
            "May be slow to adapt to sudden regime changes",
        ],
        "use_case": "Mean reversion signals, regime detection, adaptive trading systems",
    },
    "Demeaning": {
        "category": "Time-Series",
        "formula": "x_demeaned = x_t - μ_rolling",
        "why": "Removes the trend or level, isolating deviations from the moving average. Simpler than z-score when you only want to remove the level.",
        "when": [
            "Isolating short-term deviations from trend",
            "Removing non-stationary level from price series",
            "Preparing data for mean-reversion analysis",
            "When volatility normalization is not desired",
        ],
        "how": "Subtract the rolling mean from each observation. Positive values are above recent average, negative below.",
        "pitfalls": [
            "Doesn't account for changing volatility",
            "Units remain in original scale (dollars, not standardized)",
            "Still affected by outliers in the rolling window",
        ],
        "use_case": "Trend deviation analysis, pairs trading spreads, momentum indicators",
    },
    "Volatility Scaling": {
        "category": "Time-Series",
        "formula": "r_scaled = r_t / σ_rolling",
        "why": "Equalizes risk contribution across time and assets by scaling returns by their volatility. Essential for risk parity and comparable signals.",
        "when": [
            "Risk parity portfolio construction",
            "Comparing signals across assets with different volatilities",
            "Standardizing position sizes by risk",
            "Volatility targeting strategies",
        ],
        "how": "Divide returns by rolling realized volatility (often annualized). A vol-scaled return of 1 means a 1-sigma move.",
        "pitfalls": [
            "Volatility estimation is noisy and backward-looking",
            "Can amplify positions in low-vol periods (which may precede vol spikes)",
            "Requires choosing volatility estimation method and lookback",
        ],
        "use_case": "Risk parity, volatility targeting, cross-asset signal comparison",
    },
    "Log Transform": {
        "category": "Ratio-Based",
        "formula": "x_log = ln(x)",
        "why": "Compresses large values, stabilizes variance, and converts multiplicative relationships to additive. Makes percentage changes comparable across price levels.",
        "when": [
            "Working with prices, market caps, or other right-skewed data",
            "When percentage changes matter more than absolute changes",
            "Preparing data for linear regression on growth rates",
            "Stabilizing variance that increases with level",
        ],
        "how": "Take the natural logarithm of each value. Only works for positive values.",
        "pitfalls": [
            "Cannot handle zero or negative values",
            "Interpretation requires understanding log scale",
            "May over-compress meaningful variation in large values",
        ],
        "use_case": "Price series modeling, market cap comparisons, growth rate analysis",
    },
    "Log Returns": {
        "category": "Ratio-Based",
        "formula": "r_log = ln(P_t / P_{t-1})",
        "why": "Continuously compounded returns that are additive over time and approximately normal. The standard return measure in quantitative finance.",
        "when": [
            "Any return-based analysis or modeling",
            "Portfolio return calculations (log returns are additive)",
            "Statistical modeling assuming normality",
            "Multi-period return aggregation",
        ],
        "how": "Take the natural log of the price ratio (today/yesterday). For small returns, approximately equal to simple returns.",
        "pitfalls": [
            "Not intuitive for reporting to non-quants",
            "Doesn't exactly match P&L for large moves",
            "Still has fat tails despite being more normal than simple returns",
        ],
        "use_case": "Return modeling, risk calculations, statistical analysis, backtesting",
    },
    "Percent Change": {
        "category": "Ratio-Based",
        "formula": "r_simple = (P_t - P_{t-1}) / P_{t-1}",
        "why": "Simple, intuitive returns that directly represent profit/loss percentage. Easy to explain and matches actual P&L.",
        "when": [
            "Reporting returns to stakeholders",
            "Calculating actual portfolio P&L",
            "When interpretability is more important than statistical properties",
            "Short holding periods where log vs simple doesn't matter",
        ],
        "how": "Divide the price change by the starting price. A return of 0.05 means a 5% gain.",
        "pitfalls": [
            "Not additive over time (compounding)",
            "Asymmetric: +50% then -50% doesn't return to start",
            "Cross-sectional aggregation requires weighting",
        ],
        "use_case": "Performance reporting, P&L calculation, client communication",
    },
}

# Category-level educational content
CATEGORY_INFO = {
    "Level-Based Scaling": {
        "description": "Methods that rescale variables to comparable numeric ranges without changing the shape of the distribution.",
        "key_principle": "Preserve relative relationships while standardizing scale",
        "best_for": "Combining variables with different units, ML preprocessing, visualization",
        "rule_of_thumb": "Use Min-Max for bounded inputs, Max-Abs when zero matters, Unit Norm for vector operations",
    },
    "Distribution-Based": {
        "description": "Methods that standardize based on statistical properties of the distribution, making variables statistically comparable.",
        "key_principle": "Center and scale based on distribution characteristics",
        "best_for": "Cross-sectional comparisons, factor models, outlier detection",
        "rule_of_thumb": "Use Z-Score as default, Median/MAD for outliers, Rank for extreme robustness",
    },
    "Time-Series": {
        "description": "Methods that normalize relative to recent history, adapting to changing market conditions over time.",
        "key_principle": "Compare current values to recent context, avoiding look-ahead bias",
        "best_for": "Trading signals, regime detection, real-time applications",
        "rule_of_thumb": "Use Rolling Z-Score for signals, Volatility Scaling for risk normalization",
    },
    "Ratio-Based": {
        "description": "Methods that express values as ratios or transformations, converting levels to comparable dynamics.",
        "key_principle": "Transform absolute values to relative changes or comparable scales",
        "best_for": "Return calculations, growth analysis, variance stabilization",
        "rule_of_thumb": "Use Log Returns for modeling, Percent Change for reporting, Log Transform for levels",
    },
}

# Sidebar controls
with st.sidebar:
    st.header("Normalization Options")
    ticker = st.text_input("Ticker Symbol", value="AAPL")
    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=365),
    )
    end_date = st.date_input("End Date", value=date.today())

    st.subheader("Method Selection")
    method_category = st.selectbox(
        "Category",
        list(NORMALIZATION_METHODS.keys()),
    )
    normalization_method = st.selectbox(
        "Normalization Method",
        NORMALIZATION_METHODS[method_category],
    )

    st.subheader("Parameters")
    window = st.slider("Rolling Window (days)", min_value=5, max_value=100, value=20)
    winsorize_pct = st.slider("Winsorization (%)", min_value=0, max_value=10, value=0)


def apply_winsorization(series: pd.Series, pct: float) -> pd.Series:
    """Apply winsorization to cap extreme values."""
    if pct <= 0:
        return series
    lower = series.quantile(pct / 100)
    upper = series.quantile(1 - pct / 100)
    return series.clip(lower=lower, upper=upper)


def apply_normalization(prices: pd.Series, method: str, window: int) -> pd.Series:
    """Apply the selected normalization method to price series."""
    if method == "Min-Max [0,1]":
        return (prices - prices.min()) / (prices.max() - prices.min())

    elif method == "Max-Abs Scaling":
        return prices / prices.abs().max()

    elif method == "Unit Norm (L2)":
        return prices / np.sqrt((prices ** 2).sum())

    elif method == "Z-Score":
        return (prices - prices.mean()) / prices.std()

    elif method == "Median/MAD (Robust)":
        median = prices.median()
        mad = (prices - median).abs().median()
        return (prices - median) / (mad * 1.4826)  # 1.4826 for normal consistency

    elif method == "Rank Percentile":
        return prices.rank(pct=True)

    elif method == "Rolling Z-Score":
        roll_mean = prices.rolling(window=window).mean()
        roll_std = prices.rolling(window=window).std()
        return (prices - roll_mean) / roll_std

    elif method == "Demeaning":
        roll_mean = prices.rolling(window=window).mean()
        return prices - roll_mean

    elif method == "Volatility Scaling":
        returns = prices.pct_change()
        roll_vol = returns.rolling(window=window).std()
        return returns / roll_vol

    elif method == "Log Transform":
        return np.log(prices)

    elif method == "Log Returns":
        return np.log(prices / prices.shift(1))

    elif method == "Percent Change":
        return prices.pct_change()

    return prices


# Fetch data once for all tabs
df = DataInjestor.get(ticker, str(start_date), str(end_date))

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Raw vs Normalized", "Distribution Comparison", "Rolling Statistics", "Learning Center"])

with tab1:
    if df.empty:
        st.warning(f"No data found for {ticker}. Please check the ticker symbol.")
    else:
        # Show contextual help for selected method
        method_info = METHOD_INFO.get(normalization_method, {})
        with st.expander(f"About {normalization_method}", expanded=False):
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.markdown(f"**Formula:** `{method_info.get('formula', 'N/A')}`")
                st.markdown(f"**Why use it?** {method_info.get('why', 'N/A')}")
                st.markdown("**When to use:**")
                for use in method_info.get('when', []):
                    st.markdown(f"- {use}")
            with col_info2:
                st.markdown(f"**How it works:** {method_info.get('how', 'N/A')}")
                st.markdown("**Pitfalls to watch:**")
                for pitfall in method_info.get('pitfalls', []):
                    st.markdown(f"- {pitfall}")
                st.markdown(f"**Common use cases:** {method_info.get('use_case', 'N/A')}")

        close_prices = df["Close"].dropna()
        close_prices = apply_winsorization(close_prices, winsorize_pct)
        normalized = apply_normalization(close_prices, normalization_method, window)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Raw Price")
            raw_df = pd.DataFrame({
                "Day": range(len(close_prices)),
                "Value": close_prices.values,
                "Series": "Raw Price"
            })
            raw_chart = (
                alt.Chart(raw_df)
                .mark_line(color=THEME["primary"])
                .encode(
                    x=alt.X("Day:Q", title="Trading Day"),
                    y=alt.Y("Value:Q", title="Price ($)"),
                    tooltip=["Day", "Value"]
                )
                .properties(width=CHART_WIDTH // 2, height=CHART_HEIGHT // 2, title=f"{ticker} Raw Close Price")
            )
            st.altair_chart(configure_altair_theme(raw_chart), use_container_width=True)

        with col2:
            st.subheader(f"Normalized ({normalization_method})")
            norm_values = normalized.dropna()
            norm_df = pd.DataFrame({
                "Day": range(len(close_prices) - len(norm_values), len(close_prices)),
                "Value": norm_values.values,
                "Series": "Normalized"
            })
            norm_chart = (
                alt.Chart(norm_df)
                .mark_line(color=THEME["secondary"])
                .encode(
                    x=alt.X("Day:Q", title="Trading Day"),
                    y=alt.Y("Value:Q", title="Normalized Value"),
                    tooltip=["Day", "Value"]
                )
                .properties(width=CHART_WIDTH // 2, height=CHART_HEIGHT // 2, title=f"{ticker} {normalization_method}")
            )
            st.altair_chart(configure_altair_theme(norm_chart), use_container_width=True)

        # Combined overlay chart
        st.subheader("Overlay Comparison")
        st.caption("Normalized values scaled to raw price range for visual comparison")

        # Scale normalized to raw range for overlay
        norm_clean = normalized.dropna().values
        if len(norm_clean) > 0:
            raw_min, raw_max = close_prices.min(), close_prices.max()
            norm_min, norm_max = norm_clean.min(), norm_clean.max()
            if norm_max != norm_min:
                scaled_norm = raw_min + (norm_clean - norm_min) * (raw_max - raw_min) / (norm_max - norm_min)
            else:
                scaled_norm = np.full_like(norm_clean, (raw_min + raw_max) / 2)

            overlay_df = pd.concat([
                pd.DataFrame({"Day": range(len(close_prices)), "Value": close_prices.values, "Series": "Raw Price"}),
                pd.DataFrame({"Day": range(len(close_prices) - len(scaled_norm), len(close_prices)), "Value": scaled_norm, "Series": f"Normalized (scaled)"})
            ], ignore_index=True)

            overlay_chart = create_time_series(overlay_df, "Day", "Value", "Series", f"{ticker} - Raw vs {normalization_method}")
            st.altair_chart(overlay_chart, use_container_width=True)

        # Statistics comparison
        st.subheader("Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Raw Price**")
            st.write(f"Mean: {close_prices.mean():.2f}")
            st.write(f"Std: {close_prices.std():.2f}")
            st.write(f"Min: {close_prices.min():.2f}")
            st.write(f"Max: {close_prices.max():.2f}")
        with col2:
            norm_clean_series = normalized.dropna()
            st.markdown(f"**{normalization_method}**")
            st.write(f"Mean: {norm_clean_series.mean():.4f}")
            st.write(f"Std: {norm_clean_series.std():.4f}")
            st.write(f"Min: {norm_clean_series.min():.4f}")
            st.write(f"Max: {norm_clean_series.max():.4f}")

with tab2:
    if df.empty:
        st.warning(f"No data found for {ticker}. Please check the ticker symbol.")
    else:
        close_prices = df["Close"].dropna()
        close_prices = apply_winsorization(close_prices, winsorize_pct)

        st.subheader("Distribution Comparison Across Methods")
        st.caption("Compare how different normalization methods transform the data distribution")

        # Let user select methods to compare
        methods_to_compare = st.multiselect(
            "Select methods to compare",
            [m for methods in NORMALIZATION_METHODS.values() for m in methods],
            default=["Z-Score", "Rank Percentile", "Log Returns"]
        )

        if methods_to_compare:
            # Build comparison dataframe
            distributions = []
            for method in methods_to_compare:
                norm_data = apply_normalization(close_prices, method, window).dropna()
                for val in norm_data.values:
                    distributions.append({"Value": val, "Method": method})

            dist_df = pd.DataFrame(distributions)

            # Density plot
            density_chart = create_density_plot(dist_df, "Value", "Method", "Distribution Comparison")
            st.altair_chart(density_chart, use_container_width=True)

            # Histogram comparison
            st.subheader("Histogram Comparison")
            hist_chart = (
                alt.Chart(dist_df)
                .mark_bar(opacity=0.6)
                .encode(
                    x=alt.X("Value:Q", bin=alt.Bin(maxbins=50), title="Value"),
                    y=alt.Y("count()", stack=None, title="Frequency"),
                    color=alt.Color("Method:N", scale=alt.Scale(scheme="category10")),
                    tooltip=["Method", "count()"]
                )
                .properties(width=CHART_WIDTH, height=CHART_HEIGHT // 2, title="Histogram by Method")
            )
            st.altair_chart(configure_altair_theme(hist_chart), use_container_width=True)

            # Statistics table
            st.subheader("Distributional Statistics")
            stats_data = []
            for method in methods_to_compare:
                norm_data = apply_normalization(close_prices, method, window).dropna()
                stats_data.append({
                    "Method": method,
                    "Mean": f"{norm_data.mean():.4f}",
                    "Std": f"{norm_data.std():.4f}",
                    "Skewness": f"{norm_data.skew():.4f}",
                    "Kurtosis": f"{norm_data.kurtosis():.4f}",
                    "Min": f"{norm_data.min():.4f}",
                    "Max": f"{norm_data.max():.4f}",
                })
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        else:
            st.info("Select at least one method to compare distributions.")

with tab3:
    if df.empty:
        st.warning(f"No data found for {ticker}. Please check the ticker symbol.")
    else:
        close_prices = df["Close"].dropna()
        returns = close_prices.pct_change().dropna()

        st.subheader("Rolling Statistics Over Time")
        st.caption("Visualize how normalization metrics evolve through time")

        col1, col2 = st.columns(2)

        with col1:
            # Rolling mean and std bands
            st.markdown("**Rolling Mean ± Std Bands**")
            roll_mean = close_prices.rolling(window=window).mean()
            roll_std = close_prices.rolling(window=window).std()

            bands_df = pd.DataFrame({
                "Day": range(len(close_prices)),
                "Price": close_prices.values,
                "Mean": roll_mean.values,
                "Upper": (roll_mean + 2 * roll_std).values,
                "Lower": (roll_mean - 2 * roll_std).values,
            }).dropna()

            base = alt.Chart(bands_df).encode(x=alt.X("Day:Q", title="Trading Day"))

            band = base.mark_area(opacity=0.3, color=THEME["primary"]).encode(
                y="Lower:Q",
                y2="Upper:Q"
            )
            mean_line = base.mark_line(color=THEME["secondary"], strokeDash=[5, 5]).encode(y="Mean:Q")
            price_line = base.mark_line(color=THEME["primary"]).encode(y=alt.Y("Price:Q", title="Price"))

            bands_chart = (band + mean_line + price_line).properties(
                width=CHART_WIDTH // 2, height=CHART_HEIGHT // 2, title=f"Rolling {window}-day Mean ± 2σ"
            )
            st.altair_chart(configure_altair_theme(bands_chart), use_container_width=True)

        with col2:
            # Rolling Z-Score
            st.markdown("**Rolling Z-Score**")
            rolling_zscore = apply_normalization(close_prices, "Rolling Z-Score", window).dropna()

            zscore_df = pd.DataFrame({
                "Day": range(len(close_prices) - len(rolling_zscore), len(close_prices)),
                "Z-Score": rolling_zscore.values
            })

            zscore_base = alt.Chart(zscore_df).encode(x=alt.X("Day:Q", title="Trading Day"))

            # Add horizontal lines at ±2 sigma
            hline_data = pd.DataFrame({"y": [-2, 0, 2]})
            hlines = alt.Chart(hline_data).mark_rule(strokeDash=[3, 3], color="gray").encode(y="y:Q")

            zscore_line = zscore_base.mark_line(color=THEME["primary"]).encode(
                y=alt.Y("Z-Score:Q", title="Z-Score")
            )

            zscore_chart = (zscore_line + hlines).properties(
                width=CHART_WIDTH // 2, height=CHART_HEIGHT // 2, title=f"Rolling {window}-day Z-Score"
            )
            st.altair_chart(configure_altair_theme(zscore_chart), use_container_width=True)

        # Rolling Volatility
        st.subheader("Volatility Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Rolling Volatility (Annualized)**")
            roll_vol = returns.rolling(window=window).std() * np.sqrt(252)

            vol_df = pd.DataFrame({
                "Day": range(len(returns) - len(roll_vol.dropna()), len(returns)),
                "Volatility": roll_vol.dropna().values
            })

            vol_chart = (
                alt.Chart(vol_df)
                .mark_area(opacity=0.6, color=THEME["primary"])
                .encode(
                    x=alt.X("Day:Q", title="Trading Day"),
                    y=alt.Y("Volatility:Q", title="Annualized Volatility"),
                    tooltip=["Day", "Volatility"]
                )
                .properties(width=CHART_WIDTH // 2, height=CHART_HEIGHT // 2, title=f"Rolling {window}-day Volatility")
            )
            st.altair_chart(configure_altair_theme(vol_chart), use_container_width=True)

        with col2:
            st.markdown("**Volatility-Scaled Returns**")
            vol_scaled = apply_normalization(close_prices, "Volatility Scaling", window).dropna()

            vs_df = pd.DataFrame({
                "Day": range(len(close_prices) - len(vol_scaled), len(close_prices)),
                "Scaled Return": vol_scaled.values
            })

            vs_chart = (
                alt.Chart(vs_df)
                .mark_line(color=THEME["secondary"])
                .encode(
                    x=alt.X("Day:Q", title="Trading Day"),
                    y=alt.Y("Scaled Return:Q", title="Vol-Scaled Return"),
                    tooltip=["Day", "Scaled Return"]
                )
                .properties(width=CHART_WIDTH // 2, height=CHART_HEIGHT // 2, title="Volatility-Adjusted Returns")
            )
            st.altair_chart(configure_altair_theme(vs_chart), use_container_width=True)

        # Regime Detection
        st.subheader("Regime Detection")
        st.caption("Identify high/low volatility regimes and extreme normalized values")

        rolling_zscore = apply_normalization(close_prices, "Rolling Z-Score", window)
        roll_vol = returns.rolling(window=window).std() * np.sqrt(252)
        vol_median = roll_vol.median()

        regime_df = pd.DataFrame({
            "Day": range(len(close_prices)),
            "Price": close_prices.values,
            "Z-Score": rolling_zscore.values,
            "Volatility": roll_vol.reindex(close_prices.index).values,
        }).dropna()

        regime_df["Vol Regime"] = np.where(regime_df["Volatility"] > vol_median, "High Vol", "Low Vol")
        regime_df["Z Regime"] = pd.cut(regime_df["Z-Score"], bins=[-np.inf, -2, -1, 1, 2, np.inf],
                                        labels=["Oversold (<-2σ)", "Low (-2σ to -1σ)", "Normal", "High (1σ to 2σ)", "Overbought (>2σ)"])

        regime_chart = (
            alt.Chart(regime_df)
            .mark_circle(size=30, opacity=0.7)
            .encode(
                x=alt.X("Day:Q", title="Trading Day"),
                y=alt.Y("Price:Q", title="Price"),
                color=alt.Color("Vol Regime:N", scale=alt.Scale(domain=["Low Vol", "High Vol"], range=[THEME["primary"], THEME["secondary"]])),
                tooltip=["Day", "Price", "Z-Score", "Volatility", "Vol Regime", "Z Regime"]
            )
            .properties(width=CHART_WIDTH, height=CHART_HEIGHT // 2, title="Price Colored by Volatility Regime")
        )
        st.altair_chart(configure_altair_theme(regime_chart), use_container_width=True)

        # Summary metrics
        st.subheader("Summary Metrics")
        col1, col2, col3 = st.columns(3)

        high_vol_pct = (regime_df["Vol Regime"] == "High Vol").mean() * 100
        extreme_zscore_pct = ((regime_df["Z-Score"].abs() > 2).mean()) * 100
        avg_vol = roll_vol.mean()

        with col1:
            st.metric("High Volatility Days", f"{high_vol_pct:.1f}%")
        with col2:
            st.metric("Extreme Z-Score Days (|z|>2)", f"{extreme_zscore_pct:.1f}%")
        with col3:
            st.metric("Average Annualized Volatility", f"{avg_vol:.1%}")

with tab4:
    st.subheader("Normalization Methods: A Complete Guide")
    st.caption("Learn when, why, and how to apply different normalization techniques in financial modeling")

    # Quick reference guide
    st.markdown("### Quick Reference Guide")
    st.markdown("""
    | **Goal** | **Recommended Method** | **Why** |
    |----------|----------------------|---------|
    | Cross-sectional factor model | Z-Score or Rank Percentile | Standard in equity factors, comparable across stocks |
    | Time-series trading signal | Rolling Z-Score | Adapts to regimes, no look-ahead bias |
    | Risk parity / position sizing | Volatility Scaling | Equalizes risk contribution |
    | ML model preprocessing | Min-Max [0,1] | Bounded inputs for neural networks |
    | Data with outliers | Median/MAD or Rank | Robust to extreme values |
    | Return calculations | Log Returns | Additive, approximately normal |
    | Client reporting | Percent Change | Intuitive, matches P&L |
    | Combining different metrics | Z-Score | Makes variables comparable |
    """)

    st.divider()

    # Category deep dives
    st.markdown("### Method Categories")

    for category, methods in NORMALIZATION_METHODS.items():
        cat_info = CATEGORY_INFO.get(category, {})
        with st.expander(f"**{category}**", expanded=False):
            st.markdown(f"**Description:** {cat_info.get('description', '')}")
            st.markdown(f"**Key Principle:** {cat_info.get('key_principle', '')}")
            st.markdown(f"**Best For:** {cat_info.get('best_for', '')}")
            st.info(f"**Rule of Thumb:** {cat_info.get('rule_of_thumb', '')}")

            st.markdown("---")
            st.markdown("**Methods in this category:**")

            for method in methods:
                info = METHOD_INFO.get(method, {})
                st.markdown(f"#### {method}")
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
                    st.markdown("**Watch out for:**")
                    for pitfall in info.get('pitfalls', []):
                        st.markdown(f"- {pitfall}")

                st.caption(f"Common use cases: {info.get('use_case', '')}")
                st.markdown("---")

    st.divider()

    # Decision flowchart
    st.markdown("### Decision Framework")
    st.markdown("""
    **Step 1: What type of analysis?**
    - Cross-sectional (comparing across assets at one time) → Distribution-Based methods
    - Time-series (one asset over time) → Time-Series methods
    - Preparing for ML/optimization → Level-Based Scaling
    - Computing returns → Ratio-Based methods

    **Step 2: How sensitive is your data to outliers?**
    - Clean data, roughly normal → Z-Score, Min-Max
    - Some outliers → Median/MAD, Winsorization + Z-Score
    - Extreme outliers → Rank Percentile

    **Step 3: Does the absolute magnitude matter?**
    - Yes, preserve relationships → Min-Max, Z-Score
    - No, only ordering matters → Rank Percentile
    - Need risk-adjusted → Volatility Scaling

    **Step 4: Real-time or historical analysis?**
    - Real-time/trading → Rolling methods (avoid look-ahead bias)
    - Historical research → Full-sample methods okay
    """)

    st.divider()

    # Practical examples
    st.markdown("### Practical Examples")

    with st.expander("Example 1: Building a Multi-Factor Equity Model"):
        st.markdown("""
        **Scenario:** You want to combine Value (P/E ratio), Momentum (12-month return), and Quality (ROE) into a single score.

        **Challenge:** These metrics have completely different scales and distributions.

        **Solution:**
        1. **Cross-sectional Z-Score** each factor within each date
        2. **Winsorize** at 3 standard deviations to limit outlier impact
        3. **Combine** with equal or optimized weights

        **Why this works:** Z-scoring makes each factor have mean 0 and std 1, so they contribute equally before weighting.

        **Alternative:** Use **Rank Percentile** if the data has extreme outliers or you only care about relative ordering.
        """)

    with st.expander("Example 2: Mean Reversion Trading Signal"):
        st.markdown("""
        **Scenario:** You want to identify when a stock is "oversold" relative to its recent history.

        **Challenge:** What counts as oversold varies by stock and market regime.

        **Solution:**
        1. Calculate **Rolling Z-Score** with a 20-60 day window
        2. Signal when Z-Score < -2 (more than 2 std below rolling mean)
        3. Exit when Z-Score returns to 0

        **Why this works:** Rolling z-score adapts to each stock's recent behavior and avoids look-ahead bias.

        **Tip:** Combine with **Volatility Scaling** to normalize position sizes across stocks.
        """)

    with st.expander("Example 3: Risk Parity Portfolio"):
        st.markdown("""
        **Scenario:** You want each asset to contribute equally to portfolio risk.

        **Challenge:** A 10% position in bonds contributes less risk than 10% in stocks.

        **Solution:**
        1. Calculate rolling volatility for each asset
        2. **Volatility Scale** returns: `weight ∝ 1/volatility`
        3. Rebalance periodically as volatilities change

        **Why this works:** Lower-volatility assets get higher weights, equalizing risk contribution.

        **Watch out:** Volatility is estimated from the past - low-vol periods can precede vol spikes.
        """)

    with st.expander("Example 4: Preparing Data for Machine Learning"):
        st.markdown("""
        **Scenario:** Training a neural network to predict returns using price and volume features.

        **Challenge:** Neural networks work best with normalized, bounded inputs.

        **Solution:**
        1. Convert prices to **Log Returns** (unbounded but comparable)
        2. Apply **Min-Max** scaling to fit [0,1] or [-1,1] range
        3. Or use **Z-Score** (standard scaling in sklearn)

        **Why this works:** Neural networks train faster and more stably with normalized inputs.

        **Important:** Fit the scaler on training data only, then transform test data with the same parameters.
        """)

    st.divider()

    # Common mistakes
    st.markdown("### Common Mistakes to Avoid")

    st.error("""
    **1. Look-Ahead Bias**
    Using full-sample z-scores in a backtest means you're using future information.
    **Fix:** Use rolling or expanding window calculations.
    """)

    st.error("""
    **2. Ignoring Outliers**
    A single extreme value can dominate z-score calculations.
    **Fix:** Winsorize first, or use robust methods (Median/MAD, Rank).
    """)

    st.error("""
    **3. Normalizing Non-Stationary Data**
    Z-scoring a trending price series doesn't make it stationary.
    **Fix:** Transform to returns first, then normalize.
    """)

    st.error("""
    **4. Cross-Sectional vs Time-Series Confusion**
    Normalizing a stock against its own history vs against other stocks are different operations.
    **Fix:** Be explicit about which dimension you're normalizing.
    """)

    st.error("""
    **5. Over-Normalizing**
    Rank percentile discards magnitude information - a stock at P/E 5 vs P/E 8 might both be "cheap."
    **Fix:** Choose the method that preserves the information you need.
    """)
