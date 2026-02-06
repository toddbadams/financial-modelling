"""Portfolio Optimization page - Optimize portfolio weights using Modern Portfolio Theory."""

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
    "Portfolio Optimization",
    "Learn how to construct optimal portfolios using Modern Portfolio Theory. "
    "Explore the efficient frontier, find minimum volatility and maximum Sharpe portfolios, "
    "and understand the trade-offs between risk and return.",
)

# Educational content for optimization methods
OPTIMIZATION_INFO = {
    "Mean-Variance (Markowitz)": {
        "description": "Harry Markowitz's Modern Portfolio Theory (1952) models how different assets can be combined to form portfolios that maximize expected return for a given level of risk.",
        "key_insight": "Rather than analyzing investments individually, MPT focuses on how assets move together through their correlations, showing that diversification can reduce overall portfolio risk without necessarily reducing return.",
        "formula": "min w'Σw subject to w'μ = target return, w'1 = 1",
        "best_for": "Building diversified portfolios, understanding risk-return trade-offs",
    },
    "Minimum Volatility": {
        "description": "Finds the portfolio with the lowest possible volatility (risk) among all feasible combinations of assets.",
        "key_insight": "Conservative investors may prefer this allocation as it minimizes price swings, though it may sacrifice some expected return.",
        "formula": "min w'Σw subject to w'1 = 1, w ≥ 0",
        "best_for": "Risk-averse investors, capital preservation, stable income generation",
    },
    "Maximum Sharpe": {
        "description": "Finds the portfolio that delivers the highest return per unit of risk (Sharpe ratio), assuming a risk-free rate.",
        "key_insight": "This is the tangency portfolio where the Capital Market Line touches the efficient frontier. It represents the optimal risky portfolio.",
        "formula": "max (w'μ - rf) / √(w'Σw) subject to w'1 = 1, w ≥ 0",
        "best_for": "Growth-oriented investors, maximizing risk-adjusted returns",
    },
    "Risk Parity": {
        "description": "Allocates weights so that each asset contributes equally to overall portfolio risk, rather than equal capital allocation.",
        "key_insight": "Traditional 60/40 portfolios are dominated by equity risk. Risk parity balances risk contributions for more stable performance across market regimes.",
        "formula": "w_i × (Σw)_i / (w'Σw) = 1/n for all assets",
        "best_for": "All-weather portfolios, institutional investors, reducing concentration risk",
    },
}

# Concept explanations for the learning center
CONCEPT_INFO = {
    "Efficient Frontier": {
        "what": "The efficient frontier is the set of optimal portfolios that offer the highest expected return for each level of risk (or equivalently, the lowest risk for each level of return).",
        "why": "Any portfolio below the frontier is inefficient because you could achieve higher return for the same risk, or lower risk for the same return.",
        "how": "Generate many random portfolios, then identify those on the upper-left boundary of the risk-return scatter plot.",
        "interpretation": "Points on the frontier represent the best possible trade-offs. Your choice depends on your risk tolerance.",
    },
    "Portfolio Volatility": {
        "what": "The standard deviation of portfolio returns, measuring how much the portfolio value fluctuates over time.",
        "why": "Unlike individual asset volatility, portfolio volatility accounts for diversification effects through correlations.",
        "how": "σ_p = √(w'Σw) where w is the weight vector and Σ is the covariance matrix.",
        "interpretation": "Lower volatility means more stable returns. Diversification typically reduces portfolio volatility below the weighted average of individual volatilities.",
    },
    "Diversification": {
        "what": "The risk reduction achieved by combining assets that don't move perfectly together.",
        "why": "When one asset falls, another may rise or stay stable, smoothing overall portfolio returns.",
        "how": "Measured by comparing portfolio volatility to the weighted average of individual volatilities.",
        "interpretation": "Lower correlation between assets provides greater diversification benefit.",
    },
    "Sharpe Ratio": {
        "what": "Measures excess return (above risk-free rate) per unit of volatility: (R_p - R_f) / σ_p",
        "why": "Allows comparison of portfolios with different risk levels on a common scale.",
        "how": "Calculate portfolio return and volatility, subtract risk-free rate from return, divide by volatility.",
        "interpretation": "Higher is better. Sharpe > 1 is good, > 2 is very good. Negative means return below risk-free rate.",
    },
    "Correlation Matrix": {
        "what": "A matrix showing how each pair of assets moves together, ranging from -1 (opposite) to +1 (identical).",
        "why": "Correlations are key inputs for portfolio optimization. Low or negative correlations provide diversification.",
        "how": "Calculate the Pearson correlation coefficient between each pair of asset returns.",
        "interpretation": "Assets with low correlation to each other are valuable for diversification even if individually risky.",
    },
    "Systematic vs Unsystematic Risk": {
        "what": "Systematic risk affects the entire market (recessions, rate hikes) while unsystematic risk is specific to individual assets (earnings miss, scandal).",
        "why": "Diversification can eliminate unsystematic risk but not systematic risk.",
        "how": "Unsystematic risk decreases as you add more uncorrelated assets. Systematic risk remains constant.",
        "interpretation": "Even a perfectly diversified portfolio is still exposed to market-wide downturns.",
    },
}

# MPT limitations for educational purposes
MPT_LIMITATIONS = [
    {
        "limitation": "Treats all volatility as risk",
        "explanation": "MPT penalizes upside volatility (large gains) equally with downside volatility (large losses), though investors typically welcome gains.",
        "mitigation": "Consider Sortino ratio or downside deviation for asymmetric risk assessment.",
    },
    {
        "limitation": "Assumes stable correlations",
        "explanation": "Correlations tend to increase during market crises, exactly when diversification is most needed.",
        "mitigation": "Test portfolios under stress scenarios where correlations approach 1.",
    },
    {
        "limitation": "Assumes normal distribution",
        "explanation": "Real returns have fat tails with more extreme events than a normal distribution predicts.",
        "mitigation": "Use VaR/CVaR and examine skewness/kurtosis of returns.",
    },
    {
        "limitation": "Sensitive to input estimates",
        "explanation": "Small changes in expected returns can produce very different 'optimal' portfolios.",
        "mitigation": "Use robust optimization or Black-Litterman model to stabilize estimates.",
    },
]


# Calculation functions
def portfolio_return(weights: np.ndarray, expected_returns: np.ndarray) -> float:
    """Calculate portfolio expected return."""
    return float(np.dot(weights, expected_returns))


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Calculate portfolio volatility."""
    return float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))))


def portfolio_sharpe(weights: np.ndarray, expected_returns: np.ndarray,
                     cov_matrix: np.ndarray, rf: float = 0.0) -> float:
    """Calculate portfolio Sharpe ratio."""
    ret = portfolio_return(weights, expected_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    if vol == 0:
        return 0.0
    return (ret - rf) / vol


def generate_random_portfolios(n_portfolios: int, n_assets: int,
                                expected_returns: np.ndarray,
                                cov_matrix: np.ndarray,
                                rf: float = 0.0) -> tuple:
    """Generate random long-only portfolios using Dirichlet distribution."""
    weights = np.random.dirichlet(np.ones(n_assets), size=n_portfolios)

    port_returns = np.array([portfolio_return(w, expected_returns) for w in weights])
    port_vols = np.array([portfolio_volatility(w, cov_matrix) for w in weights])
    port_sharpes = np.array([portfolio_sharpe(w, expected_returns, cov_matrix, rf) for w in weights])

    return weights, port_returns, port_vols, port_sharpes


def find_efficient_frontier(port_returns: np.ndarray, port_vols: np.ndarray,
                            n_points: int = 30) -> tuple:
    """Approximate the efficient frontier from random portfolios."""
    target_returns = np.linspace(port_returns.min(), port_returns.max(), n_points)

    frontier_vols = []
    frontier_rets = []

    for tr in target_returns:
        mask = port_returns >= tr
        if mask.sum() == 0:
            continue
        frontier_vols.append(port_vols[mask].min())
        frontier_rets.append(tr)

    return np.array(frontier_vols), np.array(frontier_rets)


def calculate_risk_parity_weights(cov_matrix: np.ndarray, n_iterations: int = 100) -> np.ndarray:
    """Calculate risk parity weights using iterative approach."""
    n_assets = cov_matrix.shape[0]
    weights = np.ones(n_assets) / n_assets

    for _ in range(n_iterations):
        # Calculate marginal risk contribution
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol

        # Risk contribution of each asset
        risk_contrib = weights * marginal_contrib

        # Target: equal risk contribution
        target_risk = portfolio_vol / n_assets

        # Adjust weights
        weights = weights * (target_risk / (risk_contrib + 1e-10))
        weights = weights / weights.sum()

    return weights


# Sidebar controls
with st.sidebar:
    st.header("Portfolio Options")

    tickers_input = st.text_area(
        "Tickers (one per line)",
        value="AAPL\nMSFT\nAMZN\nKO\nXOM\nTLT\nGLD",
        height=150,
        help="Enter stock/ETF symbols. Include a mix of asset types for better diversification.",
    )

    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=365 * 3),
        help="Longer history provides more reliable estimates but may not reflect current conditions.",
    )
    end_date = st.date_input("End Date", value=date.today())

    st.subheader("Optimization Settings")

    n_portfolios = st.slider(
        "Number of Random Portfolios",
        min_value=1000,
        max_value=20000,
        value=5000,
        step=1000,
        help="More portfolios provide better frontier approximation but take longer to compute.",
    )

    risk_free_rate = st.number_input(
        "Risk-Free Rate (%)",
        value=4.0,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
        help="Current short-term Treasury rate for Sharpe ratio calculation.",
    )

    st.subheader("Display Options")
    show_individual_assets = st.checkbox("Show individual assets on chart", value=True)
    show_correlation_heatmap = st.checkbox("Show correlation matrix", value=True)

# Parse tickers
tickers = [t.strip().upper() for t in tickers_input.strip().split('\n') if t.strip()]

if len(tickers) < 2:
    st.error("Please enter at least 2 tickers to construct a portfolio.")
    st.stop()

# Fetch data for all tickers
@st.cache_data(show_spinner="Fetching price data...")
def fetch_portfolio_data(tickers: list, start: str, end: str):
    """Fetch price data for multiple tickers."""
    all_prices = {}
    failed_tickers = []

    for ticker in tickers:
        try:
            df = DataInjestor.get(ticker, start, end)
            if not df.empty and "Close" in df.columns:
                all_prices[ticker] = df["Close"]
            else:
                failed_tickers.append(ticker)
        except Exception:
            failed_tickers.append(ticker)

    if not all_prices:
        return pd.DataFrame(), failed_tickers

    prices_df = pd.DataFrame(all_prices).dropna()
    return prices_df, failed_tickers


prices_df, failed_tickers = fetch_portfolio_data(tickers, str(start_date), str(end_date))

if failed_tickers:
    st.warning(f"Could not fetch data for: {', '.join(failed_tickers)}")

if prices_df.empty or len(prices_df.columns) < 2:
    st.error("Insufficient data. Please check ticker symbols and date range.")
    st.stop()

# Calculate returns and MPT inputs
returns_df = prices_df.pct_change().dropna()
assets = list(returns_df.columns)
n_assets = len(assets)

# Annualized expected returns and covariance
mu = returns_df.mean() * 252
cov = returns_df.cov() * 252

# Convert to numpy for calculations
mu_array = mu.values
cov_array = cov.values

# Generate random portfolios
rf = risk_free_rate / 100

with st.spinner("Generating random portfolios..."):
    weights, port_returns, port_vols, port_sharpes = generate_random_portfolios(
        n_portfolios, n_assets, mu_array, cov_array, rf
    )

# Find key portfolios
idx_min_vol = np.argmin(port_vols)
idx_max_sharpe = np.argmax(port_sharpes)

w_min_vol = weights[idx_min_vol]
w_max_sharpe = weights[idx_max_sharpe]

# Risk parity weights
w_risk_parity = calculate_risk_parity_weights(cov_array)

# Efficient frontier
frontier_vols, frontier_rets = find_efficient_frontier(port_returns, port_vols)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Efficient Frontier",
    "Allocation Weights",
    "Performance Metrics",
    "Learning Center"
])

with tab1:
    st.subheader("Modern Portfolio Theory: Efficient Frontier")

    with st.expander("What is the Efficient Frontier?", expanded=False):
        st.markdown("""
        The **efficient frontier** represents the set of portfolios that offer the highest expected return
        for each level of risk. Any portfolio below the frontier is **inefficient** because you could
        achieve higher return for the same risk, or lower risk for the same return.

        **Key portfolios:**
        - **Minimum Volatility**: Lowest possible risk (leftmost point on frontier)
        - **Maximum Sharpe**: Best risk-adjusted return (highest return per unit of risk)

        The shape of the frontier depends on asset correlations. Lower correlations create a more
        curved frontier, offering better diversification benefits.
        """)

    # Build scatter chart data
    scatter_df = pd.DataFrame({
        "Volatility": port_vols,
        "Return": port_returns,
        "Sharpe": port_sharpes
    })

    # Main scatter plot
    scatter_chart = (
        alt.Chart(scatter_df)
        .mark_circle(size=8, opacity=0.2)
        .encode(
            x=alt.X("Volatility:Q", title="Annualized Volatility (Risk)",
                   axis=alt.Axis(format=".0%")),
            y=alt.Y("Return:Q", title="Annualized Return",
                   axis=alt.Axis(format=".0%")),
            color=alt.Color("Sharpe:Q", scale=alt.Scale(scheme="viridis"),
                           legend=alt.Legend(title="Sharpe Ratio")),
            tooltip=[
                alt.Tooltip("Volatility:Q", format=".2%", title="Volatility"),
                alt.Tooltip("Return:Q", format=".2%", title="Return"),
                alt.Tooltip("Sharpe:Q", format=".3f", title="Sharpe"),
            ]
        )
    )

    # Efficient frontier line
    frontier_df = pd.DataFrame({
        "Volatility": frontier_vols,
        "Return": frontier_rets
    })

    frontier_line = (
        alt.Chart(frontier_df)
        .mark_line(color="white", strokeWidth=2)
        .encode(
            x=alt.X("Volatility:Q"),
            y=alt.Y("Return:Q")
        )
    )

    # Key portfolio points
    key_portfolios = pd.DataFrame({
        "Volatility": [port_vols[idx_min_vol], port_vols[idx_max_sharpe]],
        "Return": [port_returns[idx_min_vol], port_returns[idx_max_sharpe]],
        "Portfolio": ["Minimum Volatility", "Maximum Sharpe"]
    })

    key_points = (
        alt.Chart(key_portfolios)
        .mark_point(size=200, filled=True)
        .encode(
            x=alt.X("Volatility:Q"),
            y=alt.Y("Return:Q"),
            color=alt.Color("Portfolio:N", scale=alt.Scale(
                domain=["Minimum Volatility", "Maximum Sharpe"],
                range=[THEME["secondary"], "#ff6b6b"]
            ), legend=alt.Legend(title="Key Portfolios")),
            shape=alt.value("diamond"),
            tooltip=["Portfolio:N",
                    alt.Tooltip("Volatility:Q", format=".2%"),
                    alt.Tooltip("Return:Q", format=".2%")]
        )
    )

    # Individual assets
    if show_individual_assets:
        asset_vols = np.sqrt(np.diag(cov_array))
        asset_rets = mu_array

        asset_df = pd.DataFrame({
            "Volatility": asset_vols,
            "Return": asset_rets,
            "Asset": assets
        })

        asset_points = (
            alt.Chart(asset_df)
            .mark_point(size=100, filled=True, color=THEME["primary"])
            .encode(
                x=alt.X("Volatility:Q"),
                y=alt.Y("Return:Q"),
                tooltip=["Asset:N",
                        alt.Tooltip("Volatility:Q", format=".2%"),
                        alt.Tooltip("Return:Q", format=".2%")]
            )
        )

        asset_labels = (
            alt.Chart(asset_df)
            .mark_text(align="left", dx=8, color="white")
            .encode(
                x=alt.X("Volatility:Q"),
                y=alt.Y("Return:Q"),
                text="Asset:N"
            )
        )

        combined_chart = scatter_chart + frontier_line + key_points + asset_points + asset_labels
    else:
        combined_chart = scatter_chart + frontier_line + key_points

    final_chart = combined_chart.properties(
        width=CHART_WIDTH,
        height=CHART_HEIGHT,
        title="Risk-Return Profile of Random Portfolios"
    )

    st.altair_chart(configure_altair_theme(final_chart), use_container_width=True)

    # Summary statistics
    st.subheader("Key Portfolio Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Minimum Volatility Portfolio**")
        st.metric("Return", f"{port_returns[idx_min_vol]:.2%}")
        st.metric("Volatility", f"{port_vols[idx_min_vol]:.2%}")
        st.metric("Sharpe Ratio", f"{port_sharpes[idx_min_vol]:.3f}")

    with col2:
        st.markdown("**Maximum Sharpe Portfolio**")
        st.metric("Return", f"{port_returns[idx_max_sharpe]:.2%}")
        st.metric("Volatility", f"{port_vols[idx_max_sharpe]:.2%}")
        st.metric("Sharpe Ratio", f"{port_sharpes[idx_max_sharpe]:.3f}")

    with col3:
        # Risk parity portfolio stats
        rp_ret = portfolio_return(w_risk_parity, mu_array)
        rp_vol = portfolio_volatility(w_risk_parity, cov_array)
        rp_sharpe = portfolio_sharpe(w_risk_parity, mu_array, cov_array, rf)

        st.markdown("**Risk Parity Portfolio**")
        st.metric("Return", f"{rp_ret:.2%}")
        st.metric("Volatility", f"{rp_vol:.2%}")
        st.metric("Sharpe Ratio", f"{rp_sharpe:.3f}")

    # Correlation matrix
    if show_correlation_heatmap:
        st.subheader("Asset Correlation Matrix")

        with st.expander("Why correlation matters", expanded=False):
            st.markdown("""
            **Diversification depends on correlation:**
            - **Correlation = 1**: Assets move identically (no diversification benefit)
            - **Correlation = 0**: Assets move independently (good diversification)
            - **Correlation = -1**: Assets move opposite (perfect hedge)

            In practice, most stocks have positive correlations (0.3-0.8). Adding assets with lower
            correlations (bonds, gold, international) can significantly reduce portfolio volatility.

            **Warning:** Correlations tend to increase during market crises, reducing diversification
            exactly when it's most needed.
            """)

        corr_matrix = returns_df.corr()

        # Convert to long format for heatmap
        corr_long = corr_matrix.reset_index().melt(
            id_vars="index",
            var_name="Asset2",
            value_name="Correlation"
        ).rename(columns={"index": "Asset1"})

        heatmap = (
            alt.Chart(corr_long)
            .mark_rect()
            .encode(
                x=alt.X("Asset1:N", title=None),
                y=alt.Y("Asset2:N", title=None),
                color=alt.Color("Correlation:Q",
                               scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                               legend=alt.Legend(title="Correlation")),
                tooltip=["Asset1:N", "Asset2:N",
                        alt.Tooltip("Correlation:Q", format=".2f")]
            )
        )

        text = (
            alt.Chart(corr_long)
            .mark_text(color="white")
            .encode(
                x=alt.X("Asset1:N"),
                y=alt.Y("Asset2:N"),
                text=alt.Text("Correlation:Q", format=".2f")
            )
        )

        corr_chart = (heatmap + text).properties(
            width=400,
            height=400,
            title="Correlation Matrix"
        )

        st.altair_chart(configure_altair_theme(corr_chart), use_container_width=True)


with tab2:
    st.subheader("Portfolio Allocation Weights")

    with st.expander("Understanding allocation weights", expanded=False):
        st.markdown("""
        **Allocation weights** show how to divide your capital across assets:
        - Weight of 25% in AAPL means 25% of your portfolio value is in Apple stock
        - Weights sum to 100% (fully invested, no leverage)
        - These are long-only portfolios (no short selling)

        **Different allocations serve different goals:**
        - **Minimum Volatility**: Heavy in low-correlation assets (bonds, gold) for stability
        - **Maximum Sharpe**: Tilts toward high-return assets while maintaining diversification
        - **Risk Parity**: Balances risk contribution, not capital allocation
        """)

    # Create weight comparison table
    weight_df = pd.DataFrame({
        "Asset": assets,
        "Min Volatility": w_min_vol,
        "Max Sharpe": w_max_sharpe,
        "Risk Parity": w_risk_parity,
        "Equal Weight": np.ones(n_assets) / n_assets
    })

    # Display as percentage
    weight_display = weight_df.copy()
    for col in ["Min Volatility", "Max Sharpe", "Risk Parity", "Equal Weight"]:
        weight_display[col] = weight_display[col].apply(lambda x: f"{x:.1%}")

    st.dataframe(weight_display, use_container_width=True, hide_index=True)

    # Pie charts
    st.subheader("Visual Comparison")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Minimum Volatility Portfolio**")
        st.caption("Optimized for lowest risk")

        # Create pie chart data
        pie_data_mv = pd.DataFrame({
            "Asset": assets,
            "Weight": w_min_vol
        })
        pie_data_mv = pie_data_mv[pie_data_mv["Weight"] > 0.01]  # Filter tiny weights

        pie_mv = (
            alt.Chart(pie_data_mv)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("Weight:Q", stack=True),
                color=alt.Color("Asset:N", scale=alt.Scale(scheme="category10")),
                tooltip=["Asset:N", alt.Tooltip("Weight:Q", format=".1%")]
            )
            .properties(width=300, height=300)
        )
        st.altair_chart(pie_mv, use_container_width=True)

    with col2:
        st.markdown("**Maximum Sharpe Portfolio**")
        st.caption("Optimized for risk-adjusted return")

        pie_data_ms = pd.DataFrame({
            "Asset": assets,
            "Weight": w_max_sharpe
        })
        pie_data_ms = pie_data_ms[pie_data_ms["Weight"] > 0.01]

        pie_ms = (
            alt.Chart(pie_data_ms)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("Weight:Q", stack=True),
                color=alt.Color("Asset:N", scale=alt.Scale(scheme="category10")),
                tooltip=["Asset:N", alt.Tooltip("Weight:Q", format=".1%")]
            )
            .properties(width=300, height=300)
        )
        st.altair_chart(pie_ms, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Risk Parity Portfolio**")
        st.caption("Equal risk contribution from each asset")

        pie_data_rp = pd.DataFrame({
            "Asset": assets,
            "Weight": w_risk_parity
        })
        pie_data_rp = pie_data_rp[pie_data_rp["Weight"] > 0.01]

        pie_rp = (
            alt.Chart(pie_data_rp)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("Weight:Q", stack=True),
                color=alt.Color("Asset:N", scale=alt.Scale(scheme="category10")),
                tooltip=["Asset:N", alt.Tooltip("Weight:Q", format=".1%")]
            )
            .properties(width=300, height=300)
        )
        st.altair_chart(pie_rp, use_container_width=True)

    with col4:
        st.markdown("**Equal Weight Portfolio**")
        st.caption("Simple baseline allocation")

        pie_data_eq = pd.DataFrame({
            "Asset": assets,
            "Weight": np.ones(n_assets) / n_assets
        })

        pie_eq = (
            alt.Chart(pie_data_eq)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta("Weight:Q", stack=True),
                color=alt.Color("Asset:N", scale=alt.Scale(scheme="category10")),
                tooltip=["Asset:N", alt.Tooltip("Weight:Q", format=".1%")]
            )
            .properties(width=300, height=300)
        )
        st.altair_chart(pie_eq, use_container_width=True)

    # Weight bar chart comparison
    st.subheader("Weight Comparison by Asset")

    weight_long = weight_df.melt(
        id_vars="Asset",
        value_vars=["Min Volatility", "Max Sharpe", "Risk Parity", "Equal Weight"],
        var_name="Strategy",
        value_name="Weight"
    )

    weight_bar = (
        alt.Chart(weight_long)
        .mark_bar()
        .encode(
            x=alt.X("Asset:N", title=None),
            y=alt.Y("Weight:Q", title="Weight", axis=alt.Axis(format=".0%")),
            color=alt.Color("Strategy:N", scale=alt.Scale(scheme="category10")),
            xOffset="Strategy:N",
            tooltip=["Asset:N", "Strategy:N", alt.Tooltip("Weight:Q", format=".1%")]
        )
        .properties(width=CHART_WIDTH, height=350, title="Portfolio Weights by Strategy")
    )

    st.altair_chart(configure_altair_theme(weight_bar), use_container_width=True)


with tab3:
    st.subheader("Performance Metrics Comparison")

    with st.expander("How to interpret these metrics", expanded=False):
        st.markdown("""
        | **Metric** | **Meaning** | **Better** |
        |-----------|------------|-----------|
        | Return | Annualized expected return | Higher |
        | Volatility | Annualized standard deviation | Lower |
        | Sharpe Ratio | Return per unit of risk | Higher |
        | Max Weight | Largest position in portfolio | Depends on concentration preference |
        | Effective N | Diversification measure (1 = concentrated, n = equal weight) | Higher |
        """)

    # Calculate metrics for all portfolios
    portfolios = {
        "Min Volatility": w_min_vol,
        "Max Sharpe": w_max_sharpe,
        "Risk Parity": w_risk_parity,
        "Equal Weight": np.ones(n_assets) / n_assets
    }

    metrics_data = []
    for name, weights_arr in portfolios.items():
        ret = portfolio_return(weights_arr, mu_array)
        vol = portfolio_volatility(weights_arr, cov_array)
        sharpe = portfolio_sharpe(weights_arr, mu_array, cov_array, rf)
        max_weight = weights_arr.max()
        # Effective N (inverse Herfindahl index)
        eff_n = 1 / (weights_arr ** 2).sum()

        metrics_data.append({
            "Portfolio": name,
            "Return": ret,
            "Volatility": vol,
            "Sharpe": sharpe,
            "Max Weight": max_weight,
            "Effective N": eff_n
        })

    metrics_df = pd.DataFrame(metrics_data)

    # Display formatted metrics
    metrics_display = metrics_df.copy()
    metrics_display["Return"] = metrics_display["Return"].apply(lambda x: f"{x:.2%}")
    metrics_display["Volatility"] = metrics_display["Volatility"].apply(lambda x: f"{x:.2%}")
    metrics_display["Sharpe"] = metrics_display["Sharpe"].apply(lambda x: f"{x:.3f}")
    metrics_display["Max Weight"] = metrics_display["Max Weight"].apply(lambda x: f"{x:.1%}")
    metrics_display["Effective N"] = metrics_display["Effective N"].apply(lambda x: f"{x:.2f}")

    st.dataframe(metrics_display, use_container_width=True, hide_index=True)

    # Visual comparison
    st.subheader("Risk-Return Trade-off")

    portfolio_scatter = pd.DataFrame({
        "Volatility": [m["Volatility"] for m in metrics_data],
        "Return": [m["Return"] for m in metrics_data],
        "Portfolio": [m["Portfolio"] for m in metrics_data],
        "Sharpe": [m["Sharpe"] for m in metrics_data]
    })

    portfolio_chart = (
        alt.Chart(portfolio_scatter)
        .mark_point(size=200, filled=True)
        .encode(
            x=alt.X("Volatility:Q", title="Annualized Volatility",
                   axis=alt.Axis(format=".0%")),
            y=alt.Y("Return:Q", title="Annualized Return",
                   axis=alt.Axis(format=".0%")),
            color=alt.Color("Portfolio:N", scale=alt.Scale(scheme="category10")),
            tooltip=["Portfolio:N",
                    alt.Tooltip("Volatility:Q", format=".2%"),
                    alt.Tooltip("Return:Q", format=".2%"),
                    alt.Tooltip("Sharpe:Q", format=".3f")]
        )
        .properties(width=500, height=400, title="Portfolio Comparison")
    )

    st.altair_chart(configure_altair_theme(portfolio_chart), use_container_width=True)

    # Individual asset performance
    st.subheader("Individual Asset Performance")

    asset_metrics = []
    for i, asset in enumerate(assets):
        asset_ret = mu_array[i]
        asset_vol = np.sqrt(cov_array[i, i])
        asset_sharpe = (asset_ret - rf) / asset_vol if asset_vol > 0 else 0

        asset_metrics.append({
            "Asset": asset,
            "Return": asset_ret,
            "Volatility": asset_vol,
            "Sharpe": asset_sharpe
        })

    asset_metrics_df = pd.DataFrame(asset_metrics)

    asset_display = asset_metrics_df.copy()
    asset_display["Return"] = asset_display["Return"].apply(lambda x: f"{x:.2%}")
    asset_display["Volatility"] = asset_display["Volatility"].apply(lambda x: f"{x:.2%}")
    asset_display["Sharpe"] = asset_display["Sharpe"].apply(lambda x: f"{x:.3f}")

    st.dataframe(asset_display, use_container_width=True, hide_index=True)

    # Downside risk metrics
    st.subheader("Downside Risk Analysis")

    with st.expander("Understanding downside risk", expanded=False):
        st.markdown("""
        **VaR (Value at Risk)**: The loss threshold exceeded only in the worst X% of days.
        VaR at 5% means "on 95% of days, losses won't exceed this amount."

        **CVaR (Conditional VaR)**: The average loss when VaR is exceeded.
        Shows how bad the truly bad days are.

        **Downside Deviation**: Standard deviation of only negative returns.
        Used in Sortino ratio instead of total volatility.
        """)

    # Calculate VaR and CVaR for each portfolio
    downside_data = []
    for name, weights_arr in portfolios.items():
        portfolio_returns = returns_df.values @ weights_arr
        var_5 = np.percentile(portfolio_returns, 5)
        cvar_5 = portfolio_returns[portfolio_returns <= var_5].mean()
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0

        downside_data.append({
            "Portfolio": name,
            "VaR (5%)": var_5,
            "CVaR (5%)": cvar_5,
            "Downside Deviation": downside_dev
        })

    downside_df = pd.DataFrame(downside_data)

    downside_display = downside_df.copy()
    downside_display["VaR (5%)"] = downside_display["VaR (5%)"].apply(lambda x: f"{x:.4f}")
    downside_display["CVaR (5%)"] = downside_display["CVaR (5%)"].apply(lambda x: f"{x:.4f}")
    downside_display["Downside Deviation"] = downside_display["Downside Deviation"].apply(lambda x: f"{x:.2%}")

    st.dataframe(downside_display, use_container_width=True, hide_index=True)


with tab4:
    st.subheader("Portfolio Optimization: A Complete Guide")
    st.caption("Learn the theory behind modern portfolio construction")

    # Quick reference
    st.markdown("### Quick Reference Guide")
    st.markdown("""
    | **Goal** | **Use This Portfolio** | **Why** |
    |---------|----------------------|--------|
    | Minimize risk | Minimum Volatility | Lowest possible price swings |
    | Maximize risk-adjusted return | Maximum Sharpe | Best return per unit of risk |
    | Balance risk across assets | Risk Parity | No single asset dominates risk |
    | Simple baseline | Equal Weight | Easy to implement and rebalance |
    """)

    st.divider()

    # Optimization methods
    st.markdown("### Optimization Methods")

    for method, info in OPTIMIZATION_INFO.items():
        with st.expander(f"**{method}**", expanded=False):
            st.markdown(f"**Description:** {info['description']}")
            st.info(f"**Key Insight:** {info['key_insight']}")
            st.code(info['formula'], language=None)
            st.markdown(f"**Best For:** {info['best_for']}")

    st.divider()

    # Core concepts
    st.markdown("### Core Concepts")

    for concept, info in CONCEPT_INFO.items():
        with st.expander(f"**{concept}**", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**What is it?**")
                st.write(info['what'])
                st.markdown("**Why does it matter?**")
                st.write(info['why'])
            with col2:
                st.markdown("**How to calculate?**")
                st.write(info['how'])
                st.markdown("**Interpretation:**")
                st.write(info['interpretation'])

    st.divider()

    # MPT limitations
    st.markdown("### Limitations of Modern Portfolio Theory")

    st.warning("""
    While MPT provides a powerful framework, it relies on simplifying assumptions that often break down
    in real markets. Understanding these limitations is crucial for practical portfolio construction.
    """)

    for item in MPT_LIMITATIONS:
        st.error(f"""
        **{item['limitation']}**

        {item['explanation']}

        **Mitigation:** {item['mitigation']}
        """)

    st.divider()

    # Practical examples
    st.markdown("### Practical Examples")

    with st.expander("Example 1: Choosing Between Portfolios"):
        st.markdown("""
        **Scenario:** You've generated the efficient frontier and need to choose an allocation.

        **Key questions to ask:**
        1. What's your investment horizon? (Longer = can tolerate more volatility)
        2. How would you react to a 20% drawdown? (Can't sleep = need lower risk)
        3. Are you accumulating or preserving wealth? (Growth vs stability)

        **Decision framework:**
        - **Conservative (retirement, short horizon):** Minimum Volatility or underweight from frontier
        - **Moderate:** Risk Parity for balanced exposure
        - **Aggressive (young, long horizon):** Maximum Sharpe or above
        """)

    with st.expander("Example 2: Understanding Diversification Benefit"):
        st.markdown("""
        **Scenario:** You hold 100% in a high-return stock with 30% volatility.

        **What adding a second asset does:**
        - If correlation = 1.0: No diversification, portfolio vol = weighted average
        - If correlation = 0.5: Portfolio vol drops significantly below weighted average
        - If correlation = 0.0: Maximum diversification benefit
        - If correlation = -0.5: Can reduce volatility dramatically

        **Example calculation (50/50 split, 30% vol each):**
        - Correlation 1.0: Portfolio vol = 30%
        - Correlation 0.5: Portfolio vol = 26%
        - Correlation 0.0: Portfolio vol = 21%
        - Correlation -0.5: Portfolio vol = 15%

        **Lesson:** Even modest correlation reduction significantly improves risk-adjusted returns.
        """)

    with st.expander("Example 3: Risk Parity in Action"):
        st.markdown("""
        **Scenario:** Traditional 60/40 stock/bond portfolio.

        **Problem:** Stocks have ~3x the volatility of bonds.

        **Risk contribution analysis:**
        - 60% stocks × 15% vol = 9% contribution
        - 40% bonds × 5% vol = 2% contribution
        - **Result:** 80%+ of portfolio risk comes from stocks!

        **Risk Parity solution:**
        - Weight stocks less, bonds more
        - Might be 25% stocks, 75% bonds (unlevered)
        - Each contributes ~50% of portfolio risk

        **Trade-off:** Lower expected return for more balanced risk exposure.
        """)

    st.divider()

    # Common mistakes
    st.markdown("### Common Mistakes to Avoid")

    st.error("""
    **1. Optimizing on Short History**
    Using 1-2 years of data produces unstable estimates. Small sample sizes mean high estimation error.
    **Fix:** Use at least 3-5 years of data, or consider shrinkage estimators.
    """)

    st.error("""
    **2. Ignoring Transaction Costs**
    Optimal weights change daily, but frequent rebalancing erodes returns through costs and taxes.
    **Fix:** Rebalance only when weights drift significantly (e.g., 5% bands).
    """)

    st.error("""
    **3. Trusting Extreme Weights**
    Optimization may put 80%+ in one asset. This often reflects estimation error, not true opportunity.
    **Fix:** Add position limits (e.g., max 25% per asset) or use robust optimization.
    """)

    st.error("""
    **4. Assuming Past Correlations Persist**
    Correlations change over time and spike during crises.
    **Fix:** Stress test with correlation = 0.9 for all risky assets.
    """)

    st.error("""
    **5. Forgetting About Rebalancing**
    Portfolios drift from target weights as prices change.
    **Fix:** Establish systematic rebalancing rules (calendar or threshold-based).
    """)
