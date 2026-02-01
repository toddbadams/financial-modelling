"""Portfolio Optimization page - Optimize portfolio weights using various techniques."""

import streamlit as st
from datetime import date, timedelta

from components.layouts import render_page_header, render_placeholder_tab

# Page header
render_page_header(
    "Portfolio Optimization",
    "Optimize portfolio weights using mean-variance optimization and other techniques. "
    "Find the optimal allocation for your investment goals.",
)

# Sidebar controls
with st.sidebar:
    st.header("Portfolio Options")
    tickers = st.text_area(
        "Tickers (one per line)",
        value="AAPL\nMSFT\nGOOG\nAMZN",
        height=100,
    )
    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=365 * 2),
    )
    end_date = st.date_input("End Date", value=date.today())
    optimization_method = st.selectbox(
        "Optimization Method",
        [
            "Mean-Variance (Markowitz)",
            "Risk Parity",
            "Maximum Sharpe",
            "Minimum Volatility",
        ],
    )
    target_return = st.slider(
        "Target Return (%)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=0.5,
    )
    risk_free_rate = st.number_input(
        "Risk-Free Rate (%)",
        value=4.0,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
    )

# Tabs with two-column layout
tab1, tab2, tab3 = st.tabs(["Efficient Frontier", "Allocation Weights", "Performance Metrics"])

with tab1:
    render_placeholder_tab("Efficient Frontier", "Portfolio Optimization")

with tab2:
    render_placeholder_tab("Allocation Weights", "Portfolio Optimization")

with tab3:
    render_placeholder_tab("Performance Metrics", "Portfolio Optimization")
