"""Asset Risks page - Analyze volatility, drawdowns, and risk metrics."""

import streamlit as st
from datetime import date, timedelta

from components.layouts import render_page_header, render_placeholder_tab

# Page header
render_page_header(
    "Asset Risks",
    "Analyze volatility, drawdowns, and risk metrics for selected assets. "
    "Understand the risk profile of your investments.",
)

# Sidebar controls
with st.sidebar:
    st.header("Risk Analysis Options")
    ticker = st.text_input("Ticker Symbol", value="AAPL")
    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=365),
    )
    end_date = st.date_input("End Date", value=date.today())
    risk_free_rate = st.number_input(
        "Risk-Free Rate (%)",
        value=4.0,
        min_value=0.0,
        max_value=20.0,
        step=0.1,
    )
    window = st.slider("Rolling Window (days)", min_value=5, max_value=252, value=21)
    confidence_level = st.slider(
        "VaR Confidence Level (%)",
        min_value=90,
        max_value=99,
        value=95,
    )

# Tabs with two-column layout
tab1, tab2, tab3 = st.tabs(["Volatility Analysis", "Drawdown Chart", "VaR/CVaR"])

with tab1:
    render_placeholder_tab("Volatility Analysis", "Asset Risks")

with tab2:
    render_placeholder_tab("Drawdown Chart", "Asset Risks")

with tab3:
    render_placeholder_tab("VaR/CVaR", "Asset Risks")
