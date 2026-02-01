"""Asset Normalization page - Explore normalization techniques for financial time series."""

import streamlit as st
from datetime import date, timedelta

from components.layouts import render_page_header, render_placeholder_tab

# Page header
render_page_header(
    "Asset Normalization",
    "Explore different normalization techniques for financial time series data. "
    "Normalization helps compare assets with different price scales and volatilities.",
)

# Sidebar controls
with st.sidebar:
    st.header("Normalization Options")
    ticker = st.text_input("Ticker Symbol", value="AAPL")
    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=365),
    )
    end_date = st.date_input("End Date", value=date.today())
    normalization_method = st.selectbox(
        "Normalization Method",
        ["Z-Score", "Min-Max", "Log Returns", "Percent Change"],
    )
    window = st.slider("Rolling Window (days)", min_value=5, max_value=100, value=20)

# Tabs with two-column layout
tab1, tab2, tab3 = st.tabs(["Raw vs Normalized", "Distribution Comparison", "Rolling Statistics"])

with tab1:
    render_placeholder_tab("Raw vs Normalized", "Asset Normalization")

with tab2:
    render_placeholder_tab("Distribution Comparison", "Asset Normalization")

with tab3:
    render_placeholder_tab("Rolling Statistics", "Asset Normalization")
