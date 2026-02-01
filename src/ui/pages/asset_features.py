"""Asset Features page - Explore technical indicators from the Features class."""

import streamlit as st
from datetime import date, timedelta

from components.layouts import render_page_header, render_placeholder_tab


def get_features_by_category(category: str) -> list[str]:
    """Return feature names based on category from Features class."""
    categories = {
        "Momentum": ["rsi_5", "rsi_10", "rsi_15", "roc_10", "mom_10"],
        "Oscillators": ["STOCHRSIk", "STOCHRSId", "cci_20", "wr_14", "MACD", "MACDh", "MACDs"],
        "Trend": ["sma_5", "sma_10", "sma_20", "ema_5", "ema_10", "ema_20", "vwma_20"],
        "Volatility": ["BBL", "BBM", "BBU", "BBB", "BBP", "atr_14", "KCL", "KCM", "KCU"],
        "Volume": ["obv", "ad", "efi", "nvi", "pvi"],
    }
    return categories.get(category, [])


# Page header
render_page_header(
    "Asset Features",
    "Explore 30+ technical indicators computed by the Features class. "
    "These indicators are used for machine learning model training.",
)

# Sidebar controls
with st.sidebar:
    st.header("Feature Options")
    ticker = st.text_input("Ticker Symbol", value="AAPL")
    start_date = st.date_input(
        "Start Date",
        value=date.today() - timedelta(days=365),
    )
    end_date = st.date_input("End Date", value=date.today())
    feature_category = st.selectbox(
        "Feature Category",
        ["Momentum", "Oscillators", "Trend", "Volatility", "Volume"],
    )
    available_features = get_features_by_category(feature_category)
    selected_features = st.multiselect(
        "Select Features",
        available_features,
        default=available_features[:2] if available_features else [],
    )

# Tabs with two-column layout
tab1, tab2, tab3 = st.tabs(["Feature Time Series", "Feature Correlation", "Feature Distribution"])

with tab1:
    render_placeholder_tab("Feature Time Series", "Asset Features")

with tab2:
    render_placeholder_tab("Feature Correlation", "Asset Features")

with tab3:
    render_placeholder_tab("Feature Distribution", "Asset Features")
