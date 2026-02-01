"""Main entry point for the Financial Modelling multi-page Streamlit application."""

import streamlit as st

# Page configuration must be first Streamlit command
st.set_page_config(
    page_title="Financial Modelling",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define and run the multi-page navigation
pg = st.navigation([
    st.Page("pages/hidden_factors.py", title="Hidden Factors", icon=":material/analytics:"),
    st.Page("pages/asset_normalization.py", title="Asset Normalization", icon=":material/transform:"),
    st.Page("pages/asset_risks.py", title="Asset Risks", icon=":material/warning:"),
    st.Page("pages/asset_features.py", title="Asset Features", icon=":material/category:"),
    st.Page("pages/portfolio_optimization.py", title="Portfolio Optimization", icon=":material/trending_up:"),
])

pg.run()
