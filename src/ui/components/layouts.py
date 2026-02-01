"""Reusable layout components for the financial modelling UI."""

import streamlit as st


def render_page_header(title: str, description: str = "") -> None:
    """Render consistent page header with title and optional description."""
    st.title(title)
    if description:
        st.markdown(description)


def render_placeholder_tab(tab_name: str, page_name: str) -> None:
    """Render a placeholder tab for pages under development."""
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info(f"Chart placeholder for {tab_name}")
    with col2:
        st.markdown(
            f"### {tab_name}\n\n"
            f"This tab is part of the **{page_name}** page and is under development."
        )
