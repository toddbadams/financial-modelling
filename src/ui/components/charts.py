"""Reusable Altair chart builders for the financial modelling UI."""

import altair as alt
import pandas as pd

from .theme import THEME, CHART_WIDTH, CHART_HEIGHT, configure_altair_theme


def create_scatter_with_regression(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    line_df: pd.DataFrame = None,
    x_title: str = None,
    y_title: str = None,
) -> alt.Chart:
    """Create themed scatter plot with optional regression line."""
    scatter = (
        alt.Chart(df)
        .mark_circle(
            size=80,
            opacity=0.75,
            stroke="white",
            strokeWidth=0.4,
            color=THEME["primary"],
        )
        .encode(
            x=alt.X(x_col, title=x_title or x_col),
            y=alt.Y(y_col, title=y_title or y_col),
            tooltip=[x_col, y_col],
        )
    )

    if line_df is not None:
        line = (
            alt.Chart(line_df)
            .mark_line(color=THEME["secondary"], strokeWidth=2)
            .encode(x=x_col, y="prediction")
        )
        chart = alt.layer(scatter, line)
    else:
        chart = scatter

    chart = chart.properties(width=CHART_WIDTH, height=CHART_HEIGHT)
    return configure_altair_theme(chart)


def create_time_series(
    df: pd.DataFrame,
    x_col: str,
    value_col: str,
    series_col: str,
    title: str = "",
) -> alt.Chart:
    """Create themed multi-series line chart."""
    chart = (
        alt.Chart(df)
        .mark_line(point=False)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_col.title()),
            y=alt.Y(value_col, title="Value"),
            color=alt.Color(
                f"{series_col}:N",
                scale=alt.Scale(range=[THEME["primary"], THEME["secondary"]]),
            ),
            tooltip=[x_col, series_col, value_col],
        )
        .properties(width=CHART_WIDTH, height=CHART_HEIGHT, title=title)
    )
    return configure_altair_theme(chart)


def create_density_plot(
    df: pd.DataFrame,
    value_col: str,
    series_col: str,
    title: str = "",
) -> alt.Chart:
    """Create themed kernel density plot."""
    chart = (
        alt.Chart(df)
        .transform_density(
            value_col,
            as_=[value_col, "density"],
            groupby=[series_col],
        )
        .mark_area(opacity=0.6)
        .encode(
            x=alt.X(value_col, title="Value"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color(
                f"{series_col}:N",
                scale=alt.Scale(range=[THEME["primary"], THEME["secondary"]]),
            ),
        )
        .properties(width=CHART_WIDTH, height=CHART_HEIGHT, title=title)
    )
    return configure_altair_theme(chart)
