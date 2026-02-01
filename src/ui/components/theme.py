"""Theme constants and Altair chart configuration for the financial modelling UI."""

import altair as alt

# Theme colors (from existing app.py patterns)
THEME = {
    "primary": "#66c2ff",      # Blue - primary data color
    "secondary": "#fdd835",    # Yellow - accent/regression lines
    "background": "#0d1117",   # Dark background
    "grid": "#1f262d",         # Grid lines
    "text": "white",           # Text color
}

# Standard chart dimensions
CHART_WIDTH = 650
CHART_HEIGHT = 650


def configure_altair_theme(chart: alt.Chart) -> alt.Chart:
    """Apply standard dark theme to an Altair chart."""
    return (
        chart
        .configure_view(strokeWidth=0, fill=THEME["background"])
        .configure_axis(
            labelColor=THEME["text"],
            titleColor=THEME["text"],
            gridColor=THEME["grid"]
        )
        .configure_title(fontSize=16, color=THEME["text"])
        .configure_legend(
            labelColor=THEME["text"],
            titleColor=THEME["text"]
        )
    )
