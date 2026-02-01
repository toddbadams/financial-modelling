"""Hidden Factors page - Synthetic Market Signal vs Returns analysis."""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import altair as alt

from components.theme import THEME, CHART_WIDTH, CHART_HEIGHT, configure_altair_theme
from components.layouts import render_page_header


@st.cache_data
def generate_data(observations, macro_scale, macro_sensitivity, signal_noise, returns_noise):
    """Generate synthetic market data with hidden macro factor."""
    rng = np.random.default_rng(42)
    macro = rng.normal(0, macro_scale, observations)
    signal = macro + rng.normal(0, signal_noise, observations)
    returns = macro_sensitivity * macro + rng.normal(0, returns_noise, observations)
    return pd.DataFrame({"signal": signal, "returns": returns})


# Page header
render_page_header(
    "Synthetic Market Signal vs Returns",
    "This app simulates a hidden macro factor driving returns while the observable signal "
    "merely co-moves with that factor. Adjust the knobs to see how the fitted regression behaves. "
    "The lesson is that correlation can be strong because both signal and returns share the same "
    "latent driver; the signal can help forecast returns in historical data, but it is not itself "
    "the causal mechanism.",
)

# Sidebar controls
with st.sidebar:
    st.header("Simulation Controls")
    n = st.slider("Number of observations", min_value=50, max_value=2000, value=300, step=50)
    macro_scale = st.slider(
        "Macro factor scale (std dev)",
        min_value=0.1,
        max_value=3.0,
        value=1.0,
        step=0.1,
    )
    macro_sensitivity = st.slider(
        "Macro sensitivity of returns",
        min_value=0.5,
        max_value=3.0,
        value=1.5,
        step=0.1,
    )
    signal_noise = st.slider(
        "Signal noise (std dev)",
        min_value=0.05,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )
    returns_noise = st.slider(
        "Returns noise (std dev)",
        min_value=0.05,
        max_value=1.0,
        value=0.3,
        step=0.05,
    )

# Generate data
df = generate_data(n, macro_scale, macro_sensitivity, signal_noise, returns_noise)

# Fit regression
model = LinearRegression().fit(df[["signal"]], df["returns"])
grid = np.linspace(df["signal"].min(), df["signal"].max(), 100)
prediction = model.predict(grid.reshape(-1, 1))
line_df = pd.DataFrame({"signal": grid, "prediction": prediction})

# Create charts
scatter = (
    alt.Chart(df)
    .mark_circle(size=80, opacity=0.75, stroke="white", strokeWidth=0.4, color=THEME["primary"])
    .encode(
        x=alt.X("signal", title="Market Signal"),
        y=alt.Y("returns", title="Returns"),
        tooltip=["signal", "returns"],
    )
)

line = (
    alt.Chart(line_df)
    .mark_line(color=THEME["secondary"], strokeWidth=2)
    .encode(x="signal", y="prediction")
)

chart = (
    alt.layer(scatter, line)
    .properties(width=CHART_WIDTH, height=CHART_HEIGHT)
)
chart = configure_altair_theme(chart)

# Time series overlay
df_timeseries = df.reset_index(drop=True).copy()
df_timeseries["observation"] = df_timeseries.index + 1
melted = df_timeseries.melt(
    id_vars="observation",
    value_vars=["signal", "returns"],
    var_name="series",
    value_name="value",
)

overlay = (
    alt.Chart(melted)
    .mark_line(point=False)
    .encode(
        x=alt.X("observation:Q", title="Observation"),
        y=alt.Y("value", title="Value"),
        color=alt.Color(
            "series:N",
            scale=alt.Scale(domain=["signal", "returns"], range=[THEME["primary"], THEME["secondary"]]),
        ),
        tooltip=["observation", "series", "value"],
    )
    .properties(width=CHART_WIDTH, height=CHART_HEIGHT, title="Signal vs Returns over Observations")
)
overlay = configure_altair_theme(overlay)

# Histogram / density plot
hist = (
    alt.Chart(melted)
    .transform_density(
        "value",
        as_=["value", "density"],
        groupby=["series"],
    )
    .mark_area(opacity=0.6)
    .encode(
        x=alt.X("value", title="Value"),
        y=alt.Y("density:Q", title="Density"),
        color=alt.Color(
            "series:N",
            scale=alt.Scale(domain=["signal", "returns"], range=[THEME["primary"], THEME["secondary"]]),
        ),
    )
    .properties(width=CHART_WIDTH, height=CHART_HEIGHT, title="Histogram (Kernel density)")
)
hist = configure_altair_theme(hist)

# Correlation
corr_table = df[["signal", "returns"]].corr().round(3)
corr_value = corr_table.loc["signal", "returns"]

# Tabs with two-column layout
tab1, tab2, tab3 = st.tabs(["Signal vs Returns", "Series overlay", "Distribution"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.altair_chart(chart, use_container_width=True)
    with col2:
        st.markdown(
            "### Signal vs Returns\n"
            "- The scatter shows how the observable signal co-moves with returns.\n"
            "- The yellow regression line is `returns ~ signal`, capturing their average relationship.\n"
            "- **Signal**: a real-time indicator such as intraday momentum or a short-term moving average that traders can compute.\n"
            "- **Macro factor**: a broader latent state like liquidity conditions or risk-on/risk-off sentiment circulating through risk assets.\n"
            "- **Returns**: the asset outcome (e.g., daily excess return on the index) driven by that macro factor."
        )
        st.table(corr_table)
        st.markdown(
            f"- Correlation (~{corr_value:.2f}) means the signal and returns move together.\n"
            "- The signal is predictively useful in historical data.\n"
            "- Correlation does not imply the signal causes returns."
        )

with tab2:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.altair_chart(overlay, use_container_width=True)
    with col2:
        st.markdown(
            "### Series overlay\n"
            "- Both `signal` and `returns` are plotted over the same observation axis.\n"
            "- Matching up their peaks/troughs highlights how the latent macro factor drives both."
        )
        st.markdown(
            f"- **Observations:** {n}\n"
            f"- **Macro scale:** {macro_scale:.2f}\n"
            f"- **Macro sensitivity:** {macro_sensitivity:.2f}\n"
            f"- **Signal noise:** {signal_noise:.2f}\n"
            f"- **Returns noise:** {returns_noise:.2f}"
        )

with tab3:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.altair_chart(hist, use_container_width=True)
    with col2:
        st.markdown(
            "### Distribution\n"
            "- Kernel density plots show the marginal spread of both series.\n"
            "- They look similar because both are driven by the same macro."
        )
        st.markdown(
            "You can dial down the correlation by making the noise terms large relative to the shared macro factor (increase "
            "`signal_noise` or `returns_noise`, or reduce `macro_scale`/`macro_sensitivity`). When noise swamps the macro, "
            "the computed correlation drops toward zero, so you can no longer confidently say the signal and returns move "
            "together. Even then, both series still originate from that latent macro factor; the signal simply becomes a "
            "noisier proxy and loses predictive strength."
        )
