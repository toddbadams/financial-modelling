# UI Context (Streamlit Educational Dashboard)

## Purpose
Provide the canonical UI context for the Streamlit dashboard under `src/ui`. Use this when adding or modifying UI behavior, charts, or layouts.

## Scope
- Streamlit educational dashboard (pages under `src/ui/pages`).
- Focus: teaching quantitative finance concepts through interactive visualizations, live market data, and structured educational content.

## Primary users
- Finance student: learn risk, return, and portfolio concepts interactively.
- Self-directed learner: explore financial data and understand metrics with educational guidance.
- Quant developer: prototype and validate financial calculations.

## Tech stack
- Streamlit app: `src/ui/app.py` (entry point with `st.navigation()`) and `src/ui/pages/*.py`.
- Data access: `src/data-layer/data.py` — `DataInjestor.get()` fetches OHLCV data from Yahoo Finance via yfinance, caching as Parquet in `.cache/financial_data/`.
- Charts: Altair with reusable builders in `src/ui/components/charts.py` and theme in `src/ui/components/theme.py`.
- Layouts: `src/ui/components/layouts.py` provides `render_page_header()`.

## Page structure pattern
Each page is self-contained and follows this pattern (see `src/ui/pages/asset_features.py` as the canonical example):
1. Educational content dictionaries at module level (`FEATURE_INFO`, `CATEGORY_INFO`, etc.)
2. Calculation functions (pure Python/NumPy, no Streamlit calls)
3. Sidebar controls in `with st.sidebar:` block
4. Data fetching via `DataInjestor.get()`
5. Tabs: interactive analysis tabs + a "Learning Center" tab

Pages add `data-layer` to sys.path for imports:
```python
_data_layer_path = Path(__file__).parent.parent.parent / "data-layer"
if str(_data_layer_path) not in sys.path:
    sys.path.insert(0, str(_data_layer_path))
from data import DataInjestor
```

## Design system: IBM Carbon
Apply Carbon principles to Streamlit layouts and components:
- Clear hierarchy: headers, subheaders, and tight grouping for data-dense pages.
- Consistent spacing: keep filters together, then summary metrics, then tables/charts.
- Use semantic color sparingly for status (success/failure/warning).

## UX standard: Apple HIG
Apply HIG concepts within Streamlit constraints:
- Clarity: label filters and metrics explicitly; avoid jargon in UI labels.
- Deference: keep chrome minimal; make data the focus.
- Depth: use sections, tabs, and progressive disclosure for detail.

## Layout conventions
- Use `layout="wide"` (set in `app.py`).
- Sidebar: all user controls (ticker inputs, date ranges, parameter sliders).
- Summary row: `st.metric()` in column layouts with `st.caption()` for interpretation before detailed content.
- Tabs: interactive analysis tabs first, then a "Learning Center" tab.
- Educational content: `st.expander()` for concept explanations, `st.error()` blocks for "Common Mistakes to Avoid", practical examples in expanders.
- Use `st.subheader` and `st.caption` for structure and context.

## Charts
- Altair only. Apply `configure_altair_theme()` from `src/ui/components/theme.py` to all charts.
- Use `THEME` dict colors: `primary=#66c2ff` (blue), `secondary=#fdd835` (yellow), `background=#0d1117` (dark).
- Use `CHART_WIDTH` and `CHART_HEIGHT` constants for sizing.
- Reusable chart builders in `src/ui/components/charts.py`: `create_scatter_with_regression()`, `create_time_series()`, `create_density_plot()`.
- Keep tooltips informative with formatted values (e.g., `alt.Tooltip("Return:Q", format=".2%")`).

## States and messaging
- Loading: prefer `st.cache_data` with a concise spinner message.
- Empty states: use `st.info` with a plain explanation.
- Errors: use `st.error()` for missing data or invalid input, `st.stop()` to halt page rendering.
- Warnings: use `st.warning()` for partial data issues (e.g., failed ticker fetches).

## Performance notes
- Cache data fetches with `st.cache_data` and reuse DataFrames.
- Filter data before rendering tables and charts.
- Avoid expensive per-row UI elements for large tables.

## Testing expectations
- Visual checks only (manual sanity checks in the Streamlit UI).
- Confirm pages load, filters work, and charts render without errors.
