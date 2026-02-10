# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational financial modeling application built with Streamlit. The app teaches quantitative finance concepts (risk, returns, portfolio optimization) through interactive dashboards with live market data. A separate ML classifier module uses XGBoost for trading signal generation.

## Repository Structure

- **`/src`** — Application source code (Streamlit UI, data layer, reusable components)
- **`/workbooks`** — Resource material used as background information when creating the application (Jupyter notebooks, ML classifier module, educational notes organized by week)
- **`/ai_context`** — Context markdown files for AI agents (`agents.md` entry point, `plans.md` ExecPlan methodology, `ui_context.md` UI conventions)

## Commands

```bash
# Install dependencies (Poetry required, Python >=3.12 <3.14)
poetry install -v

# Activate virtual environment
poetry shell
# or: .\.venv\Scripts\Activate (PowerShell)

# Run the Streamlit app
streamlit run src/ui/app.py

# Linting & formatting
flake8
black .
isort .
mypy .

# Tests
pytest
```

## Architecture

### Streamlit App (`src/ui/`)

Multi-page app using `st.navigation()` in `app.py`. Each page in `pages/` is self-contained with its own sidebar controls, data fetching, calculations, and tab-based layout.

**Page structure pattern** (see `asset_features.py` as the canonical example):
1. Educational content dictionaries at module level (`FEATURE_INFO`, `CATEGORY_INFO`, etc.)
2. Calculation functions (pure Python/NumPy, no Streamlit calls)
3. Sidebar controls in `with st.sidebar:` block
4. Data fetching via `DataInjestor.get()`
5. Tabs: interactive analysis tabs + a "Learning Center" tab with educational content

**Pages add `data-layer` to sys.path** for imports:
```python
_data_layer_path = Path(__file__).parent.parent.parent / "data-layer"
if str(_data_layer_path) not in sys.path:
    sys.path.insert(0, str(_data_layer_path))
from data import DataInjestor
```

### Reusable Components (`src/ui/components/`)

- **`theme.py`** — `THEME` dict (dark theme colors), `CHART_WIDTH`/`CHART_HEIGHT` constants, `configure_altair_theme()` for consistent Altair chart styling
- **`layouts.py`** — `render_page_header()` for consistent page titles
- **`charts.py`** — Reusable Altair chart builders: scatter with regression, time series, density plots

### Data Layer (`src/data-layer/data.py`)

`DataInjestor.get(symbols, start_date, end_date)` fetches OHLCV data from Yahoo Finance via yfinance, caching results as Parquet files in `.cache/financial_data/`. Handles both single ticker strings and lists.

### ML Classifier (`workbooks/ml_classifier/`)

Standalone module for XGBoost-based trading signal classification. Uses `features.py` to generate 30+ technical indicators via pandas-TA, `train.py` for model training with GridSearchCV + TimeSeriesSplit.

## UI Conventions

- **Charts**: Altair only. Always apply `configure_altair_theme()`. Use `THEME` colors (`primary=#66c2ff`, `secondary=#fdd835`).
- **Layout**: `layout="wide"`, sidebar for controls, tabs for content organization, expanders for educational content.
- **Educational content**: Each page includes a "Learning Center" tab with quick reference tables, concept explanations in expanders (using `col1, col2` layout with What/Why on left, How/Interpretation on right), practical examples, and "Common Mistakes to Avoid" using `st.error()`.
- **Metrics display**: `st.metric()` in column layouts with `st.caption()` for interpretation.
- **Design principles**: IBM Carbon (clear hierarchy, consistent spacing, semantic color) and Apple HIG (clarity, deference, depth via progressive disclosure).

## Flake8 Configuration

Max line length: 400. Extends ignore: F401, F841, F601, F541, E501, W605, W291, F821.
