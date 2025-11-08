# Supermarket Analytics API Reference

This document describes every public function and Streamlit component exposed by the Supermarket Analytics workspace. Use it as a companion when integrating individual analytics features into new apps, extending the Streamlit dashboard, or writing automated tests.

---

## Quick Start
- Install dependencies: `pip install -r requirements.txt`
- Make sure `data/SuperMarket Analysis.csv` is present; most functions expect the canonical column set from the Kaggle supermarket dataset.
- Launch the full dashboard: `streamlit run app.py`
- Individual modules (e.g., `sales_forecaster`, `retail_price_suggester`) expose their own `app()` entry point so they can be embedded in custom Streamlit layouts.

---

## Module Overview

| Module | Responsibility | Key Entry Points |
| --- | --- | --- |
| `app.py` | Global Streamlit navigation shell | `main()`, `home()` |
| `sales_forecaster.py` | Demand forecasting (ARIMA & ML) | `app()`, `train_ml_model()`, `arima_forecast()` |
| `inventory_reordering.py` | Simulated stock levels & reorder planning | `app()`, `load_data()` |
| `retail_price_suggester.py` | Price elasticity analysis & optimization | `app()`, `calculate_optimal_price()` |
| `promotional_items.py` | Promotion candidate identification via clustering | `app()`, `perform_clustering()` |
| `chatbot.py` | Pattern-based Q&A over the sales dataset | `app()`, `generate_response()` |

All modules rely on Streamlit primitives for UI rendering; non-UI helper functions are documented below with their pure Python signatures.

---

## `app.py` — Streamlit Shell

### `load_basic_data() -> dict`
- **Purpose**: Read the supermarket CSV and compute headline metrics for the home dashboard.
- **Caching**: Decorated with `@st.cache_data(ttl=3600)`.
- **Returns**: `{"total_sales": float, "avg_gross_margin": float, "avg_rating": float}`. Falls back to hard-coded defaults if the CSV is missing.
- **Side Effects**: Displays a Streamlit error message when the file cannot be read.
- **Example**:
```python
from app import load_basic_data

metrics = load_basic_data()
print(f"Total sales: ${metrics['total_sales']:.2f}")
```

### `sidebar() -> str`
- **Purpose**: Render the navigation radio buttons in the Streamlit sidebar.
- **Returns**: One of `"Home"`, `"Sales Forecasting"`, `"Inventory"`, `"Pricing"`, `"Promotions"`, `"Chatbot"`.
- **Usage**: Typically called from `main()` to decide which module `app()` to mount.

### `home() -> None`
- **Purpose**: Render the landing page with KPI metrics and quick navigation buttons.
- **Dependencies**: Calls `load_basic_data()`; requires Streamlit session state.
- **Usage**: Executed automatically when `st.session_state.page` is `"Home"`.

### `main() -> None`
- **Purpose**: Orchestrate navigation between module `app()` functions.
- **Behavior**: Ensures `st.session_state.page` is initialized, synchronizes radio selection, and dispatches to the appropriate module.
- **Usage**:
```python
import streamlit as st
import app

if __name__ == "__main__":
    app.main()
```

---

## `sales_forecaster.py` — Demand Forecasting Engine

### `load_data() -> pandas.DataFrame`
- **Purpose**: Load the supermarket CSV, coerce `Date` into `datetime`, and guarantee a `Total` column.
- **Failure Modes**: Emits a Streamlit error and returns an empty `DataFrame` if the file is missing or parsing fails.
- **Example**:
```python
from sales_forecaster import load_data

df = load_data()
if df.empty:
    raise FileNotFoundError("Upload SuperMarket Analysis.csv first.")
```

### `engineer_features(data: pandas.DataFrame) -> pandas.DataFrame`
- **Purpose**: Enrich raw sales rows with date-based splits and one-hot encoded categorical columns required by ML models.
- **Parameters**:
  | Name | Type | Description |
  | --- | --- | --- |
  | `data` | `pandas.DataFrame` | Raw transactional data, ideally produced by `load_data()`. |
- **Returns**: A feature-rich DataFrame including new numeric columns such as `DayOfWeek`, `Branch_H/K`, etc.
- **Side Effects**: Adds dummy columns; safe to reuse original frame by passing `data.copy()`.
- **Example**:
```python
from sales_forecaster import engineer_features, load_data

features = engineer_features(load_data())
model_inputs = features.select_dtypes("number")
```

### `aggregate_data_for_timeseries(data, group_by='Date', agg_column='Total') -> pandas.DataFrame`
- **Purpose**: Collapse granular sales into a time series (or other grouping) for ARIMA training.
- **Parameters**:
  | Name | Type | Description |
  | --- | --- | --- |
  | `data` | `pandas.DataFrame` | Source data with the aggregation key and target column. |
  | `group_by` | `str` | Column used as the grouping key (`'Date'` by default). |
  | `agg_column` | `str` | Column to sum within each group (`'Total'` by default). |
- **Returns**: Sorted DataFrame with two columns: `group_by` and the aggregated total.
- **Example**:
```python
daily = aggregate_data_for_timeseries(features, group_by="Date", agg_column="Total")
```

### `train_ml_model(features, target, model_type='random_forest') -> tuple`
- **Purpose**: Fit either a linear regression or random forest regressor to predict transaction totals.
- **Parameters**:
  | Name | Type | Description |
  | --- | --- | --- |
  | `features` | `pandas.DataFrame` | Engineered numeric features. |
  | `target` | `pandas.Series` | Actual sales figures (`Total` or `Sales`). |
  | `model_type` | `str` | `'linear'` or `'random_forest'` (case insensitive). |
- **Returns**: `(model, feature_columns, scaler, metrics_dict)` where `metrics_dict` includes `MAE`, `RMSE`, and `R²`.
- **Side Effects**: Displays an error via Streamlit when no numeric features are available.
- **Example**:
```python
from sales_forecaster import engineer_features, train_ml_model, load_data

raw = load_data()
X = engineer_features(raw)
y = raw["Total"]
model, cols, scaler, metrics = train_ml_model(X, y, model_type="random_forest")
print(metrics)
```

### `train_arima_model(sales_data: pandas.DataFrame) -> statsmodels.tsa.arima.model.ARIMAResults | None`
- **Purpose**: Fit a (5,1,1) ARIMA model on daily totals, falling back to (1,0,0) if needed.
- **Requirements**: At least 14 aggregated observations; expects columns `Date` and `Total`.
- **Failure Modes**: Emits Streamlit errors/warnings and returns `None` on insufficient data or model failures.

### `arima_forecast(model, days=30, last_date=None) -> pandas.DataFrame`
- **Purpose**: Generate forward-looking forecasts from a fitted ARIMA model.
- **Parameters**:
  | Name | Type | Description |
  | --- | --- | --- |
  | `model` | `ARIMAResults` | Fitted model from `train_arima_model()`. |
  | `days` | `int` | Horizon length. |
  | `last_date` | `datetime | None` | Last observed date; auto-inferred when omitted. |
- **Returns**: DataFrame with `Date` and `Forecast` columns.

### `ml_forecast(model, features, feature_cols, scaler, branch=None, product_line=None, days=30, last_date=None) -> pandas.DataFrame`
- **Purpose**: Extrapolate ML predictions into future dates by cloning the latest feature rows.
- **Key Parameters**: Optional `branch` and `product_line` filters select template rows; `last_date` anchors the forecast window.
- **Returns**: DataFrame with `Date` & `Forecast`.
- **Side Effects**: Emits Streamlit errors if required feature columns are missing.

### `estimate_profit(sales_forecast: pandas.Series, gross_margin_percentage: float = 4.76) -> pandas.Series`
- **Purpose**: Transform forecasted revenue into profit estimates using a margin percentage.
- **Returns**: Series aligned with the input, containing dollar profit estimates.

### `get_feature_importance(model, feature_cols) -> pandas.DataFrame | None`
- **Purpose**: Surface native random forest feature importances. Generates synthetic weights when all importances are zero.
- **Returns**: DataFrame with `Feature` and `Importance` columns, sorted descending. Returns `None` for models without `feature_importances_`.

### `create_download_link(df, filename, text) -> str`
- **Purpose**: Convert a DataFrame into an HTML anchor that downloads CSV data.
- **Usage**: Embed the returned HTML with `st.markdown(..., unsafe_allow_html=True)`.

### `app() -> None`
- **Purpose**: Render the interactive forecasting UI.
- **Usage**: Mount inside any Streamlit script after ensuring data is loaded:
```python
import streamlit as st
from sales_forecaster import app as forecast_app

forecast_app()
```

---

## `inventory_reordering.py` — Inventory Management

### `load_data() -> pandas.DataFrame`
- **Purpose**: Load the supermarket CSV and synthesize inventory metrics per branch/product-line combination.
- **Behavior**: Randomizes certain metrics (e.g., days of supply) to simulate inventory, based on average sales volume.
- **Returns**: DataFrame with fields such as `CurrentStock`, `ReorderPoint`, `StockOutRisk`.
- **Example**:
```python
from inventory_reordering import load_data

inventory_df = load_data()
critical = inventory_df[inventory_df["StockOutRisk"] == "High"]
```

### `create_download_link(df, filename, text) -> str`
- Same semantics as the forecasting module.

### `highlight_risk(val: str) -> str`
- **Purpose**: Provide CSS styles for Streamlit `Styler` objects based on risk level (`High`, `Medium`, `Low`).
- **Returns**: Background-color CSS fragment or empty string.
- **Usage**:
```python
styled = df.style.applymap(highlight_risk, subset=["StockOutRisk"])
```

### `app() -> None`
- **Purpose**: Render the inventory overview, reorder recommendations, visuals, and download links.

---

## `retail_price_suggester.py` — Price Optimization

### `load_sales_data() -> pandas.DataFrame | None`
- **Purpose**: Load the core dataset and parse `Date` columns.
- **Returns**: DataFrame or `None` on failure (with Streamlit warnings).

### `prepare_price_data(sales_data: pandas.DataFrame) -> pandas.DataFrame`
- **Purpose**: Aggregate transactional data to branch/product resolution and compute descriptive statistics plus price elasticity proxies.
- **Returns**: Wide DataFrame with columns like `Unit price_mean`, `Quantity_std`, `PriceElasticity`.
- **Failure Modes**: Emits Streamlit errors when required columns are absent.

### `train_price_model(sales_data, product_line=None, branch=None) -> tuple`
- **Purpose**: Train a random forest regressor that predicts `Quantity` from price and categorical features.
- **Returns**: `(model, feature_names, feature_frame, metrics_dict)`; returns `(None, None, None, {})` when filtered data is empty.
- **Example**:
```python
from retail_price_suggester import train_price_model, load_sales_data

sales = load_sales_data()
model, feature_names, feature_frame, metrics = train_price_model(
    sales, product_line="Health and beauty", branch="A"
)
```

### `calculate_optimal_price(price_data, product_line, branch) -> dict | None`
- **Purpose**: Compute a profit-maximizing price using elasticity estimates.
- **Returns**: Dict containing current vs optimal price, quantity, profit deltas, elasticity, and cost. Returns `None` if the product-line/branch combination is missing.
- **Example**:
```python
result = calculate_optimal_price(price_data, "Food and beverages", "B")
if result:
    print(result["OptimalPrice"])
```

### `get_feature_importance(model, feature_names) -> pandas.DataFrame | None`
- **Purpose**: Expose feature importances for trained sklearn models with the attribute available.

### `generate_price_recommendations(price_data, min_profit_increase=5.0) -> pandas.DataFrame`
- **Purpose**: Batch optimal price calculations across all product-line/branch pairs and filter by profit lift.
- **Returns**: DataFrame sorted by `ProfitChange`. May be empty if no product meets the threshold.

### `create_download_link(...) -> str`
- Identical download utility.

### `app() -> None`
- **Purpose**: Serve the interactive price optimization workflow, including single-product analysis and portfolio recommendations.

---

## `promotional_items.py` — Promotion Candidate Discovery

### `load_sales_data() -> pandas.DataFrame | None`
- **Purpose**: Load the supermarket dataset with `Date` parsing; emits Streamlit warnings on failure.

### `prepare_promotion_data(sales_data: pandas.DataFrame) -> pandas.DataFrame`
- **Purpose**: Derive promotion-relevant metrics (velocity, growth rate, inventory proxies) per branch/product line.
- **Returns**: Aggregated DataFrame with additional engineered features like `InventoryDays` and `SalesGrowthRate`.
- **Behavior**: Fills missing growth rates with zero when insufficient historical coverage exists.

### `perform_clustering(promotion_data: pandas.DataFrame, n_clusters: int = 4) -> tuple`
- **Purpose**: Run K-Means clustering over selected features to categorize promotion candidates.
- **Returns**: `(cluster_centers_df, interpretations_df, clustered_data_df)` or `(None, None, empty_df)` when required features are missing.
- **Example**:
```python
from promotional_items import prepare_promotion_data, perform_clustering, load_sales_data

promo_df = prepare_promotion_data(load_sales_data())
centers, interpretations, clustered = perform_clustering(promo_df, n_clusters=4)
```

### `generate_promotion_recommendations(clustered_data: pandas.DataFrame) -> pandas.DataFrame`
- **Purpose**: Filter clusters marked High/Medium priority and compute discount, lift, and profit impact heuristics.
- **Returns**: DataFrame sorted by `ProfitImpact`; empty if no qualifying candidates.

### `visualize_clusters(clustered_data: pandas.DataFrame, features: list[str]) -> tuple[pandas.DataFrame, numpy.ndarray]`
- **Purpose**: Reduce selected features to two PCA dimensions for plotting.
- **Returns**: `(plot_df, explained_variance_ratio_array)` ready for visualization via Matplotlib/Seaborn.

### `create_download_link(...) -> str`
- Same CSV export helper.

### `app() -> None`
- **Purpose**: Render the end-to-end promotion recommendation UI with clustering controls, metrics, charts, and downloads.

---

## `chatbot.py` — Pattern-Based Sales Assistant

### `load_data() -> pandas.DataFrame | None`
- **Purpose**: Load the canonical dataset and coerce `Date` values.
- **Side Effects**: Calls `st.error` when the CSV is missing.

### `generate_response(query: str, data: pandas.DataFrame) -> str`
- **Purpose**: Answer natural-language prompts with rule-based templates that inspect the supplied DataFrame.
- **Behavior**: Handles themed intents (sales overview, product, branch, customer, payment) via regex matches. Returns a default help string otherwise.
- **Example**:
```python
from chatbot import load_data, generate_response

data = load_data()
answer = generate_response("Which branch performs best?", data)
print(answer)
```

### `app() -> None`
- **Purpose**: Render a conversational Streamlit chatbot with persistent session history and typing simulation.
- **Usage**: Embed into custom Streamlit apps to provide FAQ-style answers with no additional configuration.

---

## Shared Utilities and Patterns
- **Download Links**: Multiple modules implement `create_download_link` with identical semantics. Factor this helper into a shared utility if you plan to reuse it elsewhere.
- **Streamlit State**: UI entry points (`app()` functions and `home()`) rely on Streamlit widgets and `st.session_state`. Call them only within a `streamlit run` context.
- **Data Expectations**: Functions assume the supermarket dataset includes columns such as `Date`, `Branch`, `Product line`, `Quantity`, `Unit price`, `Sales`/`Total`, `cogs`, `gross margin percentage`, and `gross income`. Validate or adapt your data before invoking helper functions directly.

---

## Extending the Platform
1. Import the relevant module and reuse its helper functions to prepare data or train models inside your own scripts/notebooks.
2. Wrap any module `app()` within another Streamlit application to compose custom dashboards.
3. When adding new features, follow the existing pattern: create a dedicated module with helper functions and an `app()` entry point, then register it in `app.sidebar()` & `app.main()`.

For questions or contributions, see `README.md` for project-level context.

