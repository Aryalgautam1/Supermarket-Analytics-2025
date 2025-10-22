# Supermarket Analytics

A comprehensive analytics platform for supermarket sales data with predictive modeling, inventory management, and pricing optimization.

## Project Structure

```
Supermarket-Analytics/
├── app.py                      # Main application entry point
├── sales_forecaster.py         # Sales prediction module
├── inventory_reordering.py     # Inventory management module
├── retail_price_suggester.py   # Price optimization module
├── promotional_items.py        # Promotion recommendation module
├── chatbot.py                  # Conversational interface
├── requirements.txt            # Python dependencies
├── data/                       # Data storage directory
│   └── SuperMarket Analysis.csv # Core dataset
├── models/                     # Trained model storage
└── downloads/                  # Generated reports directory
```

## Core Features

* **Sales Forecasting**
  Uses time series models (ARIMA, ETS) and machine learning (Random Forest, Linear Regression) to predict future sales.

* **Inventory Reordering**
  Calculates optimal order quantities based on historical sales and stock gaps to prevent stockouts.

* **Retail Price Suggestion**
  Recommends price adjustments using price elasticity models to improve profit margins.

* **Promotional Item Selector**
  Identifies items that should be promoted based on performance and forecasted lift using clustering algorithms.

* **Chatbot Assistant**
  A Streamlit-integrated assistant to handle inventory and performance queries.

## Dataset

Located at: `data/SuperMarket Analysis.csv`

Includes:
* Product categories
* Sales transactions
* Branch-specific data
* Customer information
* Payment methods
* Pricing and cost data

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download the repository

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Usage

The application provides an intuitive sidebar navigation with the following modules:

1. **Home** - Dashboard with overview metrics
2. **Sales Forecasting** - Generate sales predictions using various models
3. **Inventory** - View inventory status and reorder recommendations
4. **Pricing** - Optimize pricing for maximum profit
5. **Promotions** - Identify products for promotional campaigns
6. **Chatbot** - Ask questions about your sales data

## Technologies Used

- **Streamlit** - Web application framework
- **Pandas & NumPy** - Data manipulation
- **Scikit-learn** - Machine learning models
- **Statsmodels** - Time series analysis (ARIMA)
- **Matplotlib & Seaborn** - Data visualization

## Notes

- The application creates necessary directories (`data/`, `models/`, `downloads/`) automatically on first run
- Generated forecasts and reports can be downloaded as CSV files
- All visualizations are interactive and update based on selected filters
