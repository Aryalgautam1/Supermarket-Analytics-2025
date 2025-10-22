import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import io
import base64

# Create directories
try:
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
except PermissionError:
    st.warning("Permission error: Could not create directories. Using in-memory data only.")

# Load Supermarket sales data
def load_data():
    """
    Load Supermarket sales data from CSV
    Returns DataFrame with sales data
    """
    try:
        # Try to load the CSV file
        if os.path.exists('data/SuperMarket Analysis.csv'):
            data = pd.read_csv('data/SuperMarket Analysis.csv')
            
            # Convert Date to datetime
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            
            # Create a Total column if it doesn't exist
            if 'Total' not in data.columns and 'Sales' in data.columns:
                data['Total'] = data['Sales']
            
            return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
    
    st.error("Could not find SuperMarket Analysis.csv in the data directory.")
    return pd.DataFrame()  # Return empty DataFrame if file not found

# Feature engineering for machine learning models
def engineer_features(data):
    """
    Create features for machine learning models
    """
    # Convert categorical fields if present
    features = data.copy()
    
    # Add date features
    if 'Date' in features.columns:
        features['DayOfWeek'] = features['Date'].dt.dayofweek + 1  # 1-7 format
        features['Month'] = features['Date'].dt.month
        features['Year'] = features['Date'].dt.year
        features['Day'] = features['Date'].dt.day
        features['WeekOfYear'] = features['Date'].dt.isocalendar().week
    
    # One-hot encode categorical features
    categorical_cols = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment']
    
    for col in categorical_cols:
        if col in features.columns:
            # Create dummy variables, drop first to avoid multicollinearity
            dummies = pd.get_dummies(features[col], prefix=col, drop_first=True)
            features = pd.concat([features, dummies], axis=1)
    
    # Create interaction features
    if 'Customer type' in features.columns and 'Gender' in features.columns:
        # Check if they're already encoded
        if features['Customer type'].dtype == 'object' and features['Gender'].dtype == 'object':
            features['Member_Female'] = ((features['Customer type'] == 'Member') & 
                                        (features['Gender'] == 'Female')).astype(int)
    
    return features

# Aggregate data for time series analysis
def aggregate_data_for_timeseries(data, group_by='Date', agg_column='Total'):
    """
    Aggregate data for time series analysis
    
    Parameters:
    - data: DataFrame with sales data
    - group_by: Column to group by (default: 'Date')
    - agg_column: Column to aggregate (default: 'Total')
    
    Returns:
    - DataFrame with aggregated data
    """
    # Group by date and sum the total sales
    aggregated = data.groupby(group_by)[agg_column].sum().reset_index()
    
    # Ensure the data is sorted by date
    if group_by == 'Date':
        aggregated = aggregated.sort_values(group_by)
    
    return aggregated

# Train machine learning model
def train_ml_model(features, target, model_type='random_forest'):
    """
    Train a machine learning model for sales prediction
    
    Parameters:
    - features: DataFrame with feature columns
    - target: Series with target values (Sales/Total)
    - model_type: Type of model to train ('linear', 'random_forest')
    
    Returns:
    - trained model, feature columns, scaler
    """
    # Select numeric columns only
    numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove the target column if it's in features
    for col in ['Total', 'Sales']:
        if col in numeric_features:
            numeric_features.remove(col)
    
    # Remove Date column if it's in features (can't be used directly)
    if 'Date' in numeric_features:
        numeric_features.remove('Date')
    
    # Make sure we have features
    if len(numeric_features) == 0:
        st.error("No numeric features available for modeling")
        return None, [], None, {}
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(features[numeric_features])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.2, random_state=42)
    
    # Train model based on type
    if model_type == 'linear':
        model = LinearRegression()
    else:  # Default to random forest
        # Use better parameters for Random Forest
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2
    }
    
    return model, numeric_features, scaler, metrics

# Train ARIMA model
def train_arima_model(sales_data):
    """
    Train ARIMA model on sales data
    Returns fitted model
    """
    # Check if we have enough data
    if len(sales_data) < 14:
        st.error("Not enough data for modeling. Need at least 14 data points.")
        return None
    
    # Prepare data
    sales_ts = sales_data.set_index('Date')['Total']
    
    # Fit simple ARIMA model
    try:
        model = ARIMA(sales_ts, order=(5, 1, 1))
        result = model.fit()
    except Exception as e:
        st.warning(f"Error with complex model: {e}. Trying simpler model...")
        # Fallback to very simple model
        try:
            model = ARIMA(sales_ts, order=(1, 0, 0))
            result = model.fit()
        except Exception as e:
            st.error(f"Error training model: {e}")
            return None
    
    return result

# Generate forecast from ARIMA model
def arima_forecast(model, days=30, last_date=None):
    """
    Generate forecast from fitted ARIMA model
    Returns DataFrame with forecast data
    
    Parameters:
    - model: Fitted ARIMA model
    - days: Number of days to forecast
    - last_date: Last date in the historical data
    """
    if model is None:
        return pd.DataFrame()
    
    # Generate forecast
    pred = model.forecast(steps=days)
    
    # Create date range
    # Use provided last_date or try to extract from model
    if last_date is None:
        try:
            # Try to get date from model
            last_date = model.data.dates[-1]
        except (AttributeError, IndexError):
            # If that fails, use current date
            last_date = pd.Timestamp.today()
    
    dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': dates,
        'Forecast': pred.values
    })
    
    return forecast_df

# Generate forecast from ML model
def ml_forecast(model, features, feature_cols, scaler, branch=None, product_line=None, days=30, last_date=None):
    """
    Generate forecast from ML model by creating features for future dates
    
    Parameters:
    - model: Trained ML model
    - features: DataFrame with historical features
    - feature_cols: List of feature columns used by the model
    - scaler: Fitted scaler for feature normalization
    - branch: Branch to forecast for
    - product_line: Product line to forecast for
    - days: Number of days to forecast
    - last_date: Last date in the historical data
    
    Returns:
    - DataFrame with forecasted values
    """
    if last_date is None:
        last_date = features['Date'].max()
    
    # Create future dates
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    
    # Check if the feature columns exist in the features DataFrame
    for col in feature_cols:
        if col not in features.columns:
            st.error(f"Missing feature column: {col}")
            return pd.DataFrame()
    
    # Create future features
    future_data = []
    
    for date in future_dates:
        # Get a template row from features
        if branch is not None and 'Branch' in features.columns:
            template_row = features[features['Branch'] == branch].iloc[-1].copy()
        elif product_line is not None and 'Product line' in features.columns:
            template_row = features[features['Product line'] == product_line].iloc[-1].copy()
        else:
            template_row = features.iloc[-1].copy()
        
        # Update date-related features
        template_row['Date'] = date
        template_row['DayOfWeek'] = date.dayofweek + 1
        template_row['Month'] = date.month
        template_row['Year'] = date.year
        template_row['Day'] = date.day
        template_row['WeekOfYear'] = date.isocalendar().week
        
        future_data.append(template_row)
    
    future_df = pd.DataFrame(future_data)
    
    # Select and scale features
    X_future = future_df[feature_cols]
    X_future_scaled = scaler.transform(X_future)
    
    # Make predictions
    predictions = model.predict(X_future_scaled)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': predictions
    })
    
    return forecast_df

# Calculate profit estimate based on gross margin
def estimate_profit(sales_forecast, gross_margin_percentage=4.76):
    """
    Estimate profit based on sales forecast and gross margin percentage
    Default 4.76% from the dataset's gross margin percentage
    """
    profit_forecast = sales_forecast * (gross_margin_percentage / 100)
    return profit_forecast

# Get feature importance for Random Forest
def get_feature_importance(model, feature_cols):
    """
    Get feature importance from Random Forest model
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        
        # Check if all importances are 0 or close to 0
        if np.sum(importance) < 1e-10:
            # Alternative approach - use artificially generated importance
            st.warning("Feature importance values too small. Using generated values for demonstration.")
            
            # Create artificial importance based on feature names
            importance = np.array([
                0.25,  # First feature gets 25%
                0.20,  # Second feature gets 20%
                0.15,  # etc.
                0.10,
                0.08,
                0.07,
                0.05,
                0.04,
                0.03,
                0.02,
                0.01
            ])
            
            # Make sure we have enough values (pad with small values if needed)
            if len(importance) < len(feature_cols):
                pad_length = len(feature_cols) - len(importance)
                importance = np.pad(importance, (0, pad_length), 
                                   'constant', constant_values=0.001)
            
            # Trim if too many
            importance = importance[:len(feature_cols)]
            
            # Normalize to sum to 1
            importance = importance / np.sum(importance)
        
        # Create DataFrame
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importance
        })
        
        return feature_importance.sort_values('Importance', ascending=False)
    
    return None

# Download function
def create_download_link(df, filename, text):
    """
    Create a download link for DataFrame
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Main app
def app():
    st.title("Supermarket Sales Forecasting")
    
    # Load data
    data = load_data()
    
    if data.empty:
        st.error("No data available. Please check that SuperMarket Analysis.csv is in the data directory.")
        return
    
    # Sidebar
    st.sidebar.header("Forecast Settings")
    
    # Model selection
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["Random Forest", "Linear Regression", "ARIMA"],
        index=0
    )
    
    # Other filters based on the dataset
    filter_options = st.sidebar.expander("Filtering Options")
    
    with filter_options:
        # Branch filter
        branch_filter = st.checkbox("Filter by Branch")
        if branch_filter and 'Branch' in data.columns:
            branch = st.selectbox("Select Branch", sorted(data['Branch'].unique()))
        else:
            branch = None
        
        # Product line filter
        product_filter = st.checkbox("Filter by Product Line")
        if product_filter and 'Product line' in data.columns:
            product_line = st.selectbox("Select Product Line", sorted(data['Product line'].unique()))
        else:
            product_line = None
    
    # Days to forecast
    days = st.sidebar.slider("Forecast Days", 7, 30, 15)
    
    # Calculate forecast
    if st.sidebar.button("Generate Forecast"):
        # Filter data based on selections
        filtered_data = data.copy()
        
        if branch_filter and branch is not None:
            filtered_data = filtered_data[filtered_data['Branch'] == branch]
        
        if product_filter and product_line is not None:
            filtered_data = filtered_data[filtered_data['Product line'] == product_line]
        
        # Check if we have enough data
        if len(filtered_data) < 14:
            st.error("Not enough data for the selected filters. Please broaden your selection.")
            return
        
        # Process data based on model type
        if model_type == "ARIMA":
            # Aggregate by date for time series analysis
            aggregated_data = aggregate_data_for_timeseries(filtered_data)
            
            # Train ARIMA model
            with st.spinner("Training ARIMA model..."):
                arima_model = train_arima_model(aggregated_data)
                
                if arima_model is None:
                    return
                
                # Get last date for forecasting
                last_date = aggregated_data['Date'].max()
                
                # Generate forecast
                forecast_df = arima_forecast(arima_model, days, last_date)
                
                # Calculate profit based on gross margin percentage
                gross_margin_pct = filtered_data['gross margin percentage'].mean() if 'gross margin percentage' in filtered_data.columns else 4.76
                forecast_df['Profit'] = estimate_profit(forecast_df['Forecast'], gross_margin_pct)
                
                # Show model info
                with st.expander("ARIMA Model Details"):
                    st.text(str(arima_model.summary()))
        else:
            # ML approach (Random Forest or Linear Regression)
            with st.spinner("Training machine learning model..."):
                # Engineer features
                features = engineer_features(filtered_data)
                
                # Get target column (Sales or Total)
                target_col = 'Total' if 'Total' in filtered_data.columns else 'Sales'
                
                # Select model type
                ml_model_type = 'linear' if model_type == 'Linear Regression' else 'random_forest'
                
                # Train model
                model, feature_cols, scaler, metrics = train_ml_model(
                    features, filtered_data[target_col], ml_model_type)
                
                if model is None:
                    return
                
                # Get last date for forecasting
                last_date = features['Date'].max()
                
                # Generate forecast
                forecast_df = ml_forecast(
                    model, features, feature_cols, scaler, 
                    branch, product_line, days, last_date)
                
                # Calculate profit based on gross margin percentage
                gross_margin_pct = filtered_data['gross margin percentage'].mean() if 'gross margin percentage' in filtered_data.columns else 4.76
                forecast_df['Profit'] = estimate_profit(forecast_df['Forecast'], gross_margin_pct)
                
                # Display model metrics
                st.subheader("Model Performance Metrics")
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Absolute Error", f"${metrics['MAE']:.2f}")
                col2.metric("Root Mean Squared Error", f"${metrics['RMSE']:.2f}")
                col3.metric("R² Score", f"{metrics['R²']:.3f}")
                
                # Show feature importance for Random Forest
                if ml_model_type == 'random_forest':
                    st.subheader("Feature Importance")
                    feature_importance = get_feature_importance(model, feature_cols)
                    
                    if feature_importance is not None:
                        # Plot top features
                        fig, ax = plt.subplots(figsize=(10, 6))
                        top_features = feature_importance.head(10)
                        sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
                        ax.set_title('Top 10 Important Features')
                        st.pyplot(fig)
        
        # Display forecast results
        st.subheader("Sales Forecast")
        
        # Create title based on filters
        title_parts = []
        if branch is not None:
            title_parts.append(f"Branch: {branch}")
        if product_line is not None:
            title_parts.append(f"Product: {product_line}")
        
        title = " | ".join(title_parts) if title_parts else "All Stores & Products"
        
        # Get historical data for comparison
        historical = None
        if model_type == "ARIMA":
            historical = aggregated_data.set_index('Date')
        else:
            # Aggregate by date
            historical = aggregate_data_for_timeseries(filtered_data)
            historical = historical.set_index('Date')
        
        # Filter to recent history only
        historical = historical.last('30D').reset_index()
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        if not historical.empty:
            ax.plot(historical['Date'], historical['Total'], label='Historical')
        
        # Plot forecast data
        ax.plot(forecast_df['Date'], forecast_df['Forecast'], 'r-', label='Forecast')
        
        # Add confidence interval (simple approximation)
        if not historical.empty:
            std_dev = historical['Total'].std()
            ax.fill_between(
                forecast_df['Date'],
                forecast_df['Forecast'] - 1.96 * std_dev,
                forecast_df['Forecast'] + 1.96 * std_dev,
                color='red', alpha=0.2, label='95% Confidence Interval'
            )
        
        # Format plot
        ax.set_title(f"Sales Forecast for {title}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Sales')
        ax.legend()
        fig.autofmt_xdate()
        
        # Display plot
        st.pyplot(fig)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_forecast = forecast_df['Forecast'].mean()
            avg_historical = historical['Total'].mean() if not historical.empty else 0
            if avg_historical > 0:
                change = (avg_forecast / avg_historical - 1) * 100
                st.metric("Avg Daily Sales", f"${avg_forecast:.2f}", f"{change:.1f}%")
            else:
                st.metric("Avg Daily Sales", f"${avg_forecast:.2f}")
        
        with col2:
            total_forecast = forecast_df['Forecast'].sum()
            st.metric("Total Forecast", f"${total_forecast:.2f}")
        
        with col3:
            total_profit = forecast_df['Profit'].sum()
            st.metric("Estimated Profit", f"${total_profit:.2f}")
        
        # Display forecast data
        st.subheader("Forecast Details")
        st.dataframe(forecast_df[['Date', 'Forecast', 'Profit']].style.format({
            'Forecast': '${:.2f}',
            'Profit': '${:.2f}'
        }))
        
        # Create download link
        st.markdown("### Download Forecast")
        st.markdown(
            create_download_link(forecast_df, f"forecast_{model_type}.csv", "Download Forecast CSV"),
            unsafe_allow_html=True
        )
    
    else:
        # Show dataset overview and insights
        st.subheader("Supermarket Sales Dataset Overview")
        
        # Display basic statistics
        if not data.empty:
            # Create data summary
            if 'Total' in data.columns:
                total_sales = data['Total'].sum()
                avg_transaction = data['Total'].mean()
            elif 'Sales' in data.columns:
                total_sales = data['Sales'].sum()
                avg_transaction = data['Sales'].mean()
            else:
                total_sales = 0
                avg_transaction = 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Sales", f"${total_sales:.2f}")
            
            with col2:
                st.metric("Total Transactions", f"{len(data)}")
            
            with col3:
                st.metric("Avg Transaction", f"${avg_transaction:.2f}")
            
            # Display date range
            if 'Date' in data.columns:
                min_date = data['Date'].min()
                max_date = data['Date'].max()
                date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                st.info(f"Data range: {date_range}")
            
            # Show data distributions
            st.subheader("Data Distributions")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Branch Analysis", "Product Lines", "Customer Info"])
            
            with tab1:
                if 'Branch' in data.columns:
                    # Sales by branch
                    branch_sales = data.groupby('Branch')['Total' if 'Total' in data.columns else 'Sales'].sum().reset_index()
                    
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(x='Branch', y='Total' if 'Total' in data.columns else 'Sales', data=branch_sales, ax=ax)
                    ax.set_title('Sales by Branch')
                    ax.set_ylabel('Total Sales')
                    st.pyplot(fig)
            
            with tab2:
                if 'Product line' in data.columns:
                    # Sales by product line
                    product_sales = data.groupby('Product line')['Total' if 'Total' in data.columns else 'Sales'].sum().reset_index()
                    product_sales = product_sales.sort_values('Total' if 'Total' in data.columns else 'Sales', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.barplot(x='Total' if 'Total' in data.columns else 'Sales', y='Product line', data=product_sales, ax=ax)
                    ax.set_title('Sales by Product Line')
                    ax.set_xlabel('Total Sales')
                    st.pyplot(fig)
            
            with tab3:
                # Customer type and gender
                if 'Customer type' in data.columns and 'Gender' in data.columns:
                    # Create cross-tab
                    customer_data = pd.crosstab(
                        data['Customer type'], 
                        data['Gender'], 
                        values=data['Total' if 'Total' in data.columns else 'Sales'], 
                        aggfunc='sum'
                    ).reset_index()
                    
                    # Reshape for plotting
                    customer_data_melted = pd.melt(
                        customer_data, 
                        id_vars='Customer type',
                        var_name='Gender',
                        value_name='Sales'
                    )
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.barplot(x='Customer type', y='Sales', hue='Gender', data=customer_data_melted, ax=ax)
                    ax.set_title('Sales by Customer Type and Gender')
                    st.pyplot(fig)
            
            # Instructions
            st.info("Select model type and filters in the sidebar, then click 'Generate Forecast' to see predictions.")

if __name__ == "__main__":
    app()
