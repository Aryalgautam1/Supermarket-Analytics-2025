import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import base64

# Create directories
try:
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
except PermissionError:
    st.warning("Permission error: Could not create directories. Using in-memory data only.")

# Load Supermarket sales data
def load_sales_data():
    """
    Load Supermarket sales data
    """
    try:
        if os.path.exists('data/SuperMarket Analysis.csv'):
            data = pd.read_csv('data/SuperMarket Analysis.csv')
            
            # Convert Date to datetime
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            
            return data
    except Exception as e:
        st.warning(f"Error loading sales data: {e}")
        return None
    
    st.warning("SuperMarket Analysis.csv not found in data directory.")
    return None

# Prepare data for price analysis
def prepare_price_data(sales_data):
    """
    Prepare sales data for price analysis
    """
    if sales_data is None or sales_data.empty:
        return pd.DataFrame()
    
    # Check if required columns exist
    required_cols = ['Product line', 'Unit price', 'Quantity', 'Date', 'Branch']
    for col in required_cols:
        if col not in sales_data.columns:
            st.error(f"Required column '{col}' not found in data")
            return pd.DataFrame()
    
    # Group data by product line and branch
    price_data = sales_data.groupby(['Branch', 'Product line']).agg({
        'Quantity': ['sum', 'mean', 'std'],
        'Unit price': ['mean', 'min', 'max', 'std'],
        'cogs': 'mean',
        'gross margin percentage': 'mean',
        'gross income': 'sum',
        'Rating': 'mean'
    }).reset_index()
    
    # Flatten multi-level columns
    price_data.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                       for col in price_data.columns]
    
    # Calculate price elasticity (if enough data)
    # Price elasticity = % change in quantity / % change in price
    # This is a simplified approach using standard deviations
    
    # Add a small value to avoid division by zero
    epsilon = 1e-6
    
    # Calculate approximate elasticity using variation
    price_data['PriceElasticity'] = -(price_data['Quantity_std'] / (price_data['Quantity_mean'] + epsilon)) / \
                                  (price_data['Unit price_std'] / (price_data['Unit price_mean'] + epsilon))
    
    # If elasticity is invalid or infinite, use a default value
    price_data['PriceElasticity'] = price_data['PriceElasticity'].replace([np.inf, -np.inf, np.nan], -1.0)
    
    # Clip elasticity to reasonable range (-3 to 0)
    price_data['PriceElasticity'] = np.clip(price_data['PriceElasticity'], -3.0, 0.0)
    
    return price_data

# Train price sensitivity model
def train_price_model(sales_data, product_line=None, branch=None):
    """
    Train a model to predict quantity based on price
    """
    # Filter data if product line or branch is specified
    filtered_data = sales_data.copy()
    
    if product_line is not None:
        filtered_data = filtered_data[filtered_data['Product line'] == product_line]
    
    if branch is not None:
        filtered_data = filtered_data[filtered_data['Branch'] == branch]
    
    if filtered_data.empty:
        return None, None, None, {}
    
    # Create features
    # Convert categorical variables to dummy variables
    features = pd.get_dummies(filtered_data[['Unit price', 'Product line', 'Branch', 
                                          'Customer type', 'Gender', 'Payment']], 
                            drop_first=True)
    
    # Add date features if available
    if 'Date' in filtered_data.columns:
        features['DayOfWeek'] = filtered_data['Date'].dt.dayofweek
        features['Month'] = filtered_data['Date'].dt.month
        features['Weekend'] = (filtered_data['Date'].dt.dayofweek >= 5).astype(int)
    
    # Target is quantity
    target = filtered_data['Quantity']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R²': r2_score(y_test, y_pred)
    }
    
    return model, features.columns.tolist(), features, metrics

# Calculate optimal price
def calculate_optimal_price(price_data, product_line, branch):
    """
    Calculate optimal price based on price elasticity
    """
    # Filter for the specific product line and branch
    product_info = price_data[(price_data['Branch'] == branch) & 
                            (price_data['Product line'] == product_line)]
    
    if product_info.empty:
        return None
    
    # Get the first row (should be only one)
    product_info = product_info.iloc[0]
    
    # Get current price and cost
    current_price = product_info['Unit price_mean']
    cost = product_info['cogs_mean']
    
    # Get price elasticity
    elasticity = product_info['PriceElasticity']
    
    # Calculate optimal price
    # For profit maximization, optimal markup = -1 / (1 + elasticity)
    if elasticity > -1:  # Inelastic demand
        # If demand is inelastic, a higher price is typically better
        # But we'll cap it at a reasonable level
        optimal_markup = 0.5  # 50% markup
    else:
        optimal_markup = -1 / (1 + elasticity)
    
    # Calculate optimal price
    optimal_price = cost * (1 + optimal_markup)
    
    # Ensure the price is within reasonable bounds
    min_price = cost * 1.05  # At least 5% margin
    max_price = cost * 2.0   # At most 100% markup
    
    optimal_price = max(min_price, min(max_price, optimal_price))
    
    # Calculate expected impact
    current_quantity = product_info['Quantity_mean']
    
    # Predict new quantity with price elasticity
    price_ratio = optimal_price / current_price
    new_quantity = current_quantity * (price_ratio ** elasticity)
    
    # Calculate financials
    current_revenue = current_price * current_quantity
    current_profit = current_revenue - (cost * current_quantity)
    
    new_revenue = optimal_price * new_quantity
    new_profit = new_revenue - (cost * new_quantity)
    
    # Calculate percentage changes
    price_change_pct = (optimal_price / current_price - 1) * 100
    quantity_change_pct = (new_quantity / current_quantity - 1) * 100
    profit_change_pct = (new_profit / current_profit - 1) * 100 if current_profit > 0 else 0
    
    # Create results dictionary
    results = {
        'CurrentPrice': current_price,
        'OptimalPrice': optimal_price,
        'PriceChange': price_change_pct,
        'CurrentQuantity': current_quantity,
        'NewQuantity': new_quantity,
        'QuantityChange': quantity_change_pct,
        'CurrentProfit': current_profit,
        'NewProfit': new_profit,
        'ProfitChange': profit_change_pct,
        'Elasticity': elasticity,
        'Cost': cost
    }
    
    return results

# Get feature importance for a trained model
def get_feature_importance(model, feature_names):
    """
    Get feature importance from a trained model
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    return None

# Calculate price recommendations for all products
def generate_price_recommendations(price_data, min_profit_increase=5.0):
    """
    Generate price recommendations for all product lines
    """
    recommendations = []
    
    # Iterate through each branch and product line
    for branch in price_data['Branch'].unique():
        for product_line in price_data['Product line'].unique():
            # Calculate optimal price for this product line in this branch
            result = calculate_optimal_price(price_data, product_line, branch)
            
            if result is not None:
                # Add branch and product line
                result['Branch'] = branch
                result['ProductLine'] = product_line
                
                # Add to recommendations
                recommendations.append(result)
    
    # Convert to DataFrame
    recommendations_df = pd.DataFrame(recommendations)
    
    # Filter by minimum profit increase
    if min_profit_increase > 0:
        recommendations_df = recommendations_df[recommendations_df['ProfitChange'] >= min_profit_increase]
    
    # Sort by profit increase potential
    if not recommendations_df.empty:
        recommendations_df = recommendations_df.sort_values('ProfitChange', ascending=False)
    
    return recommendations_df

# Create download link
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
    st.title("Retail Price Optimizer")
    
    # Load sales data
    with st.spinner("Loading sales data..."):
        sales_data = load_sales_data()
    
    if sales_data is None or sales_data.empty:
        st.error("No sales data available. Please upload data file.")
        return
    
    # Prepare data for price analysis
    with st.spinner("Analyzing price sensitivity..."):
        price_data = prepare_price_data(sales_data)
    
    if price_data.empty:
        st.error("Could not prepare price data. Check required columns.")
        return
    
    # Sidebar
    st.sidebar.header("Optimization Settings")
    
    # Branch and product line filters
    branches = sorted(price_data['Branch'].unique())
    selected_branch = st.sidebar.selectbox("Select Branch", branches)
    
    product_lines = sorted(price_data['Product line'].unique())
    selected_product_line = st.sidebar.selectbox("Select Product Line", product_lines)
    
    # Filter for minimum profit improvement
    min_profit = st.sidebar.slider(
        "Minimum Profit Improvement (%)", 
        min_value=0.0, 
        max_value=50.0, 
        value=5.0,
        step=1.0
    )
    
    # Main content - Show two tabs: Single Product Analysis and All Products Recommendations
    tab1, tab2 = st.tabs(["Price Sensitivity Analysis", "Price Recommendations"])
    
    with tab1:
        st.subheader(f"Price Sensitivity Analysis for {selected_product_line} in {selected_branch}")
        
        # Get data for this product line
        product_data = price_data[(price_data['Branch'] == selected_branch) & 
                                (price_data['Product line'] == selected_product_line)]
        
        if not product_data.empty:
            # Show current metrics
            col1, col2, col3 = st.columns(3)
            
            product_info = product_data.iloc[0]
            
            with col1:
                st.metric(
                    "Average Price", 
                    f"${product_info['Unit price_mean']:.2f}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Price Range", 
                    f"${product_info['Unit price_min']:.2f} - ${product_info['Unit price_max']:.2f}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Price Elasticity", 
                    f"{product_info['PriceElasticity']:.2f}",
                    delta=None,
                    help="Price elasticity measures how quantity demanded changes with price. Values close to 0 indicate inelastic demand (price has little effect). Values less than -1 indicate elastic demand (price has strong effect)."
                )
            
            # Calculate optimal price
            if st.button("Calculate Optimal Price"):
                # Get optimal price
                result = calculate_optimal_price(price_data, selected_product_line, selected_branch)
                
                if result is not None:
                    st.subheader("Price Optimization Results")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Current Price", 
                            f"${result['CurrentPrice']:.2f}"
                        )
                        st.metric(
                            "Optimal Price", 
                            f"${result['OptimalPrice']:.2f}",
                            f"{result['PriceChange']:.1f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Current Quantity", 
                            f"{result['CurrentQuantity']:.1f}"
                        )
                        st.metric(
                            "Expected Quantity", 
                            f"{result['NewQuantity']:.1f}",
                            f"{result['QuantityChange']:.1f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "Current Profit", 
                            f"${result['CurrentProfit']:.2f}"
                        )
                        st.metric(
                            "Expected Profit", 
                            f"${result['NewProfit']:.2f}",
                            f"{result['ProfitChange']:.1f}%"
                        )
                    
                    # Create price-demand curve
                    st.subheader("Price-Demand Curve")
                    
                    # Generate price points
                    min_price = result['Cost'] * 1.05
                    max_price = result['Cost'] * 2.0
                    price_points = np.linspace(min_price, max_price, 20)
                    
                    # Calculate demand at each price point
                    elasticity = result['Elasticity']
                    base_quantity = result['CurrentQuantity']
                    base_price = result['CurrentPrice']
                    
                    quantities = []
                    profits = []
                    
                    for price in price_points:
                        # Calculate quantity using elasticity
                        price_ratio = price / base_price
                        quantity = base_quantity * (price_ratio ** elasticity)
                        
                        # Calculate profit
                        profit = (price - result['Cost']) * quantity
                        
                        quantities.append(quantity)
                        profits.append(profit)
                    
                    # Create a DataFrame for plotting
                    curve_data = pd.DataFrame({
                        'Price': price_points,
                        'Quantity': quantities,
                        'Profit': profits
                    })
                    
                    # Create plot with two y-axes
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    
                    # Plot quantity on left y-axis
                    ax1.set_xlabel('Price ($)')
                    ax1.set_ylabel('Quantity', color='blue')
                    ax1.plot(curve_data['Price'], curve_data['Quantity'], color='blue')
                    ax1.tick_params(axis='y', labelcolor='blue')
                    
                    # Create second y-axis for profit
                    ax2 = ax1.twinx()
                    ax2.set_ylabel('Profit ($)', color='green')
                    ax2.plot(curve_data['Price'], curve_data['Profit'], color='green')
                    ax2.tick_params(axis='y', labelcolor='green')
                    
                    # Mark current and optimal prices
                    ax1.axvline(x=result['CurrentPrice'], color='blue', linestyle='--', alpha=0.7, label='Current Price')
                    ax1.axvline(x=result['OptimalPrice'], color='green', linestyle='--', alpha=0.7, label='Optimal Price')
                    
                    # Add text labels
                    plt.text(result['CurrentPrice'], ax1.get_ylim()[1]*0.9, 'Current', rotation=90, color='blue')
                    plt.text(result['OptimalPrice'], ax1.get_ylim()[1]*0.9, 'Optimal', rotation=90, color='green')
                    
                    plt.title(f'Price-Demand Curve for {selected_product_line}')
                    plt.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # Display additional information
                    st.info(f"""
                    **Analysis Summary:**
                    - Current price (${result['CurrentPrice']:.2f}) {'is below' if result['PriceChange'] > 0 else 'is above'} the optimal price (${result['OptimalPrice']:.2f}).
                    - A price change of {result['PriceChange']:.1f}% would result in a quantity change of {result['QuantityChange']:.1f}%.
                    - This would increase profit by {result['ProfitChange']:.1f}%.
                    - The price elasticity of demand is {result['Elasticity']:.2f}, which means demand is {'elastic' if result['Elasticity'] < -1 else 'inelastic'}.
                    """)
                else:
                    st.error("Could not calculate optimal price. Insufficient data.")
        else:
            st.error(f"No data available for {selected_product_line} in {selected_branch}.")
    
    with tab2:
        st.subheader("Price Recommendations for All Products")
        
        # Calculate price recommendations for all products
        if st.button("Generate All Recommendations"):
            with st.spinner("Calculating price recommendations..."):
                recommendations = generate_price_recommendations(price_data, min_profit)
            
            if not recommendations.empty:
                # Display recommendations
                st.success(f"Found {len(recommendations)} products with profit improvement potential of at least {min_profit:.1f}%")
                
                # Show recommendations in a table
                display_cols = ['Branch', 'ProductLine', 'CurrentPrice', 'OptimalPrice', 
                              'PriceChange', 'QuantityChange', 'ProfitChange']
                
                st.dataframe(recommendations[display_cols].style.format({
                    'CurrentPrice': '${:.2f}',
                    'OptimalPrice': '${:.2f}',
                    'PriceChange': '{:.1f}%',
                    'QuantityChange': '{:.1f}%',
                    'ProfitChange': '{:.1f}%'
                }))
                
                # Create download link
                st.markdown("### Download Recommendations")
                
                st.markdown(
                    create_download_link(
                        recommendations, 
                        f"price_recommendations_{datetime.now().strftime('%Y%m%d')}.csv", 
                        "Download Price Recommendations (CSV)"
                    ),
                    unsafe_allow_html=True
                )
                
                # Visualize top recommendations
                st.subheader("Top Price Optimization Opportunities")
                
                # Display top 10 products by profit improvement
                top_recommendations = recommendations.head(10)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Create horizontal bar chart
                bars = ax.barh(
                    top_recommendations['ProductLine'] + ' (' + top_recommendations['Branch'] + ')',
                    top_recommendations['ProfitChange'],
                    color='green'
                )
                
                # Add data labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    label_position = width if width > 0 else 0
                    ax.text(
                        label_position + 1, 
                        bar.get_y() + bar.get_height()/2, 
                        f"{width:.1f}%", 
                        va='center'
                    )
                
                # Set labels and title
                ax.set_xlabel('Profit Improvement (%)')
                ax.set_title('Top 10 Products by Profit Improvement Potential')
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Display price change distribution
                st.subheader("Price Change Distribution")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create histogram of price changes
                sns.histplot(recommendations['PriceChange'], bins=20, kde=True, ax=ax)
                
                # Add vertical line at 0
                ax.axvline(x=0, color='red', linestyle='--')
                
                # Set labels and title
                ax.set_xlabel('Price Change (%)')
                ax.set_ylabel('Number of Products')
                ax.set_title('Distribution of Recommended Price Changes')
                
                st.pyplot(fig)
            else:
                st.info(f"No products found with profit improvement potential of at least {min_profit:.1f}%.")
        else:
            st.info("Click 'Generate All Recommendations' to calculate optimal prices for all products.")

if __name__ == "__main__":
    app()
