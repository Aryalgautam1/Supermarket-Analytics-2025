import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# Prepare data for promotion analysis
def prepare_promotion_data(sales_data):
    """
    Prepare sales data for promotion analysis
    """
    if sales_data is None or sales_data.empty:
        return pd.DataFrame()
    
    # Check if required columns exist
    required_cols = ['Product line', 'Unit price', 'Quantity', 'Date', 'Branch']
    for col in required_cols:
        if col not in sales_data.columns:
            st.error(f"Required column '{col}' not found in data")
            return pd.DataFrame()
    
    # Get most recent date
    most_recent_date = sales_data['Date'].max()
    
    # Calculate date ranges
    past_30_days = most_recent_date - timedelta(days=30)
    past_60_days = most_recent_date - timedelta(days=60)
    past_90_days = most_recent_date - timedelta(days=90)
    
    # Calculate sales trends for different time periods
    recent_30_days = sales_data[sales_data['Date'] >= past_30_days]
    previous_30_days = sales_data[(sales_data['Date'] >= past_60_days) & (sales_data['Date'] < past_30_days)]
    previous_60_days = sales_data[(sales_data['Date'] >= past_90_days) & (sales_data['Date'] < past_30_days)]
    
    # Group data by product line and branch
    promotion_data = sales_data.groupby(['Branch', 'Product line']).agg({
        'Quantity': ['sum', 'mean', 'std'],
        'Unit price': ['mean', 'min', 'max'],
        'cogs': 'mean',
        'gross margin percentage': 'mean',
        'gross income': 'sum',
        'Rating': 'mean'
    }).reset_index()
    
    # Flatten multi-level columns
    promotion_data.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                           for col in promotion_data.columns]
    
    # Calculate sales trends if we have enough data
    if not recent_30_days.empty and not previous_30_days.empty:
        recent_sales = recent_30_days.groupby(['Branch', 'Product line'])['Quantity'].sum().reset_index()
        previous_sales = previous_30_days.groupby(['Branch', 'Product line'])['Quantity'].sum().reset_index()
        
        # Merge the data
        sales_trend = pd.merge(
            recent_sales, 
            previous_sales, 
            on=['Branch', 'Product line'], 
            suffixes=('_recent', '_previous')
        )
        
        # Calculate growth rate
        sales_trend['SalesGrowthRate'] = (sales_trend['Quantity_recent'] / 
                                        sales_trend['Quantity_previous'] - 1) * 100
        
        # Merge with promotion data
        promotion_data = pd.merge(
            promotion_data,
            sales_trend[['Branch', 'Product line', 'SalesGrowthRate']],
            on=['Branch', 'Product line'],
            how='left'
        )
    else:
        # If we don't have enough data, add a dummy column
        promotion_data['SalesGrowthRate'] = 0
    
    # Fill missing values with 0
    promotion_data['SalesGrowthRate'] = promotion_data['SalesGrowthRate'].fillna(0)
    
    # Calculate sales velocity (average daily sales)
    days_in_data = (sales_data['Date'].max() - sales_data['Date'].min()).days + 1
    if days_in_data > 0:
        promotion_data['SalesVelocity'] = promotion_data['Quantity_sum'] / days_in_data
    else:
        promotion_data['SalesVelocity'] = promotion_data['Quantity_mean']
    
    # Calculate coefficient of variation (sales variability)
    promotion_data['SalesVariability'] = promotion_data['Quantity_std'] / promotion_data['Quantity_mean']
    promotion_data['SalesVariability'] = promotion_data['SalesVariability'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Calculate more features for clustering
    promotion_data['PriceRange'] = promotion_data['Unit price_max'] - promotion_data['Unit price_min']
    promotion_data['PriceVariability'] = promotion_data['PriceRange'] / promotion_data['Unit price_mean']
    promotion_data['PriceVariability'] = promotion_data['PriceVariability'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Generate mock inventory data (since we don't have actual inventory)
    # Assuming inventory level based on sales velocity
    # Fast-moving items have less inventory, slow-moving items have more
    
    # Calculate average days of inventory
    avg_inventory_days = 30  # Assume average 30 days of inventory
    
    # Adjust based on sales velocity compared to mean
    mean_velocity = promotion_data['SalesVelocity'].mean()
    promotion_data['InventoryDays'] = avg_inventory_days * (mean_velocity / 
                                                         promotion_data['SalesVelocity'])
    
    # Cap at reasonable values (10-90 days)
    promotion_data['InventoryDays'] = promotion_data['InventoryDays'].clip(10, 90)
    
    # Calculate estimated inventory
    promotion_data['EstimatedInventory'] = promotion_data['SalesVelocity'] * promotion_data['InventoryDays']
    
    # Calculate inventory value
    promotion_data['InventoryValue'] = promotion_data['EstimatedInventory'] * promotion_data['cogs_mean']
    
    # Calculate inventory turnover rate (annualized)
    promotion_data['InventoryTurnoverRate'] = 365 / promotion_data['InventoryDays']
    
    return promotion_data

# Perform K-Means clustering
def perform_clustering(promotion_data, n_clusters=4):
    """
    Cluster products based on promotion-relevant features
    """
    # Select features for clustering
    features = [
        'SalesVelocity',
        'SalesVariability',
        'SalesGrowthRate',
        'gross margin percentage_mean',
        'InventoryDays',
        'InventoryTurnoverRate',
        'Rating_mean'
    ]
    
    # Check if we have all features
    for feature in features:
        if feature not in promotion_data.columns:
            st.error(f"Missing feature for clustering: {feature}")
            return None, None, pd.DataFrame()
    
    # Extract features
    X = promotion_data[features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add cluster assignment to data
    clustered_data = promotion_data.copy()
    clustered_data['Cluster'] = clusters
    
    # Calculate cluster centers
    cluster_centers = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=features
    )
    
    # Interpret clusters
    cluster_interpretations = []
    
    for i in range(n_clusters):
        center = cluster_centers.iloc[i]
        
        # Interpret based on center values
        # High inventory days, low sales velocity, low growth rate = Good promotion candidate
        # Low inventory days, high sales velocity, high growth rate = Not a promotion candidate
        
        # Calculate promotion score (higher is better candidate)
        promo_score = (
            center['InventoryDays'] / 30 +  # Higher inventory days → better candidate
            (1 - center['SalesVelocity'] / promotion_data['SalesVelocity'].mean()) +  # Lower velocity → better candidate
            (1 - center['SalesGrowthRate'] / 100) +  # Lower growth → better candidate
            center['gross margin percentage_mean'] / 10  # Higher margin → can afford discount
        ) / 4  # Average the factors (scale 0-1)
        
        # Determine if this cluster is a good candidate for promotion
        if promo_score > 0.7:
            interpretation = "Strong Promotion Candidate"
            recommendation = "15-25% discount promotion"
            priority = "High"
        elif promo_score > 0.5:
            interpretation = "Moderate Promotion Candidate"
            recommendation = "10-15% discount promotion"
            priority = "Medium"
        elif promo_score > 0.3:
            interpretation = "Weak Promotion Candidate"
            recommendation = "5-10% discount or bundle"
            priority = "Low"
        else:
            interpretation = "Not a Promotion Candidate"
            recommendation = "No promotion needed"
            priority = "None"
        
        cluster_interpretations.append({
            'Cluster': i,
            'Interpretation': interpretation,
            'Recommendation': recommendation,
            'Priority': priority,
            'PromoScore': promo_score
        })
    
    # Convert to DataFrame
    interpretation_df = pd.DataFrame(cluster_interpretations)
    
    # Merge interpretations with clustered data
    clustered_data = clustered_data.merge(
        interpretation_df[['Cluster', 'Interpretation', 'Recommendation', 'Priority']],
        on='Cluster'
    )
    
    return cluster_centers, interpretation_df, clustered_data

# Generate specific promotion recommendations
def generate_promotion_recommendations(clustered_data):
    """
    Generate specific promotion recommendations based on clustering results
    """
    # Focus on promotion candidates
    candidates = clustered_data[clustered_data['Priority'].isin(['High', 'Medium'])].copy()
    
    if candidates.empty:
        return pd.DataFrame()
    
    # Calculate specific discount percentages
    def calculate_discount(row):
        base_discount = 0.0
        
        # Higher discount for higher inventory days
        if row['InventoryDays'] > 60:
            base_discount += 0.10
        elif row['InventoryDays'] > 30:
            base_discount += 0.05
        
        # Higher discount for negative growth rate
        if row['SalesGrowthRate'] < -20:
            base_discount += 0.10
        elif row['SalesGrowthRate'] < 0:
            base_discount += 0.05
        
        # Higher discount for higher margin
        if row['gross margin percentage_mean'] > 15:
            base_discount += 0.05
        
        # Adjust based on priority
        if row['Priority'] == 'High':
            base_discount += 0.05
        
        # Ensure minimum discount
        base_discount = max(base_discount, 0.05)
        
        # Ensure maximum discount
        base_discount = min(base_discount, 0.25)
        
        return base_discount
    
    # Calculate discount percentages
    candidates['DiscountPercentage'] = candidates.apply(calculate_discount, axis=1)
    
    # Calculate new price after discount
    candidates['DiscountedPrice'] = candidates['Unit price_mean'] * (1 - candidates['DiscountPercentage'])
    
    # Calculate expected sales lift (simple model based on discount percentage)
    # Assume 2% lift for each 1% discount (elasticity of -2)
    candidates['ExpectedSalesLift'] = candidates['DiscountPercentage'] * 200  # percentage points
    
    # Calculate financial impact
    candidates['CurrentRevenue'] = candidates['SalesVelocity'] * candidates['Unit price_mean'] * 30  # monthly
    candidates['CurrentProfit'] = candidates['CurrentRevenue'] * (candidates['gross margin percentage_mean'] / 100)
    
    # Calculate new sales velocity with lift
    candidates['NewSalesVelocity'] = candidates['SalesVelocity'] * (1 + candidates['ExpectedSalesLift'] / 100)
    
    # Calculate new revenue and profit
    candidates['NewRevenue'] = candidates['NewSalesVelocity'] * candidates['DiscountedPrice'] * 30  # monthly
    candidates['NewProfit'] = candidates['NewRevenue'] * ((candidates['gross margin percentage_mean'] - 
                                                       candidates['DiscountPercentage'] * 100) / 100)
    
    # Calculate profit impact
    candidates['ProfitImpact'] = candidates['NewProfit'] - candidates['CurrentProfit']
    candidates['ProfitImpactPercentage'] = (candidates['ProfitImpact'] / candidates['CurrentProfit']) * 100
    
    # Sort by impact
    candidates = candidates.sort_values('ProfitImpact', ascending=False)
    
    return candidates

# Visualize clusters with PCA
def visualize_clusters(clustered_data, features):
    """
    Visualize clusters using PCA for dimensionality reduction
    """
    # Select features
    X = clustered_data[features].copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': clustered_data['Cluster'],
        'Interpretation': clustered_data['Interpretation'],
        'Branch': clustered_data['Branch'],
        'Product line': clustered_data['Product line'],
        'Priority': clustered_data['Priority']
    })
    
    return plot_df, pca.explained_variance_ratio_

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
    st.title("Promotional Item Identification")
    
    # Load sales data
    with st.spinner("Loading sales data..."):
        sales_data = load_sales_data()
    
    if sales_data is None or sales_data.empty:
        st.error("No sales data available. Please upload data file.")
        return
    
    # Sidebar
    st.sidebar.header("Clustering Settings")
    
    # Number of clusters
    n_clusters = st.sidebar.slider("Number of Clusters", min_value=3, max_value=6, value=4)
    
    # Branch filter
    branches = sorted(sales_data['Branch'].unique())
    selected_branch = st.sidebar.selectbox("Select Branch", branches)
    
    # Filter by promotion priority
    priorities = ["All", "High", "Medium", "Low", "None"]
    selected_priority = st.sidebar.selectbox("Promotion Priority", priorities, index=0)
    
    # Prepare data for analysis
    with st.spinner("Analyzing sales patterns..."):
        branch_data = sales_data[sales_data['Branch'] == selected_branch]
        promotion_data = prepare_promotion_data(branch_data)
    
    if promotion_data.empty:
        st.error("Could not prepare promotion data. Check data format.")
        return
    
    # Perform clustering
    with st.spinner("Identifying promotion candidates..."):
        cluster_centers, interpretations, clustered_data = perform_clustering(promotion_data, n_clusters)
    
    if clustered_data.empty:
        st.error("Clustering failed. Check data and features.")
        return
    
    # Filter by priority if not "All"
    if selected_priority != "All":
        clustered_data = clustered_data[clustered_data['Priority'] == selected_priority]
        
        if clustered_data.empty:
            st.warning(f"No products found with priority: {selected_priority}")
            return
    
    # Generate specific recommendations
    recommendations = generate_promotion_recommendations(clustered_data)
    
    # Display inventory and sales overview
    st.subheader(f"Inventory and Sales Overview for {selected_branch}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_inventory = clustered_data['InventoryValue'].sum()
        st.metric("Total Inventory Value", f"${total_inventory:.2f}")
    
    with col2:
        avg_days = clustered_data['InventoryDays'].mean()
        st.metric("Avg Days of Inventory", f"{avg_days:.1f}")
    
    with col3:
        total_sales = clustered_data['SalesVelocity'].sum() * 30
        st.metric("Monthly Sales", f"${total_sales * clustered_data['Unit price_mean'].mean():.2f}")
    
    with col4:
        promotion_candidates = len(clustered_data[clustered_data['Priority'].isin(['High', 'Medium'])])
        st.metric("Promotion Candidates", f"{promotion_candidates}")
    
    # Display cluster analysis
    st.subheader("Product Clustering Analysis")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Cluster Visualization", "Cluster Details"])
    
    with tab1:
        # Select features for visualization
        visualization_features = [
            'SalesVelocity',
            'SalesVariability',
            'SalesGrowthRate',
            'gross margin percentage_mean',
            'InventoryDays',
            'InventoryTurnoverRate'
        ]
        
        # Visualize clusters
        pca_data, variance_ratio = visualize_clusters(clustered_data, visualization_features)
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use different colors for different priorities
        priority_colors = {
            'High': 'red',
            'Medium': 'orange',
            'Low': 'blue',
            'None': 'green'
        }
        
        # Create a scatter plot for each priority
        for priority in pca_data['Priority'].unique():
            priority_data = pca_data[pca_data['Priority'] == priority]
            ax.scatter(
                priority_data['PCA1'],
                priority_data['PCA2'],
                label=f"{priority} Priority",
                alpha=0.7,
                color=priority_colors.get(priority, 'gray')
            )
        
        # Annotate some points with product names
        for i, row in pca_data.head(10).iterrows():
            ax.annotate(
                row['Product line'],
                xy=(row['PCA1'], row['PCA2']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        # Set labels and title
        ax.set_xlabel(f"Principal Component 1 ({variance_ratio[0]:.1%} variance)")
        ax.set_ylabel(f"Principal Component 2 ({variance_ratio[1]:.1%} variance)")
        ax.set_title("Product Clusters based on Sales and Inventory Patterns")
        ax.legend()
        
        # Display the plot
        st.pyplot(fig)
        
        # Interpretation
        st.markdown("""
        **Interpretation Guide:**
        - **Red Clusters (High Priority)**: Products with high inventory, slow sales, and good margins - ideal for promotions
        - **Orange Clusters (Medium Priority)**: Products that could benefit from moderate promotions
        - **Blue Clusters (Low Priority)**: Products with some promotion potential but limited impact
        - **Green Clusters (None)**: Products that don't need promotions - typically fast-moving items with good sales
        """)
    
    with tab2:
        # Display cluster interpretations
        st.subheader("Cluster Interpretations")
        
        # Create a table of cluster interpretations
        for i, row in interpretations.iterrows():
            st.markdown(f"**Cluster {row['Cluster']}**: {row['Interpretation']} - {row['Recommendation']} (Priority: {row['Priority']})")
        
        # Show cluster centers
        with st.expander("Cluster Centers"):
            st.dataframe(cluster_centers.style.format("{:.2f}"))
        
        # Display count of products in each cluster
        cluster_counts = clustered_data['Interpretation'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster Interpretation', 'Count']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x='Count', y='Cluster Interpretation', data=cluster_counts, ax=ax)
        ax.set_title("Number of Products in Each Cluster")
        
        st.pyplot(fig)
    
    # Display promotion recommendations
    st.subheader("Promotion Recommendations")
    
    if not recommendations.empty:
        # Display count by priority
        st.write(f"Found {len(recommendations)} products requiring promotion action")
        
        # Display recommendations table
        display_cols = [
            'Product line', 'DiscountPercentage', 'Unit price_mean', 'DiscountedPrice',
            'ExpectedSalesLift', 'ProfitImpact', 'Priority'
        ]
        
        # Create a styled table
        def highlight_priority(val):
            if val == 'High':
                return 'background-color: #ffcccc'
            elif val == 'Medium':
                return 'background-color: #ffffcc'
            else:
                return ''
        
        st.dataframe(recommendations[display_cols].style.format({
            'DiscountPercentage': '{:.1%}',
            'Unit price_mean': '${:.2f}',
            'DiscountedPrice': '${:.2f}',
            'ExpectedSalesLift': '{:.1f}%',
            'ProfitImpact': '${:.2f}'
        }).applymap(highlight_priority, subset=['Priority']))
        
        # Create a visualization of top recommendations
        st.subheader("Top 10 Products for Promotion")
        
        top_recommendations = recommendations.head(10)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create a horizontal bar chart of profit impact
        bars = ax.barh(
            top_recommendations['Product line'],
            top_recommendations['ProfitImpact'],
            color='green'
        )
        
        # Add data labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            label_position = width if width > 0 else 0
            ax.text(
                label_position + 1, 
                bar.get_y() + bar.get_height()/2, 
                f"${width:.2f}", 
                va='center'
            )
        
        # Set labels and title
        ax.set_xlabel('Profit Impact ($)')
        ax.set_title('Potential Profit Impact of Promotions')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Create download link
        st.markdown("### Download Recommendations")
        
        # Include more columns for the download
        download_cols = [
            'Branch', 'Product line', 'DiscountPercentage', 'Unit price_mean', 
            'DiscountedPrice', 'ExpectedSalesLift', 'InventoryDays', 
            'SalesVelocity', 'SalesGrowthRate', 'ProfitImpact', 'Priority'
        ]
        
        st.markdown(
            create_download_link(
                recommendations[download_cols], 
                f"promotion_recommendations_{selected_branch}.csv", 
                "Download Promotion Recommendations (CSV)"
            ),
            unsafe_allow_html=True
        )
    else:
        st.info("No promotion recommendations generated. Try adjusting cluster settings or branch selection.")
    
    # Display detailed product analysis
    with st.expander("Product Analysis"):
        # Show all products with key metrics
        display_cols = [
            'Product line', 'SalesVelocity', 'SalesGrowthRate', 
            'InventoryDays', 'gross margin percentage_mean', 'Interpretation'
        ]
        
        st.dataframe(clustered_data[display_cols].style.format({
            'SalesVelocity': '{:.2f}',
            'SalesGrowthRate': '{:.1f}%',
            'InventoryDays': '{:.1f}',
            'gross margin percentage_mean': '{:.2f}%'
        }))

if __name__ == "__main__":
    app()
