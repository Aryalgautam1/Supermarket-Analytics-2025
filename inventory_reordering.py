import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import base64

# Load data
def load_data():
    """
    Load the supermarket data and create inventory estimates
    """
    try:
        if os.path.exists('data/SuperMarket Analysis.csv'):
            # Load original data
            data = pd.read_csv('data/SuperMarket Analysis.csv')
            
            # Convert date if present
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            
            # Make sure we have the necessary columns
            if 'Product line' not in data.columns or 'Branch' not in data.columns:
                st.error("Required columns 'Product line' or 'Branch' not found in dataset.")
                return pd.DataFrame()
            
            # Generate inventory data directly from product lines
            products = []
            for branch in data['Branch'].unique():
                branch_data = data[data['Branch'] == branch]
                
                for product in branch_data['Product line'].unique():
                    # Get product-specific data
                    product_data = branch_data[branch_data['Product line'] == product]
                    
                    # Calculate average sales (per transaction)
                    avg_sales = len(product_data) / len(branch_data) * 10  # Simulate 10 transactions per day
                    
                    # Generate random inventory data
                    days_supply = np.random.uniform(15, 45)
                    current_stock = int(avg_sales * days_supply)
                    reorder_point = int(avg_sales * np.random.uniform(7, 14))
                    
                    # Calculate risk
                    if current_stock <= reorder_point * 0.5:
                        risk = "High" 
                    elif current_stock <= reorder_point:
                        risk = "Medium"
                    else:
                        risk = "Low"
                    
                    # Use unit price if available
                    if 'Unit price' in data.columns:
                        unit_price = product_data['Unit price'].mean()
                        cost = unit_price * 0.7  # Assume 30% markup
                    else:
                        unit_price = np.random.uniform(20, 100)
                        cost = unit_price * 0.7
                    
                    # Add to inventory
                    products.append({
                        'Branch': branch,
                        'ProductLine': product,
                        'CurrentStock': current_stock,
                        'ReorderPoint': reorder_point,
                        'DaysOfSupply': days_supply,
                        'AvgDailySales': avg_sales,
                        'StockOutRisk': risk,
                        'CostPerUnit': cost,
                        'SellingPrice': unit_price,
                        'OrderQuantity': max(0, reorder_point - current_stock) if current_stock <= reorder_point else 0,
                        'OrderValue': max(0, reorder_point - current_stock) * cost if current_stock <= reorder_point else 0,
                        'TotalValue': current_stock * cost
                    })
            
            return pd.DataFrame(products)
        else:
            st.error("Data file not found. Please make sure 'SuperMarket Analysis.csv' is in the data folder.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Create download link
def create_download_link(df, filename, text):
    """
    Create a download link for DataFrame
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to highlight risk
def highlight_risk(val):
    """
    Apply color formatting based on risk level
    """
    if val == 'High':
        return 'background-color: #ffcccc'
    elif val == 'Medium':
        return 'background-color: #ffffcc'
    else:
        return ''

# Main app
def app():
    st.title("Inventory Management")
    
    # Load data
    with st.spinner("Analyzing inventory data..."):
        inventory_data = load_data()
    
    if inventory_data.empty:
        return
    
    # Sidebar
    st.sidebar.header("Filter Options")
    
    # Branch filter
    branches = sorted(inventory_data['Branch'].unique())
    selected_branch = st.sidebar.selectbox("Select Branch", branches)
    
    # Risk filter
    risk_options = ['All', 'High', 'Medium', 'Low']
    selected_risk = st.sidebar.selectbox("Stock Out Risk", risk_options, index=0)
    
    # Filter data by branch
    branch_inventory = inventory_data[inventory_data['Branch'] == selected_branch].copy()
    
    # Apply risk filter
    if selected_risk != "All":
        branch_inventory = branch_inventory[branch_inventory['StockOutRisk'] == selected_risk]
    
    # Find items that need reordering
    reorder_items = branch_inventory[branch_inventory['OrderQuantity'] > 0]
    
    # Display metrics
    st.subheader(f"Inventory Overview for {selected_branch}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_value = branch_inventory['TotalValue'].sum()
        st.metric("Total Inventory Value", f"${total_value:.2f}")
    
    with col2:
        avg_days = branch_inventory['DaysOfSupply'].mean()
        st.metric("Avg Days of Supply", f"{avg_days:.1f}")
    
    with col3:
        items_to_reorder = len(reorder_items)
        st.metric("Items To Reorder", f"{items_to_reorder}")
    
    with col4:
        high_risk = len(branch_inventory[branch_inventory['StockOutRisk'] == 'High'])
        st.metric("High Risk Items", f"{high_risk}")
    
    # Display recommendations
    st.subheader("Reorder Recommendations")
    
    if len(reorder_items) > 0:
        # Sort by risk level
        risk_order = {'High': 0, 'Medium': 1, 'Low': 2}
        reorder_items['RiskOrder'] = reorder_items['StockOutRisk'].map(risk_order)
        reorder_items = reorder_items.sort_values(['RiskOrder', 'DaysOfSupply'])
        
        # Display table
        display_cols = ['ProductLine', 'CurrentStock', 'ReorderPoint', 'OrderQuantity', 
                      'OrderValue', 'DaysOfSupply', 'StockOutRisk']
        
        st.dataframe(reorder_items[display_cols].style.applymap(
            highlight_risk, subset=['StockOutRisk']
        ).format({
            'OrderValue': '${:.2f}',
            'DaysOfSupply': '{:.1f}'
        }))
        
        # Plot
        st.subheader("Current Stock vs Reorder Point")
        
        # Get top items to plot
        top_items = reorder_items.head(10)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for plotting
        products = top_items['ProductLine'].tolist()
        current_stock = top_items['CurrentStock'].tolist()
        reorder_point = top_items['ReorderPoint'].tolist()
        
        # Set up bar chart
        x = np.arange(len(products))
        width = 0.35
        
        # Create bars
        ax.bar(x - width/2, current_stock, width, label='Current Stock', color='skyblue')
        ax.bar(x + width/2, reorder_point, width, label='Reorder Point', color='salmon')
        
        # Add labels and legend
        ax.set_ylabel('Units')
        ax.set_title('Current Stock vs. Reorder Point (Top 10 Items Needing Reordering)')
        ax.set_xticks(x)
        ax.set_xticklabels(products, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Create download link
        st.markdown("### Download Recommendations")
        st.markdown(
            create_download_link(
                reorder_items, 
                f"reorder_list_{selected_branch}.csv", 
                "Download Reorder List (CSV)"
            ),
            unsafe_allow_html=True
        )
    
    else:
        st.info("No items need reordering based on current filters.")
    
    # Display all inventory items
    st.subheader("Complete Inventory")
    
    display_cols = ['ProductLine', 'CurrentStock', 'AvgDailySales', 
                  'DaysOfSupply', 'ReorderPoint', 'StockOutRisk']
    
    st.dataframe(branch_inventory[display_cols].style.applymap(
        highlight_risk, subset=['StockOutRisk']
    ).format({
        'AvgDailySales': '{:.2f}',
        'DaysOfSupply': '{:.1f}'
    }))

if __name__ == "__main__":
    app()
