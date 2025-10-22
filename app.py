import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Import modules
import sales_forecaster
import inventory_reordering
import retail_price_suggester
import promotional_items
import chatbot

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('downloads', exist_ok=True)

# Set page config
st.set_page_config(page_title="Supermarket Analytics", layout="wide")

# Load basic data for home page
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_basic_data():
    try:
        data = pd.read_csv('data/SuperMarket Analysis.csv')
        
        # Calculate basic metrics
        total_sales = data['Sales'].sum() if 'Sales' in data.columns else data['Total'].sum()
        avg_gross_margin = data['gross margin percentage'].mean() if 'gross margin percentage' in data.columns else 0
        avg_rating = data['Rating'].mean() if 'Rating' in data.columns else 0
        
        return {
            'total_sales': total_sales,
            'avg_gross_margin': avg_gross_margin,
            'avg_rating': avg_rating
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return default values if file not found
        return {
            'total_sales': 286543,
            'avg_gross_margin': 4.76,
            'avg_rating': 7.8
        }

# Sidebar navigation
def sidebar():
    st.sidebar.title("Supermarket Analytics")
    page = st.sidebar.radio("Navigate:", ["Home", "Sales Forecasting", "Inventory", "Pricing", "Promotions", "Chatbot"])
    return page

# Home page
def home():
    st.title("Supermarket Analytics Dashboard")
    st.write("Select a module from the sidebar to analyze your supermarket sales data.")
    
    # Load basic metrics
    metrics = load_basic_data()
    
    # Basic metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sales", f"${metrics['total_sales']:.2f}")
    with col2:
        st.metric("Gross Margin", f"{metrics['avg_gross_margin']:.2f}%")
    with col3:
        st.metric("Customer Rating", f"{metrics['avg_rating']:.1f}/10")
    
    # Quick buttons
    st.subheader("Quick Actions")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Sales Forecast"):
            st.session_state.page = "Sales Forecasting"
            st.rerun()
    with col2:
        if st.button("Check Inventory Status"):
            st.session_state.page = "Inventory"
            st.rerun()

# Main app
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "Home"
    
    # Get page from sidebar
    page = sidebar()
    
    # Update session state if page changed
    if page != st.session_state.page:
        st.session_state.page = page
    
    # Display selected page
    if st.session_state.page == "Home":
        home()
    elif st.session_state.page == "Sales Forecasting":
        sales_forecaster.app()
    elif st.session_state.page == "Inventory":
        inventory_reordering.app()
    elif st.session_state.page == "Pricing":
        retail_price_suggester.app()
    elif st.session_state.page == "Promotions":
        promotional_items.app()
    elif st.session_state.page == "Chatbot":
        chatbot.app()

if __name__ == "__main__":
    main()
