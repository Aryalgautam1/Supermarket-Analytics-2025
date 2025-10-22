import streamlit as st
import pandas as pd
import time
import re

# Load supermarket data
def load_data():
    try:
        data = pd.read_csv('data/SuperMarket Analysis.csv')
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
        return data
    except:
        st.error("Could not load data. Make sure 'SuperMarket Analysis.csv' is in the data folder.")
        return None

# Generate responses based on patterns
def generate_response(query, data):
    query = query.lower()
    
    # Sales overview
    if re.search(r'sales|revenue|overview', query):
        total_sales = data['Sales'].sum() if 'Sales' in data.columns else data['Total'].sum()
        avg_transaction = total_sales / len(data)
        return f"Total sales are ${total_sales:.2f} with an average transaction of ${avg_transaction:.2f}."
    
    # Product questions
    elif re.search(r'product|category|item', query):
        if 'Product line' in data.columns:
            top_product = data.groupby('Product line')['Quantity'].sum().idxmax()
            return f"The best-selling product category is {top_product}."
        else:
            return "I don't have product information in the current dataset."
    
    # Branch questions
    elif re.search(r'branch|store|location', query):
        if 'Branch' in data.columns:
            branch_sales = data.groupby('Branch')['Sales'].sum() if 'Sales' in data.columns else data.groupby('Branch')['Total'].sum()
            top_branch = branch_sales.idxmax()
            return f"The top performing branch is {top_branch}."
        else:
            return "I don't have branch information in the current dataset."
    
    # Customer questions
    elif re.search(r'customer|consumer', query):
        if 'Customer type' in data.columns:
            customer_count = data['Customer type'].value_counts()
            return f"There are {customer_count.get('Member', 0)} member customers and {customer_count.get('Normal', 0)} normal customers."
        else:
            return "I don't have customer information in the current dataset."
    
    # Payment questions
    elif re.search(r'payment|transaction', query):
        if 'Payment' in data.columns:
            payment_methods = data['Payment'].value_counts()
            methods = ", ".join([f"{method}: {count}" for method, count in payment_methods.items()])
            return f"Payment methods breakdown: {methods}"
        else:
            return "I don't have payment information in the current dataset."
    
    # Default response
    else:
        return """I can answer questions about:
- Sales overview
- Product categories
- Branch performance
- Customer information
- Payment methods

What would you like to know?"""

# Main app
def app():
    st.title("Supermarket Sales Assistant")
    
    # Load data
    data = load_data()
    
    if data is None:
        return
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your supermarket sales..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Generate response
            response = generate_response(prompt, data)
            
            # Simulate typing
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.02)
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Initial message
    if not st.session_state.messages:
        st.chat_message("assistant").markdown("""
        👋 Hello! I'm your Supermarket Sales Assistant. I can help you understand your sales data.
        
        Try asking me questions like:
        - "What are the total sales?"
        - "Which product category sells best?"
        - "How are the branches performing?"
        - "Tell me about our customers"
        - "What payment methods do customers use?"
        """)

if __name__ == "__main__":
    app()
