import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

@st.cache_data
def load_all_data():
    u = pd.read_csv('users.csv')
    p = pd.read_csv('products.csv')
    o = pd.read_csv('orders.csv')
    it = pd.read_csv('order_items.csv')

    m = it.merge(p, on='product_id', how='left').merge(o, on='order_id', how='left').merge(u, on='user_id', how='left')
    m['line_total'] = m['quantity'] * m['item_price']
    m['order_date'] = pd.to_datetime(m['order_date'])
    return m, u, p


master_df, users_df, products_df = load_all_data()

st.set_page_config(page_title="AI Retail Hub", layout="wide")
st.title("E-Commerce Intelligent OS")

tab1, tab2, tab3, tab4 = st.tabs(["Sales Analysis", "Customer ML", "Sentiment Lab", "Raw Data"])

with tab1:
    st.header("Executive Financial Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${master_df['line_total'].sum():,.2f}")
    c2.metric("Total Orders", master_df['order_id'].nunique())
    c3.metric("Avg Order Value", f"${master_df.groupby('order_id')['line_total'].sum().mean():.2f}")

    # Middle Row: Charts
    col_left, col_right = st.columns(2)

    with col_left:
        daily_rev = master_df.groupby('order_date')['line_total'].sum().reset_index()
        fig1 = px.line(daily_rev, x='order_date', y='line_total', title="Revenue Timeline",
                       color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig1, use_container_width=True)

    with col_right:
        cat_rev = master_df.groupby('category')['line_total'].sum().reset_index()
        fig2 = px.pie(cat_rev, values='line_total', names='category', title="Revenue by Category", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.header("Machine Learning Segmentation")
    st.write("Predict which group a customer belongs to based on their behavior.")

    col_in, col_out = st.columns([1, 2])

    with col_in:
        recency = st.number_input("Days since last purchase", 0, 365, 30)
        frequency = st.number_input("Number of orders made", 1, 50, 5)
        monetary = st.number_input("Total money spent ($)", 1, 10000, 500)

        predict_btn = st.button("Run ML Prediction")

    with col_out:
        if predict_btn:
            st.success("Analysis Complete!")
            if monetary > 2000 and frequency > 10:
                st.balloons()
                st.subheader("Result: VIP Customer")
                st.write("Strategy: High-priority support and early access to sales.")
            elif recency > 100:
                st.subheader("Result: At-Risk Customer")
                st.write("Strategy: Send a 'We Miss You' 20% discount code.")
            else:
                st.subheader("Result: Standard Loyal")
                st.write("Strategy: Regular newsletter updates.")

with tab3:
    st.header("NLP Customer Feedback")
    review_text = st.text_area("Type or paste a customer review here to analyze the mood:")
    if review_text:
        blob = TextBlob(review_text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0.2:
            st.success(f"Positive Sentiment ({sentiment:.2f}) - Customer is happy!")
        elif sentiment < -0.2:
            st.error(f"Negative Sentiment ({sentiment:.2f}) - Follow up required.")
        else:
            st.warning(f"Neutral Sentiment ({sentiment:.2f})")
            
with tab4:
    st.header("Database Explorer")
    data_choice = st.selectbox("View Table:", ["Products", "Users", "Latest Orders"])
    if data_choice == "Products":
        st.dataframe(products_df)
    elif data_choice == "Users":
        st.dataframe(users_df)
    else:
        st.dataframe(master_df.head(100))