# ==========================================
# SUPPLY CHAIN DASHBOARD + PREDICTION
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(layout="wide")

st.title("📦 Supply Chain Analytics & Prediction System")

# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("supply_chain_data.csv")  # change file name

data = load_data()

# -----------------------------------
# SIDEBAR - PREDICTION SECTION
# -----------------------------------
st.sidebar.header("🔮 Predict Revenue")

numeric_data = data.select_dtypes(include=np.number)

if "Revenue generated" in numeric_data.columns:

    X = numeric_data.drop("Revenue generated", axis=1)
    y = numeric_data["Revenue generated"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Sidebar Inputs
    user_input = []

    for col in X.columns:
        value = st.sidebar.number_input(
            f"{col}",
            float(X[col].min()),
            float(X[col].max()),
            float(X[col].mean())
        )
        user_input.append(value)

    if st.sidebar.button("Predict"):
        prediction = model.predict([user_input])
        st.sidebar.success(f"Predicted Revenue: {prediction[0]:,.2f}")

# -----------------------------------
# MAIN PAGE - KPI METRICS
# -----------------------------------
st.subheader("📊 Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"{data['Revenue generated'].sum():,.0f}")
col2.metric("Total Cost", f"{data['Costs'].sum():,.0f}")
col3.metric("Total Products", data['Product type'].nunique())

st.markdown("---")

# -----------------------------------
# ALL PLOTS IN ONE PAGE
# -----------------------------------

col1, col2 = st.columns(2)

# 1️⃣ Product Type Count
with col1:
    st.subheader("Product Type Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Product type", data=data, palette="Set2", ax=ax1)
    plt.xticks(rotation=45)
    st.pyplot(fig1)

# 2️⃣ Revenue Distribution
with col2:
    st.subheader("Revenue Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(data["Revenue generated"], kde=True, ax=ax2)
    st.pyplot(fig2)

# -----------------------------------

col3, col4 = st.columns(2)

# 3️⃣ Revenue vs Cost
with col3:
    st.subheader("Revenue vs Cost")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(
        x="Revenue generated",
        y="Costs",
        data=data,
        ax=ax3
    )
    st.pyplot(fig3)

# 4️⃣ Stock Levels Boxplot
if "Stock levels" in data.columns:
    with col4:
        st.subheader("Stock Levels by Product")
        fig4, ax4 = plt.subplots()
        sns.boxplot(
            x="Product type",
            y="Stock levels",
            data=data,
            ax=ax4
        )
        plt.xticks(rotation=45)
        st.pyplot(fig4)

# -----------------------------------
# Correlation Heatmap
# -----------------------------------

st.subheader("🔥 Correlation Heatmap")

fig5, ax5 = plt.subplots(figsize=(10,6))
sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    ax=ax5
)
st.pyplot(fig5)

st.markdown("---")
st.success("Dashboard Loaded Successfully 🚀")
