# ===============================
# SUPPLY CHAIN ML STREAMLIT APP
# ===============================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Supply Chain Cost Prediction",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Supply Chain Cost Prediction System")
st.subheader("AI-Powered Business Analytics Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("supply_chain_data.csv")

data = load_data()
data = data.dropna()

# -------------------------------
# FEATURE ENGINEERING
# -------------------------------
data["Profit"] = data["Revenue generated"] - data["Costs"]

# Encode categorical
cat_cols = data.select_dtypes(include="object").columns
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

X = data.drop("Manufacturing costs", axis=1)
y = data["Manufacturing costs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = r2_score(y_test, preds)

# -------------------------------
# TOP METRICS
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Model Used", "Random Forest")
col2.metric("R2 Score", f"{accuracy:.2f}")
col3.metric("Dataset Size", f"{len(data)} Rows")

# -------------------------------
# SIDEBAR INPUT
# -------------------------------
st.sidebar.header("📥 Enter Business Inputs")

revenue = st.sidebar.number_input("Revenue Generated", min_value=0.0)
costs = st.sidebar.number_input("Costs", min_value=0.0)
shipping = st.sidebar.number_input("Shipping Costs", min_value=0.0)

predict_button = st.sidebar.button("🚀 Predict Manufacturing Cost")

# -------------------------------
# PREDICTION
# -------------------------------
if predict_button:

    input_data = pd.DataFrame(
        np.zeros((1, X.shape[1])),
        columns=X.columns
    )

    if "Revenue generated" in input_data.columns:
        input_data["Revenue generated"] = revenue

    if "Costs" in input_data.columns:
        input_data["Costs"] = costs

    if "Shipping costs" in input_data.columns:
        input_data["Shipping costs"] = shipping

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"💰 Predicted Manufacturing Cost: {prediction[0]:.2f}")

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

st.success("✅ App Ready for Deployment")
