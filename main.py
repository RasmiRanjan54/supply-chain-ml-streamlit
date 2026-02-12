import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(layout="wide")
st.title("📦 Supply Chain Analytics & ML Prediction App")

# ---------------------------------------
# LOAD DATA
# ---------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("supply_chain_data.csv")

data = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

st.write("Shape:", data.shape)

# ---------------------------------------
# CLEANING
# ---------------------------------------
data = data.dropna()

# ---------------------------------------
# BASIC VISUALIZATION
# ---------------------------------------
st.header("📈 Exploratory Data Analysis")

fig1, ax1 = plt.subplots()
sns.histplot(data['Manufacturing costs'], kde=True, ax=ax1, color="#2E86C1")
ax1.set_title("Distribution of Manufacturing Costs")
st.pyplot(fig1)

# Correlation
numeric_data = data.select_dtypes(include=np.number)
fig2, ax2 = plt.subplots(figsize=(10,6))
sns.heatmap(numeric_data.corr(), cmap="coolwarm", ax=ax2)
ax2.set_title("Correlation Heatmap")
st.pyplot(fig2)

# ---------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------
data['Profit'] = data['Revenue generated'] - data['Costs']
data['Profit_Status'] = data['Profit'].apply(lambda x: 1 if x > 0 else 0)

# ---------------------------------------
# MODEL TRAINING
# ---------------------------------------
st.header("🤖 Machine Learning Model Training")

cat_cols = data.select_dtypes(include='object').columns
data_encoded = pd.get_dummies(data, columns=cat_cols, drop_first=True)

X = data_encoded.drop(columns=['Manufacturing costs'])
y = data_encoded['Manufacturing costs']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5),
    "Random Forest": RandomForestRegressor(n_estimators=200),
    "Extra Trees": ExtraTreesRegressor(n_estimators=200),
    "Gradient Boosting": GradientBoostingRegressor(),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR()
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    results.append([name, mse, r2])

results_df = pd.DataFrame(results, columns=["Model", "MSE", "R2 Score"])
results_df = results_df.sort_values("MSE")

st.subheader("📊 Model Comparison")
st.dataframe(results_df)

best_model_name = results_df.iloc[0]["Model"]
st.success(f"🏆 Best Model: {best_model_name}")

# ---------------------------------------
# ACTUAL VS PREDICTED
# ---------------------------------------
best_model = models[best_model_name]
best_model.fit(X_train_scaled, y_train)
best_preds = best_model.predict(X_test_scaled)

fig3, ax3 = plt.subplots()
ax3.scatter(y_test, best_preds)
ax3.set_xlabel("Actual")
ax3.set_ylabel("Predicted")
ax3.set_title("Actual vs Predicted Manufacturing Cost")
st.pyplot(fig3)

# ---------------------------------------
# BUSINESS ANALYTICS VISUALS
# ---------------------------------------
st.header("📊 Business Insights")

# Profit by Location
fig4, ax4 = plt.subplots()
sns.barplot(data=data, x='Location', y='Profit', estimator=sum, ax=ax4)
plt.xticks(rotation=45)
ax4.set_title("Total Profit by Location")
st.pyplot(fig4)

# Revenue by Product
fig5, ax5 = plt.subplots()
sns.barplot(data=data, x='Product type', y='Revenue generated', estimator=sum, ax=ax5)
plt.xticks(rotation=45)
ax5.set_title("Revenue by Product Type")
st.pyplot(fig5)

# Shipping Cost Distribution
fig6, ax6 = plt.subplots()
sns.histplot(data['Shipping costs'], kde=True, ax=ax6, color="#8E44AD")
ax6.set_title("Shipping Cost Distribution")
st.pyplot(fig6)

# Profit Status Distribution
categ_cols = [
    'Customer demographics',
    'Location',
    'Product type',
    'Transportation modes'
]

custom_palettes = [
    ["#FF6B6B", "#1DD1A1"],
    ["#5F27CD", "#35AEAE"],
    ["#FF9F43", "#10AC84"],
    ["#EE5253", "#2E86DE"]
]

for i, col in enumerate(categ_cols):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.countplot(
        data=data,
        x=col,
        hue="Profit_Status",
        palette=custom_palettes[i % len(custom_palettes)],
        ax=ax
    )
    plt.xticks(rotation=45)
    ax.set_title(f"{col} Distribution by Profit Status")
    st.pyplot(fig)

st.success("✅ Full Streamlit App Ready for Deployment")
