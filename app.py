import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ==============================
# Load model and dataset
# ==============================
st.set_page_config(page_title="Dynamic Pricing Optimizer", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("ecommerce_dynamic_pricing_dataset.csv")

@st.cache_resource
def load_model():
    return joblib.load("pricing_model.pkl")

df = load_data()
model = load_model()

st.title("💰 Dynamic Pricing Optimization for E-commerce Store")
st.markdown("Adjust pricing based on customer and product factors to maximize revenue.")

# ==============================
# Sidebar for user inputs
# ==============================
st.sidebar.header("Product Configuration")

product_category = st.sidebar.selectbox("Product Category", df["Product_Category"].unique())
gender = st.sidebar.selectbox("Customer Gender", df["Customer_Gender"].unique())
payment = st.sidebar.selectbox("Payment Method", df["Payment_Method"].unique())
shipping = st.sidebar.selectbox("Shipping Type", df["Shipping_Type"].unique())

price = st.sidebar.slider("Current Price (₹)", float(df["Price"].min()), float(df["Price"].max()), 500.0)
discount = st.sidebar.slider("Discount (₹)", float(df["Discount"].min()), float(df["Discount"].max()), 50.0)
age = st.sidebar.slider("Customer Age", 18, 70, 30)
rating = st.sidebar.slider("Product Rating (1–5)", 1, 5, 4)
purchase_prob = st.sidebar.selectbox("Purchase Probability (0 or 1)", [0, 1])

st.sidebar.markdown("---")
optimize_button = st.sidebar.button("🔍 Optimize Price")

# ==============================
# Prepare feature input
# ==============================
product_info = {
    "Price": price,
    "Discount": discount,
    "Customer_Age": age,
    "Review_Rating": rating,
    "Purchase_Probability": purchase_prob,
}

# One-hot encode categories like training data
for col in df.select_dtypes(include="object").columns:
    unique_vals = df[col].unique()
    for val in unique_vals:
        col_name = f"{col}_{val}"
        product_info[col_name] = 1 if val in [product_category, gender, payment, shipping] else 0

# Ensure all columns match model’s training data
model_features = model.feature_names_in_
for col in model_features:
    if col not in product_info:
        product_info[col] = 0

# ==============================
# Optimization Logic
# ==============================
def optimize_price(model, product_info, price_range, discount_range):
    prices, revenues = [], []
    for p in price_range:
        for d in discount_range:
            sample = product_info.copy()
            sample["Price"] = p
            sample["Discount"] = d
            sample_df = pd.DataFrame([sample])[model_features]
            predicted_revenue = model.predict(sample_df)[0]
            prices.append(p)
            revenues.append(predicted_revenue)
    best_index = np.argmax(revenues)
    return prices[best_index], revenues[best_index], prices, revenues

# ==============================
# Run Optimization
# ==============================
if optimize_button:
    st.subheader("📈 Price Optimization Results")
    price_range = np.linspace(price * 0.5, price * 1.5, 50)
    discount_range = np.linspace(discount * 0.5, discount * 1.5, 10)

    best_price, best_revenue, prices, revenues = optimize_price(model, product_info, price_range, discount_range)

    st.success(f"✅ **Optimal Price:** ₹{best_price:.2f}")
    st.info(f"💸 **Expected Maximum Revenue:** ₹{best_revenue:.2f}")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(prices, revenues, marker='o')
    ax.axvline(best_price, color='r', linestyle='--', label=f"Best Price = ₹{best_price:.2f}")
    ax.set_title("Revenue vs Price Curve")
    ax.set_xlabel("Price (₹)")
    ax.set_ylabel("Predicted Revenue")
    ax.legend()
    st.pyplot(fig)
