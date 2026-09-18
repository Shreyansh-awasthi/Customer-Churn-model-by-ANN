# Set-Content -Path "app.py" -Value @'
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Bank Customer Churn Predictor (ANN)")
st.write("Enter the customer's metrics below to predict if they will stay or leave (`Exited`).")

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Profile Details")
    credit_score = st.slider("Credit Score", 300, 850, 600)
    geography = st.selectbox("Geography / Country", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 40)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)

with col2:
    st.subheader("💳 Financial Indicators")
    balance = st.number_input("Account Balance (\$)", min_value=0.0, value=10000.0, step=500.0)
    num_of_products = st.slider("Number of Products", 1, 4, 2)
    has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.radio("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary (\$)", min_value=0.0, value=50000.0, step=1000.0)

st.markdown("---")

if st.button("🔮 Evaluate Churn Probability", use_container_width=True):
    gender_val = 1 if gender == "Male" else 0
    has_card_val = 1 if has_cr_card == "Yes" else 0
    active_val = 1 if is_active_member == "Yes" else 0
    
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    try:
        # Load assets
        model = tf.keras.models.load_model('churn_model.keras')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
            
        raw_features = np.array([[
            credit_score, age, tenure, balance, num_of_products, 
            has_card_val, active_val, estimated_salary, gender_val, geo_germany, geo_spain
        ]])
        
        
        scaled_features = scaler.transform(raw_features)
        
    
        prediction = model.predict(scaled_features)
        probability = float(prediction[0][0]) * 100
        
        st.subheader("📊 Output Analysis")
        if probability > 50:
            st.error(f"⚠️ **High Churn Risk!** Probability: **{probability:.2f}%** (Customer likely to exit)")
        else:
            st.success(f"✅ **Loyal Customer:** Probability: **{probability:.2f}%** (Customer likely to stay)")
            
    except Exception as e:
        st.warning("📁 **Awaiting exported model files.** Please ensure `churn_model.keras` and `scaler.pkl` are generated in this folder.")
        st.info("💡 *Showing structural UI preview:*")
        mock_prob = np.random.uniform(5, 95)
        if mock_prob > 50:
            st.error(f"⚠️ **High Churn Risk:** Mock Probability **{mock_prob:.2f}%**")
        else:
            st.success(f"✅ **Loyal Customer:** Mock Probability **{mock_prob:.2f}%**")
# '@
