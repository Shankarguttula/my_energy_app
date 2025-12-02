import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

st.title("Energy Forecasting Dashboard (Prophet Models)")

# List of all models
cols = [
    'fossil_prod','nuclear_prod','renewable_prod',
    'imports','exports',
    'stock_change','fossil_cons','nuclear_cons',
    'renewable_cons'
]

# Dropdown to choose variable
selected_col = st.selectbox("Choose a variable to forecast:", cols)

# Correct model path (inside prophet_models folder)
model_path = os.path.join("prophet_models", f"prophet_model_{selected_col}.pkl")

# Load model safely
try:
    model = joblib.load(model_path)
    st.success(f"Loaded model: {model_path}")
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Number of months to forecast
periods = st.number_input("Months to forecast:", min_value=1, max_value=36, value=12)

# Predict
if st.button("Predict"):
    # Future dates
    future = model.make_future_dataframe(periods=periods, freq="M")

    # Forecast
    forecast = model.predict(future)

    # Final result
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

    st.subheader("Forecast Results")
    st.dataframe(result)

    # Plot
    fig = model.plot(forecast)
    st.pyplot(fig)

    # Download CSV
    csv = result.to_csv(index=False)
    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name=f"{selected_col}_forecast.csv",
        mime="text/csv"
    )
