
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Wind Turbine Performance Prediction", layout="centered")

st.title("🌬️ AI-Based Wind Turbine Performance Prediction System")

# Safety check for model files
if not os.path.exists("best_power_output_model.pkl") or not os.path.exists("best_downtime_model.pkl"):
    st.error("❌ Model files not found. Please ensure .pkl files are in the repository root.")
    st.stop()

# Load models
power_model = joblib.load("best_power_output_model.pkl")
downtime_model = joblib.load("best_downtime_model.pkl")

# Extract exact feature names used during training
FEATURE_COLUMNS = list(power_model.feature_names_in_)

st.header("Enter Turbine Parameters")

wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 40.0, 12.0, key="wind_speed")
wind_direction = st.number_input("Wind Direction (°)", 0.0, 360.0, 180.0, key="wind_direction")
air_density = st.number_input("Air Density (kg/m³)", 0.5, 2.0, 1.2, key="air_density")
temperature = st.number_input("Temperature (°C)", -20.0, 60.0, 25.0, key="temperature")
vibration = st.number_input("Vibration Level (mm/s)", 0.0, 20.0, 2.0, key="vibration")
rpm = st.number_input("Rotor RPM", 0.0, 50.0, 10.0, key="rpm")

hour = st.slider("Hour of Day", 0, 23, 12, key="hour")
dayofweek = st.slider("Day of Week (0 = Monday)", 0, 6, 3, key="dow")
month = st.slider("Month", 1, 12, 6, key="month")

# Feature engineering
wind_power_density = 0.5 * air_density * (wind_speed ** 3)
is_high_vibration = 1 if vibration > 6.0 else 0

# Input dictionary
input_dict = {
    "turbine_id": 1,
    "wind_speed_mps": wind_speed,
    "wind_direction_deg": wind_direction,
    "air_density_kg_m3": air_density,
    "temperature_c": temperature,
    "vibration_level_mm_s": vibration,
    "rotor_rpm": rpm,
    "hour": hour,
    "dayofweek": dayofweek,
    "month": month,
    "wind_power_density": wind_power_density,
    "is_high_vibration": is_high_vibration
}

# Build input dataframe with exact training feature order
input_data = pd.DataFrame([[input_dict[col] for col in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)

if st.button("Predict"):
    predicted_power = power_model.predict(input_data)[0]
    downtime_prob = downtime_model.predict_proba(input_data)[0][1]
    downtime_pred = downtime_model.predict(input_data)[0]

    st.subheader("🔮 Prediction Results")
    st.metric("Predicted Power Output (kW)", f"{predicted_power:.2f}")
    st.metric("Downtime Probability", f"{downtime_prob:.2%}")

    if downtime_pred == 1:
        st.error("⚠️ High risk of turbine downtime detected.")
    else:
        st.success("✅ Turbine operating normally.")
