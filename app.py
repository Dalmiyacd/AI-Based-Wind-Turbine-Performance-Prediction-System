import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Models
power_model = joblib.load("best_power_output_model.pkl")
downtime_model = joblib.load("best_downtime_model.pkl")

st.title("🌬️ AI-Based Wind Turbine Performance Prediction System")
st.write("Predict power output and downtime probability using machine learning models.")

# Input UI
st.header("Enter Turbine Parameters")

wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 40.0, 12.0, key="wind_speed")
wind_direction = st.number_input("Wind Direction (°)", 0.0, 360.0, 180.0, key="wind_direction")
air_density = st.number_input("Air Density (kg/m³)", 0.5, 2.0, 1.2, key="air_density")
temperature = st.number_input("Temperature (°C)", -20.0, 60.0, 25.0, key="temperature")

vibration = st.number_input("Vibration Level (mm/s)", 0.0, 20.0, 2.0, key="vibration")
is_high_vibration = 1 if vibration > 6.0 else 0   # Feature used by the model

rpm = st.number_input("Rotor RPM", 0.0, 50.0, 10.0, key="rotor_rpm")

hour = st.slider("Hour of Day", 0, 23, 12, key="hour")
dayofweek = st.slider("Day of Week (0 = Monday)", 0, 6, 3, key="dow")
month = st.slider("Month", 1, 12, 6, key="month")



# Derived feature
wind_power_density = 0.5 * air_density * (wind_speed ** 3)

# Create dataframe for prediction
input_data = pd.DataFrame({
    "turbine_id": [1],
    "wind_speed_mps": [wind_speed],
    "wind_direction_deg": [wind_direction],
    "air_density_kg_m3": [air_density],
    "temperature_c": [temperature],
    "vibration_level_mm_s": [vibration],
    "rotor_rpm": [rpm],
    "hour": [hour],
    "dayofweek": [dayofweek],
    "month": [month],
    "wind_power_density": [wind_power_density],
    "is_high_vibration": [is_high_vibration]   # <-- Add this inside DataFrame
})


if st.button("Predict"):
    predicted_power = power_model.predict(input_data)[0]
    predicted_downtime = downtime_model.predict(input_data)[0]
    downtime_prob = downtime_model.predict_proba(input_data)[0][1]

    st.subheader("🔮 Predictions")
    st.write(f"⚡ **Estimated Power Output:** {predicted_power:.2f} kW")
    st.write(f"🛑 **Downtime Probability:** {downtime_prob:.2f}")

    if predicted_downtime == 1:
        st.error("⚠️ High downtime risk detected! (Model Output: 1)")
    else:
        st.success("✔ Normal operation expected (Model Output: 0)")
