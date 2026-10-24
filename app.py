import streamlit as st
import pandas as pd
import joblib
from datetime import date, time

# Page configuration
st.set_page_config(
    page_title="TrafficSense AI",
    page_icon="🚦",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("trafficsense_model.pkl")

st.title("🚦 TrafficSense AI")
st.subheader("Machine Learning-Based Traffic Prediction")
st.write(
    "Predict traffic levels using time and weather conditions."
)

try:
    model = load_model()
except Exception:
    st.error(
        "Model not found. Place trafficsense_model.pkl "
        "in the same folder as app.py."
    )
    st.stop()

st.divider()

st.header("Enter Traffic Conditions")

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        "Select date",
        value=date.today()
    )

    selected_time = st.time_input(
        "Select time",
        value=time(18, 0)
    )

    temperature_c = st.number_input(
        "Temperature (°C)",
        min_value=-50.0,
        max_value=60.0,
        value=15.0
    )

with col2:
    rain = st.number_input(
        "Rainfall (mm in 1 hour)",
        min_value=0.0,
        value=0.0
    )

    snow = st.number_input(
        "Snowfall (mm in 1 hour)",
        min_value=0.0,
        value=0.0
    )

    clouds = st.slider(
        "Cloud coverage (%)",
        min_value=0,
        max_value=100,
        value=40
    )

if st.button("Predict Traffic", type="primary"):
    hour = selected_time.hour
    weekday = selected_date.weekday()
    month = selected_date.month
    is_weekend = int(weekday >= 5)

    # Convert Celsius to Kelvin for this dataset
    temperature_k = temperature_c + 273.15

    input_data = pd.DataFrame([{
        "hour": hour,
        "day": weekday,
        "month": month,
        "temp": temperature_k,
        "rain_1h": rain,
        "snow_1h": snow,
        "clouds_all": clouds
    }])

    try:
        model = model["model"]
        input_data = input_data.rename(columns = { "Day of week": "day_of_week","Is weekend": "is_weekend"})
        prediction = model.predict(input_data)[0]

        st.divider()
        st.header("Prediction Result")

        if str(prediction) == "High":
            st.error("🚨 Predicted Traffic: HIGH")
        elif str(prediction) == "Medium":
            st.warning("⚠️ Predicted Traffic: MEDIUM")
        else:
            st.success("✅ Predicted Traffic: LOW")

        st.caption(
            "Prediction is based on historical training data, "
            "not live traffic measurements."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write(
            "Check that your model was trained with the "
            "same feature names and order."
        )

st.divider()
st.caption("TrafficSense AI | Data Science Project")