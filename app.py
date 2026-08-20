import streamlit as st
import joblib
import os

st.set_page_config(
    page_title="HealthSense AI",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ HealthSense AI")
st.subheader("Heart Disease Prediction System")

# Model path
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_heart_model.pkl"
)

# Load model
try:
    model = joblib.load(MODEL_PATH)
    st.success("✅ AI Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Could not load the model: {e}")
    st.stop()

st.write("""
Welcome to **HealthSense AI**.

This system uses Machine Learning to predict the risk
of heart disease based on patient clinical data.
""")

st.info("🚧 Patient prediction form will be added in the next step.")
