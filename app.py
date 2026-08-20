from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return 'HealthSense Application is Running'

if __name__ == '__main__':
    app.run(debug=True)


import streamlit as st

# إعدادات الصفحة
st.set_page_config(
    page_title="HealthSense AI",
    page_icon="❤️",
    layout="wide"
)

# عنوان المشروع
st.title("❤️ HealthSense AI")

st.markdown("## Heart Disease Prediction System")

st.write(
    """
    Welcome to **HealthSense AI**.

    This application predicts the probability of heart disease
    using a Machine Learning model trained on clinical patient data.

    Fill in the patient's information, then click **Predict**.
    """
)

st.divider()

st.info("👈 In the next step, we will add the patient information form.")