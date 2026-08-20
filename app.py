import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="HealthSense AI",
    page_icon="❤️",
    layout="wide"
)

# =========================
# Load Model
# =========================
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_heart_model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Could not load the model: {e}")
    st.stop()

# =========================
# Title
# =========================
st.title("❤️ HealthSense AI")
st.subheader("Heart Disease Prediction System")

st.write("""
Enter the patient's clinical information below.
The AI model will analyze the data and predict the likelihood of heart disease.
""")

st.divider()

# =========================
# Patient Information Form
# =========================

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type (cp)",
        options=[0, 1, 2, 3]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=100,
        max_value=600,
        value=200
    )


with col2:

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    restecg = st.selectbox(
        "Resting ECG",
        options=[0, 1, 2]
    )

    thalach = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )


with col3:

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "ST Slope",
        options=[0, 1, 2]
    )

    ca = st.selectbox(
        "Number of Major Vessels (ca)",
        options=[0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thal",
        options=[0, 1, 2, 3]
    )

st.divider()

# =========================
# Prediction
# =========================

if st.button("🔍 Predict Heart Disease Risk", use_container_width=True):

    input_data = pd.DataFrame(
        [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]],
        columns=[
            'age',
            'sex',
            'cp',
            'trestbps',
            'chol',
            'fbs',
            'restecg',
            'thalach',
            'exang',
            'oldpeak',
            'slope',
            'ca',
            'thal'
        ]
    )

    prediction = model.predict(input_data)[0]

    # Get probability
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]

        # Probability of class 1
        risk_probability = probabilities[1] * 100
    else:
        risk_probability = None

    st.divider()

    # =========================
    # Result Dashboard
    # =========================

    st.subheader("🩺 Prediction Result")

    if prediction == 1:

        st.error("⚠️ Higher Risk of Heart Disease")

        if risk_probability is not None:

            st.metric(
                label="Heart Disease Risk Score",
                value=f"{risk_probability:.2f}%"
            )

            st.progress(
                min(risk_probability / 100, 1.0)
            )

            st.write(
    f"The model estimates a **{risk_probability:.2f}%** "
    f"probability for the positive class."
)

    else:

        st.success("✅ Lower Risk of Heart Disease")

        if risk_probability is not None:

            st.metric(
                label="Heart Disease Risk Score",
                value=f"{risk_probability:.2f}%"
            )

            st.progress(
                min(risk_probability / 100, 1.0)
            )

            st.write(
                f"The model estimates a **{risk_probability:.2f}%**
                probability for the positive class."
            )

    # =========================
    # Patient Summary
    # =========================

    st.subheader("📋 Patient Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.write(f"**Age:** {age}")
        st.write(f"**Sex:** {'Male' if sex == 1 else 'Female'}")
        st.write(f"**Cholesterol:** {chol}")

    with summary_col2:
        st.write(f"**Blood Pressure:** {trestbps}")
        st.write(f"**Maximum Heart Rate:** {thalach}")
        st.write(f"**Chest Pain Type:** {cp}")

    with summary_col3:
        st.write(f"**Exercise Angina:** {'Yes' if exang == 1 else 'No'}")
        st.write(f"**Oldpeak:** {oldpeak}")
        st.write(f"**Major Vessels:** {ca}")

    st.warning(
        "⚠️ This prediction is generated by a Machine Learning model "
        "for educational and research purposes only. "
        "It is not a medical diagnosis."
    )
