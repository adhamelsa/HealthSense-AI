import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HealthSense AI",
    page_icon="❤️",
    layout="wide"
)

# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_heart_model.pkl"
)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Could not load the model: {e}")
    st.stop()

# =========================================================
# HEADER
# =========================================================

st.title("❤️ HealthSense AI")
st.subheader("Heart Disease Prediction System")

st.write(
    """
    Welcome to **HealthSense AI**.

    This application uses a Machine Learning model to estimate
    the risk of heart disease based on clinical patient data.
    """
)

st.divider()

# =========================================================
# PATIENT INFORMATION
# =========================================================

st.subheader("👤 Patient Information")

col1, col2, col3 = st.columns(3)

# -------------------------
# Column 1
# -------------------------

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

# -------------------------
# Column 2
# -------------------------

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

# -------------------------
# Column 3
# -------------------------

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

# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔍 Predict Heart Disease Risk",
    use_container_width=True
):

    # =====================================================
    # CREATE INPUT DATAFRAME
    # =====================================================

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
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]
    )

    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    try:

        prediction = model.predict(input_data)[0]

        # Get probability if supported
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]

            risk_probability = probabilities[1] * 100

        else:

            risk_probability = None

    except Exception as e:

        st.error(f"❌ Prediction Error: {e}")
        st.stop()

    # =====================================================
    # RESULT
    # =====================================================

    st.divider()

    st.subheader("🩺 Prediction Result")

    # =====================================================
    # HIGH RISK
    # =====================================================

    if prediction == 1:

        st.error(
            "⚠️ Higher Risk of Heart Disease Detected"
        )

        if risk_probability is not None:

            st.metric(
                label="Heart Disease Risk Score",
                value=f"{risk_probability:.2f}%"
            )

            st.progress(
                min(max(risk_probability / 100, 0.0), 1.0)
            )

            st.write(
                f"The model estimates a **{risk_probability:.2f}%** "
                "probability for the positive class."
            )

    # =====================================================
    # LOW RISK
    # =====================================================

    else:

        st.success(
            "✅ Lower Risk of Heart Disease Detected"
        )

        if risk_probability is not None:

            st.metric(
                label="Heart Disease Risk Score",
                value=f"{risk_probability:.2f}%"
            )

            st.progress(
                min(max(risk_probability / 100, 0.0), 1.0)
            )

            st.write(
                f"The model estimates a **{risk_probability:.2f}%** "
                "probability for the positive class."
            )

    # =====================================================
    # PATIENT SUMMARY
    # =====================================================

    st.divider()

    st.subheader("📋 Patient Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    # -------------------------
    # Summary Column 1
    # -------------------------

    with summary_col1:

        st.write(f"**Age:** {age}")

        st.write(
            f"**Sex:** {'Male' if sex == 1 else 'Female'}"
        )

        st.write(f"**Cholesterol:** {chol}")

    # -------------------------
    # Summary Column 2
    # -------------------------

    with summary_col2:

        st.write(
            f"**Blood Pressure:** {trestbps}"
        )

        st.write(
            f"**Maximum Heart Rate:** {thalach}"
        )

        st.write(
            f"**Chest Pain Type:** {cp}"
        )

    # -------------------------
    # Summary Column 3
    # -------------------------

    with summary_col3:

        st.write(
            f"**Exercise Angina:** {'Yes' if exang == 1 else 'No'}"
        )

        st.write(
            f"**ST Depression:** {oldpeak}"
        )

        st.write(
            f"**Major Vessels:** {ca}"
        )

    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.warning(
        "⚠️ This prediction is generated by a Machine Learning "
        "model for educational and research purposes only. "
        "It is not a medical diagnosis and should not replace "
        "professional medical advice."
    )


# =========================================================
# EXPLAINABLE AI - FEATURE IMPORTANCE
# =========================================================

st.divider()

st.subheader("🧠 Explainable AI")

st.write(
    "These are the clinical features that had the greatest "
    "overall influence on the Random Forest model."
)

features = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal"
]

feature_names = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Slope",
    "ca": "Major Vessels",
    "thal": "Thal"
}

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

# =========================================================
# MODEL PERFORMANCE DASHBOARD
# =========================================================

st.divider()

st.subheader("📊 Model Performance")

st.write(
    "The following metrics were obtained by evaluating the "
    "tuned Random Forest model on the clean test dataset."
)

# Performance metrics
metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

metric_col1.metric(
    "Accuracy",
    "78.69%"
)

metric_col2.metric(
    "Precision",
    "81.25%"
)

metric_col3.metric(
    "Recall",
    "78.79%"
)

metric_col4.metric(
    "F1-Score",
    "80.00%"
)

metric_col5.metric(
    "ROC-AUC",
    "85.93%"
)

st.divider()

# =========================================================
# CONFUSION MATRIX
# =========================================================

st.subheader("🔢 Confusion Matrix")

# Results from the clean test set
cm = [
    [22, 6],
    [7, 26]
]

fig_cm, ax_cm = plt.subplots(figsize=(6, 5))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Heart Disease"]
)

display.plot(
    ax=ax_cm,
    cmap="Blues",
    colorbar=False
)

ax_cm.set_title("Confusion Matrix")

st.pyplot(fig_cm)

st.caption(
    "The model correctly classified 48 out of 61 test samples."
)

# =========================================================
# PERFORMANCE EXPLANATION
# =========================================================

st.subheader("📈 Model Evaluation Summary")

st.write(
    """
    - **Accuracy (78.69%)**: Overall percentage of correct predictions.
    - **Precision (81.25%)**: When the model predicts heart disease, how often the prediction is correct.
    - **Recall (78.79%)**: Percentage of actual positive cases correctly detected.
    - **F1-Score (80.00%)**: Balance between precision and recall.
    - **ROC-AUC (85.93%)**: Ability of the model to distinguish between the two classes.
    """
)

st.info(
    "These evaluation results were obtained after removing duplicate "
    "records from the dataset to reduce data leakage and provide a "
    "more realistic estimate of model performance."
)
# =========================================================
# TOP 5 FEATURES
# =========================================================

st.markdown("### 🔝 Top 5 Important Factors")

top5 = importance_df.head(5).copy()

top5["Feature"] = top5["Feature"].map(feature_names)

top5["Importance (%)"] = (
    top5["Importance"] * 100
).round(2)

top5 = top5[
    ["Feature", "Importance (%)"]
]

st.dataframe(
    top5,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# FEATURE IMPORTANCE CHART
# =========================================================

st.markdown("### 📊 Feature Importance Chart")

chart_df = importance_df.copy()

chart_df["Feature"] = chart_df["Feature"].map(
    feature_names
)

chart_df = chart_df.sort_values(
    by="Importance",
    ascending=True
)

fig, ax = plt.subplots(figsize=(9, 6))

ax.barh(
    chart_df["Feature"],
    chart_df["Importance"]
)

ax.set_xlabel("Importance")
ax.set_ylabel("Clinical Feature")
ax.set_title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

st.pyplot(fig)

# =========================================================
# EXPLANATION NOTE
# =========================================================

st.info(
    "💡 Feature importance shows how strongly each feature "
    "contributed to the Random Forest model's decisions overall. "
    "It does not mean that a feature directly causes heart disease "
    "or represents an individual patient's medical risk by itself."
)
