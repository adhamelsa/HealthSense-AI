import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HealthSense AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
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
# SESSION STATE
# =========================================================

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "risk_probability" not in st.session_state:
    st.session_state.risk_probability = None

if "patient_data" not in st.session_state:
    st.session_state.patient_data = None


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("❤️ HealthSense AI")

    st.caption(
        "Intelligent Heart Disease Risk Assessment"
    )

    st.divider()

    st.subheader("🧭 Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Prediction",
            "🧠 Explainable AI",
            "📊 Model Performance",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write("**Algorithm:** Random Forest")
    st.write("**Framework:** Scikit-learn")
    st.write("**Interface:** Streamlit")

    st.divider()

    st.success("🟢 Model Online")

    st.caption(
        "HealthSense AI v2.0"
    )

    st.divider()

    st.warning(
        "For educational and research purposes only. "
        "This application is not a medical diagnosis."
    )


# =========================================================
# HEADER
# =========================================================

st.title("❤️ HealthSense AI")

st.subheader(
    "Intelligent Heart Disease Risk Assessment"
)

st.write(
    "An AI-powered machine learning application designed "
    "to estimate heart disease risk from clinical patient data."
)

header1, header2, header3, header4 = st.columns(4)

with header1:
    st.success("🟢 System Online")

with header2:
    st.info("🌲 Random Forest")

with header3:
    st.info("⚡ Streamlit")

with header4:
    st.success("🤖 AI Powered")

st.divider()


# =========================================================
# PREDICTION PAGE
# =========================================================

if page == "🏠 Prediction":

    st.header("🩺 Patient Risk Assessment")

    st.write(
        "Enter the patient's clinical information and "
        "click **Analyze Heart Health** to generate a prediction."
    )

    st.divider()

    # =====================================================
    # BASIC INFORMATION
    # =====================================================

    st.subheader("👤 Basic Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=50
        )

    with col2:

        sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x:
            "Female" if x == 0 else "Male"
        )

    with col3:

        cp = st.selectbox(
            "Chest Pain Type",
            [0, 1, 2, 3]
        )

    st.divider()

    # =====================================================
    # CLINICAL MEASUREMENTS
    # =====================================================

    st.subheader("❤️ Clinical Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:

        trestbps = st.number_input(
            "Resting Blood Pressure",
            min_value=50,
            max_value=250,
            value=120
        )

    with col2:

        chol = st.number_input(
            "Cholesterol",
            min_value=100,
            max_value=600,
            value=200
        )

    with col3:

        thalach = st.number_input(
            "Maximum Heart Rate",
            min_value=50,
            max_value=250,
            value=150
        )

    st.divider()

    # =====================================================
    # CARDIAC INDICATORS
    # =====================================================

    st.subheader("🫀 Cardiac Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )

    with col2:

        restecg = st.selectbox(
            "Resting ECG",
            [0, 1, 2]
        )

    with col3:

        exang = st.selectbox(
            "Exercise Induced Angina",
            [0, 1],
            format_func=lambda x:
            "No" if x == 0 else "Yes"
        )

    with col4:

        slope = st.selectbox(
            "ST Slope",
            [0, 1, 2]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

    with col2:

        ca = st.selectbox(
            "Number of Major Vessels",
            [0, 1, 2, 3, 4]
        )

    with col3:

        thal = st.selectbox(
            "Thal",
            [0, 1, 2, 3]
        )

    st.divider()

    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    analyze = st.button(
        "🔍 ANALYZE HEART HEALTH",
        use_container_width=True
    )

    if analyze:

        # -------------------------------------------------
        # INPUT DATA
        # -------------------------------------------------

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

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        try:

            prediction = model.predict(
                input_data
            )[0]

            risk_probability = None

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                risk_probability = (
                    probabilities[1] * 100
                )

            st.session_state.prediction_done = True
            st.session_state.prediction = prediction
            st.session_state.risk_probability = risk_probability
            st.session_state.patient_data = input_data

        except Exception as e:

            st.error(
                f"❌ Prediction Error: {e}"
            )

            st.stop()


    # =====================================================
    # RESULT
    # =====================================================

    if st.session_state.prediction_done:

        st.divider()

        st.header("🎯 Prediction Result")

        prediction = st.session_state.prediction
        risk_probability = st.session_state.risk_probability

        # -------------------------------------------------
        # RISK CLASSIFICATION
        # -------------------------------------------------

        if risk_probability is not None:

            if risk_probability < 30:

                risk_level = "LOW RISK"
                risk_icon = "🟢"
                risk_message = (
                    "The model estimates a relatively "
                    "low probability of heart disease."
                )

            elif risk_probability < 60:

                risk_level = "MODERATE RISK"
                risk_icon = "🟡"
                risk_message = (
                    "The model estimates a moderate "
                    "probability of heart disease."
                )

            else:

                risk_level = "HIGH RISK"
                risk_icon = "🔴"
                risk_message = (
                    "The model estimates a relatively "
                    "high probability of heart disease."
                )

        else:

            risk_level = (
                "POSITIVE"
                if prediction == 1
                else "NEGATIVE"
            )

            risk_icon = (
                "🔴"
                if prediction == 1
                else "🟢"
            )

            risk_message = (
                "Prediction generated by the machine "
                "learning model."
            )


        # -------------------------------------------------
        # RESULT METRICS
        # -------------------------------------------------

        result1, result2, result3 = st.columns(3)

        with result1:

            if risk_probability is not None:

                st.metric(
                    "Risk Score",
                    f"{risk_probability:.2f}%"
                )

        with result2:

            st.metric(
                "Risk Level",
                f"{risk_icon} {risk_level}"
            )

        with result3:

            st.metric(
                "Model",
                "Random Forest"
            )


        st.progress(
            min(
                max(
                    (risk_probability or 0) / 100,
                    0.0
                ),
                1.0
            )
        )

        st.info(
            risk_message
        )


        # -------------------------------------------------
        # PREDICTION STATUS
        # -------------------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ Higher Risk of Heart Disease Detected"
            )

        else:

            st.success(
                "✅ Lower Risk of Heart Disease Detected"
            )


        st.divider()

        # -------------------------------------------------
        # PATIENT SUMMARY
        # -------------------------------------------------

        st.subheader("📋 Patient Summary")

        summary1, summary2, summary3, summary4 = st.columns(4)

        with summary1:

            st.metric(
                "Age",
                age
            )

        with summary2:

            st.metric(
                "Sex",
                "Male" if sex == 1 else "Female"
            )

        with summary3:

            st.metric(
                "Cholesterol",
                f"{chol} mg/dl"
            )

        with summary4:

            st.metric(
                "Max Heart Rate",
                thalach
            )


        st.write("")

        summary1, summary2, summary3, summary4 = st.columns(4)

        with summary1:

            st.write(
                f"**Blood Pressure:** {trestbps}"
            )

        with summary2:

            st.write(
                f"**Chest Pain Type:** {cp}"
            )

        with summary3:

            st.write(
                f"**Exercise Angina:** "
                f"{'Yes' if exang == 1 else 'No'}"
            )

        with summary4:

            st.write(
                f"**ST Depression:** {oldpeak}"
            )


        st.warning(
            "⚠️ This prediction is generated by a "
            "Machine Learning model for educational "
            "and research purposes only. It is not a "
            "medical diagnosis and should not replace "
            "professional medical advice."
        )


# =========================================================
# EXPLAINABLE AI
# =========================================================

elif page == "🧠 Explainable AI":

    st.header("🧠 Explainable AI")

    st.write(
        "Understand which clinical features had the greatest "
        "overall influence on the Random Forest model."
    )

    st.divider()

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

    try:

        importance_df = pd.DataFrame(
            {
                "Feature": features,
                "Importance": model.feature_importances_
            }
        )

    except AttributeError:

        st.error(
            "Feature importance is not available "
            "for the current model."
        )

        st.stop()


    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )


    # =====================================================
    # TOP FACTORS
    # =====================================================

    st.subheader("🔝 Top Model Drivers")

    top5 = importance_df.head(5).copy()

    top5["Feature"] = top5[
        "Feature"
    ].map(
        feature_names
    )

    top5["Importance"] = (
        top5["Importance"] * 100
    ).round(2)


    driver_cols = st.columns(5)

    for i, row in top5.reset_index(drop=True).iterrows():

        with driver_cols[i]:

            st.metric(
                row["Feature"],
                f"{row['Importance']:.2f}%"
            )


    st.divider()


    # =====================================================
    # FEATURE IMPORTANCE CHART
    # =====================================================

    st.subheader(
        "📊 Feature Importance"
    )

    chart_df = importance_df.copy()

    chart_df["Feature"] = chart_df[
        "Feature"
    ].map(
        feature_names
    )

    chart_df = chart_df.sort_values(
        "Importance",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        chart_df["Feature"],
        chart_df["Importance"]
    )

    ax.set_xlabel(
        "Importance"
    )

    ax.set_ylabel(
        "Clinical Feature"
    )

    ax.set_title(
        "Random Forest Feature Importance"
    )

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)


    st.info(
        "💡 Feature importance describes the model's "
        "overall use of each feature. It does not establish "
        "causation and should not be interpreted as an "
        "individual medical risk factor."
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📊 Model Performance":

    st.header("📊 Model Performance")

    st.write(
        "Evaluation results obtained from the tuned "
        "Random Forest model on the clean test dataset."
    )

    st.divider()


    # =====================================================
    # KPI CARDS
    # =====================================================

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:

        st.metric(
            "Accuracy",
            "78.69%"
        )

    with col2:

        st.metric(
            "Precision",
            "81.25%"
        )

    with col3:

        st.metric(
            "Recall",
            "78.79%"
        )

    with col4:

        st.metric(
            "F1-Score",
            "80.00%"
        )

    with col5:

        st.metric(
            "ROC-AUC",
            "85.93%"
        )


    st.divider()


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader(
        "🔢 Confusion Matrix"
    )

    cm = np.array(
        [
            [22, 6],
            [7, 26]
        ]
    )

    cm_col1, cm_col2 = st.columns(
        [2, 1]
    )

    with cm_col1:

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "No Disease",
                "Heart Disease"
            ]
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False
        )

        ax.set_title(
            "Confusion Matrix"
        )

        st.pyplot(fig)

        plt.close(fig)


    with cm_col2:

        st.metric(
            "Correct Predictions",
            "48"
        )

        st.metric(
            "Incorrect Predictions",
            "13"
        )

        st.metric(
            "Test Samples",
            "61"
        )


    st.divider()


    # =====================================================
    # METRIC EXPLANATION
    # =====================================================

    st.subheader(
        "📈 What Do These Metrics Mean?"
    )

    with st.expander(
        "Accuracy"
    ):

        st.write(
            "The percentage of all test samples that "
            "were classified correctly."
        )

    with st.expander(
        "Precision"
    ):

        st.write(
            "Among cases predicted as heart disease, "
            "the percentage that were actually positive."
        )

    with st.expander(
        "Recall"
    ):

        st.write(
            "The percentage of actual heart disease cases "
            "that were correctly detected."
        )

    with st.expander(
        "F1-Score"
    ):

        st.write(
            "The harmonic mean of precision and recall, "
            "providing a balanced measure."
        )

    with st.expander(
        "ROC-AUC"
    ):

        st.write(
            "Measures how effectively the model separates "
            "the two classes across different thresholds."
        )


    st.info(
        "These evaluation results were obtained after "
        "removing duplicate records from the dataset "
        "to reduce potential data leakage and provide "
        "a more realistic estimate of model performance."
    )


# =========================================================
# ABOUT PROJECT
# =========================================================

elif page == "ℹ️ About Project":

    st.header("ℹ️ About HealthSense AI")

    st.write(
        """
        **HealthSense AI** is a machine learning application
        developed to demonstrate how artificial intelligence
        can be applied to cardiovascular risk assessment.
        """
    )

    st.divider()


    # =====================================================
    # PROJECT PIPELINE
    # =====================================================

    st.subheader(
        "🔬 Machine Learning Pipeline"
    )

    pipeline1, pipeline2, pipeline3, pipeline4, pipeline5 = st.columns(5)

    with pipeline1:

        st.info(
            "📥\n\n"
            "**Patient Data**"
        )

    with pipeline2:

        st.info(
            "🧹\n\n"
            "**Data Preparation**"
        )

    with pipeline3:

        st.info(
            "🌲\n\n"
            "**Random Forest**"
        )

    with pipeline4:

        st.info(
            "🎯\n\n"
            "**Prediction**"
        )

    with pipeline5:

        st.info(
            "🧠\n\n"
            "**Explanation**"
        )


    st.divider()


    # =====================================================
    # TECHNOLOGIES
    # =====================================================

    st.subheader(
        "🛠️ Technologies"
    )

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:

        st.metric(
            "Language",
            "Python"
        )

    with tech2:

        st.metric(
            "ML",
            "Scikit-learn"
        )

    with tech3:

        st.metric(
            "Data",
            "Pandas / NumPy"
        )

    with tech4:

        st.metric(
            "Deployment",
            "Streamlit"
        )


    st.divider()


    # =====================================================
    # PROJECT FEATURES
    # =====================================================

    st.subheader(
        "✨ Key Features"
    )

    st.write(
        """
        - ❤️ Heart disease risk prediction
        - 🎯 Probability-based risk estimation
        - 🌲 Random Forest machine learning model
        - 🧠 Feature importance for explainability
        - 📊 Accuracy, Precision, Recall and F1-Score
        - 📈 ROC-AUC evaluation
        - 🔢 Confusion Matrix
        - 📋 Patient summary
        - ☁️ Streamlit Cloud deployment
        """
    )


    st.divider()


    st.subheader(
        "⚠️ Important Disclaimer"
    )

    st.warning(
        "HealthSense AI is an educational and research "
        "project. Its predictions are not medical diagnoses "
        "and should not be used as a substitute for professional "
        "medical advice."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "❤️ HealthSense AI v2.0"
)

st.caption(
    "Python • Pandas • NumPy • Scikit-learn • "
    "Joblib • Matplotlib • Streamlit"
)

st.caption(
    "Educational & Research Project"
)
