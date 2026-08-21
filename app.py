import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HealthSense AI",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
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

st.subheader(
    "Intelligent Heart Disease Risk Assessment"
)

st.write(
    "An AI-powered machine learning application that "
    "analyzes clinical patient information and estimates "
    "the likelihood of heart disease."
)

header_col1, header_col2, header_col3, header_col4 = st.columns(4)

with header_col1:
    st.info("✦ AI-POWERED")

with header_col2:
    st.success("🌲 RANDOM FOREST")

with header_col3:
    st.info("⚡ STREAMLIT")

with header_col4:
    st.success("🟢 MODEL ONLINE")

st.divider()


# =========================================================
# PATIENT ASSESSMENT
# =========================================================

st.header("🩺 Patient Assessment")

st.write(
    "Enter the patient's clinical information below. "
    "The trained machine learning model will analyze "
    "the information and generate a risk estimate."
)


# =========================================================
# BASIC INFORMATION
# =========================================================

st.subheader("👤 Basic Information")

st.caption(
    "Demographic and primary patient information."
)

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
        options=[0, 1],
        format_func=lambda x:
        "Female" if x == 0 else "Male"
    )

with col3:

    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3]
    )


st.divider()


# =========================================================
# CLINICAL MEASUREMENTS
# =========================================================

st.subheader("❤️ Clinical Measurements")

st.caption(
    "Cardiovascular measurements and vital indicators."
)

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


# =========================================================
# CARDIAC ASSESSMENT
# =========================================================

st.subheader("🫀 Cardiac Assessment")

st.caption(
    "Additional clinical indicators used by the model."
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120",
        options=[0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )

with col2:

    restecg = st.selectbox(
        "Resting ECG",
        options=[0, 1, 2]
    )

with col3:

    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x:
        "No" if x == 0 else "Yes"
    )

with col4:

    slope = st.selectbox(
        "ST Slope",
        options=[0, 1, 2]
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
        options=[0, 1, 2, 3, 4]
    )

with col3:

    thal = st.selectbox(
        "Thal",
        options=[0, 1, 2, 3]
    )


st.divider()


# =========================================================
# PREDICTION BUTTON
# =========================================================

predict_button = st.button(
    "🔍 ANALYZE HEART HEALTH",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # =====================================================
    # CREATE INPUT DATA
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

    except Exception as e:

        st.error(
            f"❌ Prediction Error: {e}"
        )

        st.stop()


    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.divider()

    st.header("📊 Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ Higher Risk of Heart Disease Detected"
        )

    else:

        st.success(
            "✅ Lower Risk of Heart Disease Detected"
        )


    # =====================================================
    # RISK SCORE
    # =====================================================

    if risk_probability is not None:

        st.metric(
            label="Heart Disease Risk Score",
            value=f"{risk_probability:.2f}%"
        )

        st.progress(
            min(
                max(
                    risk_probability / 100,
                    0.0
                ),
                1.0
            )
        )

        st.caption(
            f"The model estimates a "
            f"{risk_probability:.2f}% probability "
            f"for the positive class."
        )


    # =====================================================
    # RESULT TABS
    # =====================================================

    tab_patient, tab_xai, tab_performance = st.tabs(
        [
            "📋 Patient Summary",
            "🧠 Explainable AI",
            "📊 Model Performance"
        ]
    )


    # =====================================================
    # TAB 1 — PATIENT SUMMARY
    # =====================================================

    with tab_patient:

        st.subheader(
            "📋 Patient Summary"
        )

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            st.write(
                f"**Age:** {age}"
            )

            st.write(
                f"**Sex:** "
                f"{'Male' if sex == 1 else 'Female'}"
            )

            st.write(
                f"**Cholesterol:** {chol}"
            )

        with summary_col2:

            st.write(
                f"**Blood Pressure:** "
                f"{trestbps}"
            )

            st.write(
                f"**Maximum Heart Rate:** "
                f"{thalach}"
            )

            st.write(
                f"**Chest Pain Type:** "
                f"{cp}"
            )

        with summary_col3:

            st.write(
                f"**Exercise Angina:** "
                f"{'Yes' if exang == 1 else 'No'}"
            )

            st.write(
                f"**ST Depression:** "
                f"{oldpeak}"
            )

            st.write(
                f"**Major Vessels:** "
                f"{ca}"
            )


        st.warning(
            "⚠️ This prediction is generated by a "
            "Machine Learning model for educational "
            "and research purposes only. It is not a "
            "medical diagnosis and should not replace "
            "professional medical advice."
        )


    # =====================================================
    # TAB 2 — EXPLAINABLE AI
    # =====================================================

    with tab_xai:

        st.subheader(
            "🧠 Explainable AI"
        )

        st.write(
            "These are the clinical features that had "
            "the greatest overall influence on the "
            "Random Forest model."
        )


        # -------------------------------------------------
        # FEATURES
        # -------------------------------------------------

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

            "age":
            "Age",

            "sex":
            "Sex",

            "cp":
            "Chest Pain Type",

            "trestbps":
            "Resting Blood Pressure",

            "chol":
            "Cholesterol",

            "fbs":
            "Fasting Blood Sugar",

            "restecg":
            "Resting ECG",

            "thalach":
            "Maximum Heart Rate",

            "exang":
            "Exercise Induced Angina",

            "oldpeak":
            "ST Depression",

            "slope":
            "ST Slope",

            "ca":
            "Major Vessels",

            "thal":
            "Thal"
        }


        # -------------------------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------------------------

        try:

            feature_importances = (
                model.feature_importances_
            )

        except AttributeError:

            st.warning(
                "Feature importance is not available "
                "for this model."
            )

            feature_importances = None


        if feature_importances is not None:

            importance_df = pd.DataFrame(
                {
                    "Feature": features,
                    "Importance": feature_importances
                }
            )


            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            )


            # ---------------------------------------------
            # TOP 5
            # ---------------------------------------------

            st.markdown(
                "### 🔝 Top 5 Important Factors"
            )

            top5 = importance_df.head(
                5
            ).copy()


            top5["Feature"] = top5[
                "Feature"
            ].map(
                feature_names
            )


            top5["Importance (%)"] = (
                top5["Importance"] * 100
            ).round(2)


            top5 = top5[
                [
                    "Feature",
                    "Importance (%)"
                ]
            ]


            st.dataframe(
                top5,
                use_container_width=True,
                hide_index=True
            )


            # ---------------------------------------------
            # CHART
            # ---------------------------------------------

            st.markdown(
                "### 📊 Feature Importance Chart"
            )

            chart_df = importance_df.copy()


            chart_df["Feature"] = chart_df[
                "Feature"
            ].map(
                feature_names
            )


            chart_df = chart_df.sort_values(
                by="Importance",
                ascending=True
            )


            fig, ax = plt.subplots(
                figsize=(9, 6)
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


            st.pyplot(
                fig
            )


            plt.close(
                fig
            )


            st.info(
                "💡 Feature importance shows how strongly "
                "each feature contributed to the Random "
                "Forest model's decisions overall. It does "
                "not mean that a feature directly causes "
                "heart disease."
            )


    # =====================================================
    # TAB 3 — MODEL PERFORMANCE
    # =====================================================

    with tab_performance:

        st.subheader(
            "📊 Model Performance"
        )

        st.write(
            "The following metrics were obtained by "
            "evaluating the tuned Random Forest model "
            "on the clean test dataset."
        )


        # -------------------------------------------------
        # PERFORMANCE METRICS
        # -------------------------------------------------

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


        # -------------------------------------------------
        # CONFUSION MATRIX
        # -------------------------------------------------

        st.subheader(
            "🔢 Confusion Matrix"
        )


        cm = np.array(
            [
                [22, 6],
                [7, 26]
            ]
        )


        fig_cm, ax_cm = plt.subplots(
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
            ax=ax_cm,

            cmap="Blues",

            colorbar=False
        )


        ax_cm.set_title(
            "Confusion Matrix"
        )


        st.pyplot(
            fig_cm
        )


        plt.close(
            fig_cm
        )


        st.caption(
            "The model correctly classified "
            "48 out of 61 test samples."
        )


        # -------------------------------------------------
        # EVALUATION SUMMARY
        # -------------------------------------------------

        st.subheader(
            "📈 Model Evaluation Summary"
        )


        st.write(
            """
            **Accuracy (78.69%)**  
            Overall percentage of correct predictions.

            **Precision (81.25%)**  
            When the model predicts heart disease,
            how often the prediction is correct.

            **Recall (78.79%)**  
            Percentage of actual positive cases
            correctly detected.

            **F1-Score (80.00%)**  
            Balance between precision and recall.

            **ROC-AUC (85.93%)**  
            Ability of the model to distinguish
            between the two classes.
            """
        )


        st.info(
            "These evaluation results were obtained after "
            "removing duplicate records from the dataset "
            "to reduce potential data leakage and provide "
            "a more realistic estimate of model performance."
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "❤️ HealthSense AI | Heart Disease Prediction"
)

st.caption(
    "Built with Python • Pandas • NumPy • Scikit-learn • "
    "Joblib • Matplotlib • Streamlit"
)

st.caption(
    "⚠️ For educational and research purposes only. "
    "This application is not a medical diagnosis."
)
