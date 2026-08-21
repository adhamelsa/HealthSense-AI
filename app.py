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
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef6ff 50%,
            #f8fbff 100%
        );
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ---------- HEADER ---------- */

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 24px;
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #172554 50%,
            #0f766e 100%
        );
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(15, 23, 42, 0.18);
    }

    .hero-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        opacity: 0.85;
        margin-bottom: 1.2rem;
    }

    .badge {
        display: inline-block;
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        font-size: 0.85rem;
        font-weight: 600;
    }


    /* ---------- SECTION HEADERS ---------- */

    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        color: #0f172a;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
    }

    .section-description {
        color: #64748b;
        margin-bottom: 1.2rem;
    }


    /* ---------- CARDS ---------- */

    .info-card {
        background: white;
        padding: 1.3rem;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05);
        margin-bottom: 1rem;
    }


    /* ---------- RESULT ---------- */

    .result-card {
        padding: 2rem;
        border-radius: 24px;
        background: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 15px 35px rgba(15, 23, 42, 0.08);
        text-align: center;
        margin: 1.5rem 0;
    }

    .result-title {
        color: #64748b;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
    }

    .risk-score {
        font-size: 3.5rem;
        font-weight: 850;
        margin: 0.5rem 0;
        color: #0f172a;
    }

    .risk-label {
        font-size: 1.35rem;
        font-weight: 750;
        margin-bottom: 0.5rem;
    }


    /* ---------- METRIC CARDS ---------- */

    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 18px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 5px 18px rgba(15, 23, 42, 0.05);
    }

    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
    }

    .metric-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.25rem;
    }


    /* ---------- BUTTON ---------- */

    div.stButton > button {
        width: 100%;
        height: 3.2rem;
        border-radius: 14px;
        border: none;
        font-size: 1.05rem;
        font-weight: 750;
        background: linear-gradient(
            135deg,
            #0f766e,
            #0f4c81
        );
        color: white;
        box-shadow: 0 8px 20px rgba(15, 118, 110, 0.2);
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 25px rgba(15, 118, 110, 0.3);
    }


    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        padding-top: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #e2e8f0;
    }

</style>
""", unsafe_allow_html=True)


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
# HERO HEADER
# =========================================================

st.markdown("""
<div class="hero">

    <div class="hero-title">
        ❤️ HealthSense AI
    </div>

    <div class="hero-subtitle">
        Intelligent Heart Disease Risk Assessment
    </div>

    <span class="badge">
        ● AI-POWERED
    </span>

    &nbsp;

    <span class="badge">
        Random Forest
    </span>

    &nbsp;

    <span class="badge">
        Streamlit
    </span>

</div>
""", unsafe_allow_html=True)


# =========================================================
# INTRODUCTION
# =========================================================

st.markdown(
    '<div class="section-title">🩺 Patient Assessment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Enter the clinical information below to estimate the likelihood '
    'of heart disease using the trained machine learning model.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PATIENT INPUTS
# =========================================================

with st.container():

    st.markdown(
        '<div class="info-card">',
        unsafe_allow_html=True
    )

    st.markdown("### 👤 Basic Information")

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

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# CLINICAL MEASUREMENTS
# =========================================================

st.markdown(
    '<div class="info-card">',
    unsafe_allow_html=True
)

st.markdown("### ❤️ Clinical Measurements")

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

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# CARDIAC ASSESSMENT
# =========================================================

st.markdown(
    '<div class="info-card">',
    unsafe_allow_html=True
)

st.markdown("### 🫀 Cardiac Assessment")

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
        "Exercise Angina",
        options=[0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

with col4:

    slope = st.selectbox(
        "ST Slope",
        options=[0, 1, 2]
    )

col1, col2, col3, col4 = st.columns(4)

with col1:

    oldpeak = st.number_input(
        "ST Depression",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

with col2:

    ca = st.selectbox(
        "Major Vessels",
        options=[0, 1, 2, 3, 4]
    )

with col3:

    thal = st.selectbox(
        "Thal",
        options=[0, 1, 2, 3]
    )

with col4:

    st.write("")
    st.write("")

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

predict_button = st.button(
    "🔍 ANALYZE HEART HEALTH"
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

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

    try:

        prediction = model.predict(input_data)[0]

        risk_probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            risk_probability = probabilities[1] * 100

    except Exception as e:

        st.error(f"❌ Prediction Error: {e}")
        st.stop()


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Prediction Result</div>',
        unsafe_allow_html=True
    )

    if prediction == 1:

        result_label = "Higher Risk Detected"
        result_icon = "⚠️"

    else:

        result_label = "Lower Risk Detected"
        result_icon = "✅"


    if risk_probability is not None:

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-title">
                    Heart Disease Risk Score
                </div>

                <div class="risk-score">
                    {risk_probability:.2f}%
                </div>

                <div class="risk-label">
                    {result_icon} {result_label}
                </div>

            </div>
            """,
            unsafe_allow_html=True
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


    # =====================================================
    # RESULT TABS
    # =====================================================

    tab1, tab2, tab3 = st.tabs([
        "📋 Patient Summary",
        "🧠 Explainable AI",
        "📊 Model Performance"
    ])


    # =====================================================
    # TAB 1 - PATIENT SUMMARY
    # =====================================================

    with tab1:

        st.markdown("### Patient Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            st.write(f"**Age:** {age}")

            st.write(
                f"**Sex:** {'Male' if sex == 1 else 'Female'}"
            )

            st.write(
                f"**Cholesterol:** {chol}"
            )

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

        with summary_col3:

            st.write(
                f"**Exercise Angina:** "
                f"{'Yes' if exang == 1 else 'No'}"
            )

            st.write(
                f"**ST Depression:** {oldpeak}"
            )

            st.write(
                f"**Major Vessels:** {ca}"
            )


        st.warning(
            "⚠️ This prediction is generated by a Machine Learning "
            "model for educational and research purposes only. "
            "It is not a medical diagnosis and should not replace "
            "professional medical advice."
        )


    # =====================================================
    # TAB 2 - EXPLAINABLE AI
    # =====================================================

    with tab2:

        st.markdown("### 🧠 Explainable AI")

        st.write(
            "Feature importance shows which clinical features "
            "contributed most strongly to the Random Forest "
            "model's decisions overall."
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

        top5 = importance_df.head(5).copy()

        top5["Feature"] = top5["Feature"].map(
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

        st.markdown("#### 🔝 Top 5 Important Factors")

        st.dataframe(
            top5,
            use_container_width=True,
            hide_index=True
        )

        chart_df = importance_df.copy()

        chart_df["Feature"] = chart_df["Feature"].map(
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

        ax.set_xlabel("Importance")
        ax.set_ylabel("Clinical Feature")
        ax.set_title(
            "Random Forest Feature Importance"
        )

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.info(
            "Feature importance describes the overall contribution "
            "of each feature to the model. It does not mean that "
            "a feature directly causes heart disease."
        )


    # =====================================================
    # TAB 3 - MODEL PERFORMANCE
    # =====================================================

    with tab3:

        st.markdown("### 📊 Model Performance")

        st.write(
            "Evaluation results for the tuned Random Forest model "
            "on the clean test dataset."
        )

        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

        metrics = [
            ("Accuracy", "78.69%"),
            ("Precision", "81.25%"),
            ("Recall", "78.79%"),
            ("F1-Score", "80.00%"),
            ("ROC-AUC", "85.93%")
        ]

        columns = [
            metric_col1,
            metric_col2,
            metric_col3,
            metric_col4,
            metric_col5
        ]

        for column, (label, value) in zip(
            columns,
            metrics
        ):

            with column:

                st.markdown(
                    f"""
                    <div class="metric-card">

                        <div class="metric-label">
                            {label}
                        </div>

                        <div class="metric-value">
                            {value}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


        st.write("")

        st.markdown("#### 🔢 Confusion Matrix")

        cm = np.array([
            [22, 6],
            [7, 26]
        ])

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

        st.pyplot(fig_cm)

        plt.close(fig_cm)

        st.caption(
            "The model correctly classified 48 out of 61 test samples."
        )


        st.markdown("#### 📈 Metric Interpretation")

        st.write(
            """
            - **Accuracy:** Overall percentage of correct predictions.
            - **Precision:** When the model predicts heart disease, how often the prediction is correct.
            - **Recall:** Percentage of actual positive cases correctly detected.
            - **F1-Score:** Balance between precision and recall.
            - **ROC-AUC:** Ability of the model to distinguish between the two classes.
            """
        )

        st.info(
            "The evaluation results were obtained after removing "
            "duplicate records from the dataset to reduce potential "
            "data leakage and provide a more realistic estimate "
            "of model performance."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        <strong>HealthSense AI</strong> · Heart Disease Prediction

        <br><br>

        Built with Python, Scikit-learn & Streamlit

        <br><br>

        ⚠️ For educational and research purposes only.

    </div>
    """,
    unsafe_allow_html=True
)
