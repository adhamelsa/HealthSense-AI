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

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
    ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(20, 184, 166, 0.08),
                transparent 35%
            ),
            radial-gradient(
                circle at bottom left,
                rgba(37, 99, 235, 0.07),
                transparent 35%
            ),
            #f8fafc;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =====================================================
       HERO
    ===================================================== */

    .hero {
        background:
            linear-gradient(
                135deg,
                #020617 0%,
                #0f172a 45%,
                #134e4a 100%
            );

        border-radius: 28px;
        padding: 2.4rem;
        margin-bottom: 2rem;

        box-shadow:
            0 20px 50px rgba(15, 23, 42, 0.18);

        color: white;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 850;
        letter-spacing: -1px;
        line-height: 1.1;
    }

    .hero-subtitle {
        font-size: 1.15rem;
        color: rgba(255,255,255,0.75);
        margin-top: 0.6rem;
    }

    .hero-description {
        max-width: 700px;
        margin-top: 1rem;
        color: rgba(255,255,255,0.62);
        line-height: 1.6;
    }

    .badge {
        display: inline-block;
        padding: 0.45rem 0.8rem;
        margin-top: 1.2rem;
        margin-right: 0.4rem;

        border-radius: 999px;

        background: rgba(255,255,255,0.09);
        border: 1px solid rgba(255,255,255,0.14);

        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .status-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.14);

        border-radius: 18px;

        padding: 1.2rem;

        text-align: center;

        margin-top: 0.5rem;
    }

    .status-title {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.55);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .status-value {
        margin-top: 0.4rem;
        font-size: 1.1rem;
        font-weight: 750;
    }


    /* =====================================================
       SECTION
    ===================================================== */

    .section-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 1rem;
    }

    .section-description {
        color: #64748b;
        margin-bottom: 1.4rem;
        line-height: 1.6;
    }


    /* =====================================================
       INPUT CARDS
    ===================================================== */

    .input-card {
        background: white;

        padding: 1.35rem;

        border-radius: 20px;

        border: 1px solid #e2e8f0;

        box-shadow:
            0 8px 25px rgba(15, 23, 42, 0.045);

        margin-bottom: 1.2rem;
    }

    .input-card-title {
        font-size: 1.05rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }

    .input-card-description {
        font-size: 0.82rem;
        color: #64748b;
        margin-bottom: 1rem;
    }


    /* =====================================================
       BUTTON
    ===================================================== */

    div.stButton > button {

        height: 3.5rem;

        border-radius: 16px;

        border: none;

        background:
            linear-gradient(
                135deg,
                #0f766e,
                #0f4c81
            );

        color: white;

        font-size: 1.05rem;

        font-weight: 800;

        letter-spacing: 0.2px;

        box-shadow:
            0 10px 25px rgba(15,118,110,0.22);

        transition: all 0.2s ease;
    }

    div.stButton > button:hover {

        transform: translateY(-2px);

        box-shadow:
            0 14px 30px rgba(15,118,110,0.30);
    }


    /* =====================================================
       RESULT CARD
    ===================================================== */

    .result-card {

        background: white;

        border-radius: 26px;

        padding: 2.3rem;

        text-align: center;

        border: 1px solid #e2e8f0;

        box-shadow:
            0 18px 45px rgba(15,23,42,0.08);

        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .result-caption {

        color: #64748b;

        font-size: 0.78rem;

        font-weight: 800;

        text-transform: uppercase;

        letter-spacing: 1.5px;
    }

    .risk-score {

        font-size: 4rem;

        font-weight: 900;

        color: #0f172a;

        line-height: 1.1;

        margin-top: 0.5rem;
    }

    .risk-label {

        font-size: 1.3rem;

        font-weight: 800;

        margin-top: 0.5rem;
    }


    /* =====================================================
       METRICS
    ===================================================== */

    .metric-card {

        background: white;

        border-radius: 18px;

        padding: 1.25rem;

        text-align: center;

        border: 1px solid #e2e8f0;

        box-shadow:
            0 6px 20px rgba(15,23,42,0.05);
    }

    .metric-label {

        color: #64748b;

        font-size: 0.78rem;

        font-weight: 700;

        text-transform: uppercase;

        letter-spacing: 0.4px;
    }

    .metric-value {

        color: #0f172a;

        font-size: 1.75rem;

        font-weight: 900;

        margin-top: 0.3rem;
    }


    /* =====================================================
       FOOTER
    ===================================================== */

    .footer {

        text-align: center;

        color: #64748b;

        font-size: 0.82rem;

        margin-top: 3rem;

        padding-top: 1.5rem;

        border-top: 1px solid #e2e8f0;
    }


    /* =====================================================
       MOBILE
    ===================================================== */

    @media (max-width: 768px) {

        .hero-title {
            font-size: 2.1rem;
        }

        .hero {
            padding: 1.6rem;
        }

        .risk-score {
            font-size: 3rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
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

    st.error(
        f"❌ Could not load the model: {e}"
    )

    st.stop()


# =========================================================
# HERO HEADER
# =========================================================

hero_col1, hero_col2 = st.columns(
    [4, 1],
    vertical_alignment="center"
)

with hero_col1:

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                ❤️ HealthSense AI
            </div>

            <div class="hero-subtitle">
                Intelligent Heart Disease Risk Assessment
            </div>

            <div class="hero-description">
                An AI-powered machine learning application that
                analyzes clinical patient information and estimates
                the likelihood of heart disease.
            </div>

            <span class="badge">
                ✦ AI-POWERED
            </span>

            <span class="badge">
                🌲 RANDOM FOREST
            </span>

            <span class="badge">
                ⚡ STREAMLIT
            </span>

        </div>
        """,
        unsafe_allow_html=True
    )


with hero_col2:

    st.markdown(
        """
        <div class="hero">

            <div class="status-card">

                <div class="status-title">
                    Model Status
                </div>

                <div class="status-value">
                    🟢 Online
                </div>

            </div>

            <div class="status-card">

                <div class="status-title">
                    Model Type
                </div>

                <div class="status-value">
                    Random Forest
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# INTRODUCTION
# =========================================================

st.markdown(
    """
    <div class="section-title">
        🩺 Patient Assessment
    </div>

    <div class="section-description">
        Enter the patient's clinical information below.
        The trained machine learning model will analyze the
        information and generate a risk estimate.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# BASIC INFORMATION
# =========================================================

st.markdown(
    """
    <div class="input-card">

        <div class="input-card-title">
            👤 Basic Information
        </div>

        <div class="input-card-description">
            Demographic and primary patient information
        </div>

    </div>
    """,
    unsafe_allow_html=True
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


# =========================================================
# CLINICAL MEASUREMENTS
# =========================================================

st.markdown(
    """
    <div class="input-card">

        <div class="input-card-title">
            ❤️ Clinical Measurements
        </div>

        <div class="input-card-description">
            Cardiovascular measurements and vital indicators
        </div>

    </div>
    """,
    unsafe_allow_html=True
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


# =========================================================
# CARDIAC ASSESSMENT
# =========================================================

st.markdown(
    """
    <div class="input-card">

        <div class="input-card-title">
            🫀 Cardiac Assessment
        </div>

        <div class="input-card-description">
            Additional clinical indicators used by the model
        </div>

    </div>
    """,
    unsafe_allow_html=True
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


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.write("")

predict_button = st.button(
    "🔍  ANALYZE HEART HEALTH",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # -----------------------------------------------------
    # CREATE INPUT DATA
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

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
    # RESULT
    # =====================================================

    st.divider()

    st.markdown(
        """
        <div class="section-title">
            📊 Prediction Result
        </div>

        <div class="section-description">
            AI-generated risk estimation based on the provided
            clinical information.
        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # RESULT STATUS
    # -----------------------------------------------------

    if prediction == 1:

        result_icon = "⚠️"
        result_label = "Higher Risk of Heart Disease Detected"

    else:

        result_icon = "✅"
        result_label = "Lower Risk of Heart Disease Detected"


    # -----------------------------------------------------
    # RESULT CARD
    # -----------------------------------------------------

    if risk_probability is not None:

        st.markdown(
            f"""
            <div class="result-card">

                <div class="result-caption">
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
    # PATIENT SUMMARY
    # =====================================================

    with tab_patient:

        st.markdown(
            "### 📋 Patient Summary"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

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

        with col2:

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

        with col3:

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
            "⚠️ This prediction is generated by a Machine "
            "Learning model for educational and research "
            "purposes only. It is not a medical diagnosis "
            "and should not replace professional medical advice."
        )


    # =====================================================
    # EXPLAINABLE AI
    # =====================================================

    with tab_xai:

        st.markdown(
            "### 🧠 Explainable AI"
        )

        st.write(
            "Feature importance shows how strongly each "
            "clinical feature contributed to the Random "
            "Forest model's decisions overall."
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
        # IMPORTANCE DATA
        # -------------------------------------------------

        importance_df = pd.DataFrame(
            {
                "Feature":
                    features,

                "Importance":
                    model.feature_importances_
            }
        )


        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )


        # -------------------------------------------------
        # TOP 5
        # -------------------------------------------------

        st.markdown(
            "#### 🔝 Top 5 Important Factors"
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


        # -------------------------------------------------
        # CHART
        # -------------------------------------------------

        st.markdown(
            "#### 📊 Feature Importance Chart"
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
            "💡 Feature importance represents the overall "
            "contribution of each feature to the model's "
            "decisions. It does not mean that a feature "
            "directly causes heart disease or represents "
            "an individual's medical risk by itself."
        )


    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    with tab_performance:

        st.markdown(
            "### 📊 Model Performance"
        )

        st.write(
            "Evaluation results obtained from the tuned "
            "Random Forest model on the clean test dataset."
        )


        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)


        metrics = [

            (
                "Accuracy",
                "78.69%"
            ),

            (
                "Precision",
                "81.25%"
            ),

            (
                "Recall",
                "78.79%"
            ),

            (
                "F1-Score",
                "80.00%"
            ),

            (
                "ROC-AUC",
                "85.93%"
            )

        ]


        metric_columns = [

            metric_col1,
            metric_col2,
            metric_col3,
            metric_col4,
            metric_col5

        ]


        for column, (
            label,
            value
        ) in zip(
            metric_columns,
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


        # -------------------------------------------------
        # CONFUSION MATRIX
        # -------------------------------------------------

        st.markdown(
            "#### 🔢 Confusion Matrix"
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
        # METRIC EXPLANATION
        # -------------------------------------------------

        st.markdown(
            "#### 📈 Evaluation Summary"
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

st.markdown(
    """
    <div class="footer">

        <strong>❤️ HealthSense AI</strong>

        <br>

        Intelligent Heart Disease Risk Assessment

        <br><br>

        Built with Python • Scikit-learn • Streamlit

        <br><br>

        ⚠️ Educational & Research Purposes Only

    </div>
    """,
    unsafe_allow_html=True
)
