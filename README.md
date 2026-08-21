# ❤️ HealthSense AI – Heart Disease Prediction

**HealthSense AI** is a machine learning web application that predicts the likelihood of heart disease based on clinical patient data.

The application is built with **Python, Scikit-learn, and Streamlit** and uses a tuned **Random Forest Classifier** to generate predictions and risk probabilities.

## 🚀 Live Demo

**Try HealthSense AI:**
[Open the Streamlit App](https://healthsense-ai-iyulueugvqfwkgiy2v5ncc.streamlit.app/#health-sense-ai)

---

## 📌 Project Overview

Heart disease is a major healthcare challenge, and machine learning can help identify patterns associated with cardiovascular risk.

HealthSense AI provides an interactive interface where users can enter clinical patient information and receive a machine-learning-based prediction.

The application also provides:

* Risk probability
* Patient input summary
* Model performance metrics
* Confusion Matrix
* Random Forest feature importance
* Top 5 influential clinical features

> ⚠️ **Disclaimer:** This application is developed for educational and research purposes only. It is not a medical diagnosis tool and should not replace professional medical advice.

---

## ✨ Features

* ❤️ Heart disease risk prediction
* 📊 Probability-based risk score
* 👤 Interactive patient information form
* 🧠 Explainable AI using Random Forest feature importance
* 🔝 Top 5 important clinical factors
* 📈 Model performance dashboard
* 🔢 Confusion Matrix visualization
* 📋 Patient input summary
* ⚡ Interactive Streamlit interface
* ☁️ Streamlit Cloud deployment

---

## 🧠 Machine Learning Model

The application uses a **Tuned Random Forest Classifier** to classify patients into two classes:

| Class | Meaning          |
| ----- | ---------------- |
| `0`   | No Heart Disease |
| `1`   | Heart Disease    |

The trained model is stored in:

```text
best_heart_model.pkl
```

The application loads the trained model using `joblib`.

---

## 📊 Input Features

The model uses **13 clinical features**:

| Feature    | Description                 |
| ---------- | --------------------------- |
| `age`      | Patient age                 |
| `sex`      | Patient sex                 |
| `cp`       | Chest pain type             |
| `trestbps` | Resting blood pressure      |
| `chol`     | Cholesterol level           |
| `fbs`      | Fasting blood sugar         |
| `restecg`  | Resting ECG results         |
| `thalach`  | Maximum heart rate achieved |
| `exang`    | Exercise-induced angina     |
| `oldpeak`  | ST depression               |
| `slope`    | ST segment slope            |
| `ca`       | Number of major vessels     |
| `thal`     | Thal-related feature        |

---

## 📈 Model Performance

The tuned Random Forest model achieved the following results on the clean test dataset:

| Metric        |      Score |
| ------------- | ---------: |
| **Accuracy**  | **78.69%** |
| **Precision** | **81.25%** |
| **Recall**    | **78.79%** |
| **F1-Score**  | **80.00%** |
| **ROC-AUC**   | **85.93%** |

### 🔢 Confusion Matrix

|                           | Predicted: No Disease | Predicted: Heart Disease |
| ------------------------- | --------------------: | -----------------------: |
| **Actual: No Disease**    |                    22 |                        6 |
| **Actual: Heart Disease** |                     7 |                       26 |

The model correctly classified **48 out of 61 test samples**.

The evaluation was performed after removing duplicate records from the dataset to reduce potential data leakage and provide a more realistic estimate of model performance.

---

## 🧠 Explainable AI

HealthSense AI includes **Random Forest Feature Importance** to provide insight into which clinical features contributed most to the model's decisions overall.

The application displays:

* Overall feature importance
* Top 5 most important features
* Feature importance chart

> Feature importance indicates how strongly a feature contributed to the model's decisions overall. It does not mean that a feature directly causes heart disease or represents an individual patient's medical risk by itself.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Joblib**
* **Streamlit**

---

## 📂 Project Structure

The current repository structure is intentionally simple:

```text
HealthSense-AI/
│
├── app.py
├── best_heart_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

| File                   | Description                         |
| ---------------------- | ----------------------------------- |
| `app.py`               | Streamlit application               |
| `best_heart_model.pkl` | Trained Random Forest model         |
| `requirements.txt`     | Python dependencies                 |
| `README.md`            | Project documentation               |
| `.gitignore`           | Files excluded from version control |

---

## ⚙️ Installation & Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/adhamelsa/HealthSense-AI.git
```

### 2. Navigate to the project directory

```bash
cd HealthSense-AI
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your default web browser.

---

## 📦 Requirements

The project uses the following Python packages:

```text
streamlit
pandas
joblib
numpy
matplotlib
scikit-learn
```

---

## 🎯 Project Goals

The main goals of HealthSense AI are to:

* Apply machine learning to a real-world healthcare problem.
* Build an interactive prediction system.
* Evaluate model performance using multiple classification metrics.
* Improve model interpretability using feature importance.
* Deploy a machine learning application using Streamlit.
* Demonstrate an end-to-end machine learning project from model development to deployment.

---

## 🔮 Future Improvements

Potential future improvements include:

* Adding ROC Curve visualization.
* Adding SHAP-based explainability.
* Improving model probability calibration.
* Comparing multiple machine learning models within the application.
* Improving the user interface and visualizations.
* Adding automated model monitoring.
* Improving deployment reliability and monitoring.

---

## 👨‍💻 Author

**Adham El Sayed**

Computer Engineering | Data Analyst | AI & Machine Learning Enthusiast

---

## ⚠️ Medical Disclaimer

This project is intended for **educational and research purposes only**.

The predictions generated by this application should **not** be considered medical advice, diagnosis, or treatment recommendations.

Always consult a qualified healthcare professional for medical decisions.
