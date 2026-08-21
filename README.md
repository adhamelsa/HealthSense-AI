
# ❤️ HealthSense AI – Heart Disease Prediction

A Machine Learning web application that predicts the likelihood of heart disease based on clinical patient data.

The application is built with **Python** and **Streamlit** and uses a tuned **Random Forest** model to generate predictions and risk probabilities.

---

## 📌 Project Overview

Heart disease is one of the major health challenges worldwide. Early risk assessment can help identify patients who may require further medical evaluation.

**HealthSense AI** provides an interactive interface where users can enter clinical information and receive a machine-learning-based prediction.

The system also provides model performance metrics and feature importance analysis to improve transparency and interpretability.

> ⚠️ **Disclaimer:** This application is developed for educational and research purposes only. It is not a medical diagnosis tool and should not replace professional medical advice.

---

## 🚀 Features

* ❤️ Heart disease risk prediction
* 📊 Probability-based risk score
* 👤 Interactive patient information form
* 🧠 Random Forest feature importance
* 🔝 Top 5 most important clinical factors
* 📈 Model performance dashboard
* 🔢 Confusion Matrix visualization
* 📋 Patient input summary
* ⚡ Interactive Streamlit interface

---

## 🧠 Machine Learning Model

The application uses a **Tuned Random Forest Classifier** trained to classify patients into two classes:

* **0 → No Heart Disease**
* **1 → Heart Disease**

The model is loaded using `joblib` from:

```text
best_heart_model.pkl
```

---

## 📊 Input Features

The model uses 13 clinical features:

| Feature  | Description                          |
| -------- | ------------------------------------ |
| Age      | Patient age                          |
| Sex      | Biological sex                       |
| CP       | Chest pain type                      |
| Trestbps | Resting blood pressure               |
| Chol     | Cholesterol level                    |
| FBS      | Fasting blood sugar                  |
| RestECG  | Resting electrocardiographic results |
| Thalach  | Maximum heart rate achieved          |
| Exang    | Exercise-induced angina              |
| Oldpeak  | ST depression                        |
| Slope    | ST segment slope                     |
| CA       | Number of major vessels              |
| Thal     | Thalassemia-related feature          |

---

## 📈 Model Performance

The tuned Random Forest model achieved the following results on the clean test dataset:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **78.69%** |
| Precision | **81.25%** |
| Recall    | **78.79%** |
| F1-Score  | **80.00%** |
| ROC-AUC   | **85.93%** |

### Confusion Matrix

```text
                 Predicted
              No Disease  Disease

Actual
No Disease        22         6
Heart Disease      7        26
```

The model correctly classified **48 out of 61 test samples**.

The evaluation was performed after removing duplicate records from the dataset to reduce potential data leakage and provide a more realistic estimate of model performance.

---

## 🧠 Explainable AI

HealthSense AI includes feature importance analysis based on the Random Forest model.

The application displays:

* Overall feature importance
* Top 5 most influential features
* Feature importance visualization

Feature importance describes how strongly each feature contributed to the model's decisions overall. It does **not** mean that a feature directly causes heart disease or represents an individual's medical risk by itself.

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

```text
HealthSense/
├── app/
│   └── app.py
├── data/
│   └── raw/
│       └── heart.csv
├── models/
│   └── best_heart_model.pkl
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_Model_Optimization.ipynb
│   └── 05_Model_Evaluation.ipynb
├── reports/
├── requirements.txt
└── README.md

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd HealthSense-Disease-Forecasting
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

The project requires:

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
* Evaluate model performance using multiple metrics.
* Improve model interpretability using feature importance.
* Demonstrate the deployment of a machine learning model through Streamlit.

---

## 🔮 Future Improvements

Possible future improvements include:

* Adding ROC Curve visualization.
* Adding SHAP-based explainability.
* Improving model calibration.
* Adding additional machine learning models for comparison.
* Improving the user interface and visualization.
* Deploying the application publicly.
* Adding automated model monitoring.

---

## 👨‍💻 Author

**Adham El Sayed**

Computer Engineering Student | Data Analyst | AI & Machine Learning Enthusiast

---

## ⚠️ Medical Disclaimer

This project is intended for **educational and research purposes only**.

The predictions generated by this application should not be considered medical advice, diagnosis, or treatment recommendations.

Always consult a qualified healthcare professional for medical decisions.

