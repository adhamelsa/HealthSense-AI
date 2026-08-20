# HealthSense AI

HealthSense AI is a heart disease prediction project based on clinical patient data and machine learning.

## Project Structure

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
```

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit application from the project root:

```bash
streamlit run app/app.py
```

The application currently provides the HealthSense AI interface. The patient information form and prediction workflow are planned for the next development stage.

## Model

The optimized model is stored at:

```text
models/best_heart_model.pkl
```

The model can be regenerated from `notebooks/04_Model_Optimization.ipynb`.

## Disclaimer

This project is for educational and research purposes only. It is not a medical diagnostic tool and should not replace advice from a qualified healthcare professional.
