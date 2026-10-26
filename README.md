# Credit Card Fraud Detection System

An end-to-end machine learning application for detecting potentially fraudulent credit card transactions. The project combines a trained classification model with a reusable prediction pipeline, FastAPI REST API, Streamlit dashboard, and automated API testing.

## Project Overview

Credit card fraud detection is an imbalanced binary classification problem where fraudulent transactions represent only a very small portion of all transactions.

This project trains a machine learning model to estimate the probability that a transaction is fraudulent. The trained model is packaged into a reusable prediction pipeline and exposed through a REST API. A Streamlit dashboard provides a simple interface for uploading transaction data and viewing fraud predictions.

The system is designed to demonstrate the complete workflow:

```text
Raw Transaction Data
        ↓
Data Exploration & Preprocessing
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Saved ML Pipeline
        ↓
FastAPI REST API
        ↓
Streamlit Dashboard
        ↓
Fraud Risk Predictions
```

## Key Features

- Exploratory Data Analysis using Pandas, NumPy, Matplotlib and Seaborn
- Binary fraud classification
- Stratified train/validation/test splitting
- Logistic Regression and Random Forest model comparison
- Handling of class imbalance using class weights
- Feature preprocessing using Scikit-learn pipelines
- Precision-Recall based model evaluation
- Validation-based decision threshold selection
- Reusable prediction pipeline
- Single transaction prediction API
- Batch CSV prediction API
- Interactive Streamlit dashboard
- Fraud risk score and risk-level classification
- Prediction distribution visualization
- Confusion matrix and classification metrics
- CSV download of prediction results
- Automated API tests using Pytest

## Tech Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Data Analysis

- Jupyter Notebook
- Matplotlib
- Seaborn

### Backend

- FastAPI
- Uvicorn
- REST API

### Dashboard

- Streamlit

### Testing

- Pytest
- FastAPI TestClient

## Dataset

The project uses the commonly used Credit Card Fraud Detection dataset containing transactions represented by anonymized numerical features.

The dataset contains:

- `Time`
- `V1` through `V28`
- `Amount`
- `Class`

`Class` is the target variable:

```text
0 → Legitimate transaction
1 → Fraudulent transaction
```

The `V1`–`V28` features are anonymized transformed features from the original dataset.

The dataset is highly imbalanced, with fraudulent transactions representing a very small percentage of the total transactions.

The raw dataset is intentionally excluded from Git because of its large file size.

## Machine Learning Pipeline

The training pipeline follows these stages:

```text
Load Dataset
     ↓
Separate Features and Target
     ↓
Train / Validation / Test Split
     ↓
Preprocessing
     ↓
Train Candidate Models
     ↓
Compare Models
     ↓
Select Best Model
     ↓
Choose Decision Threshold
     ↓
Evaluate on Test Set
     ↓
Save Complete Pipeline
```

### Models

Two baseline classification approaches are evaluated:

1. Logistic Regression
2. Random Forest

A majority-class dummy classifier is also used as a baseline.

Because the dataset is highly imbalanced, accuracy alone is not used to select the model.

The main model-selection metric is **Average Precision**, which is more informative for highly imbalanced classification problems.

### Class Imbalance

The model uses class weighting to give greater importance to fraudulent transactions.

For example, the Logistic Regression pipeline uses:

```python
LogisticRegression(
    class_weight="balanced"
)
```

and the Random Forest uses:

```python
RandomForestClassifier(
    class_weight="balanced_subsample"
)
```

### Decision Threshold

Instead of automatically using a probability threshold of `0.5`, the project searches for a suitable threshold using the validation set.

The selected threshold is then saved as model metadata and used consistently during prediction.

This separates:

```text
Model probability
        ↓
Decision threshold
        ↓
Fraud / Legitimate
```

## Model Evaluation

The project evaluates the final model using metrics that are more appropriate for fraud detection:

- Average Precision
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Precision-Recall Curve

The final test-set metrics should be taken from the output of `src/train.py`.

```text
Average Precision: 0.7991
Precision:         80%
Recall:            79%
F1 Score:          79%
Threshold:         0.12
```

### Why accuracy is not the main metric

Fraud detection is highly imbalanced. A model could achieve very high accuracy simply by predicting almost every transaction as legitimate.

For this reason, the project focuses more on:

```text
Precision
Recall
F1 Score
Average Precision
```

For example, the Streamlit demonstration showed:

```text
Accuracy:  99.80%
Precision: 50.00%
Recall:    100.00%
F1 Score:  66.67%
```

on the uploaded demonstration sample.

These values are demonstration-sample results and should not be confused with the final held-out test-set metrics.

## Backend Architecture

The trained model is exposed through a FastAPI backend.

```text
Client
  │
  │ HTTP Request
  ▼
FastAPI
  │
  ▼
Prediction Service
  │
  ▼
Saved Scikit-learn Pipeline
  │
  ▼
Fraud Probability
  │
  ▼
Decision Threshold
  │
  ▼
Prediction Response
```

### API Endpoints

#### `GET /health`

Checks whether the API is running.

Example response:

```json
{
  "status": "ok"
}
```

#### `POST /predict`

Accepts a single transaction's feature values and returns a prediction.

Example response:

```json
{
  "prediction": "legitimate",
  "risk_score": 0.08,
  "threshold": 0.5
}
```

#### `POST /predict-file`

Accepts a CSV containing multiple transactions and returns predictions for every row.

The endpoint returns:

```text
Rows processed
Fraud flagged
Decision threshold
Prediction results
```

## Streamlit Dashboard

The Streamlit dashboard provides a user-friendly interface over the FastAPI backend.

The dashboard allows users to:

- Upload a transaction CSV
- Preview input data
- Run fraud predictions
- View the number of transactions processed
- View the number of transactions flagged as fraud
- View the fraud rate
- View the model decision threshold
- View risk scores
- View risk levels
- View prediction distribution
- View model evaluation when ground-truth labels are available
- View the confusion matrix
- Download prediction results

### Risk Levels

The dashboard converts the model probability into an easier-to-understand risk category:

```text
0.00 – 0.30 → Low
0.30 – 0.70 → Medium
0.70 – 1.00 → High
```

The exact fraud decision is still based on the model's saved decision threshold.

## Project Structure

```text
credit-card-fraud-detection/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── artifacts/
│   ├── model_comparison.csv
│   └── precision_recall_curve.png
│
├── dashboard/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── creditcard.csv
│   └── processed/
│       └── demo_transactions.csv
│
├── models/
│   ├── fraud_pipeline.joblib
│   └── model_metadata.json
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── predict.py
│   └── train.py
│
├── tests/
│   └── test_api.py
│
├── .gitignore
├── pytest.ini
├── README.md
└── requirements.txt
```

## Prediction Pipeline

The saved model is not used directly from the API.

Instead, the project has a reusable prediction layer:

```text
CSV / JSON Input
       ↓
Input Validation
       ↓
Feature Validation
       ↓
Data Type Conversion
       ↓
Saved ML Pipeline
       ↓
Fraud Probability
       ↓
Decision Threshold
       ↓
Prediction
```

This prevents training-time preprocessing and prediction-time preprocessing from becoming inconsistent.

The complete trained pipeline is saved using Joblib:

```text
models/fraud_pipeline.joblib
```

Model configuration and metadata are stored separately:

```text
models/model_metadata.json
```

## Testing

The project uses Pytest for automated API testing.

Run:

```bash
pytest -v
```

Current test coverage includes:

```text
✓ Health check
✓ Single transaction prediction
✓ Missing feature validation
✓ Batch CSV prediction
```

Current result:

```text
4 passed
```

This verifies that the main backend functionality works as expected.

## Running the Project Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd credit-card-fraud-detection
```

### 2. Create a virtual environment

Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the original dataset at:

```text
data/raw/creditcard.csv
```

The dataset is not included in the repository because of its size.

### 5. Train the model

```bash
python src/train.py
```

This generates the trained model and evaluation artifacts.

### 6. Start the FastAPI backend

```bash
uvicorn app.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### 7. Start the Streamlit dashboard

In another terminal:

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard will open in the browser.

## Example Workflow

```text
1. Start FastAPI
        ↓
2. Start Streamlit
        ↓
3. Upload transaction CSV
        ↓
4. Streamlit sends CSV to FastAPI
        ↓
5. FastAPI validates the input
        ↓
6. Prediction pipeline processes features
        ↓
7. ML model calculates fraud probability
        ↓
8. Decision threshold determines class
        ↓
9. Results returned to Streamlit
        ↓
10. User views and downloads results
```

## Engineering Decisions

### Why FastAPI?

FastAPI provides a lightweight way to expose the trained ML model through REST endpoints and is suitable for integrating machine learning models into backend applications.

### Why Streamlit?

The project does not require a complex frontend for the demonstration. Streamlit allows the ML system to have an interactive web interface while keeping the implementation entirely in Python.

### Why a separate prediction module?

The prediction logic is separated from the API layer so that the model can be reused by different clients.

```text
FastAPI
   └── predict.py
          └── ML pipeline

Streamlit
   └── FastAPI
          └── predict.py
```

### Why use a validation set?

The validation set is used for model comparison and decision-threshold selection. The test set is kept separate for final evaluation.

This reduces the risk of tuning the model directly against the test data.

## Limitations

- The dataset uses anonymized PCA-transformed features, so the model is not directly interpretable in terms of real-world transaction attributes.
- The system expects the same feature schema as the training dataset.
- The demonstration dashboard is not intended to represent a production fraud-monitoring system.
- A production deployment would require authentication, authorization, rate limiting, logging, monitoring, and stronger input controls.
- The current system performs batch prediction rather than real-time event-stream processing.

## Future Improvements

Possible extensions include:

- Real-time transaction streaming
- PostgreSQL or another production database
- Authentication and role-based access control
- Transaction history and audit logs
- Model monitoring
- Data drift detection
- Automated retraining
- Explainable AI using SHAP
- Docker containerization
- Cloud deployment
- CI/CD pipeline
- API rate limiting
- Production logging and observability

## What This Project Demonstrates

This project demonstrates the complete process of taking a machine learning model beyond a notebook and turning it into an application.

Key areas demonstrated:

```text
Machine Learning
    +
Data Analysis
    +
Model Evaluation
    +
Python
    +
REST APIs
    +
FastAPI
    +
Backend Architecture
    +
Streamlit
    +
Automated Testing
    +
Model Deployment
```

The main engineering idea is:

> Train a model → package the model → expose it through an API → build a user interface → test the system → deploy it.