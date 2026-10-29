# AI-Based Credit Risk Prediction System

A machine learning-based web application for predicting loan default risk using historical LendingClub loan data. The system combines a trained credit-risk classification model with a Django REST API and a Bootstrap-based web interface.

> **Educational Project:** This application is intended for academic and educational purposes. Its predictions should not be used as the sole basis for real-world lending or credit decisions.

## Project Overview

The goal of this project is to estimate the probability that a loan applicant may default based on financial, credit, and loan-related characteristics.

The project covers the complete machine learning workflow:

- Dataset collection and preparation
- Exploratory data analysis
- Data preprocessing
- Feature engineering
- Model training and comparison
- Validation-based decision threshold selection
- Final evaluation on an untouched test set
- Model serialization
- Django REST API integration
- Web-based prediction interface
- Prediction history storage
- Automated testing

## Dataset

The project uses the **LendingClub accepted loan dataset (2007–2018 Q4)** available through Kaggle.

Original dataset:

- Approximately 2.26 million accepted-loan records
- 151 columns

For supervised learning, only loans with resolved repayment outcomes were retained:

- `Fully Paid` → Non-default (`0`)
- `Charged Off` → Default (`1`)

Other unresolved loan statuses were excluded.

Final resolved dataset:

- **1,345,310 loans**
- Non-default: **1,076,751 (80.04%)**
- Default: **268,559 (19.96%)**

The processed dataset itself is excluded from this repository because of its size. Scripts for downloading and preparing the data are included in the `data/` directory.

## Features

The model uses loan and applicant characteristics including:

- Loan amount
- Loan term
- Interest rate
- Installment
- Loan grade
- Employment length
- Home ownership
- Annual income
- Verification status
- Loan purpose
- Debt-to-income ratio
- Delinquencies
- FICO score
- Recent credit inquiries
- Open accounts
- Public records
- Revolving balance
- Revolving utilization
- Total accounts
- Mortgage accounts
- Public-record bankruptcies

Post-origination variables that could introduce target leakage were excluded.

## Data Preprocessing

The preprocessing workflow includes:

- Converting loan term to numeric values
- Converting employment length to numeric years
- Combining FICO range values into a single FICO score
- Median imputation for missing numerical values
- Most-frequent imputation for missing categorical values
- One-hot encoding of categorical variables
- Handling unknown categories during prediction

The preprocessing steps and classifier are integrated using a Scikit-learn pipeline.

## Machine Learning Models

The following models were evaluated:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 64.02% | 31.27% | 66.97% | 42.63% | 70.94% |
| Random Forest | 65.02% | 31.90% | 66.31% | 43.08% | 71.44% |
| HistGradientBoosting | **65.05%** | **32.16%** | **67.65%** | **43.59%** | **72.06%** |

The final selected model is **HistGradientBoostingClassifier**.

## Model Evaluation Methodology

The resolved dataset was divided into:

- **70% Training Set**
- **15% Validation Set**
- **15% Test Set**

The model was trained only on the training set.

The validation set was used to select the classification threshold. A threshold of **0.20** produced the best validation F1 score among the evaluated thresholds.

The final model was then evaluated once on the untouched test set.

### Final Test Performance

- Accuracy: **65.05%**
- Precision: **32.16%**
- Recall: **67.65%**
- F1 Score: **43.59%**
- ROC-AUC: **72.06%**
- Decision Threshold: **0.20**

The lower decision threshold prioritizes identifying a larger proportion of actual defaults, resulting in higher default-class recall at the cost of additional false-positive risk classifications.

## Web Application

The trained model is integrated into a Django application.

Main functionality includes:

- Credit-risk prediction form
- Default probability estimation
- Higher/lower default-risk classification
- Prediction history
- Django admin interface
- REST API
- Model information page

## REST API

Prediction endpoint:

```text
POST /api/predict/
```

The API validates applicant information, runs the trained machine learning pipeline, stores the result in the database, and returns information including:

- Default probability
- Default probability percentage
- Risk classification
- Decision threshold
- Prediction record ID

## Technology Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

### Backend

- Django
- Django REST Framework
- SQLite (development)

### Frontend

- HTML
- CSS
- JavaScript
- Bootstrap

### Development

- PyCharm
- Git
- GitHub

## Project Structure

```text
AI_Credit_Risk_Prediction/
├── credit_risk_web/       # Django project configuration
├── data/                  # Dataset download/preparation scripts
├── ml/                    # EDA, preprocessing and model training
├── models/                # Serialized trained ML model
├── predictor/             # Django prediction application
├── .gitignore
├── manage.py
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/padashettishiva711/AI_Credit_Risk_Prediction.git
cd AI_Credit_Risk_Prediction
```

Create and activate a virtual environment.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Apply database migrations:

```powershell
python manage.py migrate
```

Run the development server:

```powershell
python manage.py runserver
```

Open the application at:

```text
http://127.0.0.1:8000/
```

## Automated Tests

Run the Django test suite with:

```powershell
python manage.py test
```

The tests cover:

- Machine learning prediction service
- Prediction API
- Input validation and rejection of invalid data

## Important Limitations

This system is an educational credit-risk prediction prototype.

The LendingClub `grade` and `int_rate` variables may encode information from LendingClub's own underwriting and pricing process. Therefore, although useful for a benchmark model, they may make the model less independent of the original lender's risk assessment.

Model predictions should not be interpreted as guaranteed outcomes or used as the sole basis for actual lending decisions. A production credit system would require additional validation, fairness analysis, explainability, regulatory review, security controls, monitoring, and governance.

## Author

**Shiva Padashetti**

College Project — AI-Based Credit Risk Prediction System