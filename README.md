# AI-Based Credit Risk Prediction System

An end-to-end machine learning web application that predicts loan default risk using historical LendingClub loan data. The system combines a trained credit-risk classification model, Django REST API, cloud database, and responsive web interface.

**Live Application:** [https://ai-credit-risk-prediction.onrender.com/](https://ai-credit-risk-prediction.onrender.com/)

**GitHub Repository:** [https://github.com/padashettishiva711/AI_Credit_Risk_Prediction](https://github.com/padashettishiva711/AI_Credit_Risk_Prediction)

> **Educational Project:** This application is intended for academic and educational purposes. Its predictions should not be used as the sole basis for real-world lending or credit decisions.

---

## Project Overview

The objective of this project is to estimate the probability that a loan applicant may default based on financial, credit, and loan-related characteristics.

The project implements a complete machine learning workflow:

- Dataset collection and preparation
- Exploratory data analysis
- Data preprocessing and feature engineering
- Model training and comparison
- Validation-based decision threshold selection
- Final evaluation on an untouched test set
- Model serialization
- Individual prediction explanations
- Django REST API integration
- Web-based prediction interface
- Prediction history storage
- Cloud database integration
- Deployment using Render

## Key Features

- **Credit Risk Prediction:** Estimates the probability of loan default.
- **Risk Classification:** Classifies applicants into higher or lower default-risk categories.
- **Individual Explanations:** Compares important applicant features against dataset reference values.
- **Prediction History:** Stores and retrieves previous predictions.
- **Model Analytics:** Displays performance metrics, confusion matrix, ROC curve, and feature importance.
- **REST API:** Provides a prediction endpoint using Django REST Framework.
- **Cloud Database:** Uses TiDB Cloud for hosted prediction storage.
- **Responsive Interface:** Provides a web-based interface for submitting applicant information and viewing results.

## Dataset

The project uses the **LendingClub Accepted Loan Dataset (2007–2018 Q4)** available through Kaggle.

Original dataset characteristics:

- Approximately 2.26 million accepted-loan records
- 151 columns

For supervised learning, only loans with resolved repayment outcomes were retained:

| Loan Status |          Target |
| ----------- | --------------: |
| Fully Paid  | 0 – Non-default |
| Charged Off |     1 – Default |

Other unresolved loan statuses were excluded.

Final resolved dataset:

- **1,345,310 loans**
- Non-default: **1,076,751 (80.04%)**
- Default: **268,559 (19.96%)**

The original processed dataset is excluded from GitHub because of its large file size. Dataset download and preparation scripts are available in the `data/` directory.

**Important deployment optimization:** The live prediction service does not require the original 190 MB CSV dataset. Dataset reference values are calculated locally and stored in `models/ml_analytics.json`, allowing Render to generate explanations without loading the original dataset.

## Features Used by the Model

The model uses the following applicant and loan characteristics:

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

- Converting loan term into numeric values
- Converting employment length into numeric years
- Combining FICO range values into a single FICO score
- Median imputation for missing numerical values
- Most-frequent imputation for missing categorical values
- One-hot encoding of categorical variables
- Handling unknown categories during prediction

The preprocessing steps and classifier are integrated using a Scikit-learn pipeline.

## Machine Learning Models

The following classification models were evaluated:

| Model                |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| -------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression  |     64.02% |     31.27% |     66.97% |     42.63% |     70.94% |
| Random Forest        |     65.02% |     31.90% |     66.31% |     43.08% |     71.44% |
| HistGradientBoosting | **65.05%** | **32.16%** | **67.65%** | **43.59%** | **72.06%** |

**Final selected model: HistGradientBoostingClassifier**

## Model Evaluation Methodology

The resolved dataset was divided into:

- **70% Training Set**
- **15% Validation Set**
- **15% Test Set**

The training set was used to train the model.

The validation set was used to select the classification threshold. A threshold of **0.20** produced the best validation F1 score among the evaluated thresholds.

The final model was evaluated on the untouched test set.

### Final Test Performance

| Metric             | Result |
| ------------------ | -----: |
| Accuracy           | 65.05% |
| Precision          | 32.16% |
| Recall             | 67.65% |
| F1 Score           | 43.59% |
| ROC-AUC            | 72.06% |
| Decision Threshold |   0.20 |

The lower decision threshold prioritizes identifying a larger proportion of actual defaults, at the cost of additional false-positive classifications.

## Individual Prediction Explanations

The application provides individual feature explanations using:

1. Global permutation feature importance.
2. Dataset reference values calculated from numerical medians and categorical modes.
3. One-feature-at-a-time comparison against the applicant's actual values.
4. Changes in predicted default probability.

The reference values are stored in:

`models/ml_analytics.json`

This approach avoids loading the original dataset during live prediction.

These comparisons describe model behavior and should not be interpreted as causal explanations.

## Web Application and API

The trained model is integrated into a Django web application.

Main functionality:

- Credit-risk prediction form
- Default probability estimation
- Risk classification
- Individual feature explanations
- Prediction history
- Dashboard and model analytics
- Django administration
- REST API

### REST API

**Prediction Endpoint**

`POST /api/predict/`

The API:

1. Validates applicant information.
2. Passes the validated data to the ML prediction service.
3. Calculates default probability.
4. Applies the decision threshold.
5. Generates individual feature explanations.
6. Stores the prediction in the database.
7. Returns the prediction response.

Example response structure:

```json
{
  "id": 1,
  "default_probability": 0.5967,
  "default_probability_percent": 59.68,
  "prediction": 1,
  "risk_label": "Higher Default Risk",
  "threshold": 0.2,
  "explanation": []
}
```

## System Architecture

```text
              Applicant
                  |
                  v
          Django Web Interface
                  |
                  v
           Django REST API
                  |
                  v
         Input Validation
                  |
                  v
       Machine Learning Service
                  |
          +-------+--------+
          |                |
          v                v
     Trained Model     Analytics JSON
     (Joblib)          (Reference Values
          |             and Importance)
          |                |
          +-------+--------+
                  |
                  v
        Default Probability
                  |
                  v
         Risk Classification
                  |
                  v
        Individual Explanation
                  |
                  v
          TiDB Cloud Database
                  |
                  v
         Prediction Response
```

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
- MySQL-compatible database
- TiDB Cloud
- django-tidb

### Frontend

- HTML5
- CSS3
- JavaScript
- Bootstrap

### Database and Deployment

- Local MySQL for development
- TiDB Cloud Starter for hosted database storage
- Render for web application hosting
- Git and GitHub for version control

### Development Tools

- PyCharm
- PowerShell
- Git
- GitHub

## Project Structure

```text
AI_Credit_Risk_Prediction/
│
├── credit_risk_web/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── data/
│   ├── download_dataset.py
│   ├── prepare_dataset.py
│   ├── inspect_dataset.py
│   └── check_target.py
│
├── ml/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── generate_analytics.py
│   ├── generate_reference_values.py
│   ├── train_gradient_boosting.py
│   ├── train_random_forest.py
│   └── train_logistic.py
│
├── models/
│   ├── credit_risk_model.joblib
│   └── ml_analytics.json
│
├── predictor/
│   ├── migrations/
│   ├── templates/
│   ├── ml_service.py
│   ├── models.py
│   ├── serializers.py
│   ├── views.py
│   ├── urls.py
│   └── admin.py
│
├── certs/
│   └── isrgrootx1.pem
│
├── manage.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/padashettishiva711/AI_Credit_Risk_Prediction.git

cd AI_Credit_Risk_Prediction
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file in the project root.

Example for local MySQL:

```env
DJANGO_DEBUG=True
DJANGO_SECRET_KEY=your-local-secret-key
DJANGO_ALLOWED_HOSTS=127.0.0.1,localhost

DB_NAME=credit_risk_db
DB_USER=root
DB_PASSWORD=your-local-mysql-password
DB_HOST=127.0.0.1
DB_PORT=3306
```

Do not commit your `.env` file or expose database credentials.

Create the local MySQL database before running migrations.

### 5. Apply database migrations

```powershell
python manage.py migrate
```

### 6. Start the development server

```powershell
python manage.py runserver
```

Open:

[http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## Cloud Deployment

The application is deployed using:

- **Web Hosting:** Render
- **Database:** TiDB Cloud Starter
- **Database Compatibility:** MySQL
- **Cloud Provider:** AWS
- **Region:** Singapore (`ap-southeast-1`)

Live application:

[https://ai-credit-risk-prediction.onrender.com/](https://ai-credit-risk-prediction.onrender.com/)

### Production Environment Variables

Configure the following variables in the Render service environment settings:

| Variable               | Purpose                            |
| ---------------------- | ---------------------------------- |
| `DJANGO_SECRET_KEY`    | Django secret key                  |
| `DJANGO_DEBUG`         | Production debug setting (`False`) |
| `DJANGO_ALLOWED_HOSTS` | Allowed production hostname        |
| `DB_ENGINE`            | `django_tidb`                      |
| `DB_NAME`              | `credit_risk_db`                   |
| `DB_USER`              | TiDB database username             |
| `DB_PASSWORD`          | TiDB database password             |
| `DB_HOST`              | TiDB endpoint                      |
| `DB_PORT`              | `4000`                             |
| `DB_SSL_CA`            | `certs/isrgrootx1.pem`             |

Keep passwords and secret keys in Render environment settings, not in GitHub.

### Cloud Prediction Optimization

The original CSV dataset is excluded from GitHub because of its size.

To support cloud deployment:

1. Dataset reference values are generated locally.
2. Reference values are stored in `models/ml_analytics.json`.
3. The analytics JSON is committed to GitHub.
4. The production ML service loads reference values directly from JSON.
5. Render performs predictions without requiring the original CSV dataset.

This reduces deployment dependencies and avoids loading a large dataset for live prediction requests.

**Note:** Render's free web service may sleep after inactivity, so the first request after an idle period can take longer.

## Automated Tests

Run the Django test suite:

```powershell
python manage.py test
```

Run the Django configuration check:

```powershell
python manage.py check
```

## Important Limitations

This application is an educational credit-risk prediction prototype.

The LendingClub `grade` and `int_rate` variables may encode information from LendingClub's own underwriting and pricing process. Although useful for benchmark prediction, they may make the model less independent of the original lender's risk assessment.

Other limitations include:

- Historical dataset bias
- Class imbalance
- False-positive and false-negative predictions
- Potential differences between historical and current lending conditions
- Limited generalization to other lending institutions

A real-world credit-risk system would require additional validation, fairness analysis, explainability, regulatory review, security controls, continuous monitoring, and governance.

Model predictions are estimates, not guaranteed outcomes, and should never be used as the sole basis for actual lending decisions.

## Author

**Shivanand Padashetti**

College Project — AI-Based Credit Risk Prediction System

---

*Built using Python, Machine Learning, Django, REST APIs, MySQL-compatible cloud storage, and Render.*
