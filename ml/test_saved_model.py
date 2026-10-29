import joblib
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Model path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "credit_risk_model.joblib"
)


# --------------------------------------------------
# Load saved model package
# --------------------------------------------------

print("Loading saved model package...")

model_package = joblib.load(MODEL_FILE)

model = model_package["model"]
threshold = model_package["threshold"]
metrics = model_package["metrics"]

print("Model package loaded successfully.")
print("Saved threshold:", threshold)


# --------------------------------------------------
# Example applicant
# --------------------------------------------------

applicant = {
    "loan_amnt": 15000,
    "term": 36,
    "int_rate": 12.5,
    "installment": 500,
    "emp_length": 5,
    "home_ownership": "RENT",
    "annual_inc": 65000,
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "dti": 18.5,
    "delinq_2yrs": 0,
    "inq_last_6mths": 1,
    "open_acc": 10,
    "pub_rec": 0,
    "revol_bal": 12000,
    "revol_util": 45.0,
    "total_acc": 25,
    "mort_acc": 1,
    "pub_rec_bankruptcies": 0,
    "fico_score": 700,
    "grade": "B",
}

applicant_df = pd.DataFrame(
    [applicant]
)


# --------------------------------------------------
# Predict probability
# --------------------------------------------------

default_probability = float(
    model.predict_proba(
        applicant_df
    )[0][1]
)


# --------------------------------------------------
# Apply saved threshold
# --------------------------------------------------

prediction = int(
    default_probability >= threshold
)


# --------------------------------------------------
# Display prediction
# --------------------------------------------------

print("\n" + "=" * 50)
print("CREDIT RISK PREDICTION")
print("=" * 50)

print(
    f"Default probability: "
    f"{default_probability:.4f}"
)

print(
    f"Default probability (%): "
    f"{default_probability * 100:.2f}%"
)

print(
    f"Decision threshold: "
    f"{threshold:.2f}"
)

if prediction == 1:
    print(
        "Prediction: HIGHER DEFAULT RISK"
    )
else:
    print(
        "Prediction: LOWER DEFAULT RISK"
    )


# --------------------------------------------------
# Display saved final test metrics
# --------------------------------------------------

print("\n" + "=" * 50)
print("SAVED FINAL TEST METRICS")
print("=" * 50)

print(
    f"Accuracy:  "
    f"{metrics['accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{metrics['precision']:.4f}"
)

print(
    f"Recall:    "
    f"{metrics['recall']:.4f}"
)

print(
    f"F1 Score:  "
    f"{metrics['f1_score']:.4f}"
)

print(
    f"ROC-AUC:   "
    f"{metrics['roc_auc']:.4f}"
)