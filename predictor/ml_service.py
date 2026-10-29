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
# Load model package
# --------------------------------------------------

print("Loading credit risk ML model...")

model_package = joblib.load(MODEL_FILE)

model = model_package["model"]
THRESHOLD = model_package["threshold"]
MODEL_METRICS = model_package["metrics"]

print("Credit risk ML model loaded successfully.")
print("Decision threshold:", THRESHOLD)


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_credit_risk(applicant_data):

    applicant_df = pd.DataFrame(
        [applicant_data]
    )

    default_probability = float(
        model.predict_proba(
            applicant_df
        )[0][1]
    )

    prediction = int(
        default_probability >= THRESHOLD
    )

    return {
        "default_probability":
            default_probability,

        "default_probability_percent":
            round(
                default_probability * 100,
                2
            ),

        "prediction":
            prediction,

        "risk_label": (
            "Higher Default Risk"
            if prediction == 1
            else "Lower Default Risk"
        ),

        "threshold":
            THRESHOLD,
    }