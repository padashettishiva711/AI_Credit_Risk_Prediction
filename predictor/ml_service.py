
import joblib
import json
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = PROJECT_ROOT / "models" / "credit_risk_model.joblib"
ANALYTICS_FILE = PROJECT_ROOT / "models" / "ml_analytics.json"


# --------------------------------------------------
# Load trained ML model
# --------------------------------------------------

print("Loading credit risk ML model...")

model_package = joblib.load(MODEL_FILE)

model = model_package["model"]
THRESHOLD = model_package["threshold"]
MODEL_METRICS = model_package["metrics"]

print("Credit risk ML model loaded successfully.")
print("Decision threshold:", THRESHOLD)


# --------------------------------------------------
# Load analytics and reference values
# --------------------------------------------------

with ANALYTICS_FILE.open("r", encoding="utf-8") as file:
    analytics = json.load(file)

FEATURE_IMPORTANCE = analytics.get("feature_importance", [])

REFERENCE_VALUES = analytics.get("reference_values", {})

print("ML analytics loaded successfully.")
print("Reference features:", len(REFERENCE_VALUES))


# --------------------------------------------------
# Generate individual applicant explanation
# --------------------------------------------------

def generate_explanation(applicant_data, actual_probability):

    explanations = []

    applicant_df = pd.DataFrame([applicant_data])

    for item in FEATURE_IMPORTANCE[:5]:

        feature = item["feature"]

        if feature not in applicant_data:
            continue

        if feature not in REFERENCE_VALUES:
            continue

        applicant_value = applicant_data[feature]
        reference_value = REFERENCE_VALUES[feature]

        # Replace only one feature with its reference value.

        comparison_df = applicant_df.copy()

        comparison_df.at[0, feature] = reference_value

        comparison_probability = float(
            model.predict_proba(comparison_df)[0][1]
        )

        change = (
            actual_probability - comparison_probability
        ) * 100

        if change > 0.0001:
            effect = "higher_risk_than_reference"

        elif change < -0.0001:
            effect = "lower_risk_than_reference"

        else:
            effect = "little_or_no_change"

        explanations.append({

            "feature": feature.replace(
                "_", " "
            ).title(),

            "applicant_value": (
                applicant_value
                if applicant_value is not None
                else "Not provided"
            ),

            "reference_value": reference_value,

            "comparison_probability_percent": round(
                comparison_probability * 100, 2
            ),

            "change_percentage_points": round(
                change, 2
            ),

            "effect": effect,

            "global_importance": round(
                float(item["importance"]), 6
            ),

        })

    return explanations


# --------------------------------------------------
# Main prediction function
# --------------------------------------------------

def predict_credit_risk(applicant_data):

    applicant_df = pd.DataFrame([applicant_data])

    default_probability = float(
        model.predict_proba(applicant_df)[0][1]
    )

    prediction = int(
        default_probability >= THRESHOLD
    )

    explanation = generate_explanation(
        applicant_data,
        default_probability
    )

    return {

        "default_probability": default_probability,

        "default_probability_percent": round(
            default_probability * 100, 2
        ),

        "prediction": prediction,

        "risk_label": (
            "Higher Default Risk"
            if prediction == 1
            else "Lower Default Risk"
        ),

        "threshold": THRESHOLD,

        "explanation": explanation,

    }
