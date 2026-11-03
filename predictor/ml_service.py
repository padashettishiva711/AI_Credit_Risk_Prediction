
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
DATA_FILE = PROJECT_ROOT / "data" / "credit_risk_dataset.csv"


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
# Load global feature importance
# --------------------------------------------------

FEATURE_IMPORTANCE = []

if ANALYTICS_FILE.exists():

    with ANALYTICS_FILE.open("r", encoding="utf-8") as file:
        analytics = json.load(file)

    FEATURE_IMPORTANCE = analytics.get(
        "feature_importance", []
    )


# --------------------------------------------------
# Prepare dataset reference values
# --------------------------------------------------

_REFERENCE_VALUES = None


def get_reference_values():

    global _REFERENCE_VALUES

    if _REFERENCE_VALUES is not None:
        return _REFERENCE_VALUES

    print("Loading dataset reference values...")

    df = pd.read_csv(DATA_FILE, low_memory=False)

    # Apply the same transformations used during training.

    df["term"] = (
        df["term"]
        .astype("string")
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
    )

    employment_mapping = {
        "< 1 year": 0,
        "1 year": 1,
        "2 years": 2,
        "3 years": 3,
        "4 years": 4,
        "5 years": 5,
        "6 years": 6,
        "7 years": 7,
        "8 years": 8,
        "9 years": 9,
        "10+ years": 10,
    }

    df["emp_length"] = df["emp_length"].map(
        employment_mapping
    )

    df["fico_score"] = (
        df["fico_range_low"] + df["fico_range_high"]
    ) / 2

    df.drop(
        columns=["fico_range_low", "fico_range_high"],
        inplace=True
    )

    X = df.drop(columns=["default"])

    numerical_features = X.select_dtypes(
        include="number"
    ).columns

    categorical_features = X.select_dtypes(
        exclude="number"
    ).columns

    reference = {}

    for feature in numerical_features:

        value = X[feature].median()

        if pd.notna(value):
            reference[feature] = float(value)

    for feature in categorical_features:

        modes = X[feature].mode(dropna=True)

        if not modes.empty:
            reference[feature] = modes.iloc[0]

    _REFERENCE_VALUES = reference

    print("Dataset reference values loaded successfully.")

    return _REFERENCE_VALUES


# --------------------------------------------------
# Generate individual applicant explanation
# --------------------------------------------------

def generate_explanation(applicant_data, actual_probability):

    reference_values = get_reference_values()

    applicant_df = pd.DataFrame([applicant_data])

    explanations = []

    for item in FEATURE_IMPORTANCE[:5]:

        feature = item["feature"]

        if feature not in applicant_data:
            continue

        if feature not in reference_values:
            continue

        applicant_value = applicant_data[feature]
        reference_value = reference_values[feature]

        # Replace only one feature with its reference value.

        comparison_df = applicant_df.copy()

        comparison_df.at[0, feature] = reference_value

        comparison_probability = float(
            model.predict_proba(comparison_df)[0][1]
        )

        # Positive means the applicant's actual value
        # is associated with a higher model score
        # than the reference value.

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

            "reference_value": (
                reference_value.item()
                if hasattr(reference_value, "item")
                else reference_value
            ),

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
