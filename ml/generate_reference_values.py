
import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "credit_risk_dataset.csv"
ANALYTICS_FILE = PROJECT_ROOT / "models" / "ml_analytics.json"

print("Loading dataset to calculate reference values...")

df = pd.read_csv(DATA_FILE, low_memory=False)

# Apply the same transformations used by the ML training process.

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

df["emp_length"] = df["emp_length"].map(employment_mapping)

df["fico_score"] = (
    df["fico_range_low"] + df["fico_range_high"]
) / 2

df.drop(
    columns=["fico_range_low", "fico_range_high"],
    inplace=True
)

X = df.drop(columns=["default"])

reference_values = {}

# Numerical feature medians
for feature in X.select_dtypes(include="number").columns:
    value = X[feature].median()

    if pd.notna(value):
        reference_values[feature] = float(value)

# Categorical feature modes
for feature in X.select_dtypes(exclude="number").columns:
    modes = X[feature].mode(dropna=True)

    if not modes.empty:
        value = modes.iloc[0]

        if hasattr(value, "item"):
            value = value.item()

        reference_values[feature] = value

# Preserve existing analytics and add reference values.
with ANALYTICS_FILE.open("r", encoding="utf-8") as file:
    analytics = json.load(file)

analytics["reference_values"] = reference_values

with ANALYTICS_FILE.open("w", encoding="utf-8") as file:
    json.dump(analytics, file, indent=4)

print("Reference values generated successfully!")
print("Features:", len(reference_values))
print("Analytics file:", ANALYTICS_FILE)
