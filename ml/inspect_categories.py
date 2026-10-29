import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "credit_risk_dataset.csv"

categorical_columns = [
    "term",
    "grade",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose"
]

print("Loading categorical columns...")

df = pd.read_csv(
    DATA_FILE,
    usecols=categorical_columns
)

for column in categorical_columns:

    print("\n" + "=" * 60)
    print(f"COLUMN: {column}")
    print("=" * 60)

    print(df[column].value_counts(dropna=False))

    print("\nNumber of unique values:")
    print(df[column].nunique(dropna=False))