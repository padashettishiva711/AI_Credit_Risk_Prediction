import pandas as pd
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "credit_risk_dataset.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_FILE, low_memory=False)

print("\nDataset loaded successfully.")

print("\nShape:")
print(df.shape)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
missing = df.isnull().sum()
print(missing.sort_values(ascending=False))

print("\nMissing percentages:")
missing_percentage = (
    df.isnull()
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print(missing_percentage.round(2))

print("\nDuplicate rows:")
print(df.duplicated().sum())