import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "credit_risk_dataset.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_FILE, low_memory=False)

print("Original shape:", df.shape)


# --------------------------------------------------
# 1. Convert term to numeric
# --------------------------------------------------

df["term"] = (
    df["term"]
    .str.extract(r"(\d+)")
    .astype(float)
)


# --------------------------------------------------
# 2. Convert employment length to numeric
# --------------------------------------------------

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
    "10+ years": 10
}

df["emp_length"] = df["emp_length"].map(employment_mapping)


# --------------------------------------------------
# 3. Combine FICO range
# --------------------------------------------------

df["fico_score"] = (
    df["fico_range_low"] + df["fico_range_high"]
) / 2

df.drop(
    columns=["fico_range_low", "fico_range_high"],
    inplace=True
)


# --------------------------------------------------
# 4. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["default"])
y = df["default"]


# --------------------------------------------------
# 5. Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining set:")
print(X_train.shape)

print("\nTesting set:")
print(X_test.shape)


print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).mul(100).round(2))


print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).mul(100).round(2))


print("\nRemaining categorical columns:")
print(X_train.select_dtypes(include=["object", "str"]).columns.tolist())


print("\nMissing values in training data:")
print(
    X_train
    .isnull()
    .sum()
    .sort_values(ascending=False)
    .head(10)
)