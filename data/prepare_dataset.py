import pandas as pd
from pathlib import Path

SOURCE_FILE = (
    r"C:\Users\shiva\.cache\kagglehub\datasets"
    r"\wordsforthewise\lending-club\versions\3"
    r"\accepted_2007_to_2018Q4.csv.gz"
)

OUTPUT_FILE = Path(__file__).parent / "credit_risk_dataset.csv"

features = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "dti",
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "loan_status",
]

print("Reading dataset...")

df = pd.read_csv(
    SOURCE_FILE,
    compression="gzip",
    usecols=features,
    low_memory=False
)

print("Original selected records:", len(df))

# Keep only loans with resolved outcomes
df = df[
    df["loan_status"].isin(
        ["Fully Paid", "Charged Off"]
    )
].copy()

# Create binary target
df["default"] = df["loan_status"].map({
    "Fully Paid": 0,
    "Charged Off": 1
})

# We no longer need the original text target
df.drop(columns=["loan_status"], inplace=True)

print("Resolved loans:", len(df))

print("\nTarget distribution:")
print(df["default"].value_counts())

print("\nTarget percentages:")
print(
    df["default"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nDataset shape:")
print(df.shape)

# Save our ML dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\nDataset saved to:")
print(OUTPUT_FILE)