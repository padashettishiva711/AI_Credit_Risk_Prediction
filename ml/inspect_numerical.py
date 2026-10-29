import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "credit_risk_dataset.csv"

numerical_columns = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
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
    "pub_rec_bankruptcies"
]

print("Loading numerical columns...")

df = pd.read_csv(
    DATA_FILE,
    usecols=numerical_columns
)

print("\nNumerical summary:")
print(
    df[numerical_columns]
    .describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
    .T
    .to_string()
)