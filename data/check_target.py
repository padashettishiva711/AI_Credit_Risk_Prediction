import pandas as pd

file_path = (
    r"C:\Users\shiva\.cache\kagglehub\datasets"
    r"\wordsforthewise\lending-club\versions\3"
    r"\accepted_2007_to_2018Q4.csv.gz"
)

# We only need the target column
df = pd.read_csv(
    file_path,
    compression="gzip",
    usecols=["loan_status"],
    low_memory=False
)

print("Number of records:")
print(len(df))

print("\nLoan status values:")
print(df["loan_status"].value_counts(dropna=False))