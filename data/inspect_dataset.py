import pandas as pd

file_path = (
    r"C:\Users\shiva\.cache\kagglehub\datasets"
    r"\wordsforthewise\lending-club\versions\3"
    r"\accepted_2007_to_2018Q4.csv.gz"
)

# Read only 5 rows from the compressed dataset
df = pd.read_csv(
    file_path,
    compression="gzip",
    nrows=5,
    low_memory=False
)

print("Dataset shape:")
print(df.shape)

print("\nNumber of columns:")
print(len(df.columns))

print("\nColumn names:")
for column in df.columns:
    print(column)