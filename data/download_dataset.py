import kagglehub

# Download the LendingClub dataset
path = kagglehub.dataset_download(
    "wordsforthewise/lending-club"
)

print("Dataset downloaded successfully!")
print("Path to dataset files:", path)