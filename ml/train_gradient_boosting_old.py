import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "credit_risk_dataset.csv"



MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODEL_DIR / "credit_risk_model.joblib"

MODEL_DIR.mkdir(exist_ok=True)





# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_FILE, low_memory=False)


# --------------------------------------------------
# Feature transformations
# --------------------------------------------------

df["term"] = (
    df["term"]
    .str.extract(r"(\d+)")
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
    "10+ years": 10
}

df["emp_length"] = df["emp_length"].map(employment_mapping)

df["fico_score"] = (
    df["fico_range_low"] + df["fico_range_high"]
) / 2

df = df.drop(
    columns=["fico_range_low", "fico_range_high"]
)


# --------------------------------------------------
# Features and target
# --------------------------------------------------

X = df.drop(columns=["default"])
y = df["default"]


# --------------------------------------------------
# Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Feature groups
# --------------------------------------------------

categorical_features = [
    "grade",
    "home_ownership",
    "verification_status",
    "purpose"
]

numerical_features = [
    column
    for column in X.columns
    if column not in categorical_features
]


# --------------------------------------------------
# Numerical preprocessing
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


# --------------------------------------------------
# Categorical preprocessing
# --------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# --------------------------------------------------
# Preprocessor
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# --------------------------------------------------
# Gradient Boosting model
# --------------------------------------------------

classifier = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_iter=200,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42
)


# --------------------------------------------------
# Complete pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ]
)


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nTraining HistGradientBoosting...")
print("This may take several minutes.")

model.fit(X_train, y_train)

print("Training completed.")

print("\nSaving trained model...")

joblib.dump(model, MODEL_FILE)

print("Model saved successfully:")
print(MODEL_FILE)




# --------------------------------------------------
# Predictions
# --------------------------------------------------

print("\nMaking predictions...")

y_probability = model.predict_proba(X_test)[:, 1]

# Test different classification thresholds
thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

print("\nThreshold Comparison")
print("-" * 70)
print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'Accuracy':<12}"
)

for threshold in thresholds:

    predictions = (
        y_probability >= threshold
    ).astype(int)

    print(
        f"{threshold:<12.2f}"
        f"{precision_score(y_test, predictions):<12.4f}"
        f"{recall_score(y_test, predictions):<12.4f}"
        f"{f1_score(y_test, predictions):<12.4f}"
        f"{accuracy_score(y_test, predictions):<12.4f}"
    )


# Keep 0.50 temporarily for the existing report below
# Selected classification threshold
SELECTED_THRESHOLD = 0.20

y_pred = (
    y_probability >= SELECTED_THRESHOLD
).astype(int)

print(
    f"\nSelected classification threshold: "
    f"{SELECTED_THRESHOLD}"
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)


print("\n" + "=" * 50)
print("HIST GRADIENT BOOSTING RESULTS")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Non-Default",
            "Default"
        ]
    )
)