import joblib
import pandas as pd
from pathlib import Path

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
    classification_report,
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "credit_risk_dataset.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = (
    MODEL_DIR
    / "credit_risk_model.joblib"
)

MODEL_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(
    str(DATA_FILE),
    low_memory=False,
)

print("Dataset loaded.")
print("Shape:", df.shape)


# --------------------------------------------------
# Feature transformations
# --------------------------------------------------

df["term"] = (
    df["term"]
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

df["emp_length"] = (
    df["emp_length"]
    .map(employment_mapping)
)

df["fico_score"] = (
    df["fico_range_low"]
    + df["fico_range_high"]
) / 2

df = df.drop(
    columns=[
        "fico_range_low",
        "fico_range_high",
    ]
)


# --------------------------------------------------
# Features and target
# --------------------------------------------------

X = df.drop(columns=["default"])
y = df["default"]


# --------------------------------------------------
# Train / Validation / Test Split
# --------------------------------------------------

# 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

# Remaining 30% becomes:
# 15% validation + 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

print("\nDataset split:")
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True))

print("\nValidation target distribution:")
print(y_val.value_counts(normalize=True))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True))


# --------------------------------------------------
# Feature groups
# --------------------------------------------------

categorical_features = [
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
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
            SimpleImputer(
                strategy="median"
            ),
        ),
    ]
)


# --------------------------------------------------
# Categorical preprocessing
# --------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        ),
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
            numerical_features,
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features,
        ),
    ]
)


# --------------------------------------------------
# HistGradientBoosting model
# --------------------------------------------------

classifier = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_iter=200,
    max_leaf_nodes=31,
    l2_regularization=1.0,
    random_state=42,
)


# --------------------------------------------------
# Complete pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            classifier,
        ),
    ]
)


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nTraining HistGradientBoosting...")
print("This may take several minutes.")

model.fit(
    X_train,
    y_train,
)

print("Training completed.")


# --------------------------------------------------
# Threshold Selection — Validation Set Only
# --------------------------------------------------

print(
    "\nSelecting threshold using validation set..."
)

y_val_prob = model.predict_proba(
    X_val
)[:, 1]

thresholds = [
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]

best_threshold = thresholds[0]
best_f1 = -1.0

print("\nValidation Threshold Comparison")
print("-" * 65)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'Accuracy':<12}"
)

for threshold in thresholds:

    y_val_pred = (
        y_val_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        y_val_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        y_val_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        y_val_pred,
        zero_division=0,
    )

    accuracy = accuracy_score(
        y_val,
        y_val_pred,
    )

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{accuracy:<12.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


print(
    "\nSelected threshold:",
    best_threshold,
)

print(
    "Validation F1:",
    round(best_f1, 4),
)


# --------------------------------------------------
# Final Evaluation — Untouched Test Set
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINAL TEST SET EVALUATION")
print("=" * 65)

y_test_prob = model.predict_proba(
    X_test
)[:, 1]

y_test_pred = (
    y_test_prob >= best_threshold
).astype(int)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred,
)

test_precision = precision_score(
    y_test,
    y_test_pred,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    y_test_pred,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    y_test_pred,
    zero_division=0,
)

test_roc_auc = roc_auc_score(
    y_test,
    y_test_prob,
)

print(
    f"\nSelected Threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Accuracy:  "
    f"{test_accuracy:.4f}"
)

print(
    f"Precision: "
    f"{test_precision:.4f}"
)

print(
    f"Recall:    "
    f"{test_recall:.4f}"
)

print(
    f"F1 Score:  "
    f"{test_f1:.4f}"
)

print(
    f"ROC-AUC:   "
    f"{test_roc_auc:.4f}"
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_test_pred,
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=[
            "Non-default",
            "Default",
        ],
        zero_division=0,
    )
)


# --------------------------------------------------
# Save Model + Threshold + Final Metrics
# --------------------------------------------------

print(
    "\nSaving trained model package..."
)

model_package = {
    "model": model,
    "threshold": float(
        best_threshold
    ),
    "metrics": {
        "accuracy": float(
            test_accuracy
        ),
        "precision": float(
            test_precision
        ),
        "recall": float(
            test_recall
        ),
        "f1_score": float(
            test_f1
        ),
        "roc_auc": float(
            test_roc_auc
        ),
    },
}

joblib.dump(
    model_package,
    MODEL_FILE,
)

print(
    "Model package saved successfully:"
)

print(MODEL_FILE)

print(
    "Saved threshold:",
    best_threshold,
)

print("\nSaved final test metrics:")

for metric_name, metric_value in (
    model_package["metrics"].items()
):
    print(
        f"{metric_name}: "
        f"{metric_value:.4f}"
    )