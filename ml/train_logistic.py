import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

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


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(DATA_FILE, low_memory=False)


# --------------------------------------------------
# Basic feature transformations
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
# Identify column types
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

print("\nNumerical features:")
print(numerical_features)

print("\nCategorical features:")
print(categorical_features)


# --------------------------------------------------
# Numerical preprocessing
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
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
                handle_unknown="ignore"
            )
        )
    ]
)


# --------------------------------------------------
# Combine preprocessing
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
# Complete ML pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)


# --------------------------------------------------
# Train
# --------------------------------------------------

print("\nTraining Logistic Regression...")

model.fit(X_train, y_train)

print("Training completed.")


# --------------------------------------------------
# Predictions
# --------------------------------------------------

print("\nMaking predictions...")

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION RESULTS")
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