import json
import joblib
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_ROOT / "data" / "credit_risk_dataset.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "credit_risk_model.joblib"
OUTPUT_FILE = PROJECT_ROOT / "models" / "ml_analytics.json"

# --------------------------------------------------
# Load existing model
# --------------------------------------------------

print("Loading existing ML model...")

package = joblib.load(MODEL_FILE)

model = package["model"]
threshold = package["threshold"]

print("Model loaded.")
print("Threshold:", threshold)

# --------------------------------------------------
# Load and prepare original dataset
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE, low_memory=False)

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

df["emp_length"] = df["emp_length"].map(employment_mapping)

df["fico_score"] = (
    df["fico_range_low"] + df["fico_range_high"]
) / 2

df = df.drop(columns=["fico_range_low", "fico_range_high"])

X = df.drop(columns=["default"])
y = df["default"]

# --------------------------------------------------
# Reproduce original test split
# --------------------------------------------------

print("\nReproducing original test split...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

print("Test samples:", len(X_test))

# --------------------------------------------------
# Evaluate existing model
# --------------------------------------------------

print("\nEvaluating existing model...")

probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= threshold).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)
roc_auc = roc_auc_score(y_test, probabilities)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1],
)

# --------------------------------------------------
# ROC curve
# --------------------------------------------------

fpr, tpr, thresholds = roc_curve(y_test, probabilities)

# Keep at most approximately 100 points for chart display.
step = max(1, len(fpr) // 100)

roc_points = [
    {
        "fpr": round(float(fpr[i]), 6),
        "tpr": round(float(tpr[i]), 6),
    }
    for i in range(0, len(fpr), step)
]

if roc_points[-1]["fpr"] != float(fpr[-1]):
    roc_points.append({
        "fpr": round(float(fpr[-1]), 6),
        "tpr": round(float(tpr[-1]), 6),
    })

# --------------------------------------------------
# Permutation feature importance
# --------------------------------------------------

print("\nCalculating permutation feature importance...")
print("This may take a few minutes.")

importance_result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=3,
    random_state=42,
    n_jobs=-1,
)

importance = [
    {
        "feature": str(feature),
        "importance": round(float(score), 6),
    }
    for feature, score in zip(
        X_test.columns,
        importance_result.importances_mean,
    )
]

importance.sort(
    key=lambda item: item["importance"],
    reverse=True,
)

# --------------------------------------------------
# Prepare analytics output
# --------------------------------------------------

analytics = {
    "threshold": float(threshold),
    "test_samples": int(len(X_test)),

    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
    },

    "confusion_matrix": {
        "true_negative": int(matrix[0][0]),
        "false_positive": int(matrix[0][1]),
        "false_negative": int(matrix[1][0]),
        "true_positive": int(matrix[1][1]),
    },

    "roc_curve": roc_points,

    "feature_importance": importance[:15],
}

# --------------------------------------------------
# Save analytics
# --------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(analytics, file, indent=4)

print("\nAnalytics generated successfully!")
print("Output:", OUTPUT_FILE)

print("\nModel metrics:")
for name, value in analytics["metrics"].items():
    print(f"{name}: {value:.4f}")

print("\nConfusion matrix:")
print(matrix)

print("\nTop 10 important features:")
for item in importance[:10]:
    print(
        f"{item['feature']}: {item['importance']:.6f}"
    )
