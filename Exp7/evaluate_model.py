"""Model evaluation stage for DVC pipeline"""
import json
import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Create metrics directory
Path("metrics").mkdir(exist_ok=True)

# Load test data
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Load trained model
with open("models/model.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Make predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)

# Calculate metrics
metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, average="weighted")),
    "recall": float(recall_score(y_test, y_pred, average="weighted")),
    "f1": float(f1_score(y_test, y_pred, average="weighted")),
    "roc_auc": float(roc_auc_score(y_test, y_pred_proba, multi_class="ovr", average="weighted")),
}

# Save metrics
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Save detailed metrics
with open("metrics/confusion_matrix.json", "w") as f:
    json.dump(confusion_matrix(y_test, y_pred).tolist(), f)

with open("metrics/classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred, target_names=["setosa", "versicolor", "virginica"]))

print("✓ Model evaluation completed")
print("\nMetrics:")
for metric_name, metric_value in metrics.items():
    print(f"  - {metric_name}: {metric_value:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["setosa", "versicolor", "virginica"]))
