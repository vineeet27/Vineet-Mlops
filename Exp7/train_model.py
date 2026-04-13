"""Model training stage for DVC pipeline"""
import json
import pickle
import pandas as pd
import yaml
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Create models directory
Path("models").mkdir(exist_ok=True)

# Load training data
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()

# Load hyperparameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

model_params = params["train"]["model_params"]

# Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(**model_params))
])

# Train model
pipeline.fit(X_train, y_train)

# Save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✓ Model training completed")
print(f"  - Model parameters: {model_params}")
print(f"  - Model saved to: models/model.pkl")
