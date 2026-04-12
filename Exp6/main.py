import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')
iris.drop_duplicates(inplace = True)

import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ensure MLflow UI reads the same local store for runs
tracking_dir = Path(__file__).resolve().parent / "mlruns"
mlflow.set_tracking_uri(tracking_dir.as_uri())
# mlflow.set_tracking_uri("sqlite:///D:/WORK FROM HOME/Github 2/Shrikant_mlops/Exp_6/mlflow.db")
mlflow.set_experiment("exp_6_Trial_2")

# logging start kar
mlflow.sklearn.autolog()

with mlflow.start_run():

    # Assume last column is target
    X = iris.iloc[:, :-1].values
    y = iris.iloc[:, -1].values

    # Parameters (you can tune these later)
    test_size = 0.2
    random_state = 42

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Model parameter
    model = LogisticRegression(max_iter=200)

    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)

    # # Manual logging kar
    # mlflow.log_param("test_size", test_size)
    # mlflow.log_param("random_state", random_state)
    # mlflow.log_metric("accuracy", acc)

    # Print results (for your notebook)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))