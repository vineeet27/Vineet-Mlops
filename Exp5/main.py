from typing import List

import numpy as np
import seaborn as sns
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class IrisInput(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)


class PredictionResponse(BaseModel):
    predicted_class: str
    probabilities: List[float]


app = FastAPI(title="Iris Inference API (Dockerized)", version="1.0")


def train_exp5_model() -> Pipeline:
    """Train Logistic Regression model on Iris dataset"""
    iris = sns.load_dataset("iris")
    X = iris.iloc[:, :-1].values
    y = iris.iloc[:, -1].values

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression()),
        ]
    ).fit(X_train, y_train)


# Train model on startup
model = train_exp5_model()
classes = list(model.named_steps["model"].classes_)


@app.get("/health")
def health_check() -> dict:
    """Health check endpoint"""
    return {"status": "ok", "service": "Iris Inference API"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: IrisInput) -> PredictionResponse:
    """Make predictions on Iris features"""
    features = np.array(
        [[
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width,
        ]]
    )
    probabilities = model.predict_proba(features)[0].tolist()
    predicted_index = int(np.argmax(probabilities))
    return PredictionResponse(
        predicted_class=classes[predicted_index],
        probabilities=probabilities,
    )


@app.get("/")
def root() -> dict:
    """Root endpoint with API information"""
    return {
        "service": "Iris Inference API",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces (0.0.0.0) for Docker
    uvicorn.run(app, host="0.0.0.0", port=8000)
