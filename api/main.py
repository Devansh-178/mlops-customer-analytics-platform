from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from api.schemas import CustomerFeatures, PredictionResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"

ml_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded once at startup, not per-request -- this is the whole point
    ml_artifacts["preprocessor"] = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    ml_artifacts["model"] = joblib.load(ARTIFACTS_DIR / "model.joblib")
    yield
    ml_artifacts.clear()

app = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    input_df = pd.DataFrame([customer.model_dump()])
    X_transformed = ml_artifacts["preprocessor"].transform(input_df)

    prediction = ml_artifacts["model"].predict(X_transformed)[0]
    probability = ml_artifacts["model"].predict_proba(X_transformed)[0][1]

    return PredictionResponse(
        churn_prediction="Yes" if prediction == 1 else "No",
        churn_probability=round(float(probability), 4),
    )