# api/main.py
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import CustomerFeatures, PredictionResponse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("churn_api")

ml_artifacts = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts...")
    ml_artifacts["preprocessor"] = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    ml_artifacts["model"] = joblib.load(ARTIFACTS_DIR / "model.joblib")
    logger.info("Artifacts loaded successfully.")
    yield
    ml_artifacts.clear()

app = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    start = time.perf_counter()

    try:
        input_df = pd.DataFrame([customer.model_dump()])
        
        X_transformed = ml_artifacts["preprocessor"].transform(input_df)
        prediction = ml_artifacts["model"].predict(X_transformed)[0]
        probability = ml_artifacts["model"].predict_proba(X_transformed)[0][1]
    except Exception:
        logger.exception("Prediction failed for incoming request.")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while generating the prediction.",
        )

    latency_ms = (time.perf_counter() - start) * 1000
    result = "Yes" if prediction == 1 else "No"

    logger.info(
        "Prediction served | contract=%s | tenure=%d | result=%s | probability=%.4f | latency_ms=%.2f",
        customer.Contract, customer.tenure, result, probability, latency_ms,
    )

    return PredictionResponse(
        churn_prediction=result,
        churn_probability=round(float(probability), 4),
    )