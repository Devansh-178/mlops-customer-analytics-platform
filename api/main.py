# api/main.py
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from api.schemas import (
    CustomerFeatures,
    PredictionResponse,
    BatchPredictionResult,
    BatchPredictionResponse,
)

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


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: Request):
    start = time.perf_counter()
    payload = await request.json()

    results: List[BatchPredictionResult] = []
    valid_customers = []
    valid_indices = []

    for idx, row in enumerate(payload):
        customer_id = row.get("customerID")
        try:
            customer = CustomerFeatures.model_validate(row)
            valid_customers.append(customer)
            valid_indices.append((idx, customer_id))
        except ValidationError as e:
            results.append(BatchPredictionResult(
                row_index=idx,
                customer_id=customer_id,
                status="error",
                error=e.errors()[0]["msg"],
            ))

    if valid_customers:
        try:
            df = pd.DataFrame([c.model_dump() for c in valid_customers])
            X_transformed = ml_artifacts["preprocessor"].transform(df)
            probs = ml_artifacts["model"].predict_proba(X_transformed)[:, 1]
        except Exception:
            logger.exception("Batch prediction failed during model inference.")
            raise HTTPException(
                status_code=500,
                detail="An internal error occurred while generating batch predictions.",
            )

        for (idx, customer_id), prob in zip(valid_indices, probs):
            results.append(BatchPredictionResult(
                row_index=idx,
                customer_id=customer_id,
                status="success",
                prediction=PredictionResponse(
                    churn_probability=round(float(prob), 4),
                    churn_prediction="Yes" if prob >= 0.5 else "No",
                ),
            ))

    results.sort(key=lambda r: r.row_index)
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Batch prediction served | total=%d | succeeded=%d | failed=%d | latency_ms=%.2f",
        len(payload), len(valid_customers), len(payload) - len(valid_customers), latency_ms,
    )

    return BatchPredictionResponse(
        total=len(payload),
        succeeded=len(valid_customers),
        failed=len(payload) - len(valid_customers),
        results=results,
    )