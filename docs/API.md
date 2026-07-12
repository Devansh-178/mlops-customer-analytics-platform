# API Layer — Design & Testing

## Overview
A FastAPI service exposing the trained churn model via a single `/predict`
endpoint. Accepts a raw-feature customer record (mirroring the original
Telco CSV row shape) and returns a churn prediction with probability.

## Framework choice: FastAPI
- Pydantic-based request validation is built in — no separate validation
  layer needed.
- Automatic OpenAPI/Swagger docs (`/docs`) generated from type hints,
  useful for manual testing and demoing without extra tooling.
- ASGI-native, though `/predict` is a synchronous endpoint since sklearn's
  `.predict()` is CPU-bound and wouldn't benefit from async.

## Model artifact
Logistic Regression (`class_weight="balanced"`) was promoted to production
over the tuned Random Forest. Both had comparable F1 (~0.61-0.63) and
ROC-AUC after tuning; when performance is tied, simplicity wins —
LR is smaller, faster at inference, and its coefficients are directly
interpretable, which matters when explaining predictions to stakeholders.
Retrained and persisted via `ml/train_final_model.py` → `ml/artifacts/model.joblib`.

## Startup-loading pattern
Both `preprocessor.joblib` and `model.joblib` are loaded once via FastAPI's
`lifespan` context manager, not per-request. Avoids repeated disk I/O and
deserialization on every call — the whole point of a persistent service
over ad-hoc script execution.

## Schema design: raw features in, not encoded vectors
`CustomerFeatures` (in `api/schemas.py`) mirrors the original Telco CSV
column shape. Callers shouldn't need to know the model internally uses
one-hot encoding — that mapping is the API's responsibility, done inside
`/predict` via the persisted `preprocessor.joblib`.

## Validation strategy: two layers, different jobs
- **Pydantic `Literal` types** on every categorical field: reject unknown
  category values immediately with a `422`, before the request reaches
  the preprocessor. This is a deliberate fail-fast choice — an API
  client sending a typo'd or unexpected category should get a clear,
  immediate error, not a silently-degraded prediction.
- **`OneHotEncoder(handle_unknown="ignore")`** in the preprocessor
  remains as a defense-in-depth backstop from training time. With
  `Literal` validation in place, it should rarely if ever trigger at
  the API boundary — but it's still the right choice for the
  preprocessor to guard against, e.g., programmatic callers that
  bypass schema validation entirely.
- Numeric fields (`tenure`, `MonthlyCharges`, `TotalCharges`) are
  constrained with `ge=0` bounds; `tenure` additionally capped at
  `le=100` as a sanity bound.

## Endpoints
| Method | Path       | Purpose                                  |
|--------|-----------|-------------------------------------------|
| GET    | `/health`  | Liveness check                           |
| POST   | `/predict` | Runs preprocessing + model inference     |

## Testing performed (manual, via Swagger UI)
1. **Valid request** — realistic high-risk profile (month-to-month,
   no online security/tech support, electronic check payment) →
   `200`, `churn_prediction: "Yes"`, probability `0.8063`. Confirms
   prediction direction aligns with Day 1 EDA churn drivers.
2. **Invalid category** — `InternetService: "FiberOptics"` (typo) →
   `422 literal_error`, pointing at the exact field and listing valid
   values. Confirms fail-fast validation works before reaching the model.
3. **Boundary case** — `tenure: 0`, `TotalCharges: 0.0` (the same edge
   case behind the Day 1 whitespace bug) → `200`, `churn_prediction: "Yes"`,
   probability `0.8125`. No crash, no NaN propagation through the
   preprocessor's numeric scaling.
   - Note: whether `tenure=0` records were present in the training
     split (vs. dropped during Day 1 cleanup) has not yet been
     re-verified against `preprocessing.py`. The endpoint handles the
     input gracefully regardless; flagged here as an open item rather
     than an assumption.

## Known limitations / future work
- No authentication/rate limiting — acceptable for local development,
  would need addressing before any real deployment.
- Single-record prediction only; no batch endpoint yet.
- No request logging/monitoring — worth revisiting once this moves
  toward containerization .