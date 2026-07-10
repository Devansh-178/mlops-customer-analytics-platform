# Preprocessing Pipeline

## Purpose

Transform the raw Telco Customer Churn data into a model-ready numeric feature
matrix, as a reusable script (`ml/pipeline/preprocessing.py`) rather than
notebook-only code — so the exact same transformation can be reused by the
Week 3 training script, the Week 4 API, and any future retraining job.

## Feature groups

The Week 1 EDA identified 16 categorical + 4 numeric columns by dtype, out of
20 columns (21 total minus `customerID`). That count includes `Churn` itself,
since it's stored as an object column — but `Churn` is the target, not an
input feature, so it's excluded from one-hot encoding and instead mapped
directly to a binary label (`Yes` -> 1, `No` -> 0).

- **Numeric input features (4):** `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
- **Categorical input features (15):** `gender`, `Partner`, `Dependents`, `PhoneService`,
  `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`,
  `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Dropped:** `customerID` (identifier, no predictive signal)
- **Target:** `Churn` (binary-mapped separately, not part of the feature transformer)

## Design decisions

- **`ColumnTransformer`**: bundles the numeric scaler and categorical encoder into
  a single fitted object, so one artifact (`preprocessor.joblib`) captures the
  entire transformation instead of two separate pieces to keep in sync.
- **`StandardScaler`** on numeric features: needed for scale-sensitive models
  (e.g. logistic regression); harmless for tree-based models.
- **`OneHotEncoder(handle_unknown="ignore")`**: if a category appears at
  inference time that wasn't seen during training, it encodes as all-zeros
  instead of raising an error.
- **`drop="if_binary"`**: for the 5 two-category columns, only one dummy column
  is kept (the second is redundant — knowing one tells you the other).
- **`sparse_output=False`**: dataset is small (7,043 rows), so a dense array is
  easier to inspect than a sparse matrix, with no meaningful memory cost.
- **Stratified train/test split**: Churn is ~73% No / ~27% Yes (Week 1 finding).
  A plain random split risks skewing that ratio differently between train and
  test; stratifying keeps both sets representative.
- **Fit on train only, `.transform()` (not `.fit_transform()`) on test**: prevents
  data leakage — the scaler's mean/std and the encoder's known categories must
  come only from the training data.

## Output

- Train shape: (5634, 40)
- Test shape: (1409, 40)
- 40 = 4 numeric + 5 binary categoricals (1 col each) + 9 three-category
  categoricals (3 cols each) + `PaymentMethod` (4 categories, 4 cols)
- Fitted preprocessor saved to `ml/artifacts/preprocessor.joblib`
