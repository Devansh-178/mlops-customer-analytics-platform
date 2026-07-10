"""
ml/pipeline/preprocessing.py

Preprocessing pipeline for the IBM Telco Customer Churn dataset.

What this script does:
1. Loads the raw CSV and re-applies the TotalCharges fix found in the Week 1 EDA.
2. Splits columns into numeric vs categorical feature groups.
3. Builds a single, reusable sklearn ColumnTransformer that:
   - scales the 4 numeric features (StandardScaler)
   - one-hot encodes the 15 categorical input features (OneHotEncoder)
4. Does a stratified train/test split (stratified because classes are ~73/27
   imbalanced, per Week 1 EDA).
5. Fits the transformer on TRAIN only, applies it to TRAIN and TEST (no leakage),
   and saves the fitted transformer to disk so Week 3 (model training) and
   Week 4 (API) can reuse the exact same transformation at inference time.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Paths — adjust RAW_DATA_PATH if your CSV lives somewhere else
# ---------------------------------------------------------------------------
RAW_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
ARTIFACTS_DIR = Path("ml/artifacts")

# ---------------------------------------------------------------------------
# Column groups — locked in from Week 1 EDA
# ---------------------------------------------------------------------------
ID_COL = "customerID"
TARGET_COL = "Churn"

# 4 numeric input features
NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# 15 categorical input features.
# Note: the EDA's "16 categorical columns" count includes Churn itself
# (it's dtype object), but Churn is the target, not an input feature —
# so it's handled separately below via a binary map, not one-hot encoded here.
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and fix the TotalCharges whitespace bug found in EDA."""
    df = pd.read_csv(path)

    # Week 1 finding: 11 rows have " " (whitespace) instead of a number,
    # all tied to tenure == 0 (brand-new customers, no charges yet).
    df["TotalCharges"] = df["TotalCharges"].replace(" ", pd.NA)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer that scales numeric features and one-hot
    encodes categorical features, all as a single fitted object.
    """
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",   # unseen category at inference -> all-zero row, no crash
        drop="if_binary",          # 2-category columns (e.g. Partner: Yes/No) -> 1 column, not 2
        sparse_output=False,       # dense array output, easy to inspect at this data size
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # explicitly drops customerID (and anything not listed)
    )
    return preprocessor


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Stratified train/test split.

    Stratified because Churn is ~73% No / ~27% Yes (Week 1 finding) — a plain
    random split risks skewing that ratio differently between train and test,
    which would make evaluation numbers less reliable.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL].map({"Yes": 1, "No": 0})

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def run_pipeline():
    """End-to-end run: load -> split -> fit on train -> transform both -> save artifact."""
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor()

    # Fit ONLY on train — the test set must stay "unseen," or scaling/encoding
    # statistics would leak information from test into train.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")

    print(f"Train shape (processed): {X_train_processed.shape}")
    print(f"Test shape  (processed): {X_test_processed.shape}")
    print(f"Total encoded feature count: {len(preprocessor.get_feature_names_out())}")
    print(f"Preprocessor saved to: {ARTIFACTS_DIR / 'preprocessor.joblib'}")

    return X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    run_pipeline()
