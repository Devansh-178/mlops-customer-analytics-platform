import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier

PROCESSED_DATA_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("ml/artifacts")

mlflow.set_tracking_uri("sqlite:///mlflow.db")  # SQLite backend, not deprecated filesystem store
mlflow.set_experiment("customer-churn-baseline")


def load_processed_data():
    X_train = joblib.load(PROCESSED_DATA_DIR / "X_train.joblib")
    X_test = joblib.load(PROCESSED_DATA_DIR / "X_test.joblib")
    y_train = joblib.load(PROCESSED_DATA_DIR / "y_train.joblib")
    y_test = joblib.load(PROCESSED_DATA_DIR / "y_test.joblib")
    return X_train, X_test, y_train, y_test

def train_logistic_regression(class_weight=None, run_name="logreg_baseline_unweighted"):
    X_train, X_test, y_train, y_test = load_processed_data()

    params = {
        "model_type": "LogisticRegression",
        "class_weight": str(class_weight),
        "max_iter": 1000,
        "random_state": 42,
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        model = LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metrics({
            "f1_score": f1,
            "roc_auc": roc_auc,
        })

        mlflow.sklearn.log_model(model, name="model")  # 'name' replaces deprecated 'artifact_path'

        print(f"\n=== Run: {run_name} ===")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC:  {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    return model, f1, roc_auc

def train_random_forest(class_weight=None, run_name="rf_baseline"):
    X_train, X_test, y_train, y_test = load_processed_data()

    params = {
        "model_type": "RandomForestClassifier",
        "class_weight": str(class_weight),
        "n_estimators": 200,
        "max_depth": 10,
        "random_state": 42,
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metrics({
            "f1_score": f1,
            "roc_auc": roc_auc,
        })

        mlflow.sklearn.log_model(model, name="model")

        print(f"\n=== Run: {run_name} ===")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC-AUC:  {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    return model, f1, roc_auc

if __name__ == "__main__":
    # Run 1: honest baseline, no imbalance correction
    train_logistic_regression(class_weight=None, run_name="logreg_baseline_unweighted")

    # Run 2: same model, correcting for the ~73/27 imbalance
    train_logistic_regression(class_weight="balanced", run_name="logreg_baseline_balanced")

    # Run 3: tree-based model, same imbalance correction for fair comparison
    train_random_forest(class_weight="balanced", run_name="rf_baseline_balanced")
