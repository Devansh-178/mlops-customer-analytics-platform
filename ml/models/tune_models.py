"""
ml/models/tune_models.py

Day 4: Hyperparameter tuning for logistic regression and random forest,
building on the Day 3 baselines. Logs tuned runs to the same MLflow
experiment ("customer-churn-baseline") so they're directly comparable
against logreg_baseline_unweighted, logreg_baseline_balanced, and
rf_baseline_balanced.
"""

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("ml/artifacts")


def load_processed_data():
    X_train = joblib.load(PROCESSED_DIR / "X_train.joblib")
    X_test = joblib.load(PROCESSED_DIR / "X_test.joblib")
    y_train = joblib.load(PROCESSED_DIR / "y_train.joblib")
    y_test = joblib.load(PROCESSED_DIR / "y_test.joblib")
    return X_train, X_test, y_train, y_test


def evaluate_on_test(model, X_test, y_test):
    """Same metrics used for the Day 3 baseline comparison table."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }


def tune_logistic_regression(X_train, y_train):
    """
    GridSearchCV: small, cheap parameter space -> exhaustive search is
    tractable and gives certainty within the grid.

    class_weight is fixed to 'balanced'. Day 3 already established that
    balanced meaningfully improved recall on the churn class -- we're
    refining that specific model, not re-litigating the weighting choice.
    """
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
    }

    base_model = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",  # only solver that supports both l1 and l2 cleanly
        max_iter=1000,
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def tune_random_forest(X_train, y_train, n_iter=40):
    """
    RandomizedSearchCV: larger parameter space + heavier per-fit cost ->
    a full grid would be combinatorially expensive for little extra gain.
    Sampling n_iter combinations from distributions covers the space more
    efficiently than a fine grid (Bergstra & Bengio, 2012).

    class_weight fixed to 'balanced' for the same reason as logreg above.
    """
    param_distributions = {
        "n_estimators": randint(100, 500),
        "max_depth": [None, 5, 10, 15, 20, 30],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 20),
        "max_features": ["sqrt", "log2", None],
    }

    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        # n_jobs left at default (1) here on purpose -- see note below
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def log_tuned_run(run_name, search, X_test, y_test, search_type):
    """Log only the best estimator from the search as one MLflow run."""
    metrics = evaluate_on_test(search.best_estimator_, X_test, y_test)

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("stage", "tuned")
        mlflow.set_tag("search_type", search_type)

        mlflow.log_params(search.best_params_)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_metric("cv_best_f1", search.best_score_)

        for metric_name, value in metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        mlflow.sklearn.log_model(search.best_estimator_, "model")

        # Full search history as an artifact -- useful for the write-up,
        # and lets you show *how* you searched, not just the winner
        cv_results_path = ARTIFACTS_DIR / f"cv_results_{run_name}.csv"
        pd.DataFrame(search.cv_results_).to_csv(cv_results_path, index=False)
        mlflow.log_artifact(str(cv_results_path))
        cv_results_path.unlink()  # logged to MLflow; don't keep a redundant local copy

    print(f"{run_name}: {metrics}")
    return metrics


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("customer-churn-baseline")

    X_train, X_test, y_train, y_test = load_processed_data()

    print("Tuning logistic regression (GridSearchCV)...")
    logreg_search = tune_logistic_regression(X_train, y_train)
    log_tuned_run("logreg_tuned_gridsearch", logreg_search, X_test, y_test, "GridSearchCV")

    print("Tuning random forest (RandomizedSearchCV)...")
    rf_search = tune_random_forest(X_train, y_train)
    log_tuned_run("rf_tuned_randomsearch", rf_search, X_test, y_test, "RandomizedSearchCV")


if __name__ == "__main__":
    main()
