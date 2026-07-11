"""
ml/models/feature_importance.py

Day 4, Step 2: Feature importance analysis for the tuned logistic
regression and random forest models. Pulls the already-logged models
back from MLflow (no re-training) and maps raw coefficient/importance
arrays back to human-readable feature names via the fitted preprocessor.
"""

import joblib
import mlflow
import pandas as pd
from pathlib import Path

ARTIFACTS_DIR = Path("ml/artifacts")
EXPERIMENT_NAME = "customer-churn-baseline"


def get_feature_names():
    """Recover human-readable column names from the fitted preprocessor."""
    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.joblib")
    return preprocessor.get_feature_names_out()


def load_tuned_model(run_name):
    """
    Fetch a specific run by name from the experiment and load its logged
    sklearn model. Avoids re-running the search just to inspect the model.
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
    )
    if runs.empty:
        raise ValueError(f"No run found with name '{run_name}'")

    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)


def logreg_importance(model, feature_names, top_n=15):
    """
    Coefficient sign + magnitude. Positive coefficient -> pushes toward
    churn (class 1); negative -> pushes toward retention. Magnitude
    reflects strength given standardized inputs (StandardScaler means
    coefficients are already on a comparable scale -- this comparability
    is *why* we scaled numeric features in Week 2, not just for
    convergence).
    """
    coefs = model.coef_[0]
    df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    df["abs_coefficient"] = df["coefficient"].abs()
    df = df.sort_values("abs_coefficient", ascending=False).head(top_n)
    return df[["feature", "coefficient"]].reset_index(drop=True)


def rf_importance(model, feature_names, top_n=15):
    """
    RF feature_importances_ = mean decrease in impurity (Gini) attributable
    to each feature across all trees. Always positive, no directionality --
    tells you *what* matters, not *which way* it pushes the prediction.
    That directional gap vs. logreg is worth calling out explicitly.
    """
    importances = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def main():
    feature_names = get_feature_names()

    print("Loading tuned logistic regression from MLflow...")
    logreg_model = load_tuned_model("logreg_tuned_gridsearch")
    logreg_top = logreg_importance(logreg_model, feature_names)

    print("Loading tuned random forest from MLflow...")
    rf_model = load_tuned_model("rf_tuned_randomsearch")
    rf_top = rf_importance(rf_model, feature_names)

    print("\n=== Logistic Regression: Top 15 Coefficients ===")
    print(logreg_top.to_string(index=False))

    print("\n=== Random Forest: Top 15 Feature Importances ===")
    print(rf_top.to_string(index=False))

    # Save both for the docs write-up
    out_dir = Path("ml/artifacts")
    logreg_top.to_csv(out_dir / "logreg_top_features.csv", index=False)
    rf_top.to_csv(out_dir / "rf_top_features.csv", index=False)
    print(f"\nSaved to {out_dir}/logreg_top_features.csv and rf_top_features.csv")


if __name__ == "__main__":
    main()