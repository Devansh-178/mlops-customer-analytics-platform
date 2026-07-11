"""
ml/models/save_production_model.py

Day 4: Persist the selected production model (tuned Random Forest) as a
standalone artifact, mirroring the single-source-of-truth pattern used
for the preprocessor and processed data splits.

Source of truth for *why* RF was selected: docs/MODEL_TUNING.md
Source of truth for the model's actual parameters: the MLflow run
"rf_tuned_randomsearch" under experiment "customer-churn-baseline".

This script pulls that logged model back out of MLflow rather than
retraining, so ml/artifacts/model.joblib is guaranteed to be byte-for-byte
the same model that was evaluated and logged on Day 4 -- not a fresh
retrain that could drift due to randomness in RF's bootstrapping.
"""

import joblib
import mlflow
from pathlib import Path

ARTIFACTS_DIR = Path("ml/artifacts")
EXPERIMENT_NAME = "customer-churn-baseline"
PRODUCTION_RUN_NAME = "rf_tuned_randomsearch"


def load_tuned_model(run_name):
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
    print(f"Loading model from run_id={run_id} (run_name='{run_name}')")
    model_uri = f"runs:/{run_id}/model"
    return mlflow.sklearn.load_model(model_uri)


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model = load_tuned_model(PRODUCTION_RUN_NAME)

    output_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(model, output_path)

    print(f"Saved production model to {output_path}")
    print(f"Model type: {type(model).__name__}")
    print(f"Params: {model.get_params()}")


if __name__ == "__main__":
    main()