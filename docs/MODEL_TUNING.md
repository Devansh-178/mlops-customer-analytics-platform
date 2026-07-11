# Model Tuning — Day 4

## Objective

Take the three Day 3 baseline models (logistic regression unweighted, logistic
regression balanced, Random Forest balanced) and determine whether tuning
improves on them meaningfully, which features actually drive predictions, and
which model should be carried forward as the production model for the API
layer in Week 4.

## Search strategy

Two different search methods were used deliberately, not arbitrarily:

- **Logistic regression: GridSearchCV.** The parameter space is small (`C` ×
  `penalty`), so an exhaustive grid is cheap and gives certainty that the best
  combination *within the grid* was found. `class_weight="balanced"` was fixed
  rather than searched — Day 3 already established that balancing improved
  recall on the churn class; Day 4 refines *given* that choice, rather than
  re-litigating it.
- **Random Forest: RandomizedSearchCV.** RF's parameter space
  (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`,
  `max_features`) is large enough that a full grid would be combinatorially
  expensive, and each RF fit is heavier than a logreg fit. Sampling 40
  combinations from parameter distributions covers the space more efficiently
  than a fine-grained grid for the same compute budget (Bergstra & Bengio,
  2012). `class_weight="balanced"` was fixed here too, for the same
  apples-to-apples reasoning as logreg.

Both searches used `StratifiedKFold(n_splits=5)`, scoring on `f1` to stay
consistent with the primary metric justified in `docs/PROBLEM.md`.

## Results

| Run | Search type | F1 | ROC-AUC | Precision (churn) | Recall (churn) |
|---|---|---|---|---|---|
| logreg_baseline_balanced (Day 3) | — | 0.6143 | 0.8417 | 0.51 | 0.78 |
| **logreg_tuned_gridsearch** | GridSearchCV | 0.6183 | 0.8411 | 0.5095 | 0.7861 |
| rf_baseline_balanced (Day 3) | — | 0.6123 | 0.8373 | 0.52 | 0.74 |
| **rf_tuned_randomsearch** | RandomizedSearchCV | **0.6327** | **0.8423** | 0.5319 | 0.7807 |

**Tuned RF parameters:** `n_estimators=486`, `max_depth=15`,
`max_features='log2'`, `min_samples_leaf=7`, `min_samples_split=3`,
`class_weight='balanced'`, `criterion='gini'`, `random_state=42`.

**Tuned logreg parameters:** best `C` and `penalty` from the grid search (see
MLflow run `logreg_tuned_gridsearch` for exact values), `class_weight='balanced'`,
`solver='liblinear'`.

### Interpretation

Logistic regression barely moved with tuning (F1 +0.004, ROC-AUC essentially
flat). This is consistent with the Day 3 observation that all three baseline
models converged to similar F1 through different precision/recall tradeoffs —
it suggests the model is close to its structural ceiling as a linear
classifier, not under-tuned.

Random Forest moved more substantially (F1 +0.021 over its own baseline, and
it overtook logreg on both F1 and ROC-AUC). This is a meaningful revision to
the Day 3 conclusion: **default RF hyperparameters were masking RF's real
potential.** The Day 3 finding that "RF doesn't outperform logistic
regression" was true only for RF at default settings — once tuned, RF is the
strongest model produced so far on both target metrics.

## Feature importance analysis

Both tuned models were inspected for which features drive predictions.
Feature names were recovered from the fitted `ColumnTransformer` via
`get_feature_names_out()` to map raw coefficient/importance arrays back to
human-readable columns.

### Logistic regression (coefficient sign + magnitude)

Top drivers, by absolute coefficient:

| Feature | Coefficient | Direction |
|---|---|---|
| tenure | -0.979 | Longer tenure → lower churn risk (strongest signal overall) |
| Contract_Two year | -0.714 | Long contract → lower churn risk |
| Contract_Month-to-month | +0.606 | Flexible contract → higher churn risk |
| InternetService_DSL | -0.379 | DSL → lower churn risk |
| PaperlessBilling_Yes | +0.318 | Paperless billing → higher churn risk |
| InternetService_Fiber optic | +0.304 | Fiber → higher churn risk |
| TotalCharges | +0.292 | Higher lifetime charges → higher churn risk |

Because numeric features were standardized in the Week 2 preprocessing
pipeline, these coefficients are on a comparable scale — the standardization
done for convergence purposes also happens to make this coefficient
comparison meaningful, not just an incidental benefit.

### Random Forest (mean decrease in impurity)

Top drivers, by importance:

| Feature | Importance |
|---|---|
| Contract_Month-to-month | 0.146 |
| tenure | 0.117 |
| TotalCharges | 0.102 |
| MonthlyCharges | 0.069 |
| Contract_Two year | 0.063 |
| OnlineSecurity_No | 0.056 |
| InternetService_Fiber optic | 0.052 |

RF importances reflect magnitude only — they indicate *what* matters, not
*which direction* it pushes a prediction. This is a real interpretability gap
relative to logistic regression's signed coefficients, not just a formatting
difference.

### Cross-model agreement

Tenure, Contract_Month-to-month, Contract_Two year, TotalCharges, and
InternetService_Fiber optic all appear in the top tier of **both** models,
despite the two algorithms making fundamentally different assumptions (linear
decision boundary vs. an ensemble of threshold-based tree splits). Independent
convergence on the same features from structurally different models is
stronger evidence that these are genuine drivers of churn in the data, rather
than an artifact of one algorithm's biases.

**Business interpretation:** churn is driven primarily by contract/commitment
structure and account tenure, not by service-quality-adjacent features (e.g.
tech support usage). This points toward a retention strategy focused on
converting month-to-month customers to longer contracts early in their
tenure, since tenure itself is protective and month-to-month is consistently
the strongest risk signal across both models.

One caveat: `PaymentMethod_Electronic check` appears as a risk factor in both
models. This may be a genuine behavioral signal or a proxy for a segment not
directly captured elsewhere in the feature set — treated here as correlational,
not a claimed causal mechanism.

## Model selection decision

**Random Forest (tuned) was selected as the production model** carried
forward into Week 4's API layer.

Reasoning:
- RF outperforms tuned logistic regression on both F1 (0.6327 vs. 0.6183) and
  ROC-AUC (0.8423 vs. 0.8411), the two metrics justified as primary in
  `docs/PROBLEM.md`.
- The two models agree on the dominant churn drivers, which increases
  confidence that RF's edge reflects real (likely mild non-linear or
  interaction) structure in the data that a linear model can't fully capture,
  rather than overfitting to noise.

**Logistic regression is retained as a documented interpretable reference
model**, not discarded. Its coefficients provide clear, signed,
stakeholder-legible reasoning ("longer tenure reduces churn risk by X") that
RF cannot provide without additional tooling (e.g. SHAP values, out of scope
for Day 4). This is a deliberate scope decision: best-metric and
best-fit-for-purpose are not always the same model, and both are documented
here rather than silently defaulting to whichever number was higher.

## Known tradeoffs of the RF choice, carried into Week 4

- **Interpretability:** per-prediction explanations (e.g. "why did this
  specific customer get flagged") will require additional tooling like SHAP
  if the dashboard needs to surface reasoning to business users. Not solved
  in Day 4; noted for a future week.
- **Artifact size / inference cost:** RF with 486 trees is a larger artifact
  and requires walking every tree at inference time versus a single dot
  product for logreg. At this dataset's scale (7K rows, low-dimensional,
  batch or low-throughput serving), this is a non-issue — noted only because
  it would matter if the project's scope changed.
- **Reproducibility surface:** RF has 5 tuned hyperparameters vs. logreg's 2.
  All are logged to MLflow (`rf_tuned_randomsearch` run) and restated above,
  so the exact configuration is traceable.

## Artifacts produced

- `ml/models/tune_models.py` — GridSearchCV (logreg) and RandomizedSearchCV
  (RF), both logged to MLflow experiment `customer-churn-baseline` with
  `stage: tuned` tag.
- `ml/models/feature_importance.py` — pulls tuned models back from MLflow
  (no retraining), maps coefficients/importances to feature names.
- `ml/models/save_production_model.py` — persists the selected production
  model (tuned RF) to `ml/artifacts/model.joblib`, pulled directly from its
  MLflow run to guarantee it's byte-identical to the evaluated model, not a
  fresh retrain.
- `ml/artifacts/logreg_top_features.csv`, `ml/artifacts/rf_top_features.csv`
  — top-15 feature tables for both models.

## MLflow experiment state

All Day 3 and Day 4 runs live under experiment `customer-churn-baseline` in
`sqlite:///mlflow.db`, distinguishable by the `stage` tag (`baseline` vs.
`tuned`). This allows direct side-by-side comparison in the MLflow UI without
needing separate experiments.