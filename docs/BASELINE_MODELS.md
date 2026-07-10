# Baseline Models

## Objective

Establish baseline performance using the Week 2 preprocessing pipeline, before any hyperparameter
tuning or feature engineering. Three models were trained and logged via MLflow for comparison.

## Models Trained

| Run name | Model | class_weight |
|---|---|---|
| `logreg_baseline_unweighted` | Logistic Regression | `None` |
| `logreg_baseline_balanced` | Logistic Regression | `"balanced"` |
| `rf_baseline_balanced` | Random Forest (n_estimators=200, max_depth=10) | `"balanced"` |

All models trained on the same processed train/test split persisted to `data/processed/`
(see `docs/PREPROCESSING.md`), ensuring a fair, leakage-free comparison.

## Why F1 and ROC-AUC

Accuracy is misleading here due to the ~73/27 class imbalance (documented in `docs/PROBLEM.md`) —
a model that always predicts "no churn" would score ~73% accuracy while being useless. F1 balances
precision and recall on the minority (churn) class; ROC-AUC measures ranking quality independent
of a specific decision threshold.

## Results

| Metric (class 1 / churn) | LogReg (unweighted) | LogReg (balanced) | Random Forest (balanced) |
|---|---|---|---|
| Precision | 0.66 | 0.51 | 0.52 |
| Recall | 0.56 | 0.78 | 0.74 |
| F1 | 0.61 | 0.61 | 0.61 |
| ROC-AUC | 0.8420 | 0.8417 | 0.8373 |
| Accuracy | 0.81 | 0.74 | 0.75 |

## Interpretation

**F1 alone suggests all three models are roughly equivalent (~0.61).** Looking only at that number
would be a mistake — the precision/recall breakdown reveals meaningfully different behavior:

- The **unweighted** logistic regression favors precision: when it predicts churn, it's usually
  right (0.66 precision), but it misses nearly half of actual churners (0.56 recall).
- The **balanced** logistic regression inverts this tradeoff: it catches 78% of churners, at the
  cost of a much higher false-positive rate (0.51 precision).
- **Random Forest, trained with the same `class_weight="balanced"` setting, sits almost exactly
  between the two logistic regression variants** on precision/recall, and did not clearly
  outperform logistic regression on ROC-AUC (0.8373 vs. 0.8420/0.8417).

Which model is "better" depends on the business cost of a false negative (a missed at-risk
customer) versus a false positive (unnecessary retention outreach to a customer who wouldn't
have churned) — this is a business decision, not a purely statistical one.

**Random Forest not outperforming logistic regression is a meaningful finding, not a null
result.** It suggests the relationship between these features and churn is largely linear/
additive, which is a legitimate reason to prefer the simpler, more interpretable logistic
regression model going forward — pending further work (hyperparameter tuning, feature
engineering) that could still change this picture.

## Next Steps

- Hyperparameter tuning for both model families
- Feature importance analysis (coefficients for logistic regression, feature importances for RF)
- Consider whether feature engineering (e.g., interaction terms) could close the gap further