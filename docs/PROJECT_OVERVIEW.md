# MLOps Customer Analytics Platform — Full Summary & Presentation Guide

*Covers Day 1 – Day 4 (Weeks 1–3 of the six-week plan)*

---

## PART 1: What We've Actually Built, Chronologically

### Day 1 (Week 1) — Data Understanding

**What happened:**
- Loaded the IBM Telco Customer Churn dataset (7,043 rows × 21 columns).
- Ran exploratory data analysis in `notebooks/01_eda.ipynb`.
- Found a data quality bug: `TotalCharges` was stored as a string, and 11 rows
  (all with `tenure == 0`, i.e. brand-new customers) had whitespace instead of
  a number. Fixed by converting to float and imputing those 11 rows as 0.
- Confirmed class imbalance: ~73% of customers did **not** churn, ~27% did.
- Wrote `docs/PROBLEM.md` — business context, why F1/ROC-AUC were chosen as
  the evaluation metrics (not plain accuracy), and data quality notes.

**Output artifacts:** `notebooks/01_eda.ipynb`, `docs/PROBLEM.md`

---

### Day 2 (Week 2) — Preprocessing Pipeline

**What happened:**
- Built `ml/pipeline/preprocessing.py`, a reusable pipeline with four
  functions: loading data, building a `ColumnTransformer`, splitting data,
  and running the full pipeline end to end.
- Preprocessing design:
  - **Numeric features** → `StandardScaler` (mean 0, std 1).
  - **Categorical features** → `OneHotEncoder(handle_unknown="ignore",
    drop="if_binary", sparse_output=False)`.
  - **Stratified 80/20 train/test split** — preserves the 73/27 churn ratio
    in both sets.
  - **Fit the preprocessor on train data only**, then transform both train
    and test — the single most important rule in the whole pipeline (see
    Part 2 below for why).
  - Saved the fitted preprocessor to `ml/artifacts/preprocessor.joblib` so
    it can be reused identically later (e.g., in the API).
- Verified output shape: (5634, 40) train matrix, (1409, 40) test matrix —
  40 columns because one-hot encoding expanded the categorical columns.
- Wrote `docs/PREPROCESSING.md`.

**Output artifacts:** `ml/pipeline/preprocessing.py`,
`ml/artifacts/preprocessor.joblib`, `docs/PREPROCESSING.md`

---

### Day 3 (Week 3, part 1) — Experiment Tracking + Baselines

**What happened:**
- Modified the pipeline to persist the actual processed train/test arrays
  (`X_train`, `X_test`, `y_train`, `y_test`) as `.joblib` files in
  `data/processed/` — so every later script loads the *exact same* data,
  instead of silently re-running preprocessing and risking drift.
- Fixed a `.gitignore` gap: `ml/artifacts/*.joblib` wasn't excluded, so
  binary artifacts were at risk of being committed. Cleaned up the
  convention: `ml/models/` = code only, `ml/artifacts/` = all binaries.
- Installed MLflow 3.14.0. Hit a deprecation warning on the default
  filesystem tracking backend, switched to a SQLite backend:
  `mlflow.set_tracking_uri("sqlite:///mlflow.db")`.
- Trained and logged three baseline models under one MLflow experiment,
  `"customer-churn-baseline"`:

  | Run | class_weight | F1 | ROC-AUC | Precision | Recall |
  |---|---|---|---|---|---|
  | logreg_baseline_unweighted | None | 0.6061 | 0.8420 | 0.66 | 0.56 |
  | logreg_baseline_balanced | balanced | 0.6143 | 0.8417 | 0.51 | 0.78 |
  | rf_baseline_balanced | balanced | 0.6123 | 0.8373 | 0.52 | 0.74 |

- **Key finding:** all three landed at nearly the same F1, but got there
  through very different precision/recall tradeoffs. Random Forest did not
  beat logistic regression — an early signal that the relationship in the
  data might be largely linear (later revised on Day 4, see below).
- Wrote `docs/BASELINE_MODELS.md`.

**Output artifacts:** `data/processed/*.joblib`, `ml/models/train_baseline.py`,
`mlflow.db`, `docs/BASELINE_MODELS.md`

---

### Day 4 (Week 3, part 2) — Tuning, Feature Importance, Model Selection

**What happened:**

**1. Hyperparameter tuning** (`ml/models/tune_models.py`):
- Logistic regression tuned with **GridSearchCV** (small parameter space —
  `C`, `penalty` — exhaustive search is cheap and certain).
- Random Forest tuned with **RandomizedSearchCV**, 40 sampled combinations
  (large parameter space — `n_estimators`, `max_depth`, `min_samples_split`,
  `min_samples_leaf`, `max_features` — a full grid would be too expensive).
- Both used `StratifiedKFold(5)`, scored on `f1` to stay consistent with the
  metric justified in `docs/PROBLEM.md`.
- `class_weight="balanced"` was fixed (not searched) for both — Day 3
  already decided that was the right choice; Day 4 only refined *within*
  that choice, keeping the comparison apples-to-apples.
- Results:

  | Run | F1 | ROC-AUC |
  |---|---|---|
  | logreg_tuned_gridsearch | 0.6183 | 0.8411 |
  | **rf_tuned_randomsearch** | **0.6327** | **0.8423** |

- **Key finding:** logistic regression barely moved with tuning (near its
  structural ceiling as a linear model). Random Forest moved meaningfully
  and became the best model overall — revising the Day 3 conclusion. RF
  wasn't a bad fit for the data; it was just badly configured at defaults.

**2. Feature importance analysis** (`ml/models/feature_importance.py`):
- Pulled both tuned models back from MLflow (no retraining) and mapped
  their coefficients/importances back to real column names using the
  fitted preprocessor's `get_feature_names_out()`.
- Both models independently agreed that **tenure** and **contract type**
  (month-to-month = risk, two-year = protective) are the dominant churn
  drivers — strong evidence since two structurally different model types
  converged on the same signal.
- Logistic regression gives **signed** coefficients (direction + magnitude);
  Random Forest gives **magnitude only** — a real interpretability gap, not
  just a formatting difference.

**3. Model selection:**
- **Random Forest (tuned) selected as the production model** — better F1
  and ROC-AUC, and its top drivers agree with logreg's, increasing
  confidence the improvement is real signal, not noise.
- **Logistic regression retained as an interpretable reference model** —
  not deleted. Its signed coefficients are useful for stakeholder-facing
  explanations that raw RF importances can't give without extra tooling
  (e.g. SHAP, not yet built).
- Production model persisted to `ml/artifacts/model.joblib`, pulled
  directly from its MLflow run (not retrained) so it's guaranteed to be the
  exact model that was evaluated.
- Wrote `docs/MODEL_TUNING.md` covering all of the above with full reasoning.

**Output artifacts:** `ml/models/tune_models.py`,
`ml/models/feature_importance.py`, `ml/models/save_production_model.py`,
`ml/artifacts/model.joblib`, `ml/artifacts/logreg_top_features.csv`,
`ml/artifacts/rf_top_features.csv`, `docs/MODEL_TUNING.md`

---

## PART 2: Concepts You Need to Actually Understand (Not Just Have Working)

This is the "if someone asks you *why*, not just *what*" section. If you can
explain each of these in your own words without looking at the docs, you
understand the project — not just executed it.

### Data & preprocessing concepts

- **Why fit-on-train-only matters (data leakage).** If you fit a
  `StandardScaler` or `OneHotEncoder` on the *full* dataset before splitting,
  the scaler's mean/std (or the encoder's known categories) are computed
  using information from the test set. That means your test set is no
  longer a clean, unseen simulation of "new" data — the model's evaluation
  becomes optimistic and unreliable. Fitting on train only, then applying
  that same fitted transformer to test, keeps the test set honest.
- **Why stratified splitting matters.** With a 73/27 class imbalance, a
  random (non-stratified) split could by chance put too many or too few
  churn cases in the test set, making your evaluation metrics noisy or
  misleading. Stratifying forces both splits to preserve the original ratio.
- **Why StandardScaler for numeric, OneHotEncoder for categorical.**
  Distance/gradient-based models (like logistic regression) are sensitive to
  feature scale — a feature ranging 0–10,000 (TotalCharges) would dominate
  one ranging 0–1 (tenure in years, say) if left unscaled. Categorical
  features have no inherent numeric order, so one-hot encoding avoids
  falsely implying one (e.g. encoding contract type as 0/1/2 would imply
  "two year" is "more" than "month-to-month," which is meaningless).
- **`handle_unknown="ignore"`.** If a category shows up at prediction time
  that the encoder never saw during training (e.g. a new payment method),
  this setting prevents a crash — it just encodes it as all-zeros instead
  of raising an error. Important for production robustness.

### Model & metric concepts

- **Why F1 and ROC-AUC instead of accuracy.** With ~73% "no churn," a model
  that *always* predicts "no churn" gets 73% accuracy while being useless.
  F1 balances precision and recall on the minority (churn) class, so it
  actually reflects whether the model catches real churners. ROC-AUC
  measures how well the model ranks churners above non-churners across all
  possible decision thresholds, independent of the specific cutoff chosen.
- **Precision vs. recall, and why `class_weight="balanced"` shifts them.**
  Precision: of everyone predicted to churn, how many actually did.
  Recall: of everyone who actually churned, how many did we catch.
  Balancing class weights makes misclassifying the minority (churn) class
  more costly during training, which typically trades some precision for
  higher recall — useful when missing an actual churner (a false negative)
  is more costly to the business than a false alarm.
- **Why logistic regression coefficients are directional and RF importances
  are not.** Logistic regression fits a linear equation — each feature gets
  one coefficient whose sign tells you which way it pushes the prediction.
  Random Forest is an ensemble of decision trees; "importance" is measured
  as how much each feature reduces impurity (Gini) across all the splits it
  was used in, summed and averaged — a magnitude with no built-in sense of
  direction.
- **Why GridSearchCV vs. RandomizedSearchCV.** Grid search tries every
  combination in a defined grid — exhaustive but scales multiplicatively
  with each added parameter. Random search samples a fixed number of random
  combinations from distributions — often more efficient in practice
  because usually only a few hyperparameters matter much, and random
  sampling covers that space better per unit of compute than a fine grid
  does (Bergstra & Bengio, 2012). Use grid search when the space is small
  and cheap; random search when it's large or each fit is expensive.
- **Why RF beat logreg only after tuning.** At default settings, RF's trees
  can be too shallow/too specific to be competitive. Tuning `max_depth`,
  `min_samples_leaf`, etc. lets RF properly capture whatever mild
  non-linearity or feature interactions exist in the data that a linear
  model structurally cannot represent — which is exactly what its edge over
  tuned logreg suggests is happening here.

### MLOps / tooling concepts

- **Why MLflow, and why SQLite backend.** Manually tracking model
  parameters/metrics across many experiments in a spreadsheet or notebook
  cells doesn't scale and isn't reproducible. MLflow logs every run's
  parameters, metrics, and model artifact automatically, queryable later.
  The filesystem backend was deprecated in this MLflow version, so a SQLite
  database (`mlflow.db`) is used instead — still local, no server needed,
  but a proper structured store rather than loose files.
- **Why pull models back from MLflow instead of retraining.** Retraining
  Random Forest could produce a *slightly* different model even with a
  fixed `random_state`, depending on library versions or environment.
  Loading the exact model that was already logged guarantees the file you
  ship (`model.joblib`) is byte-identical to the one whose metrics you
  actually evaluated and wrote down.
- **Why "single source of truth" artifacts matter.** Persisting
  `preprocessor.joblib`, the processed data splits, and `model.joblib` as
  fixed files (rather than recomputing them fresh in every script) prevents
  silent drift — where two scripts might otherwise process the same raw
  data slightly differently over time and produce inconsistent results
  without anyone noticing.
- **Why `.gitignore` hygiene matters for this project specifically.**
  Binary artifacts (`.joblib` files, `mlflow.db`, `mlruns/`) don't belong in
  git — they're large, not human-diffable, and regeneratable from code.
  Git should track code and small reference outputs (like the feature
  importance CSVs); MLflow and local artifact folders are the system of
  record for models and data.

---

## PART 3: How to Present / Demo This Project

A suggested order for walking someone through it — interviewer, portfolio
reviewer, or just yourself revisiting it later.

1. **Open with the business problem, not the code.** State it in one
   sentence: "Predict which telecom customers are likely to churn, so the
   business can intervene before they leave." Mention the ~73/27 imbalance
   and why that made accuracy the wrong metric — this immediately signals
   you understand the problem before the tooling.

2. **Show the data quality finding.** The `TotalCharges` whitespace bug is a
   great two-sentence story: found it, understood *why* it happened
   (brand-new customers with `tenure == 0` had no charges yet), fixed it
   deliberately rather than blindly dropping rows.

3. **Walk through the preprocessing pipeline design**, emphasizing
   fit-on-train-only. This is the single most common real-world ML mistake
   (data leakage) — correctly avoiding it and being able to explain why is
   a strong signal of rigor.

4. **Show the MLflow UI** (`mlflow ui --backend-store-uri sqlite:///mlflow.db`),
   comparing baseline vs. tuned runs side by side. This is your best visual
   — it shows real experiment tracking, not just a final number.

5. **Present the tuning methodology as a deliberate choice**, not a
   checkbox: GridSearch for the small logreg space, RandomizedSearch for
   the larger RF space, and *why*.

6. **Present feature importance as the "so what."** This is where you tie
   the technical work back to business value: contract type and tenure
   dominate, both models agree, here's what that suggests for a retention
   strategy. This is usually the most interesting part of the demo for a
   non-technical audience.

7. **Present the model selection decision as reasoning, not a scoreboard.**
   Don't just say "RF won." Say: "RF had better metrics, both models agreed
   on the underlying drivers, so I trusted the improvement was real — but I
   kept logistic regression as a reference model because its interpretability
   has real value for stakeholder communication, and that's a deliberate
   tradeoff I made rather than defaulting to the highest number." This
   single point differentiates a portfolio project from a Kaggle-notebook
   exercise.

8. **Close with the project's structure itself as a talking point:**
   reproducible pipeline, experiment tracking, single-source-of-truth
   artifacts, a doc for every major decision, conventional commits. This
   shows you're building toward *production* practice, not just a one-off
   model.

9. **If asked "what's next"** — be ready with the honest answer: Week 4
   wraps this in an API, Week 5 adds a dashboard, Week 6 handles
   deployment. Framing remaining work confidently, rather than as
   unfinished business, reads well.

### Quick presentation checklist

- [ ] Can explain the business problem and metric choice without notes
- [ ] Can explain the `TotalCharges` bug and fix in under 30 seconds
- [ ] Can explain fit-on-train-only / data leakage clearly
- [ ] Can show the MLflow UI live, baseline vs. tuned side by side
- [ ] Can explain GridSearch vs. RandomizedSearch choice and why
- [ ] Can name the top 3 churn drivers and what they mean for the business
- [ ] Can explain why RF was chosen over logreg, and why logreg wasn't deleted
- [ ] Repo is clean (`git status` shows nothing pending) before any demo
- [ ] All four docs (`PROBLEM.md`, `PREPROCESSING.md`, `BASELINE_MODELS.md`,
      `MODEL_TUNING.md`) are proofread and consistent with each other
