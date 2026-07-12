"""
Trains the final production model on the persisted processed splits
and saves it as a versioned artifact for the API layer to load.
"""
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"

# NOTE: adjust these filenames if Day 3's script named them differently
X_train = joblib.load(PROCESSED_DIR / "X_train.joblib")
X_test = joblib.load(PROCESSED_DIR / "X_test.joblib")
y_train = joblib.load(PROCESSED_DIR / "y_train.joblib")
y_test = joblib.load(PROCESSED_DIR / "y_test.joblib")

model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print(f"F1 score:  {f1_score(y_test, preds):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.4f}")
print(classification_report(y_test, preds))

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, ARTIFACTS_DIR / "model.joblib")
print(f"Saved production model to {ARTIFACTS_DIR / 'model.joblib'}")