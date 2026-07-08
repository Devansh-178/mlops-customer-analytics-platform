# Problem Definition: Customer Churn Prediction

## 1. Business Problem

Telecom companies lose significant recurring revenue when customers cancel
their subscriptions ("churn"). Acquiring a new customer typically costs far
more than retaining an existing one, so the ability to identify customers
who are *likely* to churn — before they actually leave — allows a business
to proactively intervene (e.g. targeted discounts, improved support outreach,
retention offers) rather than reactively losing revenue.

This project builds a system to predict, from a customer's account and
service usage data, whether they are likely to churn.

## 2. Target Variable

- **Column**: `Churn`
- **Values**: `Yes` / `No`
- **Definition**: Whether the customer left the company within the last
  observed billing period.

## 3. Problem Type

**Binary classification.**

Each customer is classified into one of two classes (`Churn = Yes` or
`Churn = No`) based on their account attributes (contract type, tenure,
monthly/total charges), service usage (internet type, streaming add-ons,
tech support), and demographic info (senior citizen status, dependents,
partner).

## 4. Class Distribution

From EDA (`notebooks/01_eda.ipynb`):

| Class | Count | Percentage |
|---|---|---|
| No (retained)  | 5,174 | 73.46% |
| Yes (churned)  | 1,869 | 26.54% |

This is a **moderately imbalanced** classification problem — roughly 3:1.
This has a direct consequence on how the model should be evaluated (see
below).

## 5. Success Metrics

**Primary metrics: F1-score and ROC-AUC — not raw accuracy.**

**Why not accuracy**: with a 73.46% / 26.54% split, a naive model that
always predicts "No churn" would score 73.46% accuracy while being
completely useless in practice — it would never correctly flag a single
churning customer. Accuracy alone would reward exactly the wrong behavior
here.

- **F1-score** balances precision (of the customers we flag as likely to
  churn, how many actually do?) and recall (of the customers who actually
  churn, how many did we catch?). This matters because both false positives
  (wasting a retention offer on a customer who wasn't leaving) and false
  negatives (missing a customer who does leave) have real costs.
- **ROC-AUC** measures how well the model separates the two classes across
  all possible decision thresholds, which is useful for comparing models
  independent of a single fixed cutoff — important since the "right"
  threshold for flagging a customer may be tuned later based on business
  cost tradeoffs (e.g. cost of a retention offer vs. cost of losing the
  customer).

Accuracy will still be reported for reference, but will not be used as the
deciding metric when comparing models in Week 2 and Week 3.

## 6. Known Data Quality Notes

- `TotalCharges` was stored as a string (`object`/`str` dtype) due to 11
  rows containing blank/whitespace values instead of numbers. These all
  correspond to customers with `tenure = 0` (i.e., brand-new customers who
  have not yet been billed). Fixed by converting to numeric and filling
  with `0`, which is the factually correct value rather than a statistical
  estimate.
- `customerID` is a unique identifier with no predictive value and will be
  dropped before model training.

## 7. Out of Scope (for now)

- Multi-class or churn-timing prediction (e.g. *when* a customer will
  churn) — this project predicts binary churn risk only.
- Causal analysis of *why* customers churn — this is a predictive, not
  explanatory, model in its base form (though SHAP-based explainability is
  a possible bonus feature per the roadmap).