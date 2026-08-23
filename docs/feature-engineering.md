# Feature Engineering Decisions

Decisions locked in during Étape 1 (EDA, see `notebooks/01-eda-exploration.ipynb`)
and to be implemented in `training/preprocessing.py` at Étape 2. Each decision
states *why*, because every choice in this repo must survive an interview question.

## Dataset

Telco Customer Churn — 7,043 rows × 21 columns, fetched by
`scripts/download_data.sh` (SHA256-pinned, never committed).

## Decisions

### 1. Target: `Churn` → binary flag

`Yes/No` mapped to `1/0`. Positive class = 26.54% of rows.

**Consequence**: the problem is moderately imbalanced.
- Split is stratified on the target, seed fixed for full reproducibility.
- Primary metrics are **recall and ROC-AUC**, not accuracy — for churn, missing
  a leaver costs more than a false retention alert (business framing from the
  design doc). Accuracy would look great while the model ignores churners.

### 2. `TotalCharges`: impute `0`, don't drop

The column arrives as text; `pd.to_numeric(..., errors="coerce")` exposes 11
non-numeric values. EDA shows those 11 rows are **exactly the `tenure == 0`
customers**: brand-new accounts that have not completed a billing cycle yet,
so their accumulated charges are legitimately zero.

- Imputation value: **0** (semantically exact — not median, not mean).
- Mechanism: inside the shared scikit-learn pipeline, so training and inference
  can never diverge (training-serving skew).
- Alternative rejected: dropping the 11 rows discards real (newest, most
  retention-relevant) customers for negligible gain.

Implementation note: coercion of string → numeric happens before imputation;
the pipeline owns both steps so raw JSON input at inference needs no manual cleanup.

### 3. `customerID`: dropped

Pure identifier, carries no generalizable signal. Kept only as request metadata
at the API layer (Étape 5) if useful for tracing responses.

### 4. Categorical features: one-hot encoding

All remaining categoricals are low-cardinality (2–4 values).
- `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` — unknown
  categories at inference degrade gracefully instead of crashing.
- `Contract` has a natural order (month-to-month < one-year < two-year);
  ordinal encoding is a valid alternative but one-hot keeps the pipeline simple
  and model-agnostic. Documented here as the trade-off it is.

### 5. Numeric features: scaled

`StandardScaler` on `tenure`, `MonthlyCharges`, `TotalCharges`.
- Neutral for tree-based models (scale-invariant), required for the logistic
  regression baseline used to sanity-check the gradient boosting model.
- Lives inside the shared pipeline, fitted on train only — no leakage.

### 6. Class imbalance handling: deferred to model config, not resampling

No SMOTE/upfront resampling in preprocessing. Rationale: keep the pipeline
deterministic and simple; `HistGradientBoostingClassifier` handles imbalance
reasonably via its own mechanisms, and class-weight equivalents are tuned at
Étape 3 where they are measurable via recall/ROC-AUC. Resampling is listed as a
documented fallback if recall proves poor.

## Non-decisions (explicitly out of scope)

- No new derived features (e.g., `charges_per_month = TotalCharges / tenure`)
  until a baseline exists — feature additions must prove themselves against
  the baseline metrics at Étape 3.
