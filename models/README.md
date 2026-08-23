# models/

Serialized model artifacts live here at runtime — they are **not committed**
to git (see `.gitignore`).

## Current production artifact

- `churn_classifier_v0.1.0.joblib` — full pipeline (shared preprocessing +
  `LogisticRegression`, `class_weight="balanced"`)
- Test metrics: ROC-AUC **0.842**, recall **0.783**, average precision **0.633**

Algorithm selection was metric-driven: a `HistGradientBoostingClassifier`
candidate scored lower on both headline metrics (ROC-AUC 0.828, recall 0.690),
so the logistic baseline won. Both candidates were trained through the same
CLI with identical split and weighting policy — see
`churn_classifier_v0.0.1-baseline_metrics.json` vs the v0.1.0 report.

Regenerate with:

```
uv run python -m training.train --algorithm logistic_regression --model-version 0.1.0
```

## Why are artifacts (mostly) not committed?

Artifacts are reproducible: retraining via `training/train.py` with the fixed
seed regenerates them deterministically, so experiments stay out of git by
default (`models/*` is gitignored). **Blessed production versions are
whitelisted explicitly** in `.gitignore`: at ~9 KB the cost is negligible and
an evaluator gets a working service straight after cloning (design requirement
NF8), with no LFS or download step.

## Distribution decision (pending)

How consumers get the artifact is deliberately deferred. The candidates:

1. **Plain git commit** — viable here (artifact ≈ 9 KB), best clone-and-run UX.
2. **Git LFS** — only if artifacts grow large; adds setup friction.
3. **GitHub Release asset + `scripts/download_model.sh`** — keeps the repo light,
   adds one download step (acceptable if scripted).

Each trained run writes two files into this folder:

- `<model_name>_v<version>.joblib` — full sklearn pipeline (preprocessing + model)
- `<model_name>_v<version>_metrics.json` — training date, dataset hash, evaluation metrics
