# models/

Serialized model artifacts live here at runtime — they are **not committed**
to git (see `.gitignore`).

## Why not commit artifacts?

- Model binaries bloat the repository history and are poor fits for plain git diffs.
- Artifacts must be reproducible: retraining via `training/train.py` with a fixed
  seed regenerates them deterministically.

## Distribution decision (pending)

How consumers get the artifact is deliberately deferred to Étape 3. The two
candidates, to be documented here once decided:

1. **Git LFS** — simple clone-and-go, but requires LFS quota and adds setup friction.
2. **GitHub Release asset + `scripts/download_model.sh`** — keeps the repo light,
   adds one download step (acceptable if scripted).

Each trained run writes two files into this folder:

- `<model_name>_v<version>.joblib` — full sklearn pipeline (preprocessing + model)
- `<model_name>_v<version>_metrics.json` — training date, dataset hash, evaluation metrics
