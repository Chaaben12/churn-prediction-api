# Architecture

## Overview

Two strictly separated flows share exactly one piece of ML code — the
preprocessing/classification pipeline — which structurally prevents
training-serving skew.

```
OFFLINE (not shipped in the API image)
┌────────────────────────────────────────────────────────────┐
│ raw CSV ──► training/train.py (CLI)                        │
│              ├─ churn_api.ml.preprocessing  (shared code)  │
│              ├─ stratified seeded split                    │
│              ├─ training/evaluate.py  (metrics)            │
│              └─► models/churn_classifier_vX.Y.Z.joblib     │
│                  models/..._metrics.json  (contract)       │
└────────────────────────────────────────────────────────────┘

ONLINE (src/churn_api, shipped in the image)
HTTP ─► api/routes ─► api/schemas (Pydantic, strict)
          │                │ validated CustomerProfile dicts
          ▼                ▼
      services/prediction_service ─► ml/predictor ─► ml/model_loader
                                          │                │
                                          ▼                ▼
                                   shared pipeline    artifact + report
                                    (predict_proba)    loaded once at boot

Cross-cutting: core/config (CHURN_* env), core/logging_config (JSON logs),
core/exceptions (domain errors ─► HTTP via handlers in main.py).
```

## Layer rules

| Layer | May import | Never imports |
| --- | --- | --- |
| `api` (routes/schemas) | `services`, `core`, `api.deps` | `ml` directly |
| `services` | `ml`, `core` | FastAPI `Request` |
| `ml` | `core` | HTTP concepts |
| `training` | `churn_api.ml.preprocessing` | anything HTTP |

Dependency inversion lives in `api/deps.py`: routers consume
`SettingsDep` / `PredictionServiceDep` from `app.state`, set once by the
lifespan — trivially swappable in tests.

## The artifact contract

The metrics JSON is not documentation; it is parsed and enforced at boot
(`ml/model_loader._parse_metadata`): version, algorithm, training date,
scikit-learn version, the **input column contract** and metrics must all be
present and well-typed or the process refuses to start.

Startup is deliberately fail-fast: the pipeline is deserialized exactly once
(`lru_cache`) inside the lifespan. A broken artifact means a dead container
that orchestrators will restart/detect — never a half-working service.

## Why preprocessing lives in the serving package

Pickles store functions *by reference*. The pipeline embeds
`_coerce_total_charges`, so unpickling requires its defining module on the
import path. Keeping that module under `training/` would force shipping the
whole offline tree in the runtime image. Its canonical home is therefore
`src/churn_api/ml/preprocessing.py`; training imports it from there.
This was proven necessary the hard way: CI's container-boot smoke test
caught the original layout failing with `ModuleNotFoundError: training`.

## Request lifecycle

1. `POST /predict` receives `{"customers": [...]}` (1–1000 profiles).
2. Pydantic validates against dataset-exact vocabularies (`Literal`s,
   `extra="forbid"`); invalid input never reaches business code (422).
3. `PredictionService` delegates to `ChurnPredictor`, which checks the frame
   columns against the artifact's declared contract before scoring.
4. Probabilities pass through the configured decision threshold;
   the response carries `model_version` + `decision_threshold` for
   traceability of every prediction.
5. Domain failures map to explicit codes: bad payload vs contract → 400,
   missing model → 503, load/inference errors → 500.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHURN_ENVIRONMENT` | `local` | surfaced by `/health` |
| `CHURN_LOG_LEVEL` | `INFO` | JSON logging threshold |
| `CHURN_DECISION_THRESHOLD` | `0.5` | probability → churn flag |
| `CHURN_MODEL_PATH` | repo `models/…joblib` | artifact location |
| `CHURN_MODEL_METADATA_PATH` | repo `models/…json` | contract/report location |

Defaults resolve relative to the installed package, so they work identically
in a venv and inside the container (`/app/models/...`).
