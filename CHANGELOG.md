# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Repository bootstrap: folder structure, tooling configuration (ruff, mypy,
  pytest), README skeleton, MIT license.
- `scripts/download_data.sh`: reproducible Telco Customer Churn download with
  pinned SHA256 checksum.
- EDA dependency group (`uv` lockfile committed for reproducible environments).
- Exploratory notebook `notebooks/01-eda-exploration.ipynb` (executed) and
  documented feature engineering decisions in `docs/feature-engineering.md`.
- Shared preprocessing pipeline `training/preprocessing.py` (type coercion,
  zero-imputation of `TotalCharges`, one-hot encoding, scaling) with unit tests.
- Reproducible training CLI `training/train.py` (stratified seeded split,
  pluggable algorithm) and evaluation module `training/evaluate.py`
  (ROC-AUC, average precision, recall, precision, F1, confusion matrix).
- First versioned model artifact `churn_classifier_v0.1.0` (logistic
  regression selected over gradient boosting on ROC-AUC/recall).
- Application core: typed settings (`pydantic-settings`, `CHURN_*` env vars),
  structured JSON logging, domain exceptions.
- Inference layer: cached model loader (artifact + metrics report loaded once)
  and `ChurnPredictor` wrapper validating payloads against the model's own
  declared feature contract.
- Preprocessing pipeline hardened for inference: frames without optional
  columns (e.g. `customerID`) no longer fail transformation.
- REST API (FastAPI): `POST /predict` (strict validation, explicit 4xx,
  batch scoring with model-version traceability), `GET /health`,
  `GET /model/info`, auto-generated OpenAPI docs; domain exceptions mapped
  to HTTP codes via handlers.
- Coverage enforcement: pytest-cov wired into the default test run with an
  80% `fail_under` gate (current: ~99%); end-to-end training CLI test,
  fail-fast startup test, and full error-path coverage for model loading,
  inference and HTTP exception mapping.
- Docker packaging: multi-stage build (uv-based dependency install in a
  builder stage, slim runtime without build tools), non-root user,
  native `HEALTHCHECK`, strict `.dockerignore`; blessed artifact baked
  into the image. `docker-compose.yml` + `.env.example` for one-command
  startup.
- Runtime fix: `scikit-learn` promoted to a core dependency (required to
  unpickle the pipeline; previously only in the `training` group).
- CI (GitHub Actions): quality job (ruff format + lint, mypy strict,
  pytest with coverage gate) then Docker job (build with GHA cache,
  container boot + live smoke test of `/model/info`, `/predict` and
  validation errors, image-size report in the job summary); images are
  published to GHCR (`:version`, `:latest`) on `v*` tags.
