# churn-prediction-api

A production-grade customer churn prediction API — project 1 of an MLOps portfolio.

**Status**: under construction (Étape 1 — data preparation done; preprocessing next).

<!-- TODO: badges once CI is in place (Étape 8): CI status, coverage, Python version, license -->

## Overview

This service exposes a machine learning model as a REST API. Given a customer
profile, it returns a churn probability so a sales team can prioritize retention
actions.

The project demonstrates end-to-end delivery of an ML service:
reproducible training → API → containerization → CI/CD — with real software
engineering hygiene (typed code, tests, linting, modular architecture), not
just data science.

## Features (planned)

- `POST /predict` — churn probability + class for one or many customer profiles
- `GET /health` — service health check
- `GET /model/info` — metadata of the loaded model (version, training date, metrics)
- Strict input validation with explicit 4xx errors
- Auto-generated OpenAPI/Swagger documentation
- Single scikit-learn pipeline shared between training and inference
  (no training-serving skew)

## Architecture

Two distinct flows, like a real MLOps system:

- **Offline (training)**: dataset → preprocessing pipeline → model training →
  evaluation → serialized artifact (`joblib`) + versioned metrics report.
- **Online (inference)**: HTTP request → Pydantic validation → shared
  preprocessing transform → prediction → JSON response with probability,
  class, and model version.

The application is layered: FastAPI routers (API) → business logic (services)
→ model wrapper (ML), with cross-cutting concerns (config, logging, exceptions)
in `core/`.

Full design rationale:

- [`docs/architecture.md`](docs/architecture.md) (to come)
- [`docs/feature-engineering.md`](docs/feature-engineering.md)

## Quickstart

TODO — available from Étape 7 (Docker). The goal: `docker build` +
`docker run` is all an evaluator needs, no hidden manual steps.

## Repository layout

```
src/churn_api/     Application package (FastAPI service, src-layout)
training/          Offline flow: preprocessing, training CLI, evaluation
models/            Serialized model artifacts (not committed - see models/README.md)
tests/             Unit and integration tests (pytest)
notebooks/         Exploratory EDA (explicitly off the critical path)
scripts/           Reproducibility helpers (data download, smoke test)
.github/workflows/ CI (lint, tests, Docker build) and image publication
```

## Roadmap

- [x] Étape 0 — Repository bootstrap
- [x] Étape 1 — Data preparation and exploration
- [x] Étape 2 — Preprocessing pipeline
- [x] Étape 3 — Model training and evaluation
- [x] Étape 4 — FastAPI application core
- [x] Étape 5 — API endpoints and schemas
- [ ] Étape 6 — Unit and integration tests (coverage ≥ 80%)
- [ ] Étape 7 — Dockerization (multi-stage image < 300 MB)
- [ ] Étape 8 — CI/CD with GitHub Actions
- [ ] Étape 9 — Final README and documentation
- [ ] Étape 10 — Final review against all requirements

## License

MIT — see [LICENSE](LICENSE).
