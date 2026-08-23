# churn-prediction-api

[![CI](https://github.com/Chaaben12/churn-prediction-api/actions/workflows/ci.yml/badge.svg)](https://github.com/Chaaben12/churn-prediction-api/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Type checking](https://img.shields.io/badge/mypy-strict-2A6DB2)
![Coverage](https://img.shields.io/badge/coverage-99%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade customer churn prediction API — project 1 of an MLOps portfolio.

**Status**: `v0.1.0` — delivered end-to-end: reproducible training → typed REST
API → Dockerized service → CI with live container smoke tests and GHCR
publication. Compliance against the original requirements:
[`docs/final-review.md`](docs/final-review.md).

## Overview

This service exposes a machine learning model as a REST API. Given a customer
profile, it returns a churn probability so a sales team can prioritize retention
actions.

The project demonstrates end-to-end delivery of an ML service:
reproducible training → API → containerization → CI/CD — with real software
engineering hygiene (typed code, tests, linting, modular architecture), not
just data science.

## Features

- `POST /predict` — churn probability + class for one to 1000 customer
  profiles per request, each response traceable to a `model_version`
- `GET /health` — liveness/readiness probe (model loads at boot, so 200 == ready)
- `GET /model/info` — metadata of the loaded model (version, training date, metrics)
- Strict input validation with explicit 4xx errors (`Literal` vocabularies,
  unknown fields rejected)
- Auto-generated OpenAPI/Swagger documentation at `/docs`
- Single scikit-learn pipeline shared between training and inference
  (no training-serving skew)
- Fail-fast startup: a broken artifact means the process refuses to start

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

Design rationale and further reading:

- [`docs/architecture.md`](docs/architecture.md) — layers, artifact contract, key decisions
- [`docs/feature-engineering.md`](docs/feature-engineering.md) — preprocessing decisions
- [`docs/final-review.md`](docs/final-review.md) — compliance review vs requirements
- [`CHANGELOG.md`](CHANGELOG.md) — notable changes per release

## Quickstart

### With Docker (recommended)

```bash
cp .env.example .env   # optional — sane defaults apply
docker compose up --build
```

The service is then available at:

- `http://localhost:8000/docs` — interactive OpenAPI/Swagger UI
- `http://localhost:8000/health` — liveness probe
- `http://localhost:8000/model/info` — loaded model metadata

Example prediction:

```bash
curl -s http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customers": [{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 24, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "No",
        "DeviceProtection": "Yes", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "Yes",
        "Contract": "Two year", "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 65.5, "TotalCharges": 1572.0
      }]}'
```

The blessed model artifact is baked into the image, so `docker build` +
`docker run` is all an evaluator needs — no hidden manual steps.

Prebuilt images are published to GHCR on version tags:

```bash
docker run -p 8000:8000 ghcr.io/chaaben12/churn-prediction-api:latest
```

### Without Docker (local development)

Requires [uv](https://docs.astral.sh/uv/):

```bash
uv sync --dev
uv run uvicorn churn_api.main:app --reload
```

A `Makefile` wraps the common workflows: `make lint`, `make test`,
`make docker-up`, …

## Repository layout

```
src/churn_api/     Application package (FastAPI service, src-layout)
training/          Offline flow: config, training CLI, evaluation
models/            Blessed artifact pair (pipeline + metrics contract), committed
tests/             Unit and integration tests (pytest, coverage-gated)
notebooks/         Exploratory EDA (explicitly off the critical path)
scripts/           Reproducibility helpers (data download)
docs/              Architecture, feature engineering, final compliance review
.github/workflows/ CI: quality gates, Docker build/smoke/size, GHCR publication
```

## Model card summary

| | |
| --- | --- |
| Algorithm | Logistic regression (selected over gradient boosting on ROC-AUC/recall) |
| Dataset | Telco Customer Churn (SHA256-pinned download) |
| Metrics (test split, seeded) | ROC-AUC 0.842 · avg precision 0.633 · recall 0.783 |
| Contract | 19 input columns declared in the metrics JSON, enforced at boot |

## Roadmap

- [x] Étape 0 — Repository bootstrap
- [x] Étape 1 — Data preparation and exploration
- [x] Étape 2 — Preprocessing pipeline
- [x] Étape 3 — Model training and evaluation
- [x] Étape 4 — FastAPI application core
- [x] Étape 5 — API endpoints and schemas
- [x] Étape 6 — Unit and integration tests (coverage ≥ 80%)
- [x] Étape 7 — Dockerization (multi-stage image, 330 MB measured in CI)
- [x] Étape 8 — CI/CD with GitHub Actions
- [x] Étape 9 — Final README and documentation
- [x] Étape 10 — Final review against all requirements

## License

MIT — see [LICENSE](LICENSE).
