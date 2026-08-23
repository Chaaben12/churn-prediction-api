# Final review — compliance against the design document

Status at tag `v0.1.0` (August 2026).

## Functional requirements

| Requirement | Status | Evidence |
| --- | --- | --- |
| Reproducible data acquisition with checksum | ✅ | `scripts/download_data.sh` (SHA256 pinned) |
| Explored dataset, documented feature decisions | ✅ | executed notebook, `docs/feature-engineering.md` |
| Shared train/inference preprocessing | ✅ | `src/churn_api/ml/preprocessing.py`, unit-tested |
| Training CLI producing versioned artifacts | ✅ | `training/train.py`, e2e-tested in CI |
| Evaluation beyond accuracy | ✅ | ROC-AUC, average precision, recall, precision, F1, confusion matrix |
| Versioned serialized artifact + metrics report | ✅ | `models/churn_classifier_v0.1.0.{joblib,_metrics.json}` |
| REST API: predict (batch), health, model info | ✅ | `src/churn_api/api/routes/`, OpenAPI at `/docs` |
| Strict input validation, explicit 4xx | ✅ | Pydantic `Literal` vocabularies, `extra="forbid"`; tested |
| Model metadata exposed | ✅ | `GET /model/info` served from the artifact's own report |

## Non-functional requirements

| ID | Target | Status | Evidence |
| --- | --- | --- | --- |
| NF1 | Prediction < 200 ms (warm) | ✅ ~29 ms local batch-of-3 warm; single-digit ms per record | dev log; CI smoke test |
| NF2 | Coverage ≥ 80% on business code | ✅ **99.1%**, hard-gated (`fail_under=80`) | pytest-cov output in CI |
| NF3 | Image < 300 MB | ⚠️ **330 MB** — accepted deviation | CI job-summary size report |
| NF4 | CI pipeline < 5 min | ✅ ≈ 2 min end-to-end | GitHub Actions run history |
| NF5 | No hardcoded secrets | ✅ none; `GITHUB_TOKEN` scoped in workflow only | repo audit, `.gitignore` |
| NF6 | Typed code + static checks | ✅ mypy `strict` + ruff (format/lint) as merge gates | CI `quality` job |
| NF7 | Starts without external services | ✅ artifact baked into image; no DB/cache dependency | container smoke test |
| NF8 | `docker build` + `docker run` suffice | ✅ quickstart = 2 commands, no hidden steps | README quickstart |

## Deviation rationale — NF3

The runtime stack required to unpickle and execute the pipeline
(numpy + pandas + scipy + scikit-learn wheels) weighs ≈300 MB unpacked on
its own, before the interpreter and application code. Remaining levers
(alpine base, stripping dist-info) either break manylinux wheels or
introspection tooling for marginal gains. The multi-stage build already
excludes every non-runtime concern (dev groups, tests caches, bytecode).
Accepting 330 MB keeps correctness and observability intact; revisiting this
budget belongs to project 2 (orchestration/cloud cost context), e.g. via
ONNX export or a slimmer serving runtime.

## Known limitations & future work

- Single-model serving: no shadow deployment/A-B switching yet (project 2).
- Synchronous batch cap of 1000 records; larger jobs need an async queue.
- GHCR images are published per tag; no nightly rebuild of the base image
  (base-image digest pinning is a project-2 concern).
