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
