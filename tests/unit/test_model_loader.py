"""Unit tests for model loading and validation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import pytest

from churn_api.core.exceptions import ModelLoadError, ModelNotFoundError
from churn_api.ml.model_loader import load_model

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "models" / "churn_classifier_v0.1.0.joblib"
REPORT = REPO_ROOT / "models" / "churn_classifier_v0.1.0_metrics.json"


def test_loads_blessed_artifact_with_full_metadata() -> None:
    loaded = load_model(ARTIFACT, REPORT)

    assert loaded.metadata.model_version == "0.1.0"
    assert loaded.metadata.algorithm == "LogisticRegression"
    assert callable(getattr(loaded.pipeline, "predict_proba", None))
    assert "tenure" in loaded.metadata.input_columns
    assert loaded.metadata.metrics["roc_auc"] > 0.8


def test_missing_artifact_raises_model_not_found(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError):
        load_model(tmp_path / "absent.joblib", tmp_path / "absent_metrics.json")


def test_corrupt_pickle_raises_model_load_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.joblib"
    broken.write_bytes(b"definitely-not-a-pickle")
    shutil.copyfile(REPORT, tmp_path / "report.json")

    with pytest.raises(ModelLoadError, match="deserialize"):
        load_model(broken, tmp_path / "report.json")


def test_incomplete_metadata_raises_model_load_error(tmp_path: Path) -> None:
    report = tmp_path / "incomplete.json"
    report.write_text(json.dumps({"model_version": "0.1.0"}), encoding="utf-8")

    with pytest.raises(ModelLoadError, match="missing fields"):
        load_model(ARTIFACT, report)


def test_malformed_json_metadata_raises_model_load_error(tmp_path: Path) -> None:
    report = tmp_path / "broken.json"
    report.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ModelLoadError, match="not valid JSON"):
        load_model(ARTIFACT, report)


def test_non_object_metadata_raises_model_load_error(tmp_path: Path) -> None:
    report = tmp_path / "array.json"
    report.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ModelLoadError, match="JSON object"):
        load_model(ARTIFACT, report)


def test_wrongly_typed_contract_raises_model_load_error(tmp_path: Path) -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    payload["input_columns"] = "tenure"
    report = tmp_path / "bad-contract.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelLoadError, match="list of strings"):
        load_model(ARTIFACT, report)


def test_non_string_metadata_value_raises_model_load_error(tmp_path: Path) -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    payload["trained_at_utc"] = 12345
    report = tmp_path / "bad-value.json"
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelLoadError, match="trained_at_utc"):
        load_model(ARTIFACT, report)


def test_artifact_without_predict_proba_raises_model_load_error(tmp_path: Path) -> None:
    not_a_pipeline = tmp_path / "plain.joblib"
    joblib.dump({"weights": [1, 2, 3]}, not_a_pipeline)

    with pytest.raises(ModelLoadError, match="predict_proba"):
        load_model(not_a_pipeline, REPORT)
