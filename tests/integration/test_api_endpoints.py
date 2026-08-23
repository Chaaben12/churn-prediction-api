"""Integration tests for the HTTP contract (real app, real artifact)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import status
from fastapi.testclient import TestClient

SAMPLES_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_customers.json"


def _load_samples() -> list[dict[str, Any]]:
    return json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))


def test_health_reports_ok_and_model_version(api_client: TestClient) -> None:
    response = api_client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "0.1.0"
    assert isinstance(body["environment"], str)


def test_model_info_exposes_provenance_and_metrics(api_client: TestClient) -> None:
    response = api_client.get("/model/info")

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["model_version"] == "0.1.0"
    assert body["algorithm"] == "LogisticRegression"
    assert body["n_input_features"] == 19
    assert body["metrics"]["roc_auc"] > 0.8


def test_predict_single_customer(api_client: TestClient) -> None:
    payload = {"customers": [_load_samples()[0]]}

    response = api_client.post("/predict", json=payload)

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["model_version"] == "0.1.0"
    assert len(body["predictions"]) == 1
    result = body["predictions"][0]
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert isinstance(result["churn"], bool)
    assert result["customer_id"] == "1001-A"


def test_predict_batch_preserves_order_and_accepts_new_customers(api_client: TestClient) -> None:
    samples = _load_samples()

    response = api_client.post("/predict", json={"customers": samples})

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert len(body["predictions"]) == len(samples)
    assert [p["customer_id"] for p in body["predictions"]] == ["1001-A", "1002-B", None]
    new_customer = body["predictions"][2]
    assert 0.0 <= new_customer["churn_probability"] <= 1.0


def test_predict_rejects_unknown_category_value(api_client: TestClient) -> None:
    invalid = dict(_load_samples()[0])
    invalid["gender"] = "Unknown"

    response = api_client.post("/predict", json={"customers": [invalid]})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert any("gender" in str(loc) for loc in [item["loc"] for item in response.json()["detail"]])


def test_predict_rejects_missing_required_field(api_client: TestClient) -> None:
    incomplete = dict(_load_samples()[0])
    del incomplete["tenure"]

    response = api_client.post("/predict", json={"customers": [incomplete]})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_rejects_unknown_extra_field(api_client: TestClient) -> None:
    leaked = dict(_load_samples()[0])
    leaked["loyalty_tier"] = "gold"

    response = api_client.post("/predict", json={"customers": [leaked]})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_predict_rejects_empty_batch(api_client: TestClient) -> None:
    response = api_client.post("/predict", json={"customers": []})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_openapi_document_is_generated(api_client: TestClient) -> None:
    response = api_client.get("/openapi.json")

    assert response.status_code == status.HTTP_200_OK
    paths = response.json()["paths"]
    assert "/predict" in paths
    assert "/health" in paths
    assert "/model/info" in paths


def test_startup_aborts_when_model_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import churn_api.main as main_module
    from churn_api.core.exceptions import ModelNotFoundError

    def _boom(*args: object, **kwargs: object) -> object:
        raise ModelNotFoundError("artifact absent")

    monkeypatch.setattr(main_module, "load_model", _boom)

    with pytest.raises(ModelNotFoundError), TestClient(main_module.create_app()):
        pass
