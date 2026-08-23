"""HTTP mappings of the domain exception handlers."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from churn_api.core.exceptions import (
    ChurnApiError,
    ModelLoadError,
    ModelNotFoundError,
    PredictionError,
    register_exception_handlers,
)


@pytest.mark.parametrize(
    ("raised", "expected_status"),
    [
        (PredictionError("bad input"), status.HTTP_400_BAD_REQUEST),
        (ModelNotFoundError("missing.joblib"), status.HTTP_503_SERVICE_UNAVAILABLE),
        (ModelLoadError("corrupt artifact"), status.HTTP_500_INTERNAL_SERVER_ERROR),
        (ChurnApiError("unexpected"), status.HTTP_500_INTERNAL_SERVER_ERROR),
    ],
)
def test_domain_errors_map_to_explicit_http_responses(
    raised: ChurnApiError, expected_status: int
) -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise raised

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom")

    assert response.status_code == expected_status
    body = response.json()
    assert body["error"] == type(raised).__name__
    assert body["detail"] == str(raised)
