"""GET /health — liveness and readiness probe."""

from __future__ import annotations

from fastapi import APIRouter

from churn_api.api.deps import PredictionServiceDep, SettingsDep
from churn_api.api.schemas.prediction import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep, service: PredictionServiceDep) -> HealthResponse:
    """Report service readiness.

    The model is loaded during startup (the process refuses to start without
    it), so a 200 from this endpoint means the service can score traffic.
    """
    return HealthResponse(
        status="ok",
        environment=settings.environment,
        model_version=service.metadata.model_version,
    )
