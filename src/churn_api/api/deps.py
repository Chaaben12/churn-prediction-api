"""FastAPI dependency providers.

All singletons (settings, prediction service) are created once in the app
lifespan and stashed on ``app.state``; these providers expose them to routers
with full type annotations and zero global mutable state.
"""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from churn_api.core.config import Settings
from churn_api.services.prediction_service import PredictionService


def get_settings(request: Request) -> Settings:
    """Return the process-wide settings stored at startup."""
    return cast("Settings", request.app.state.settings)


def get_prediction_service(request: Request) -> PredictionService:
    """Return the prediction service built at startup."""
    return cast("PredictionService", request.app.state.prediction_service)


SettingsDep = Annotated[Settings, Depends(get_settings)]
PredictionServiceDep = Annotated[PredictionService, Depends(get_prediction_service)]
