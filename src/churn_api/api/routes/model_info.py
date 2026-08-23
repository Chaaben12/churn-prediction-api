"""GET /model/info — metadata of the model currently serving traffic."""

from __future__ import annotations

from fastapi import APIRouter

from churn_api.api.deps import PredictionServiceDep
from churn_api.api.schemas.prediction import ModelInfoResponse

router = APIRouter(tags=["model"])


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(service: PredictionServiceDep) -> ModelInfoResponse:
    """Expose version, training provenance and evaluation metrics."""
    metadata = service.metadata
    return ModelInfoResponse(
        model_version=metadata.model_version,
        algorithm=metadata.algorithm,
        trained_at_utc=metadata.trained_at_utc,
        sklearn_version=metadata.sklearn_version,
        decision_threshold=service.decision_threshold,
        n_input_features=len(metadata.input_columns),
        metrics=metadata.metrics,
    )
