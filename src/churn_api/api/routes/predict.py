"""POST /predict — batch churn scoring."""

from __future__ import annotations

from fastapi import APIRouter, status

from churn_api.api.deps import PredictionServiceDep
from churn_api.api.schemas.prediction import PredictRequest, PredictResponse

router = APIRouter(tags=["predictions"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    summary="Score one or many customer profiles",
)
def predict(request: PredictRequest, service: PredictionServiceDep) -> PredictResponse:
    """Return a churn probability and class per submitted customer.

    Invalid profiles are rejected with an explicit 422 listing every failing
    field; the response always states which model version produced it.
    """
    predictions = service.predict_batch(request.customers)
    return PredictResponse(
        model_version=service.metadata.model_version,
        decision_threshold=service.decision_threshold,
        predictions=predictions,
    )
