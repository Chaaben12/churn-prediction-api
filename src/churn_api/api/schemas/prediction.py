"""Output schemas for the prediction, health and model-info endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from churn_api.api.schemas.customer import CustomerProfile

MAX_BATCH_SIZE = 1000


class PredictionResult(BaseModel):
    """Outcome for a single customer."""

    model_config = ConfigDict(frozen=True)

    customer_id: str | None = None
    churn_probability: float = Field(ge=0.0, le=1.0)
    churn: bool


class PredictRequest(BaseModel):
    """Envelope accepting one or many customer profiles."""

    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerProfile] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class PredictResponse(BaseModel):
    """Batch scoring response with full traceability."""

    model_config = ConfigDict(frozen=True)

    model_version: str
    decision_threshold: float
    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    """Liveness/readiness probe answer (model loads at boot, so 200 == ready)."""

    model_config = ConfigDict(frozen=True)

    status: Literal["ok"]
    environment: str
    model_version: str


class ModelInfoResponse(BaseModel):
    """Metadata of the model currently serving traffic."""

    model_config = ConfigDict(frozen=True)

    model_version: str
    algorithm: str
    trained_at_utc: str
    sklearn_version: str
    decision_threshold: float
    n_input_features: int
    metrics: dict[str, object]
