"""Business logic for batch churn scoring.

Routers stay free of logic: they hand validated profiles to this service,
which converts them to pipeline records, runs inference through the ML layer
and shapes the response. Everything here is testable without FastAPI.
"""

from __future__ import annotations

from collections.abc import Sequence

from churn_api.api.schemas.customer import CustomerProfile
from churn_api.api.schemas.prediction import PredictionResult
from churn_api.ml.model_loader import ModelMetadata
from churn_api.ml.predictor import ChurnPredictor


class PredictionService:
    """Orchestrates one prediction round-trip; stateless per request."""

    def __init__(self, predictor: ChurnPredictor) -> None:
        self._predictor = predictor

    @property
    def metadata(self) -> ModelMetadata:
        """Metadata of the model serving traffic."""
        return self._predictor.metadata

    @property
    def decision_threshold(self) -> float:
        """Probability threshold used to derive the boolean churn class."""
        return self._predictor.decision_threshold

    def predict_batch(self, profiles: Sequence[CustomerProfile]) -> list[PredictionResult]:
        """Score a batch of validated customer profiles.

        Args:
            profiles: Already-validated profiles (Pydantic guarantees shape).

        Returns:
            One result per input profile, in the same order.
        """
        records = [profile.to_pipeline_record() for profile in profiles]
        probabilities = self._predictor.predict_probabilities(records)
        decisions = self._predictor.decide_churn(probabilities)
        return [
            PredictionResult(
                customer_id=profile.customer_id,
                churn_probability=float(probability),
                churn=decision,
            )
            for profile, probability, decision in zip(
                profiles, probabilities, decisions, strict=True
            )
        ]
