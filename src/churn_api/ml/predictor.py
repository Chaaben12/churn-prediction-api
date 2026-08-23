"""Inference wrapper around the serialized scikit-learn pipeline.

The predictor is deliberately thin: it validates that incoming records carry
the feature columns declared by the model's own metadata, delegates all
transformation to the shared pipeline, and applies the decision threshold.
No business rule lives anywhere else, so train/inference can never diverge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from churn_api.core.exceptions import PredictionError
from churn_api.ml.model_loader import LoadedModel, ModelMetadata


class ChurnPredictor:
    """Stateless facade over a ``LoadedModel``; safe to share across requests."""

    def __init__(self, model: LoadedModel, decision_threshold: float) -> None:
        self._model = model
        self._threshold = decision_threshold

    @property
    def metadata(self) -> ModelMetadata:
        """Metadata of the underlying model (version, metrics, contract)."""
        return self._model.metadata

    @property
    def decision_threshold(self) -> float:
        """Probability threshold used to derive the boolean churn class."""
        return self._threshold

    def predict_probabilities(self, records: Sequence[Mapping[str, object]]) -> np.ndarray:
        """Score raw customer records and return churn probabilities.

        Args:
            records: One mapping per customer. Keys must cover the feature
                columns declared in the model metadata; unknown keys are
                ignored by the shared pipeline (e.g. ``customerID``).

        Returns:
            Float array of positive-class probabilities, aligned with input order.

        Raises:
            PredictionError: A required feature column is missing or inference failed.
        """
        if not records:
            return np.empty(0, dtype=np.float64)
        frame = pd.DataFrame.from_records(list(records))
        missing = sorted(set(self._model.metadata.input_columns) - set(map(str, frame.columns)))
        if missing:
            raise PredictionError(f"Missing required features: {missing}")
        try:
            probabilities = self._model.pipeline.predict_proba(frame)[:, 1]
        except Exception as exc:
            raise PredictionError(f"Inference failed on provided records: {exc}") from exc
        return np.asarray(probabilities, dtype=np.float64)

    def decide_churn(self, probabilities: np.ndarray) -> list[bool]:
        """Apply the configured decision threshold (>= threshold means churn)."""
        return [bool(value >= self._threshold) for value in probabilities]
