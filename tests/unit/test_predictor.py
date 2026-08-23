"""Unit tests for the inference wrapper."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_api.core.exceptions import PredictionError
from churn_api.ml.model_loader import load_model
from churn_api.ml.predictor import ChurnPredictor

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def predictor() -> Iterator[ChurnPredictor]:
    loaded = load_model(
        REPO_ROOT / "models" / "churn_classifier_v0.1.0.joblib",
        REPO_ROOT / "models" / "churn_classifier_v0.1.0_metrics.json",
    )
    yield ChurnPredictor(loaded, decision_threshold=0.5)


def test_probabilities_are_aligned_and_in_unit_range(
    predictor: ChurnPredictor, raw_customers: pd.DataFrame
) -> None:
    records = raw_customers.to_dict(orient="records")

    probabilities = predictor.predict_probabilities(records)

    assert probabilities.shape == (len(raw_customers),)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)


def test_customer_identifier_is_optional_and_ignored(
    predictor: ChurnPredictor, raw_customers: pd.DataFrame
) -> None:
    with_id = raw_customers.to_dict(orient="records")
    without_id = [{k: v for k, v in row.items() if k != "customerID"} for row in with_id]

    probabilities_with = predictor.predict_probabilities(with_id)
    probabilities_without = predictor.predict_probabilities(without_id)

    np.testing.assert_allclose(probabilities_with, probabilities_without)


def test_missing_required_feature_raises_explicit_error(predictor: ChurnPredictor) -> None:
    with pytest.raises(PredictionError, match="Missing required features"):
        predictor.predict_probabilities([{"gender": "Female"}])


def test_empty_input_short_circuits(predictor: ChurnPredictor) -> None:
    probabilities = predictor.predict_probabilities([])

    assert probabilities.shape == (0,)


def test_decide_churn_applies_threshold_inclusively(predictor: ChurnPredictor) -> None:
    decisions = predictor.decide_churn(np.array([0.4999, 0.5, 0.5001]))

    assert decisions == [False, True, True]
