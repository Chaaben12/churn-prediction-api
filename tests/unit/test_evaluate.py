"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest
from training.evaluate import compute_classification_metrics


def _arrays(y_true: list[int], y_scores: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray(y_true)
    scores = np.asarray(y_scores)
    predictions = (scores >= 0.5).astype(int)
    return truth, scores, predictions


def test_perfect_predictions_score_one() -> None:
    truth, scores, predictions = _arrays([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    metrics = compute_classification_metrics(truth, scores, predictions)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    confusion = metrics["confusion_matrix"]
    assert isinstance(confusion, dict)
    assert confusion["fn"] == 0
    assert confusion["fp"] == 0


def test_hand_computed_case_matches_expected_values() -> None:
    truth, scores, predictions = _arrays([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])

    metrics = compute_classification_metrics(truth, scores, predictions)

    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(2 / 3)
    confusion = metrics["confusion_matrix"]
    assert isinstance(confusion, dict)
    assert confusion == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}


def test_threshold_is_recorded_in_report() -> None:
    truth, scores, predictions = _arrays([0, 1], [0.3, 0.6])

    metrics = compute_classification_metrics(truth, scores, predictions, threshold=0.5)

    assert metrics["threshold"] == pytest.approx(0.5)
