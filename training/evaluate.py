"""Evaluation metrics for the churn classifier.

Recall and ROC-AUC are the headline metrics (see docs/feature-engineering.md):
under a 26.5% positive rate, accuracy is misleading and missing a leaver costs
more than a false alert.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD: Final[float] = 0.5


def compute_classification_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, object]:
    """Compute the evaluation report for binary churn predictions.

    Args:
        y_true: Ground-truth binary labels.
        y_scores: Predicted probability of the positive class.
        y_pred: Predicted binary labels (threshold applied by the caller).
        threshold: The decision threshold used to produce ``y_pred``.

    Returns:
        Flat metric dict with scalar values plus a nested confusion matrix.
    """
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(y_true, y_pred).ravel())
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "average_precision": float(average_precision_score(y_true, y_scores)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
