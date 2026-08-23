"""Central configuration for the training flow.

Every knob that affects reproducibility lives here: seed, split ratio,
hyperparameters, and filesystem locations. Nothing else in ``training/``
hardcodes such values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH: Final[Path] = PROJECT_ROOT / "training" / "data" / "Telco-Customer-Churn.csv"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"

MODEL_NAME: Final[str] = "churn_classifier"
RANDOM_SEED: Final[int] = 42
TEST_SIZE: Final[float] = 0.2
DECISION_THRESHOLD: Final[float] = 0.5

LOGISTIC_REGRESSION_PARAMS: Final[dict[str, object]] = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}

HIST_GRADIENT_BOOSTING_PARAMS: Final[dict[str, object]] = {
    "learning_rate": 0.08,
    "max_iter": 300,
    "max_leaf_nodes": 31,
    "l2_regularization": 0.1,
    "early_stopping": False,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
}
