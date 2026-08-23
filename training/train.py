"""Reproducible training CLI.

Produces exactly two versioned outputs in the models directory:
``<model_name>_v<version>.joblib`` (full pipeline: preprocessing + classifier)
and ``<model_name>_v<version>_metrics.json`` (provenance + evaluation report).

Example:
    uv run python -m training.train --algorithm hist_gradient_boosting --model-version 0.1.0
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn_api.ml.preprocessing import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    IDENTIFIER_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessing_pipeline,
    encode_target,
)
from training.config import (
    DECISION_THRESHOLD,
    DEFAULT_DATA_PATH,
    HIST_GRADIENT_BOOSTING_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    MODEL_NAME,
    MODELS_DIR,
    RANDOM_SEED,
    TEST_SIZE,
)
from training.evaluate import compute_classification_metrics

ALGORITHMS: Final[tuple[str, ...]] = ("hist_gradient_boosting", "logistic_regression")
REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {TARGET_COLUMN, *IDENTIFIER_COLUMNS, *NUMERIC_FEATURES, *BINARY_FEATURES, *CATEGORICAL_FEATURES}
)


def sha256_of(path: Path) -> str:
    """Compute the SHA256 hex digest of a file, streaming to stay memory-safe."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the raw dataset and validate that every expected column exists.

    Args:
        path: CSV file produced by ``scripts/download_data.sh``.

    Returns:
        The raw dataset.

    Raises:
        ValueError: If any required column is missing.
    """
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns.tolist())
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return frame


def split_dataset(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split with the fixed global seed."""
    features = frame.drop(columns=[TARGET_COLUMN])
    target = encode_target(frame[TARGET_COLUMN])
    splits = cast(
        "tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]",
        train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=target,
        ),
    )
    return splits


def _build_estimator(algorithm: str) -> Any:
    if algorithm == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**HIST_GRADIENT_BOOSTING_PARAMS)
    if algorithm == "logistic_regression":
        return LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    raise ValueError(f"Unknown algorithm '{algorithm}'. Expected one of {ALGORITHMS}.")


def build_full_pipeline(algorithm: str) -> Pipeline:
    """Assemble preprocessing + classifier into one serializable pipeline."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("classifier", _build_estimator(algorithm)),
        ]
    )


def artifact_stem(model_version: str) -> str:
    """File stem shared by the joblib artifact and its metrics report."""
    return f"{MODEL_NAME}_v{model_version}"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHMS,
        default="hist_gradient_boosting",
        help="Classifier to train.",
    )
    parser.add_argument(
        "--model-version",
        default="0.1.0",
        help="Version tag used in artifact file names.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the raw dataset CSV.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory where artifacts are written.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the full offline flow: load, split, fit, evaluate, persist."""
    args = _parse_args(argv)

    frame = load_dataset(args.data_path)
    x_train, x_test, y_train, y_test = split_dataset(frame)

    pipeline = build_full_pipeline(args.algorithm)
    pipeline.fit(x_train, y_train)

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    metrics = compute_classification_metrics(
        np.asarray(y_test), probabilities, predictions, threshold=DECISION_THRESHOLD
    )

    confusion = metrics["confusion_matrix"]
    assert isinstance(confusion, dict)
    print("=== training summary ===")
    print(f"algorithm       : {args.algorithm}")
    print(f"model version   : {args.model_version}")
    print(f"train/test rows : {len(x_train)} / {len(x_test)}")
    print(f"roc_auc         : {metrics['roc_auc']:.4f}")
    print(f"avg_precision   : {metrics['average_precision']:.4f}")
    print(f"recall          : {metrics['recall']:.4f}")
    print(f"precision       : {metrics['precision']:.4f}")
    print(f"f1              : {metrics['f1']:.4f}")
    print(f"confusion       : {confusion}")

    args.models_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact_stem(args.model_version)
    artifact_path = args.models_dir / f"{stem}.joblib"
    report_path = args.models_dir / f"{stem}_metrics.json"

    joblib.dump(pipeline, artifact_path)
    report = {
        "model_name": MODEL_NAME,
        "model_version": args.model_version,
        "algorithm": type(pipeline.named_steps["classifier"]).__name__,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "decision_threshold": DECISION_THRESHOLD,
        "input_columns": sorted(set(REQUIRED_COLUMNS) - {TARGET_COLUMN} - set(IDENTIFIER_COLUMNS)),
        "dataset": {
            "path": str(args.data_path),
            "sha256": sha256_of(args.data_path),
            "rows": int(len(frame)),
        },
        "hyperparameters": HIST_GRADIENT_BOOSTING_PARAMS
        if args.algorithm == "hist_gradient_boosting"
        else LOGISTIC_REGRESSION_PARAMS,
        "metrics": metrics,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"artifact : {artifact_path}")
    print(f"report   : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
