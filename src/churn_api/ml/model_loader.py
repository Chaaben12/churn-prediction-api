"""Loading and caching of the serialized model artifact.

The model is deserialized exactly once per process (lru_cache singleton) at
application startup, never per request — a hard requirement for the < 200 ms
response-time budget. Both the joblib pipeline and its metrics report are
loaded together so the service can always state which model it is running.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, cast

import joblib

from churn_api.core.exceptions import ModelLoadError, ModelNotFoundError

_REQUIRED_METADATA_KEYS: Final[tuple[str, ...]] = (
    "model_version",
    "algorithm",
    "trained_at_utc",
    "sklearn_version",
    "input_columns",
    "metrics",
)


@dataclass(frozen=True)
class ModelMetadata:
    """Typed view over the training metrics report."""

    model_version: str
    algorithm: str
    trained_at_utc: str
    sklearn_version: str
    input_columns: tuple[str, ...]
    metrics: dict[str, object]


@dataclass(frozen=True)
class LoadedModel:
    """Everything the inference layer needs, loaded once."""

    pipeline: Any
    metadata: ModelMetadata
    artifact_path: Path


def _required_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ModelLoadError(f"Model metadata field '{key}' is missing or not a string.")
    return value


def _parse_metadata(path: Path) -> ModelMetadata:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"Model metadata file is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ModelLoadError(f"Model metadata must be a JSON object: {path}")
    data = cast("dict[str, object]", raw)
    missing = [key for key in _REQUIRED_METADATA_KEYS if key not in data]
    if missing:
        raise ModelLoadError(f"Model metadata is missing fields: {missing}")
    columns = data["input_columns"]
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        raise ModelLoadError("Model metadata field 'input_columns' must be a list of strings.")
    return ModelMetadata(
        model_version=_required_str(data, "model_version"),
        algorithm=_required_str(data, "algorithm"),
        trained_at_utc=_required_str(data, "trained_at_utc"),
        sklearn_version=_required_str(data, "sklearn_version"),
        input_columns=tuple(columns),
        metrics=cast("dict[str, object]", data["metrics"]),
    )


@lru_cache(maxsize=1)
def load_model(model_path: Path, metadata_path: Path) -> LoadedModel:
    """Deserialize the pipeline and its report; cached for the process lifetime.

    Raises:
        ModelNotFoundError: An artifact path does not exist.
        ModelLoadError: Deserialization or contract validation failed.
    """
    if not model_path.is_file():
        raise ModelNotFoundError(f"Model artifact not found: {model_path}")
    if not metadata_path.is_file():
        raise ModelNotFoundError(f"Model metadata not found: {metadata_path}")

    try:
        pipeline = joblib.load(model_path)
    except Exception as exc:
        raise ModelLoadError(f"Failed to deserialize model artifact: {model_path}") from exc

    if getattr(pipeline, "predict_proba", None) is None:
        raise ModelLoadError("Loaded artifact does not expose predict_proba().")

    return LoadedModel(
        pipeline=pipeline,
        metadata=_parse_metadata(metadata_path),
        artifact_path=model_path,
    )
