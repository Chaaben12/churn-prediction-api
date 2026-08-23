"""Typed application settings read from environment variables.

Every knob is prefixed with ``CHURN_`` (e.g. ``CHURN_LOG_LEVEL=DEBUG``) and can
also live in a local ``.env`` file (see ``.env.example``, added with Docker).
Paths default to the source-checkout layout; deployments override them via env
vars so the same image runs anywhere.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BLESSED_ARTIFACT = "churn_classifier_v0.1.0.joblib"


class Settings(BaseSettings):
    """Runtime configuration for the service."""

    model_config = SettingsConfigDict(env_prefix="CHURN_", env_file=".env", extra="ignore")

    environment: str = "local"
    log_level: str = "INFO"
    model_path: Path = _REPO_ROOT / "models" / _BLESSED_ARTIFACT
    model_metadata_path: Path = _REPO_ROOT / "models" / f"{_BLESSED_ARTIFACT[:-7]}_metrics.json"
    decision_threshold: float = 0.5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
