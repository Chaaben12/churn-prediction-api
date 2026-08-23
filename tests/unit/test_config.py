"""Unit tests for application settings."""

from __future__ import annotations

import pytest

from churn_api.core.config import Settings, get_settings


def test_defaults_point_to_blessed_artifact() -> None:
    settings = Settings(_env_file=None)

    assert settings.model_path.name == "churn_classifier_v0.1.0.joblib"
    assert settings.model_metadata_path.name == "churn_classifier_v0.1.0_metrics.json"
    assert settings.log_level == "INFO"
    assert settings.decision_threshold == 0.5
    assert settings.model_path.is_file()


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHURN_LOG_LEVEL", "debug")
    monkeypatch.setenv("CHURN_DECISION_THRESHOLD", "0.7")

    settings = Settings(_env_file=None)

    assert settings.log_level == "debug"
    assert settings.decision_threshold == 0.7


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
