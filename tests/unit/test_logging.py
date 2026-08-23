"""Unit tests for structured JSON logging."""

from __future__ import annotations

import json
import logging

import pytest

from churn_api.core.logging_config import JsonFormatter, configure_logging


class CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.formatted: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.formatted.append(self.format(record))


@pytest.fixture()
def captured_logger() -> tuple[logging.Logger, CaptureHandler]:
    handler = CaptureHandler()
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("churn_api.test.json")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    yield logger, handler
    logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def test_formatter_emits_parseable_json(
    captured_logger: tuple[logging.Logger, CaptureHandler],
) -> None:
    logger, handler = captured_logger

    logger.info("hello %s", "world")

    payload = json.loads(handler.formatted[0])
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "churn_api.test.json"
    assert "levelname" not in payload
    assert "pathname" not in payload


def test_extra_fields_survive_into_payload(
    captured_logger: tuple[logging.Logger, CaptureHandler],
) -> None:
    logger, handler = captured_logger

    logger.info("scoring request", extra={"request_id": "abc-123", "model_version": "0.1.0"})

    payload = json.loads(handler.formatted[0])
    assert payload["request_id"] == "abc-123"
    assert payload["model_version"] == "0.1.0"


def test_configure_logging_replaces_handlers_and_sets_level() -> None:
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    try:
        configure_logging("debug")

        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)
