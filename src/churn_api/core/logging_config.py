"""Structured JSON logging on top of the standard library.

Text logs are not machine-exploitable; emitting one JSON object per line keeps
the output ready for any log aggregator (monitoring arrives in project 3).
Stdlib-only avoids an extra runtime dependency for formatting we fully control.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Final

STANDARD_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "stack_print",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """Format every record as a single-line JSON object.

    Non-standard attributes passed through ``logger.info(..., extra={...})``
    are merged into the payload so request-scoped context (request id, model
    version, ...) survives into the logs.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in STANDARD_RECORD_KEYS and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the root logger (safe to call twice)."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
