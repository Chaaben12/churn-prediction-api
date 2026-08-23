"""Domain exceptions for the churn API.

Each maps to an explicit HTTP error response via FastAPI exception handlers
registered in ``main.py`` (API layer). Nothing here imports web framework code,
so business logic stays testable without FastAPI.
"""

from __future__ import annotations


class ChurnApiError(Exception):
    """Base class for all domain errors raised by the application."""


class ModelNotFoundError(ChurnApiError):
    """The configured model artifact or its metadata file does not exist."""


class ModelLoadError(ChurnApiError):
    """The model artifact exists but could not be deserialized or validated."""


class PredictionError(ChurnApiError):
    """Inference failed, typically because input records are invalid."""
