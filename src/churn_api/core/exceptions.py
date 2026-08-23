"""Domain exceptions and their HTTP mappings.

Exceptions stay framework-free; only ``register_exception_handlers`` touches
FastAPI, so business logic remains testable without a web stack.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from fastapi import FastAPI, Request

logger = logging.getLogger("churn_api.errors")


class ChurnApiError(Exception):
    """Base class for all domain errors raised by the application."""


class ModelNotFoundError(ChurnApiError):
    """The configured model artifact or its metadata file does not exist."""


class ModelLoadError(ChurnApiError):
    """The model artifact exists but could not be deserialized or validated."""


class PredictionError(ChurnApiError):
    """Inference failed, typically because input records are invalid."""


def register_exception_handlers(app: FastAPI) -> None:
    """Map domain errors onto explicit HTTP responses."""

    def _respond(status_code: int, exc: ChurnApiError) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    @app.exception_handler(ModelNotFoundError)
    async def _handle_model_not_found(request: Request, exc: ModelNotFoundError) -> JSONResponse:
        logger.error("model artifact missing: %s", exc)
        return _respond(503, exc)

    @app.exception_handler(ModelLoadError)
    async def _handle_model_load_error(request: Request, exc: ModelLoadError) -> JSONResponse:
        logger.error("model failed to load: %s", exc)
        return _respond(500, exc)

    @app.exception_handler(PredictionError)
    async def _handle_prediction_error(request: Request, exc: PredictionError) -> JSONResponse:
        logger.warning("prediction rejected: %s", exc)
        return _respond(400, exc)

    @app.exception_handler(ChurnApiError)
    async def _handle_unexpected_domain_error(request: Request, exc: ChurnApiError) -> JSONResponse:
        logger.exception("unhandled domain error")
        return _respond(500, exc)
