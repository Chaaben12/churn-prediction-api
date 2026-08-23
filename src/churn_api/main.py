"""FastAPI application factory.

The lifespan loads the model exactly once before the service accepts traffic;
a failure aborts startup (fail-fast) rather than serving 500s at runtime.
Routers hold no logic — every request flows router -> service -> ML layer.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

import churn_api
from churn_api.api.routes import health, model_info, predict
from churn_api.core.config import get_settings
from churn_api.core.exceptions import ChurnApiError, register_exception_handlers
from churn_api.core.logging_config import configure_logging
from churn_api.ml.model_loader import load_model
from churn_api.ml.predictor import ChurnPredictor
from churn_api.services.prediction_service import PredictionService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load settings and model once; stash singletons on app.state."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger("churn_api")

    try:
        loaded = load_model(settings.model_path, settings.model_metadata_path)
    except ChurnApiError:
        logger.exception(
            "model loading failed at startup",
            extra={"artifact_path": str(settings.model_path)},
        )
        raise

    app.state.settings = settings
    app.state.prediction_service = PredictionService(
        ChurnPredictor(loaded, decision_threshold=settings.decision_threshold)
    )
    logger.info(
        "startup complete",
        extra={
            "model_version": loaded.metadata.model_version,
            "algorithm": loaded.metadata.algorithm,
            "environment": settings.environment,
        },
    )
    yield
    logger.info("shutdown complete")


def create_app() -> FastAPI:
    """Build the FastAPI application with routers and error handlers."""
    app = FastAPI(
        title="Churn Prediction API",
        description=(
            "Scores customer profiles with a churn probability so retention "
            "teams can prioritize outreach. Preprocessing is the exact "
            "scikit-learn pipeline used at training time (no skew)."
        ),
        version=churn_api.__version__,
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(model_info.router)
    app.include_router(predict.router)
    return app


app = create_app()
