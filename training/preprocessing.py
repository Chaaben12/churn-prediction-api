"""Preprocessing pipeline shared by the training and inference flows.

The single source of truth for raw-customer-DataFrame -> feature-matrix.
The exact object built here is what gets serialized at the end of training
and reloaded inside the API, which structurally prevents training-serving
skew. Decisions implemented here are documented in docs/feature-engineering.md.
"""

from __future__ import annotations

from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN: Final[str] = "Churn"
IDENTIFIER_COLUMNS: Final[tuple[str, ...]] = ("customerID",)
NUMERIC_FEATURES: Final[tuple[str, ...]] = ("tenure", "MonthlyCharges", "TotalCharges")
BINARY_FEATURES: Final[tuple[str, ...]] = ("SeniorCitizen",)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
)
TOTAL_CHARGES_IMPUTE_VALUE: Final[float] = 0.0


def _coerce_total_charges(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with ``TotalCharges`` coerced to float.

    The raw dataset stores this column as text and new customers carry blank
    values; coercion turns those into NaN that the numeric branch imputes.

    Args:
        frame: Raw customer rows.

    Returns:
        Copy of the input with a numeric ``TotalCharges`` column.
    """
    coerced = frame.copy()
    coerced["TotalCharges"] = pd.to_numeric(coerced["TotalCharges"], errors="coerce")
    return coerced


def build_preprocessing_pipeline() -> Pipeline:
    """Build the (unfitted) preprocessing pipeline shared train/inference.

    Branches:
        numeric: impute missing ``TotalCharges`` with 0 then standard-scale.
        categorical: one-hot encode, unknown categories ignored at inference.
        binary: pass through features already encoded as 0/1.
        remainder: anything not explicitly listed (identifiers, leaked columns)
            is silently dropped rather than allowed into feature space.

    Returns:
        Pipeline mapping a raw customer DataFrame to a dense float matrix.
    """
    numeric_branch = Pipeline(
        steps=[
            (
                "impute_missing_with_zero",
                SimpleImputer(strategy="constant", fill_value=TOTAL_CHARGES_IMPUTE_VALUE),
            ),
            ("scale", StandardScaler()),
        ]
    )
    branches = ColumnTransformer(
        transformers=[
            ("numeric", numeric_branch, list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
            ("binary", "passthrough", list(BINARY_FEATURES)),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("coerce_types", FunctionTransformer(_coerce_total_charges)),
            ("features", branches),
        ]
    )


def encode_target(target: pd.Series) -> pd.Series:
    """Map churn labels to a binary integer series.

    Args:
        target: Series of ``Yes`` / ``No`` labels.

    Returns:
        Integer series with 1 for churners and 0 otherwise.

    Raises:
        ValueError: If any label other than ``Yes``/``No`` is present.
    """
    allowed = {"Yes", "No"}
    unexpected = set(target.unique().tolist()) - allowed
    if unexpected:
        raise ValueError(f"Unexpected target values: {sorted(unexpected)}")
    return (target == "Yes").astype(int)
