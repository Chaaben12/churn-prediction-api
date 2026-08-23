"""Unit tests for the shared preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_api.ml.preprocessing import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessing_pipeline,
    encode_target,
)


def test_transform_output_is_dense_and_finite(raw_customers: pd.DataFrame) -> None:
    features = build_preprocessing_pipeline().fit_transform(raw_customers)

    assert isinstance(features, np.ndarray)
    assert np.isfinite(features).all()


def test_identifier_column_does_not_affect_features(raw_customers: pd.DataFrame) -> None:
    pipeline = build_preprocessing_pipeline()
    renamed_ids = raw_customers.assign(customerID=[f"id-{i}" for i in range(len(raw_customers))])

    baseline = pipeline.fit_transform(raw_customers)
    variant = pipeline.fit_transform(renamed_ids)

    np.testing.assert_array_equal(baseline, variant)


def test_blank_total_charges_match_explicit_zero(raw_customers: pd.DataFrame) -> None:
    features = build_preprocessing_pipeline().fit_transform(raw_customers)

    np.testing.assert_array_equal(features[0], features[1])
    np.testing.assert_array_equal(features[0], features[2])


def test_unknown_category_at_inference_is_ignored(raw_customers: pd.DataFrame) -> None:
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(raw_customers)
    unseen = raw_customers.head(1).assign(InternetService="Quantum")

    features = pipeline.transform(unseen)

    assert features.shape[0] == 1


def test_unlisted_columns_are_dropped(raw_customers: pd.DataFrame) -> None:
    pipeline = build_preprocessing_pipeline()
    leaked = raw_customers.assign(internal_note="should-not-leak")

    baseline = pipeline.fit_transform(raw_customers)
    variant = pipeline.fit_transform(leaked)

    np.testing.assert_array_equal(baseline, variant)


def test_expected_feature_width(raw_customers: pd.DataFrame) -> None:
    expected_width = (
        len(NUMERIC_FEATURES)
        + len(BINARY_FEATURES)
        + sum(int(raw_customers[column].nunique()) for column in CATEGORICAL_FEATURES)
    )

    features = build_preprocessing_pipeline().fit_transform(raw_customers)

    assert features.shape == (len(raw_customers), expected_width)


def test_encode_target_maps_labels_to_binary() -> None:
    target = pd.Series(["No", "Yes", "Yes", "No"])

    encoded = encode_target(target)

    assert encoded.tolist() == [0, 1, 1, 0]


def test_encode_target_rejects_unknown_label() -> None:
    target = pd.Series(["Yes", "Maybe"])

    with pytest.raises(ValueError, match="Unexpected target values"):
        encode_target(target)
