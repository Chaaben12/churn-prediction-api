"""Unit tests for training utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from training.config import MODEL_NAME, TEST_SIZE
from training.train import artifact_stem, load_dataset, split_dataset


def _synthetic_frame(rows: int = 60) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "customerID": [f"id-{i}" for i in range(rows)],
            "tenure": range(rows),
        }
    )
    frame["Churn"] = ["Yes" if i % 10 < 3 else "No" for i in range(rows)]
    return frame


def test_split_is_stratified_and_reproducible() -> None:
    frame = _synthetic_frame()
    overall_rate = float((frame["Churn"] == "Yes").mean())

    x_train_1, x_test_1, y_train_1, _ = split_dataset(frame)
    x_train_2, x_test_2, y_train_2, _ = split_dataset(frame)

    expected_test_rows = round(len(frame) * TEST_SIZE)
    assert len(x_train_1) + len(x_test_1) == len(frame)
    assert len(x_test_1) == expected_test_rows

    train_rate = float((y_train_1 == 1).mean())
    assert train_rate == pytest.approx(overall_rate, abs=0.02)

    pd.testing.assert_frame_equal(x_train_1, x_train_2)
    pd.testing.assert_frame_equal(x_test_1, x_test_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)


def test_artifact_stem_embeds_model_name_and_version() -> None:
    assert artifact_stem("9.9.9") == f"{MODEL_NAME}_v9.9.9"


def test_load_dataset_rejects_incomplete_csv(tmp_path: Path) -> None:
    incomplete = pd.DataFrame({"customerID": ["a"], "tenure": [1]})
    csv_path = tmp_path / "broken.csv"
    incomplete.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_dataset(csv_path)
