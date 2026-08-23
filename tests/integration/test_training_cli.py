"""End-to-end coverage of the offline training CLI."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from training.train import main


def _synthetic_dataset(rows: int = 80) -> pd.DataFrame:
    """Deterministic miniature Telco-shaped dataset (same vocabulary)."""
    contracts = ["Month-to-month", "One year", "Two year"]
    internet = ["DSL", "Fiber optic", "No"]
    payments = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    multiple_lines = ["No", "Yes", "No phone service"]
    records = []
    for i in range(rows):
        tenure = i % 73
        monthly = round(20.0 + (i % 70), 2)
        has_internet = internet[i % 3] != "No"
        service = ("Yes" if i % 2 else "No") if has_internet else "No internet service"
        records.append(
            {
                "customerID": f"id-{i:04d}",
                "gender": "Female" if i % 2 else "Male",
                "SeniorCitizen": i % 2,
                "Partner": "Yes" if i % 3 else "No",
                "Dependents": "Yes" if i % 4 else "No",
                "tenure": tenure,
                "PhoneService": "No" if i % 11 == 0 else "Yes",
                "MultipleLines": multiple_lines[i % 3],
                "InternetService": internet[i % 3],
                "OnlineSecurity": service,
                "OnlineBackup": service,
                "DeviceProtection": service,
                "TechSupport": service,
                "StreamingTV": service,
                "StreamingMovies": service,
                "Contract": contracts[i % 3],
                "PaperlessBilling": "Yes" if i % 2 else "No",
                "PaymentMethod": payments[i % 4],
                "MonthlyCharges": monthly,
                "TotalCharges": "" if i % 17 == 0 else str(round(tenure * monthly, 2)),
                "Churn": "Yes" if i % 10 < 3 else "No",
            }
        )
    return pd.DataFrame(records)


def test_main_runs_full_flow_and_persists_versioned_outputs(tmp_path: Path) -> None:
    data_path = tmp_path / "data.csv"
    _synthetic_dataset().to_csv(data_path, index=False)
    models_dir = tmp_path / "models"

    exit_code = main(
        [
            "--algorithm",
            "logistic_regression",
            "--model-version",
            "9.9-test",
            "--data-path",
            str(data_path),
            "--models-dir",
            str(models_dir),
        ]
    )

    assert exit_code == 0
    artifact = models_dir / "churn_classifier_v9.9-test.joblib"
    report_path = models_dir / "churn_classifier_v9.9-test_metrics.json"
    assert artifact.is_file()
    assert report_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["model_version"] == "9.9-test"
    assert report["algorithm"] == "LogisticRegression"
    assert len(report["input_columns"]) == 19
    assert len(report["dataset"]["sha256"]) == 64
    assert 0.0 <= report["metrics"]["roc_auc"] <= 1.0

    restored = joblib.load(artifact)
    assert callable(getattr(restored, "predict_proba", None))
