"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churn_api.main import create_app


def _customer_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "customerID": "0000-XXXX",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0,
    }
    row.update(overrides)
    return row


@pytest.fixture()
def raw_customers() -> pd.DataFrame:
    """A small raw batch covering every category value and edge cases."""
    rows = [
        _customer_row(customerID="0001-A", tenure=1, TotalCharges=""),
        _customer_row(customerID="0002-B", tenure=1, TotalCharges=" "),
        _customer_row(customerID="0003-C", tenure=1, TotalCharges=0.0),
        _customer_row(
            customerID="0004-D",
            gender="Male",
            SeniorCitizen=1,
            Partner="Yes",
            Dependents="Yes",
            tenure=70,
            PhoneService="No",
            MultipleLines="No phone service",
            InternetService="No",
            OnlineSecurity="No internet service",
            Contract="Two year",
            PaperlessBilling="No",
            PaymentMethod="Mailed check",
            MonthlyCharges=20.0,
            TotalCharges=1400.0,
            StreamingTV="No internet service",
            StreamingMovies="No internet service",
        ),
        _customer_row(
            customerID="0005-E",
            InternetService="Fiber optic",
            OnlineSecurity="Yes",
            Contract="One year",
            PaymentMethod="Credit card (automatic)",
            MonthlyCharges=95.0,
            TotalCharges=2280.0,
        ),
        _customer_row(
            customerID="0006-F",
            MultipleLines="Yes",
            OnlineBackup="Yes",
            DeviceProtection="Yes",
            TechSupport="Yes",
            StreamingTV="Yes",
            StreamingMovies="Yes",
            PaymentMethod="Bank transfer (automatic)",
            MonthlyCharges=85.0,
            TotalCharges=1020.0,
        ),
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def api_client() -> Iterator[TestClient]:
    """HTTP client bound to the real app (real committed model artifact)."""
    with TestClient(create_app()) as client:
        yield client
