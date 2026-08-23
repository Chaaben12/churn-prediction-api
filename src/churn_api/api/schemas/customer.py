"""Input schemas for customer profiles.

Field names are snake_case; aliases mirror the raw dataset vocabulary so the
service layer can feed records straight into the shared pipeline without any
renaming logic (single source of truth stays inside the model artifact).
Clients may send either form thanks to ``populate_by_name``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ChurnServiceFlag = Literal["Yes", "No"]
PhoneRelatedFlag = Literal["Yes", "No", "No phone service"]
InternetRelatedFlag = Literal["Yes", "No", "No internet service"]


class CustomerProfile(BaseModel):
    """One customer as described by the 19 model input features.

    ``total_charges: null`` means a brand-new customer with no billing cycle
    closed yet; the shared pipeline owns the zero-imputation rule, so the API
    never duplicates preprocessing knowledge.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    customer_id: str | None = Field(default=None, alias="customerID", max_length=64)
    gender: Literal["Female", "Male"]
    senior_citizen: int = Field(alias="SeniorCitizen", ge=0, le=1)
    partner: ChurnServiceFlag = Field(alias="Partner")
    dependents: ChurnServiceFlag = Field(alias="Dependents")
    tenure: int = Field(ge=0, le=120)
    phone_service: ChurnServiceFlag = Field(alias="PhoneService")
    multiple_lines: PhoneRelatedFlag = Field(alias="MultipleLines")
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(alias="InternetService")
    online_security: InternetRelatedFlag = Field(alias="OnlineSecurity")
    online_backup: InternetRelatedFlag = Field(alias="OnlineBackup")
    device_protection: InternetRelatedFlag = Field(alias="DeviceProtection")
    tech_support: InternetRelatedFlag = Field(alias="TechSupport")
    streaming_tv: InternetRelatedFlag = Field(alias="StreamingTV")
    streaming_movies: InternetRelatedFlag = Field(alias="StreamingMovies")
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(alias="Contract")
    paperless_billing: ChurnServiceFlag = Field(alias="PaperlessBilling")
    payment_method: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ] = Field(alias="PaymentMethod")
    monthly_charges: float = Field(alias="MonthlyCharges", ge=0)
    total_charges: float | None = Field(default=None, alias="TotalCharges", ge=0)

    def to_pipeline_record(self) -> dict[str, object]:
        """Serialize to the exact record shape expected by the pipeline."""
        record = self.model_dump(by_alias=True)
        if record["TotalCharges"] is None:
            record["TotalCharges"] = ""
        return record
