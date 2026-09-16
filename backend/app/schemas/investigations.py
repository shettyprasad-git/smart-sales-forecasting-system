import datetime as dt
from typing import Literal
from pydantic import BaseModel, Field

DriverType = Literal[
    "promotion",
    "holiday",
    "category",
    "product",
    "price",
    "discount",
    "recent_trend",
    "baseline_drift",
    "other",
]

ConfidenceLevel = Literal["low", "medium", "high"]
DriverDirection = Literal["positive", "negative", "neutral"]


class InvestigationDriver(BaseModel):
    driver_type: DriverType = Field(..., description="Category/dimension of the driver factor")
    driver_name: str = Field(..., description="Human-readable label for the driver")
    direction: DriverDirection = Field(
        ...,
        description="Direction of the driver's effect: positive, negative, or neutral",
    )
    observed_value: float | None = Field(
        default=None,
        description="Value observed on the anomaly date for this dimension",
    )
    reference_value: float | None = Field(
        default=None,
        description="Baseline or historical reference value prior to anomaly date",
    )
    difference: float | None = Field(
        default=None,
        description="Absolute difference between observed and reference (observed - reference)",
    )
    difference_percent: float | None = Field(
        default=None,
        description="Percentage shift relative to reference value",
    )
    contribution_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated relative contribution share percentage or importance score",
    )
    evidence: str = Field(
        ...,
        description="Factual evidence explanation supporting this driver's association",
    )
    confidence: ConfidenceLevel = Field(
        ...,
        description="Confidence grade based on sample size and signal consistency: low, medium, high",
    )
    is_event_related: bool = Field(
        default=False,
        description="Whether this driver represents a discrete marketing/calendar event",
    )

    model_config = {"from_attributes": True}


class EstimatedImpact(BaseModel):
    impact_metric: str = Field(..., description="Metric being quantified (quantity or sales_amount)")
    impact_value: float = Field(
        ...,
        description="Non-negative magnitude of impact (excess units/sales or lost units/revenue gap)",
    )
    signed_deviation: float = Field(
        ...,
        description="Signed difference (actual - expected baseline)",
    )
    deviation_percent: float = Field(
        ...,
        description="Percentage deviation relative to baseline",
    )
    estimated_revenue_impact: float | None = Field(
        default=None,
        description="Estimated monetary impact in currency units where applicable",
    )
    price_basis_explanation: str | None = Field(
        default=None,
        description="Explicit documentation of the pricing assumption used for monetary translation",
    )
    interpretation: str = Field(
        ...,
        description="Executive interpretation of the business and financial impact",
    )

    model_config = {"from_attributes": True}


class InvestigationResponse(BaseModel):
    anomaly_id: str = Field(..., description="Unique anomaly record identifier")
    anomaly_date: dt.date = Field(..., description="Date of the investigated anomaly")
    metric: Literal["quantity", "sales_amount"] = Field(..., description="Metric evaluated")
    entity_type: Literal["aggregate", "product", "category"] = Field(
        ..., description="Granularity level: aggregate, category, or product"
    )
    entity_id: str | None = Field(default=None, description="Optional entity ID")
    entity_name: str | None = Field(default=None, description="Optional entity name")
    actual_value: float = Field(..., description="Observed value on this date")
    expected_baseline: float = Field(..., description="Expected baseline value")
    deviation: float = Field(..., description="Signed deviation (actual - expected)")
    deviation_percent: float = Field(..., description="Percentage deviation relative to baseline")
    anomaly_score: float = Field(..., description="Robust z-score magnitude")
    severity: Literal["low", "medium", "high", "critical"] = Field(..., description="Severity level")
    direction: Literal["spike", "drop"] = Field(..., description="Spike or drop")
    estimated_impact: EstimatedImpact = Field(..., description="Quantified business impact analysis")
    impact_interpretation: str = Field(..., description="Summary impact narrative")
    drivers: list[InvestigationDriver] = Field(
        ..., description="Ranked list of contributing driver factors"
    )
    investigation_summary: str = Field(
        ..., description="Synthesized deterministic executive summary of findings"
    )
    evidence_notes: list[str] = Field(
        ..., description="Specific evidence observations and statistical references"
    )
    limitations: list[str] = Field(
        ..., description="Transparent methodological and observational data limitations"
    )

    model_config = {"from_attributes": True}
