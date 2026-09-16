import datetime as dt
from typing import Literal
from pydantic import BaseModel, Field


SeverityLevel = Literal["low", "medium", "high", "critical"]
AnomalyDirection = Literal["spike", "drop"]
MetricType = Literal["quantity", "sales_amount"]
EntityType = Literal["aggregate", "product", "category"]


class AnomalyItem(BaseModel):
    id: str = Field(..., description="Unique anomaly record identifier")
    date: dt.date = Field(..., description="Date of the anomaly observation")
    metric: MetricType = Field(..., description="Metric evaluated: quantity or sales_amount")
    entity_type: EntityType = Field(
        default="aggregate",
        description="Entity granularity: aggregate, product, or category",
    )
    entity_id: str | None = Field(
        default=None,
        description="Optional entity identifier (e.g. Product ID or Category ID)",
    )
    entity_name: str | None = Field(
        default=None,
        description="Optional human-readable entity name",
    )
    actual_value: float = Field(..., description="Observed value on this date")
    expected_value: float = Field(
        ...,
        description="Expected baseline value computed from historical rolling window",
    )
    deviation: float = Field(..., description="Absolute deviation (actual - expected)")
    deviation_percent: float = Field(
        ...,
        description="Percentage deviation relative to baseline ((actual - expected) / expected * 100)",
    )
    anomaly_score: float = Field(
        ...,
        description="Robust anomaly score (|z-score| computed via Median and MAD)",
    )
    severity: SeverityLevel = Field(
        ...,
        description="Statistically derived severity: low, medium, high, critical",
    )
    direction: AnomalyDirection = Field(
        ...,
        description="Direction of deviation: spike (above baseline) or drop (below baseline)",
    )
    explanation: str = Field(
        ...,
        description="Natural language interpretable explanation of why this was flagged",
    )

    model_config = {"from_attributes": True}


class AnomalySummary(BaseModel):
    total_anomalies: int = Field(..., description="Total anomalies detected matching criteria")
    total_records_evaluated: int = Field(..., description="Total observations analyzed")
    anomaly_rate_percent: float = Field(
        ...,
        description="Percentage of evaluated observations flagged as anomalies",
    )
    severity_breakdown: dict[str, int] = Field(
        ...,
        description="Breakdown of anomaly counts by severity level",
    )
    direction_breakdown: dict[str, int] = Field(
        ...,
        description="Breakdown of anomaly counts by direction (spike vs drop)",
    )
    metric_breakdown: dict[str, int] = Field(
        ...,
        description="Breakdown of anomaly counts by metric",
    )
    start_date: dt.date | None = Field(default=None, description="Earliest date evaluated")
    end_date: dt.date | None = Field(default=None, description="Latest date evaluated")


class AnomalyListResponse(BaseModel):
    items: list[AnomalyItem] = Field(..., description="List of detected anomalies")
    total: int = Field(..., description="Total count of anomalies matching filters")
    skip: int = Field(default=0, description="Pagination skip offset")
    limit: int = Field(default=100, description="Pagination limit")
    summary: AnomalySummary = Field(..., description="Summary statistics for the query")
