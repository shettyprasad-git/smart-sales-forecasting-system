from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class AlertStatus(str, Enum):
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class AlertType(str, Enum):
    SALES_SPIKE = "sales_spike"
    SALES_DROP = "sales_drop"
    DEMAND_SHIFT = "demand_shift"
    FORECAST_DEVIATION = "forecast_deviation"
    REPEATED_ANOMALY = "repeated_anomaly"
    TREND_CHANGE = "trend_change"
    CATEGORY_DEVIATION = "category_deviation"
    PRODUCT_DEVIATION = "product_deviation"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertPriority(str, Enum):
    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class MonitoringRunRequest(BaseModel):
    lookback_days: int = Field(default=30, ge=1, le=365, description="Number of past days to scan")
    min_severity: str = Field(default="medium", description="Minimum anomaly severity to consider ('low', 'medium', 'high', 'critical')")
    include_categories: bool = Field(default=True, description="Whether to include category-level anomalies")
    include_products: bool = Field(default=False, description="Whether to scan product-level anomalies")


class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: int
    anomaly_id: str
    alert_type: str
    title: str
    metric: str
    severity: str
    priority: str
    status: str
    event_date: dt.date
    actual_value: float
    expected_value: float
    deviation: float
    deviation_percent: float
    fingerprint: str
    evidence_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: dt.datetime
    acknowledged_at: dt.datetime | None = None
    resolved_at: dt.datetime | None = None
    human_review_required: bool = True
    automatic_execution: bool = False


class AlertDetailResponse(AlertResponse):
    explanation: str | None = None
    top_drivers: list[dict[str, Any]] = Field(default_factory=list)
    has_recommendation: bool = False
    recommendation_type: str | None = None
    simulation_available: bool = False
    recommended_scenario: str | None = None
    decision_id: str | None = None
    decision_status: str | None = None


class AlertListResponse(BaseModel):
    items: list[AlertResponse]
    total: int
    skip: int
    limit: int


class MonitoringSummary(BaseModel):
    scan_timestamp: dt.datetime
    total_anomalies: int
    new_alerts: int
    critical_alerts: int
    high_alerts: int
    medium_alerts: int
    suppressed_duplicates: int
    unresolved_alerts: int


class MonitoringRunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    status: str
    started_at: dt.datetime
    completed_at: dt.datetime | None = None
    total_records_evaluated: int
    anomalies_found: int
    alerts_created: int
    duplicates_suppressed: int
    summary: MonitoringSummary
    error_message: str | None = None
    human_review_required: bool = True
    automatic_execution: bool = False


class MonitoringRunItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    started_at: dt.datetime
    completed_at: dt.datetime | None = None
    total_records_evaluated: int
    anomalies_found: int
    alerts_created: int
    duplicates_suppressed: int
    status: str
    error_message: str | None = None
