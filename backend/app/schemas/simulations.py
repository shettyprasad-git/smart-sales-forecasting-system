from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, model_validator


class ScenarioType(str, Enum):
    DEMAND_MULTIPLIER = "demand_multiplier"
    TEMPORARY_SHOCK = "temporary_shock"
    PERSISTENT_SHIFT = "persistent_shift"
    TREND_CONTINUATION = "trend_continuation"
    PROMOTION_SCENARIO = "promotion_scenario"
    HOLIDAY_SCENARIO = "holiday_scenario"
    PRICE_CHANGE = "price_change"
    DISCOUNT_CHANGE = "discount_change"


class SimulationStatus(str, Enum):
    COMPLETED = "completed"
    REQUIRES_MODEL = "requires_model"
    REQUIRES_MODEL_SUPPORT = "requires_model_support"
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID_REQUEST = "invalid_request"


class SimulationRequest(BaseModel):
    anomaly_id: str | None = Field(
        default=None,
        description="Optional anomaly ID to anchor the simulation with contextual investigation evidence.",
    )
    horizon_days: Literal[7, 30, 90] = Field(
        default=30,
        description="Forecast horizon in days (7, 30, or 90).",
    )
    scenario_type: ScenarioType = Field(
        ...,
        description="Hypothetical scenario type to evaluate.",
    )
    demand_change_percent: float | None = Field(
        default=None,
        description="Percentage shift in demand (-50.0 to +50.0%).",
    )
    shock_duration_days: int | None = Field(
        default=None,
        description="Duration of temporary demand shock in days (1 to horizon_days).",
    )
    trend_window_days: int = Field(
        default=28,
        description="Historical trend evaluation window in days (7 to 90).",
    )
    promotion_active: bool = Field(
        default=True,
        description="Promotional campaign flag for promotion scenario.",
    )
    holiday_active: bool = Field(
        default=True,
        description="Holiday trading flag for holiday scenario.",
    )
    include_revenue: bool = Field(
        default=True,
        description="Whether to calculate revenue under the baseline realized-price assumption.",
    )
    include_explanation: bool = Field(
        default=True,
        description="Whether to include a narrative explanation of already-calculated results.",
    )

    @model_validator(mode="after")
    def validate_request_bounds(self) -> SimulationRequest:
        # Validate demand_change_percent
        if self.demand_change_percent is not None:
            if not (-50.0 <= self.demand_change_percent <= 50.0):
                raise ValueError("demand_change_percent must be between -50.0 and +50.0 percent.")

        # Validate shock_duration_days for temporary shock
        if self.scenario_type == ScenarioType.TEMPORARY_SHOCK:
            if self.shock_duration_days is None:
                self.shock_duration_days = min(7, self.horizon_days)
            if self.shock_duration_days <= 0 or self.shock_duration_days > self.horizon_days:
                raise ValueError(
                    f"shock_duration_days must be between 1 and {self.horizon_days} (horizon_days)."
                )

        # Validate trend_window_days
        if not (7 <= self.trend_window_days <= 90):
            raise ValueError("trend_window_days must be between 7 and 90 days.")

        return self


class ForecastSummary(BaseModel):
    total_quantity: float = Field(..., description="Total cumulative quantity over the horizon.")
    total_revenue: float | None = Field(default=None, description="Total revenue over the horizon in ₹.")
    average_daily_quantity: float = Field(..., description="Average daily quantity.")
    average_daily_revenue: float | None = Field(default=None, description="Average daily revenue in ₹.")


class ForecastDelta(BaseModel):
    quantity_delta: float = Field(..., description="Difference in total quantity (scenario - baseline).")
    quantity_delta_percent: float = Field(..., description="Percentage difference in total quantity.")
    revenue_delta: float | None = Field(default=None, description="Difference in total revenue (scenario - baseline) in ₹.")
    revenue_delta_percent: float | None = Field(default=None, description="Percentage difference in total revenue.")


class DailySimulationResult(BaseModel):
    date: dt.date = Field(..., description="Calendar date of the daily forecast.")
    baseline_quantity: float = Field(..., description="Baseline model forecast quantity.")
    scenario_quantity: float = Field(..., description="Hypothetical scenario forecast quantity.")
    delta_quantity: float = Field(..., description="Daily quantity delta (scenario - baseline).")
    delta_percent: float = Field(..., description="Daily percentage delta relative to baseline.")
    baseline_revenue: float | None = Field(default=None, description="Baseline daily revenue in ₹.")
    scenario_revenue: float | None = Field(default=None, description="Scenario daily revenue in ₹.")
    delta_revenue: float | None = Field(default=None, description="Daily revenue delta in ₹.")


class AnomalyContext(BaseModel):
    anomaly_id: str = Field(..., description="Associated anomaly identifier.")
    anomaly_date: dt.date = Field(..., description="Anomaly occurrence date.")
    metric: str = Field(..., description="Metric evaluated: quantity or sales_amount.")
    actual_value: float = Field(..., description="Actual value recorded on anomaly date.")
    expected_baseline: float = Field(..., description="Expected baseline on anomaly date.")
    deviation: float = Field(..., description="Measured deviation on anomaly date.")
    deviation_percent: float = Field(..., description="Percentage deviation on anomaly date.")
    severity: str = Field(..., description="Severity level: low, medium, high, critical.")
    key_contributors: list[str] = Field(default_factory=list, description="Top associated root-cause drivers.")


class SimulationResponse(BaseModel):
    status: SimulationStatus = Field(..., description="Execution status of the simulation.")
    simulation_id: str = Field(..., description="Unique simulation identifier.")
    anomaly_id: str | None = Field(default=None, description="Associated anomaly ID if anchored.")
    scenario_type: ScenarioType = Field(..., description="Evaluated scenario type.")
    horizon_days: int = Field(..., description="Forecast horizon evaluated (7, 30, or 90 days).")
    model_name: str = Field(..., description="Underlying production machine learning model.")
    anomaly_context: AnomalyContext | None = Field(default=None, description="Context from investigated anomaly.")
    assumptions: list[str] = Field(default_factory=list, description="Explicit modeling and operational assumptions.")
    limitations: list[str] = Field(default_factory=list, description="Methodological limitations and causal boundaries.")
    baseline: ForecastSummary | None = Field(default=None, description="Baseline production forecast summary.")
    scenario: ForecastSummary | None = Field(default=None, description="Scenario forecast summary.")
    delta: ForecastDelta | None = Field(default=None, description="Cumulative variance between scenario and baseline.")
    daily_results: list[DailySimulationResult] = Field(default_factory=list, description="Day-by-day simulated timeline.")
    confidence: Literal["high", "medium", "lower", "not_available"] = Field(
        default="high",
        description="Confidence grade based on scenario assumption strength.",
    )
    source: Literal["deterministic", "gemini_explanation"] = Field(
        default="deterministic",
        description="Origin of the narrative explanation.",
    )
    explanation: str | None = Field(default=None, description="Concise narrative interpretation of simulated variance.")
    reason: str | None = Field(default=None, description="Reason for unsupported or unexecuted scenarios.")
    required_model: str | None = Field(default=None, description="Specific model required for unsupported scenario.")
