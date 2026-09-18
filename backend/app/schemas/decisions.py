from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class DecisionStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


class RecommendationSnapshot(BaseModel):
    recommendation_id: str = Field(..., description="Unique recommendation identifier")
    recommendation_type: str = Field(..., description="Recommendation category")
    title: str | None = Field(default=None, description="Executive title")
    action: str = Field(..., description="Proposed action wording")
    priority: str | None = Field(default=None, description="Priority level: low, medium, high, urgent")
    risk: str | None = Field(default=None, description="Risk assessment: low, medium, high")
    confidence: str | None = Field(default=None, description="Confidence grade: low, medium, high")
    supporting_evidence: list[str] = Field(default_factory=list, description="Empirical evidence cited")
    assumptions: list[str] = Field(default_factory=list, description="Explicit assumptions")
    tradeoffs: list[str] = Field(default_factory=list, description="Evaluated trade-offs")
    limitations: list[str] = Field(default_factory=list, description="Identified limitations")


class AnomalySnapshot(BaseModel):
    anomaly_id: str = Field(..., description="Anchored anomaly ID")
    date: str = Field(..., description="Date anomaly occurred")
    metric: str = Field(..., description="Metric evaluated: quantity or sales_amount")
    actual: float = Field(..., description="Observed value on date")
    baseline: float = Field(..., description="Expected baseline")
    deviation: float = Field(..., description="Deviation from baseline")
    severity: str = Field(..., description="Severity: low, medium, high, critical")
    associated_contributors: list[str] = Field(default_factory=list, description="Top root-cause drivers")


class SimulationSnapshot(BaseModel):
    simulation_id: str = Field(..., description="Evaluated what-if simulation ID")
    scenario_type: str = Field(..., description="Scenario type explored")
    horizon_days: int = Field(..., description="Forecast horizon in days")
    baseline_quantity: float = Field(..., description="Total baseline forecasted quantity")
    scenario_quantity: float = Field(..., description="Total scenario forecasted quantity")
    quantity_delta: float = Field(..., description="Quantity delta (scenario - baseline)")
    baseline_revenue: float | None = Field(default=None, description="Total baseline revenue in ₹")
    scenario_revenue: float | None = Field(default=None, description="Total scenario revenue in ₹")
    revenue_delta: float | None = Field(default=None, description="Revenue delta in ₹")
    assumptions: list[str] = Field(default_factory=list, description="Simulation modeling assumptions")
    limitations: list[str] = Field(default_factory=list, description="Simulation limitations")
    confidence: str = Field(default="high", description="Simulation confidence grade")


class DecisionPackage(BaseModel):
    recommendation: RecommendationSnapshot = Field(..., description="Immutable snapshot of recommendation")
    anomaly_context: AnomalySnapshot | None = Field(default=None, description="Anchored anomaly snapshot")
    simulation: SimulationSnapshot | None = Field(default=None, description="Associated simulation snapshot")
    assumptions: list[str] = Field(default_factory=list, description="Combined explicit assumptions")
    limitations: list[str] = Field(default_factory=list, description="Combined operational limitations")
    evidence_quality: str = Field(default="high", description="Data evidence quality score")
    approval_requirement: str = Field(
        default="Human review and approval required before consideration/execution",
        description="Mandatory governance advisory",
    )


class DecisionCreateRequest(BaseModel):
    recommendation_id: str | None = Field(default=None, description="Identifier of the recommendation")
    anomaly_id: str | None = Field(default=None, description="Associated anomaly ID")
    simulation_id: str | None = Field(default=None, description="Optional associated what-if simulation ID")
    recommendation_type: str = Field(..., min_length=1, max_length=50, description="Recommendation category")
    proposed_action: str = Field(..., min_length=1, max_length=2000, description="Original proposed recommendation action")
    modified_action: str | None = Field(default=None, max_length=2000, description="Reviewer modified action wording")
    decision_note: str | None = Field(default=None, max_length=2000, description="Initial reviewer notes")
    recommendation_payload: dict[str, Any] | None = Field(
        default=None, description="Complete recommendation dictionary if submitted directly"
    )


class DecisionActionRequest(BaseModel):
    decision_note: str | None = Field(default=None, max_length=2000, description="Reviewer rationale/note")
    modified_action: str | None = Field(default=None, max_length=2000, description="Optional modified action wording")


class DecisionResubmitRequest(BaseModel):
    modified_action: str | None = Field(default=None, max_length=2000, description="Revised action wording")
    decision_note: str | None = Field(default=None, max_length=2000, description="Clarification or resubmission rationale")


class DecisionAuditEventResponse(BaseModel):
    id: int = Field(..., description="Audit event ID")
    decision_id: str = Field(..., description="Associated decision ID")
    actor_user_id: int = Field(..., description="User ID of decision-maker")
    event_type: str = Field(..., description="created, approved, rejected, changes_requested, reopened")
    previous_status: str | None = Field(default=None, description="Prior lifecycle state")
    new_status: str = Field(..., description="New lifecycle state")
    note: str | None = Field(default=None, description="Human decision rationale or note")
    event_metadata: dict[str, Any] | None = Field(default=None, description="Additional audit metadata")
    created_at: dt.datetime = Field(..., description="Timestamp event was permanently recorded")

    model_config = {"from_attributes": True}


class DecisionResponse(BaseModel):
    id: str = Field(..., description="Unique decision record ID")
    status: DecisionStatus = Field(..., description="Current lifecycle state")
    recommendation_type: str = Field(..., description="Recommendation category")
    proposed_action: str = Field(..., description="Original immutable proposed action")
    modified_action: str | None = Field(default=None, description="Human modified action if provided")
    rationale: str | None = Field(default=None, description="Human rationale for decision")
    decision_note: str | None = Field(default=None, description="Reviewer decision notes")
    anomaly_id: str | None = Field(default=None, description="Associated anomaly ID")
    simulation_id: str | None = Field(default=None, description="Associated simulation ID")
    created_at: dt.datetime = Field(..., description="Creation timestamp")
    reviewed_at: dt.datetime | None = Field(default=None, description="Final review timestamp")
    updated_at: dt.datetime = Field(..., description="Last update timestamp")
    evidence_snapshot: dict[str, Any] = Field(default_factory=dict, description="Immutable evidence snapshot at submission")
    simulation_snapshot: dict[str, Any] | None = Field(default=None, description="Immutable simulation snapshot if anchored")
    audit_events: list[DecisionAuditEventResponse] = Field(default_factory=list, description="Chronological immutable audit trail")
    human_approval_required: Literal[True] = Field(
        default=True,
        description="Explicit guardrail: Human approval is mandatory for all recommendations.",
    )
    automatic_execution: Literal[False] = Field(
        default=False,
        description="Explicit guardrail: Zero autonomous business execution is performed.",
    )

    model_config = {"from_attributes": True}
