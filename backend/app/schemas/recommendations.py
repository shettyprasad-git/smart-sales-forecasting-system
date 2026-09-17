from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, model_validator

RecommendationType = Literal[
    "inventory_review",
    "pricing_review",
    "promotion_review",
    "category_review",
    "product_review",
    "demand_monitoring",
    "forecast_review",
    "data_validation",
]

ConfidenceLevel = Literal["low", "medium", "high"]
PriorityLevel = Literal["low", "medium", "high", "critical"]
RiskLevel = Literal["low", "medium", "high"]
RecommendationStatus = Literal["actionable", "review_required", "no_actionable_recommendation"]


class Recommendation(BaseModel):
    id: str = Field(..., description="Unique recommendation identifier")
    anomaly_id: str = Field(..., description="Associated anomaly identifier")
    recommendation_type: RecommendationType = Field(
        ..., description="Controlled recommendation classification"
    )
    title: str = Field(..., description="Executive action headline")
    action: str = Field(..., description="Non-autonomous advisory action to consider")
    reason: str = Field(..., description="Commercial and analytical justification")
    supporting_evidence: str = Field(
        ..., description="Empirical evidence cited from prior pipeline phases"
    )
    expected_objective: str = Field(
        ..., description="Anticipated business purpose or risk mitigation target"
    )
    confidence: ConfidenceLevel = Field(
        ..., description="Confidence grade capped by underlying driver confidence"
    )
    priority: PriorityLevel = Field(
        ..., description="Deterministic urgency priority: low, medium, high, critical"
    )
    risk_level: RiskLevel = Field(
        ..., description="Operational and commercial risk assessment: low, medium, high"
    )
    tradeoffs: list[str] = Field(
        default_factory=list, description="Documented trade-offs and potential downsides"
    )
    validation_required: list[str] = Field(
        default_factory=list,
        description="Prerequisite human validations prior to operational changes",
    )
    is_actionable: bool = Field(
        default=True,
        description="Indicates whether the recommendation warrants immediate human review",
    )
    human_approval_required: bool = Field(
        default=True,
        description="Mandatory governance safeguard: human approval is required for all recommendations",
    )

    model_config = {"from_attributes": True}

    @model_validator(mode="after")
    def enforce_human_approval(self) -> Recommendation:
        # Phase 6.5 invariant: human approval is unconditionally mandatory
        if not self.human_approval_required:
            raise ValueError("human_approval_required must unconditionally be True")
        return self


class RecommendationResponse(BaseModel):
    anomaly_id: str = Field(..., description="Associated anomaly record identifier")
    recommendation_status: RecommendationStatus = Field(
        ..., description="Status indicating whether actionable advisory steps were identified"
    )
    summary: str = Field(
        ..., description="High-level synthesis of prescriptive recommendations for leadership"
    )
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="Prioritized collection of at most 3 primary recommendations",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Methodological caveats, observational constraints, and policy boundaries",
    )
    human_approval_required: bool = Field(
        default=True,
        description="Governance flag ensuring autonomous operational execution is prohibited",
    )
    source: str = Field(
        default="ai_generated",
        description="Origin indicator: 'ai_generated' or 'deterministic_fallback'",
    )
    cached: bool = Field(
        default=False,
        description="Indicates whether the response was served from the in-memory cache",
    )

    model_config = {"from_attributes": True}

    @model_validator(mode="after")
    def enforce_human_approval(self) -> RecommendationResponse:
        if not self.human_approval_required:
            raise ValueError("human_approval_required must unconditionally be True")
        return self
