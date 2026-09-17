from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

ConfidenceLevel = Literal["low", "medium", "high"]
PriorityLevel = Literal["low", "medium", "high", "critical"]
RiskLevel = Literal["low", "medium", "high"]


class AIRecommendation(BaseModel):
    recommendation_type: str = Field(
        ...,
        description="Must be one of the explicitly supplied eligible types: inventory_review, pricing_review, promotion_review, category_review, product_review, demand_monitoring, forecast_review, data_validation",
    )
    title: str = Field(
        ..., description="Executive advisory title"
    )
    action: str = Field(
        ...,
        description="Advisory action to consider. Use non-autonomous verbs: Review, Consider, Validate, Monitor, Reassess. Never prescribe exact numeric quantities or price cuts.",
    )
    reason: str = Field(
        ..., description="Analytical explanation for why this action deserves consideration"
    )
    supporting_evidence: str = Field(
        ..., description="Direct citation of empirical driver or metric from the evidence package"
    )
    expected_objective: str = Field(
        default="Mitigate commercial risk and validate observed demand trajectory",
        description="Expected analytical or risk mitigation objective",
    )
    confidence: ConfidenceLevel = Field(
        default="medium",
        description="Confidence level (capped by driver confidence): low, medium, high",
    )
    priority: PriorityLevel = Field(
        default="medium",
        description="Urgency priority: low, medium, high, critical",
    )
    risk_level: RiskLevel = Field(
        default="low",
        description="Risk level: low, medium, high",
    )
    tradeoffs: list[str] = Field(
        default_factory=list,
        description="Potential operational or commercial trade-offs",
    )
    validation_required: list[str] = Field(
        default_factory=list,
        description="Specific verification questions or data checks for human reviewers",
    )

    model_config = {"from_attributes": True}


class AIRecommendationResponse(BaseModel):
    summary: str = Field(
        ...,
        description="Executive summary synthesizing the recommended actions and commercial context",
    )
    recommendations: list[AIRecommendation] = Field(
        default_factory=list,
        description="Collection of eligible advisory recommendations",
    )

    model_config = {"from_attributes": True}
