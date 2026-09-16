from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

SectionType = Literal[
    "headline",
    "what_happened",
    "why_it_matters",
    "key_contributors",
    "event_context",
    "trend_context",
    "evidence_quality",
    "limitations",
]


class ExplanationSection(BaseModel):
    section_type: SectionType = Field(
        ..., description="Standard classification of the narrative section"
    )
    title: str = Field(..., description="Executive section heading")
    content: str = Field(..., description="Deterministic factual narrative body")
    severity: str | None = Field(
        default=None, description="Optional severity indicator (low, medium, high, critical)"
    )
    confidence: str | None = Field(
        default=None, description="Confidence assessment for the section (low, medium, high)"
    )
    evidence: list[str] | str | None = Field(
        default=None, description="Supporting empirical evidence or data observations"
    )

    model_config = {"from_attributes": True}


class ExecutiveExplanation(BaseModel):
    anomaly_id: str = Field(..., description="Unique anomaly record identifier")
    headline: str = Field(..., description="Concise factual 1-sentence executive headline")
    what_happened: str = Field(
        ..., description="Factual breakdown of observed deviation vs baseline"
    )
    why_it_matters: str = Field(
        ..., description="Quantified business and monetary impact interpretation"
    )
    key_contributors: list[str] = Field(
        ..., description="Prioritized list of top evidence-backed driver summaries"
    )
    impact_summary: str = Field(
        ..., description="Executive impact synthesis"
    )
    evidence_summary: str = Field(
        ..., description="Synthesis of empirical evidence quality across drivers"
    )
    confidence_summary: str = Field(
        ..., description="Overall confidence assessment based on sample size and signal consistency"
    )
    limitations: list[str] = Field(
        ..., description="Methodological disclaimers and observational constraints"
    )
    sections: list[ExplanationSection] = Field(
        ..., description="Ordered list of structured narrative sections"
    )

    model_config = {"from_attributes": True}
