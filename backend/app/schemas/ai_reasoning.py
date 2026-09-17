from __future__ import annotations

from typing import Any, Literal, Self
from pydantic import BaseModel, Field, model_validator

ConfidenceLevel = Literal["low", "medium", "high"]


class AIInsight(BaseModel):
    dimension: str = Field(
        default="other",
        description="Evidence dimension: category, product, promotion, holiday, price, discount, drift, other",
    )
    statement: str = Field(
        ..., description="Concise business or analytical insight statement"
    )
    supporting_evidence: str = Field(
        ...,
        description="Factual empirical evidence from the statistical investigation supporting this insight",
    )
    confidence: ConfidenceLevel = Field(
        ...,
        description="Confidence level based on sample size and signal consistency: low, medium, high",
    )
    related_driver: str | None = Field(
        default=None,
        description="Associated driver, dimension, or factor name where applicable",
    )
    associated_driver: str | None = Field(
        default=None,
        description="Associated driver, dimension, or factor name where applicable",
    )

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def sync_driver_fields_before(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "associated_driver" in data and not data.get("related_driver"):
                data["related_driver"] = data["associated_driver"]
            elif "related_driver" in data and not data.get("associated_driver"):
                data["associated_driver"] = data["related_driver"]
            if "observation" in data and not data.get("statement"):
                data["statement"] = data["observation"]
            if "commercial_implication" in data and not data.get("supporting_evidence"):
                data["supporting_evidence"] = data["commercial_implication"]
        return data

    @model_validator(mode="after")
    def sync_driver_fields_after(self) -> Self:
        if self.associated_driver and not self.related_driver:
            self.related_driver = self.associated_driver
        elif self.related_driver and not self.associated_driver:
            self.associated_driver = self.related_driver
        return self


class AIReasoningResponse(BaseModel):
    anomaly_id: str = Field(..., description="Unique anomaly record identifier")
    reasoning_headline: str = Field(
        ...,
        description="One concise factual statement synthesizing the primary reasoning outcome",
    )
    executive_interpretation: str = Field(
        ...,
        description="Executive interpretation translating statistical findings into commercial meaning",
    )
    key_insights: list[AIInsight] = Field(
        ...,
        description="2 to 4 prioritized insights directly derived from the structured evidence",
    )
    alternative_explanations: list[str] = Field(
        ...,
        description="Plausible alternative explanations supported by the data, explicitly labeled as possibilities",
    )
    evidence_assessment: str = Field(
        ...,
        description="Evaluation of which evidence dimensions are robust versus circumstantial or limited",
    )
    uncertainties: list[str] = Field(
        ...,
        description="Identified ambiguities, missing data dimensions, or historical sample constraints",
    )
    validation_questions: list[str] = Field(
        ...,
        description="Actionable questions for finance or operational teams to verify with business stakeholders",
    )
    risk_flags: list[str] = Field(
        ...,
        description="Operational, commercial, or data integrity risks highlighted by the deviation",
    )
    causal_disclaimer: str = Field(
        default=(
            "Observational associations identify statistically correlated contributors; "
            "they do not establish counterfactual causality."
        ),
        description="Mandatory scientific disclaimer on observational sales attribution",
    )
    cached: bool = Field(
        default=False,
        description="Telemetry flag indicating if the response was served from the in-memory cache",
    )

    model_config = {"from_attributes": True}

    @model_validator(mode="before")
    @classmethod
    def sync_response_fields_before(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "headline" in data and not data.get("reasoning_headline"):
                data["reasoning_headline"] = data["headline"]
            elif "reasoning_headline" in data and not data.get("headline"):
                data["headline"] = data["reasoning_headline"]
            if "interpretation" in data and not data.get("executive_interpretation"):
                data["executive_interpretation"] = data["interpretation"]
            elif "executive_interpretation" in data and not data.get("interpretation"):
                data["interpretation"] = data["executive_interpretation"]
        return data

    @property
    def headline(self) -> str:
        return self.reasoning_headline

    @property
    def interpretation(self) -> str:
        return self.executive_interpretation

