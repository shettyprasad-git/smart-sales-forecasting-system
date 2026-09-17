from __future__ import annotations

import datetime as dt
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.llm.provider import (
    LLMConfigurationError,
    LLMProvider,
    LLMProviderError,
)
from backend.app.main import app
from backend.app.schemas.ai_reasoning import (
    AIInsight,
    AIReasoningResponse,
)
from backend.app.schemas.investigations import (
    EstimatedImpact,
    InvestigationDriver,
    InvestigationResponse,
)
from backend.app.schemas.recommendation_ai import (
    AIRecommendation,
    AIRecommendationResponse,
)
from backend.app.schemas.recommendations import (
    Recommendation,
    RecommendationResponse,
)
from backend.app.services.ai_reasoning_service import STANDARD_CAUSAL_DISCLAIMER
from backend.app.services.explanation_service import ExplanationService
from backend.app.services.investigation_service import InvestigationService
from backend.app.services.recommendation_service import RecommendationService


class MockLLMProvider(LLMProvider):
    """Deterministic mock provider simulating Gemini reasoning and recommendation responses."""

    def __init__(
        self,
        canned_reasoning: AIReasoningResponse | None = None,
        canned_recommendations: AIRecommendationResponse | None = None,
        should_fail: bool = False,
        error_to_raise: Exception | None = None,
    ) -> None:
        self.canned_reasoning = canned_reasoning
        self.canned_recommendations = canned_recommendations
        self.should_fail = should_fail
        self.error_to_raise = error_to_raise
        self.reasoning_call_count = 0
        self.recommendation_call_count = 0
        self.last_evidence_package: dict[str, Any] | None = None

    def generate_reasoning(self, evidence_package: dict[str, Any]) -> AIReasoningResponse:
        self.reasoning_call_count += 1
        if self.should_fail:
            raise self.error_to_raise or LLMProviderError("Simulated Gemini reasoning failure")

        if self.canned_reasoning is not None:
            return self.canned_reasoning

        anomaly_id = evidence_package.get("anomaly", {}).get("anomaly_id", "anom-test")
        return AIReasoningResponse(
            anomaly_id=anomaly_id,
            reasoning_headline="Sales revenue deviation was associated with category shifts and price realization.",
            executive_interpretation="The observed spike represents strong customer engagement coinciding with active factors.",
            key_insights=[
                AIInsight(
                    dimension="category",
                    statement="Clothing category was a strong associated factor in total variance.",
                    supporting_evidence="Clothing sales exceeded baseline by 20.3%.",
                    confidence="high",
                    associated_driver="Category: Clothing",
                    related_driver="Category: Clothing",
                ),
            ],
            alternative_explanations=["Multiple recorded categories showed positive deviations."],
            evidence_assessment="Category evidence has robust sample sizes.",
            uncertainties=["Lack of elasticity models precludes counterfactual volume conclusions."],
            validation_questions=["Could product-mix differences contribute to higher unit price?"],
            risk_flags=["Monitor inventory levels."],
            causal_disclaimer=STANDARD_CAUSAL_DISCLAIMER,
            cached=False,
        )

    def generate_recommendations(self, evidence_package: dict[str, Any]) -> AIRecommendationResponse:
        self.recommendation_call_count += 1
        self.last_evidence_package = evidence_package

        if self.should_fail:
            raise self.error_to_raise or LLMProviderError("Simulated Gemini recommendation failure")

        if self.canned_recommendations is not None:
            return self.canned_recommendations

        eligible_types = evidence_package.get("eligible_recommendation_types", ["inventory_review"])
        primary_type = eligible_types[0] if eligible_types else "inventory_review"

        return AIRecommendationResponse(
            summary="Review inventory and baseline forecast models to preserve commercial alignment with demand shifts.",
            recommendations=[
                AIRecommendation(
                    recommendation_type=primary_type,
                    title="Review Inventory Replenishment Schedules",
                    action="Review inventory allocation and replenishment schedules across recorded categories.",
                    reason="Sales exceeded expected baseline by a material margin.",
                    supporting_evidence="Observed volume deviation was +25% over baseline.",
                    expected_objective="Mitigate potential stockout risks while avoiding excessive holding costs.",
                    confidence="high",
                    priority="high",
                    risk_level="medium",
                    tradeoffs=["Accelerated orders incur expedited shipping and holding expenses."],
                    validation_required=["Verify current on-hand warehouse inventory and supplier lead times."],
                )
            ],
        )


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_anomaly_id():
    inv_service = InvestigationService()
    anomalies = inv_service.anomaly_service.get_anomalies(limit=1)
    assert len(anomalies.items) > 0
    return anomalies.items[0].id


def _create_mock_investigation(
    anomaly_id: str = "anom-test-001",
    direction: str = "spike",
    deviation: float = 50000.0,
    deviation_percent: float = 18.5,
    severity: str = "high",
    anomaly_score: float = 3.2,
    drivers: list[InvestigationDriver] | None = None,
) -> InvestigationResponse:
    if drivers is None:
        drivers = [
            InvestigationDriver(
                driver_type="category",
                driver_name="Category: Electronics",
                direction="positive",
                observed_value=200000.0,
                reference_value=150000.0,
                difference=50000.0,
                difference_percent=33.3,
                contribution_score=25.0,
                evidence="Electronics sales rose 33.3% relative to reference.",
                confidence="high",
                is_event_related=False,
            )
        ]

    return InvestigationResponse(
        anomaly_id=anomaly_id,
        anomaly_date=dt.date(2025, 11, 15),
        metric="sales_amount",
        entity_type="aggregate",
        actual_value=300000.0,
        expected_baseline=250000.0,
        deviation=deviation,
        deviation_percent=deviation_percent,
        anomaly_score=anomaly_score,
        severity=severity,  # type: ignore
        direction=direction,  # type: ignore
        estimated_impact=EstimatedImpact(
            impact_metric="sales_amount",
            impact_value=abs(deviation),
            signed_deviation=deviation,
            deviation_percent=deviation_percent,
            estimated_revenue_impact=abs(deviation),
            interpretation="Material revenue deviation observed.",
        ),
        impact_interpretation="Material revenue deviation observed.",
        drivers=drivers,
        investigation_summary="Summary of findings.",
        evidence_notes=["Note 1"],
        limitations=["Limitation 1"],
    )


# =========================================================================
# 1. Eligibility Policy Rule Tests
# =========================================================================


def test_eligibility_inventory_review_on_spike():
    """Inventory review is eligible on a positive spike with catalog/product drivers."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation(
        direction="spike",
        deviation=50000.0,
        drivers=[
            InvestigationDriver(
                driver_type="category",
                driver_name="Category: Apparel",
                direction="positive",
                contribution_score=15.0,
                evidence="Apparel drove sales above baseline.",
                confidence="high",
            )
        ],
    )
    eligible = service.determine_eligibility(inv)
    assert "inventory_review" in eligible


def test_eligibility_pricing_review_on_price_shift():
    """Pricing review is eligible when price drivers or average unit price realization shifts."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation(
        drivers=[
            InvestigationDriver(
                driver_type="price",
                driver_name="Average Unit Price Realization",
                direction="positive",
                contribution_score=20.0,
                evidence="Unit price realization shifted upward by ₹120.",
                confidence="high",
            )
        ]
    )
    eligible = service.determine_eligibility(inv)
    assert "pricing_review" in eligible


def test_eligibility_promotion_review_when_active():
    """Promotion review is eligible when a promotion driver is present."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation(
        drivers=[
            InvestigationDriver(
                driver_type="promotion",
                driver_name="Weekend Flash Promotion",
                direction="positive",
                observed_value=1.0,
                contribution_score=30.0,
                evidence="Promotion active during the anomaly date.",
                confidence="high",
                is_event_related=True,
            )
        ]
    )
    eligible = service.determine_eligibility(inv)
    assert "promotion_review" in eligible


def test_eligibility_category_review_on_material_contribution():
    """Category review is eligible when a category contributes > 5.0%."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation(
        drivers=[
            InvestigationDriver(
                driver_type="category",
                driver_name="Category: Footwear",
                direction="positive",
                contribution_score=12.5,
                evidence="Footwear contributed 12.5% to total deviation.",
                confidence="high",
            )
        ]
    )
    eligible = service.determine_eligibility(inv)
    assert "category_review" in eligible


def test_eligibility_product_review_on_product_driver():
    """Product review is eligible when product-level attribution is present."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation(
        drivers=[
            InvestigationDriver(
                driver_type="product",
                driver_name="Product: Running Shoes",
                direction="positive",
                contribution_score=18.0,
                evidence="Top SKU accounted for majority of apparel variance.",
                confidence="high",
            )
        ]
    )
    eligible = service.determine_eligibility(inv)
    assert "product_review" in eligible


def test_eligibility_demand_monitoring_on_unstable_drift():
    """Demand monitoring is eligible on drops, unstable drift, or moderate/low confidence."""
    service = RecommendationService(provider=MockLLMProvider())
    inv_drop = _create_mock_investigation(
        direction="drop",
        deviation=-30000.0,
        drivers=[
            InvestigationDriver(
                driver_type="other",
                driver_name="External Fluctuation",
                direction="negative",
                contribution_score=5.0,
                evidence="Demand declined unexpectedly.",
                confidence="low",
            )
        ],
    )
    eligible = service.determine_eligibility(inv_drop)
    assert "demand_monitoring" in eligible


def test_eligibility_forecast_review_on_high_severity():
    """Forecast review is eligible when severity is high/critical or deviation percent >= 10.0%."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation(
        severity="high",
        deviation_percent=22.4,
    )
    eligible = service.determine_eligibility(inv)
    assert "forecast_review" in eligible


def test_eligibility_data_validation_on_low_confidence():
    """Data validation is eligible when driver confidence is low or anomaly score >= 4.0."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation(
        anomaly_score=4.8,
        drivers=[
            InvestigationDriver(
                driver_type="category",
                driver_name="Category: Accessories",
                direction="positive",
                contribution_score=6.0,
                evidence="Small sample size on record date.",
                confidence="low",
            )
        ],
    )
    eligible = service.determine_eligibility(inv)
    assert "data_validation" in eligible


# =========================================================================
# 2. Prioritization & Capacity Tests
# =========================================================================


def test_no_actionable_recommendation_when_insufficient_evidence():
    """Zero deviation or missing drivers yields no_actionable_recommendation status."""
    service = RecommendationService(provider=MockLLMProvider())
    inv_zero = _create_mock_investigation(
        deviation=0.0,
        drivers=[],
    )

    with patch.object(service.investigation_service, "investigate_anomaly", return_value=inv_zero):
        resp = service.recommend_for_anomaly("anom-zero-test")
        assert resp.recommendation_status == "no_actionable_recommendation"
        assert len(resp.recommendations) == 0
        assert resp.human_approval_required is True


def test_deterministic_prioritization_ordering():
    """Candidate recommendations are ranked deterministically by severity, contribution, confidence, and risk."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation(
        severity="critical",
        drivers=[
            InvestigationDriver(
                driver_type="category",
                driver_name="Category: Electronics",
                direction="positive",
                contribution_score=40.0,
                evidence="Major driver.",
                confidence="high",
            ),
            InvestigationDriver(
                driver_type="price",
                driver_name="Average Unit Price Realization",
                direction="positive",
                contribution_score=10.0,
                evidence="Minor price drift.",
                confidence="low",
            ),
        ],
    )

    rec_high_contrib = Recommendation(
        id="rec-1",
        anomaly_id=inv.anomaly_id,
        recommendation_type="category_review",
        title="Review High Contribution Category",
        action="Review category inventory and demand allocation.",
        reason="Large contribution.",
        supporting_evidence="40% share.",
        expected_objective="Balance inventory.",
        confidence="high",
        priority="critical",
        risk_level="low",
        tradeoffs=[],
        validation_required=[],
    )

    rec_low_contrib = Recommendation(
        id="rec-2",
        anomaly_id=inv.anomaly_id,
        recommendation_type="pricing_review",
        title="Review Pricing Structure",
        action="Review unit pricing realization.",
        reason="Minor shift.",
        supporting_evidence="10% share.",
        expected_objective="Assess pricing impact on baseline revenue.",
        confidence="low",
        priority="low",
        risk_level="high",
        tradeoffs=[],
        validation_required=[],
    )

    prioritized = service.prioritize_recommendations([rec_low_contrib, rec_high_contrib], inv)
    assert prioritized[0].id == "rec-1"
    assert prioritized[1].id == "rec-2"


def test_maximum_three_recommendations_enforced():
    """At most 3 recommendations are returned even if more are generated or eligible."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation()

    candidates = [
        Recommendation(
            id=f"rec-{i}",
            anomaly_id=inv.anomaly_id,
            recommendation_type="inventory_review",
            title=f"Action {i}",
            action=f"Review area {i}.",
            reason="Observation.",
            supporting_evidence="Evidence.",
            expected_objective="Objective.",
            confidence="medium",
            priority="medium",
            risk_level="low",
            tradeoffs=[],
            validation_required=[],
        )
        for i in range(1, 6)
    ]

    prioritized = service.prioritize_recommendations(candidates, inv)
    assert len(prioritized) <= 3


# =========================================================================
# 3. AI Structured Response & Post-Validation Tests
# =========================================================================


def test_gemini_structured_response_parsing(sample_anomaly_id):
    """Verify Gemini structured response parses cleanly into RecommendationResponse with source='ai_generated'."""
    canned_ai = AIRecommendationResponse(
        summary="Synthesized AI advisory recommendations for the investigated sales anomaly.",
        recommendations=[
            AIRecommendation(
                recommendation_type="inventory_review",
                title="Review Category Inventory Buffer",
                action="Review inventory allocation and replenishment schedules across recorded categories.",
                reason="Demand significantly exceeded baseline.",
                supporting_evidence="Sales variance was +24.5% relative to expected baseline.",
                expected_objective="Prevent out-of-stock events while evaluating lead times.",
                confidence="high",
                priority="high",
                risk_level="medium",
                tradeoffs=["Holding costs increase if surge is temporary."],
                validation_required=["Check warehouse inventory availability."],
            )
        ],
    )

    mock_provider = MockLLMProvider(canned_recommendations=canned_ai)
    service = RecommendationService(provider=mock_provider)

    resp = service.recommend_for_anomaly(sample_anomaly_id, refresh=True)
    assert resp.source == "ai_generated"
    assert resp.recommendation_status == "actionable"
    assert len(resp.recommendations) > 0
    assert resp.recommendations[0].recommendation_type == "inventory_review"
    assert resp.human_approval_required is True


def test_invalid_recommendation_type_rejected(sample_anomaly_id):
    """Recommendations with non-eligible or unregistered recommendation types are rejected."""
    canned_ai = AIRecommendationResponse(
        summary="AI advisory recommendations.",
        recommendations=[
            AIRecommendation(
                recommendation_type="unauthorized_marketing_campaign",  # Invalid type
                title="Launch Marketing Campaign",
                action="Launch new marketing campaign across channels.",
                reason="Boost demand.",
                supporting_evidence="Evidence.",
                confidence="high",
                priority="high",
                risk_level="medium",
                tradeoffs=[],
                validation_required=[],
            )
        ],
    )

    mock_provider = MockLLMProvider(canned_recommendations=canned_ai)
    service = RecommendationService(provider=mock_provider)

    # When all AI recs are invalid, system gracefully falls back to deterministic fallback
    resp = service.recommend_for_anomaly(sample_anomaly_id, refresh=True)
    assert resp.source == "deterministic_fallback"
    for r in resp.recommendations:
        assert r.recommendation_type != "unauthorized_marketing_campaign"


def test_unsupported_recommendation_rejected():
    """A valid recommendation type that is NOT in eligible_types for the anomaly is rejected."""
    service = RecommendationService(provider=MockLLMProvider())
    # Investigation without pricing driver -> pricing_review is ineligible
    inv = _create_mock_investigation(
        drivers=[
            InvestigationDriver(
                driver_type="category",
                driver_name="Category: Books",
                direction="positive",
                contribution_score=12.0,
                evidence="Books sales elevated.",
                confidence="high",
            )
        ]
    )
    eligible = service.determine_eligibility(inv)
    assert "pricing_review" not in eligible

    unsupported_ai_rec = AIRecommendation(
        recommendation_type="pricing_review",
        title="Review Pricing Structure",
        action="Review unit pricing realization across SKUs.",
        reason="Price shift.",
        supporting_evidence="Pricing evidence.",
        confidence="medium",
        priority="medium",
        risk_level="low",
        tradeoffs=[],
        validation_required=[],
    )

    validated = service._post_validate_recommendations(
        raw_recs=[unsupported_ai_rec],
        anomaly_id=inv.anomaly_id,
        investigation=inv,
        eligible_types=eligible,
        driver_confidence_map={"Category: Books": "high"},
    )
    assert len(validated) == 0


def test_numeric_target_hallucination_sanitized():
    """Imperative numeric inventory targets ('Increase inventory by 20%') are sanitized to safe advisory phrasing."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation()

    hallucinated_rec = AIRecommendation(
        recommendation_type="inventory_review",
        title="Increase Inventory Orders",
        action="Increase inventory by 25% across all regional distribution centers immediately.",
        reason="Demand surged.",
        supporting_evidence="Sales +30%.",
        expected_objective="Prevent stockout.",
        confidence="high",
        priority="high",
        risk_level="medium",
        tradeoffs=[],
        validation_required=[],
    )

    validated = service._post_validate_recommendations(
        raw_recs=[hallucinated_rec],
        anomaly_id=inv.anomaly_id,
        investigation=inv,
        eligible_types=["inventory_review"],
        driver_confidence_map={},
    )
    assert len(validated) == 1
    # Check that the imperative quantity directive was sanitized
    assert "25%" not in validated[0].action
    assert "Review inventory allocation and replenishment schedules based on observed demand" in validated[0].action
    assert not validated[0].action.lower().startswith("increase")


def test_price_cut_hallucination_sanitized():
    """Imperative price reduction directives ('Cut prices by 10%') are sanitized."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation()

    price_cut_rec = AIRecommendation(
        recommendation_type="pricing_review",
        title="Reduce Prices",
        action="Cut price by 15% to stimulate lagging demand.",
        reason="Sales slowed down.",
        supporting_evidence="Price deviation.",
        expected_objective="Boost sales.",
        confidence="medium",
        priority="high",
        risk_level="medium",
        tradeoffs=[],
        validation_required=[],
    )

    validated = service._post_validate_recommendations(
        raw_recs=[price_cut_rec],
        anomaly_id=inv.anomaly_id,
        investigation=inv,
        eligible_types=["pricing_review"],
        driver_confidence_map={},
    )
    assert len(validated) == 1
    assert "15%" not in validated[0].action
    assert "Review realized unit prices and discount compliance against baseline" in validated[0].action


def test_guaranteed_revenue_claim_sanitized():
    """Unsubstantiated promises of guaranteed revenue ('Will increase revenue by 20%') are sanitized."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation()

    guaranteed_rec = AIRecommendation(
        recommendation_type="inventory_review",
        title="Buffer Stock",
        action="Review stock levels which will increase revenue by 20% next quarter.",
        reason="Guaranteed to grow revenue by 15% over historical baseline.",
        supporting_evidence="Demand spike.",
        expected_objective="Will boost revenue by 10% across channels.",
        confidence="high",
        priority="medium",
        risk_level="low",
        tradeoffs=[],
        validation_required=[],
    )

    validated = service._post_validate_recommendations(
        raw_recs=[guaranteed_rec],
        anomaly_id=inv.anomaly_id,
        investigation=inv,
        eligible_types=["inventory_review"],
        driver_confidence_map={},
    )
    assert len(validated) == 1
    rec = validated[0]
    assert "will increase revenue by 20%" not in rec.action.lower()
    assert "guaranteed to grow revenue" not in rec.reason.lower()
    assert "will boost revenue" not in rec.expected_objective.lower()


def test_confidence_capping_against_driver():
    """Recommendation confidence is capped against the underlying driver's confidence."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation()

    rec = AIRecommendation(
        recommendation_type="category_review",
        title="Review Category Dynamics",
        action="Review category mix.",
        reason="Category variance.",
        supporting_evidence="Driver Category: Apparel exhibited moderate drift.",
        confidence="high",  # Claimed high confidence
        priority="medium",
        risk_level="low",
        tradeoffs=[],
        validation_required=[],
    )

    validated = service._post_validate_recommendations(
        raw_recs=[rec],
        anomaly_id=inv.anomaly_id,
        investigation=inv,
        eligible_types=["category_review"],
        driver_confidence_map={"Category: Apparel": "low"},  # Underlying driver has low confidence
    )
    assert len(validated) == 1
    assert validated[0].confidence == "low"


def test_causal_language_sanitization():
    """Prohibited causal phrases like 'definitely caused by' are converted to associative terminology."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation()

    causal_rec = AIRecommendation(
        recommendation_type="category_review",
        title="Review Driver",
        action="Review category mix that was definitely caused by catalog shifts.",
        reason="The shift proves that category shifts caused the revenue increase.",
        supporting_evidence="Sales rose.",
        expected_objective="Capture gains that guarantees margin recovery.",
        confidence="medium",
        priority="medium",
        risk_level="low",
        tradeoffs=[],
        validation_required=[],
    )

    validated = service._post_validate_recommendations(
        raw_recs=[causal_rec],
        anomaly_id=inv.anomaly_id,
        investigation=inv,
        eligible_types=["category_review"],
        driver_confidence_map={},
    )
    assert len(validated) == 1
    rec = validated[0]
    assert "definitely caused by" not in rec.action.lower()
    assert "strongly associated with" in rec.action.lower()
    assert "proves that" not in rec.reason.lower()
    assert "indicates that" in rec.reason.lower()
    assert "guarantees" not in rec.expected_objective.lower()


def test_human_approval_always_true():
    """Pydantic model validator and service guarantee human_approval_required is unconditionally True."""
    # Attempting to construct Recommendation with human_approval_required=False raises ValidationError
    with pytest.raises(ValidationError):
        Recommendation(
            id="rec-test",
            anomaly_id="anom-001",
            recommendation_type="inventory_review",
            title="Test Title",
            action="Test Action",
            reason="Test Reason",
            supporting_evidence="Test Evidence",
            expected_objective="Test Objective",
            confidence="medium",
            priority="medium",
            risk_level="low",
            human_approval_required=False,  # Forbidden
        )

    # Attempting to construct RecommendationResponse with human_approval_required=False raises ValidationError
    with pytest.raises(ValidationError):
        RecommendationResponse(
            anomaly_id="anom-001",
            recommendation_status="actionable",
            summary="Test Summary",
            recommendations=[],
            human_approval_required=False,  # Forbidden
            source="deterministic_fallback",
        )


def test_anomaly_id_consistency_enforced(sample_anomaly_id):
    """All recommendations in the response have an anomaly_id matching the investigated anomaly."""
    mock_provider = MockLLMProvider()
    service = RecommendationService(provider=mock_provider)

    resp = service.recommend_for_anomaly(sample_anomaly_id, refresh=True)
    assert resp.anomaly_id == sample_anomaly_id
    for rec in resp.recommendations:
        assert rec.anomaly_id == sample_anomaly_id
        assert rec.id.startswith(f"rec-{sample_anomaly_id}")


# =========================================================================
# 4. Fallback, Cache, and API Tests
# =========================================================================


def test_provider_failure_returns_deterministic_fallback(sample_anomaly_id):
    """When Gemini provider fails and fallback=True, service gracefully returns deterministic fallback with 200 OK."""
    failing_provider = MockLLMProvider(should_fail=True)
    service = RecommendationService(provider=failing_provider)

    resp = service.recommend_for_anomaly(sample_anomaly_id, refresh=True, fallback=True)
    assert resp.source == "deterministic_fallback"
    assert resp.recommendation_status == "actionable"
    assert len(resp.recommendations) > 0
    assert resp.human_approval_required is True
    assert any("Deterministic Fallback" in lim for lim in resp.limitations)


def test_unknown_anomaly_returns_404(client):
    """Querying recommendations for a nonexistent anomaly ID returns HTTP 404."""
    resp = client.get("/api/recommendations/anom-nonexistent-999999")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_cache_and_refresh_behavior(sample_anomaly_id):
    """First call evaluates and caches (cached=False), second call hits cache (cached=True), refresh=True re-evaluates."""
    mock_provider = MockLLMProvider()
    service = RecommendationService(provider=mock_provider)

    # 1. First call: miss
    res1 = service.recommend_for_anomaly(sample_anomaly_id, refresh=False)
    assert res1.cached is False
    assert mock_provider.recommendation_call_count == 1

    # 2. Second call: hit
    res2 = service.recommend_for_anomaly(sample_anomaly_id, refresh=False)
    assert res2.cached is True
    assert mock_provider.recommendation_call_count == 1
    assert res1.summary == res2.summary

    # 3. Third call with refresh=True: bypass
    res3 = service.recommend_for_anomaly(sample_anomaly_id, refresh=True)
    assert res3.cached is False
    assert mock_provider.recommendation_call_count == 2


# =========================================================================
# 5. Promotion Invariance & Business Consequence Regression Tests
# =========================================================================


def test_promotion_review_rejected_when_inactive(sample_anomaly_id):
    """When promotion is inactive (as in sample_anomaly_id), promotion_review is not eligible."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = service.investigation_service.investigate_anomaly(sample_anomaly_id)
    promo_driver = next((d for d in inv.drivers if d.driver_type == "promotion"), None)
    assert promo_driver is not None and promo_driver.observed_value == 0.0

    eligible = service.determine_eligibility(inv)
    assert "promotion_review" not in eligible


def test_fallback_never_recommends_promotion_when_inactive(sample_anomaly_id):
    """Deterministic fallback never outputs promotion_review when promotion is inactive."""
    service = RecommendationService(provider=MockLLMProvider(should_fail=True))
    resp = service.recommend_for_anomaly(sample_anomaly_id, refresh=True, fallback=True)

    assert resp.source == "deterministic_fallback"
    assert resp.recommendation_status == "actionable"
    assert len(resp.recommendations) > 0
    for r in resp.recommendations:
        assert r.recommendation_type != "promotion_review"
        assert "promot" not in r.action.lower()
        assert "promot" not in r.reason.lower()
        assert "promot" not in r.supporting_evidence.lower()


def test_gemini_promotion_recommendation_rejected_when_inactive(sample_anomaly_id):
    """If Gemini outputs promotion_review when promotion is inactive, it is strictly rejected."""
    canned_ai = AIRecommendationResponse(
        summary="AI proposing unauthorized promotion review.",
        recommendations=[
            AIRecommendation(
                recommendation_type="promotion_review",
                title="Review Promotion Campaign",
                action="Review promotional discount margins.",
                reason="Promotion drove sales volume.",
                supporting_evidence="Promotion was active.",
                confidence="high",
                priority="high",
                risk_level="medium",
                tradeoffs=[],
                validation_required=[],
            )
        ],
    )
    mock_provider = MockLLMProvider(canned_recommendations=canned_ai)
    service = RecommendationService(provider=mock_provider)

    resp = service.recommend_for_anomaly(sample_anomaly_id, refresh=True)
    # The unauthorized recommendation is rejected, falling back to deterministic fallback
    for r in resp.recommendations:
        assert r.recommendation_type != "promotion_review"


def test_promotion_claim_rejected_in_recommendation_reason(sample_anomaly_id):
    """If an AI recommendation contains promotional claims in its reason when inactive, it is rejected."""
    canned_ai = AIRecommendationResponse(
        summary="AI proposing recommendation with ungrounded promotional reason.",
        recommendations=[
            AIRecommendation(
                recommendation_type="category_review",
                title="Review Category Mix",
                action="Review category assortment across catalogs.",
                reason="Category sales surged because marketing promotion campaign was active.",
                supporting_evidence="Clothing sales exceeded baseline.",
                confidence="high",
                priority="high",
                risk_level="low",
                tradeoffs=[],
                validation_required=[],
            )
        ],
    )
    mock_provider = MockLLMProvider(canned_recommendations=canned_ai)
    service = RecommendationService(provider=mock_provider)

    resp = service.recommend_for_anomaly(sample_anomaly_id, refresh=True)
    for r in resp.recommendations:
        assert "marketing promotion campaign" not in r.reason.lower()


def test_promotion_claim_rejected_in_supporting_evidence(sample_anomaly_id):
    """If an AI recommendation contains promotional claims in supporting evidence when inactive, it is rejected."""
    canned_ai = AIRecommendationResponse(
        summary="AI proposing recommendation with ungrounded promotional evidence.",
        recommendations=[
            AIRecommendation(
                recommendation_type="category_review",
                title="Review Category Mix",
                action="Review category assortment across catalogs.",
                reason="Category sales exceeded expected baseline.",
                supporting_evidence="Measured deviation coincided with flash sale promo event.",
                confidence="high",
                priority="high",
                risk_level="low",
                tradeoffs=[],
                validation_required=[],
            )
        ],
    )
    mock_provider = MockLLMProvider(canned_recommendations=canned_ai)
    service = RecommendationService(provider=mock_provider)

    resp = service.recommend_for_anomaly(sample_anomaly_id, refresh=True)
    for r in resp.recommendations:
        assert "promo event" not in r.supporting_evidence.lower()
        assert "flash sale" not in r.supporting_evidence.lower()


def test_every_returned_recommendation_belongs_to_eligible_types(sample_anomaly_id):
    """Deterministic Invariant: Every returned recommendation type must belong to eligible_recommendation_types."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = service.investigation_service.investigate_anomaly(sample_anomaly_id)
    eligible = service.determine_eligibility(inv)

    resp = service.recommend_for_anomaly(sample_anomaly_id, refresh=True)
    assert len(resp.recommendations) > 0
    for r in resp.recommendations:
        assert r.recommendation_type in eligible


def test_unsupported_business_consequence_reframe():
    """Unmeasured business consequences like customer retention impairment or margin erosion are reframed into validation questions."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation()

    raw_rec = AIRecommendation(
        recommendation_type="pricing_review",
        title="Review Pricing Structure",
        action="Review unit pricing realization across catalogs.",
        reason="Unit price rose above baseline.",
        supporting_evidence="Price deviation was +11%.",
        expected_objective="Validate pricing stability.",
        confidence="medium",
        priority="high",
        risk_level="medium",
        tradeoffs=[
            "Over-relying on price realization without volume growth can impair customer retention.",
            "Aggressive discounting erodes gross margin.",
        ],
        validation_required=[
            "Evaluate customer acquisition cost impact before adjusting catalog list prices.",
        ],
    )

    validated = service._post_validate_recommendations(
        raw_recs=[raw_rec],
        anomaly_id=inv.anomaly_id,
        investigation=inv,
        eligible_types=["pricing_review"],
        driver_confidence_map={},
    )
    assert len(validated) == 1
    rec = validated[0]
    # Check that assertive consequence claims were reframed
    assert "impair customer retention" not in rec.tradeoffs[0].lower()
    assert "validate whether sustained price realization is consistent with customer-retention objectives" in rec.tradeoffs[0].lower()
    assert "erodes gross margin" not in rec.tradeoffs[1].lower()
    assert "validate the margin implications of discount intensity before changing promotional strategy" in rec.tradeoffs[1].lower()
    assert "customer acquisition cost impact" not in rec.validation_required[0].lower()


# =========================================================================
# 5. Recommendation Evidence Alignment & Dimension Grounding Tests
# =========================================================================


def test_pricing_recommendation_uses_price_evidence(sample_anomaly_id):
    """pricing_review must cite price driver evidence (observed/ref price, shift %) and NOT category evidence."""
    service = RecommendationService(provider=MockLLMProvider(should_fail=True))
    resp = service.recommend_for_anomaly(sample_anomaly_id, refresh=True)

    pricing_rec = next((r for r in resp.recommendations if r.recommendation_type == "pricing_review"), None)
    assert pricing_rec is not None, "pricing_review recommendation should be generated for sample anomaly"

    ev = pricing_rec.supporting_evidence.lower()
    # Must contain price metrics / driver terms
    assert "price" in ev or "realization" in ev or "3,537" in ev or "11.1%" in ev
    # Must NOT use category evidence as primary evidence
    assert "clothing actual was" not in ev
    assert "total category variance" not in ev
    # Invariant: contains INR symbol if formatted with currency
    assert "$" not in pricing_rec.supporting_evidence


def test_product_recommendation_uses_product_evidence(sample_anomaly_id):
    """product_review must cite product driver evidence (SKU/product observed/ref values, contribution) and NOT category evidence."""
    service = RecommendationService(provider=MockLLMProvider(should_fail=True))
    resp = service.recommend_for_anomaly(sample_anomaly_id, refresh=True)

    prod_rec = next((r for r in resp.recommendations if r.recommendation_type == "product_review"), None)
    assert prod_rec is not None, "product_review recommendation should be generated for sample anomaly"

    ev = prod_rec.supporting_evidence.lower()
    # Must cite product driver name or SKU terms
    assert "product" in ev or "sku" in ev or "id card holder" in ev or "shorts" in ev
    # Must NOT be pure category evidence
    assert "clothing actual was" not in ev
    assert "total category variance" not in ev
    assert "$" not in prod_rec.supporting_evidence


def test_category_recommendation_uses_category_evidence(sample_anomaly_id):
    """category_review must cite category driver evidence."""
    service = RecommendationService(provider=MockLLMProvider(should_fail=True))
    resp = service.recommend_for_anomaly(sample_anomaly_id, refresh=True)

    cat_rec = next((r for r in resp.recommendations if r.recommendation_type == "category_review"), None)
    assert cat_rec is not None, "category_review recommendation should be generated for sample anomaly"

    ev = cat_rec.supporting_evidence.lower()
    assert "clothing" in ev or "category" in ev
    assert "3,177,900" in ev or "20.3%" in ev or "20.9%" in ev
    assert "$" not in cat_rec.supporting_evidence


def test_promotion_recommendation_uses_promotion_evidence():
    """promotion_review must cite promotion evidence only, and only when promotion is active."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation(
        drivers=[
            InvestigationDriver(
                driver_type="promotion",
                driver_name="Promotion Activity",
                direction="positive",
                observed_value=1.0,
                reference_value=0.0,
                difference=1.0,
                difference_percent=100.0,
                contribution_score=15.0,
                evidence="Active flash sale promotional campaign drove high volume.",
                confidence="high",
                is_event_related=True,
            ),
            InvestigationDriver(
                driver_type="category",
                driver_name="Category: Electronics",
                direction="positive",
                observed_value=200000.0,
                reference_value=150000.0,
                contribution_score=25.0,
                evidence="Electronics sales rose 33.3%.",
                confidence="high",
            ),
        ]
    )
    eligible = service.determine_eligibility(inv)
    assert "promotion_review" in eligible

    fallback_resp = service.build_deterministic_fallback(
        investigation=inv,
        explanation=None,
        ai_reasoning=None,
        eligible_types=eligible,
    )
    promo_rec = next((r for r in fallback_resp.recommendations if r.recommendation_type == "promotion_review"), None)
    assert promo_rec is not None
    assert "promot" in promo_rec.supporting_evidence.lower() or "flash sale" in promo_rec.supporting_evidence.lower()
    assert "electronics" not in promo_rec.supporting_evidence.lower()


def test_recommendation_evidence_dimension_mismatch_rejected(sample_anomaly_id):
    """If a recommendation's supporting evidence dimension does not match its recommendation type, it is rejected."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = service.investigation_service.investigate_anomaly(sample_anomaly_id)

    # pricing_review recommendation with category evidence (no price terms)
    mismatched_rec = AIRecommendation(
        recommendation_type="pricing_review",
        title="Review Realized Unit Price",
        action="Review realized average unit prices.",
        reason="Price realization shifted.",
        supporting_evidence="Clothing actual was 3,177,900 ₹ vs 28-day baseline of 2,642,271 ₹ (+20.3%).",
        expected_objective="Validate pricing stability.",
        confidence="high",
        priority="high",
        risk_level="low",
        tradeoffs=[],
        validation_required=[],
    )

    validated = service._post_validate_recommendations(
        raw_recs=[mismatched_rec],
        anomaly_id=inv.anomaly_id,
        investigation=inv,
        eligible_types=["pricing_review"],
        driver_confidence_map={},
    )
    assert len(validated) == 0, "Recommendation with dimension mismatch must be rejected"


def test_unsupported_premium_tier_language_rejected():
    """Unsupported premium-tier assertions are rejected, and tier wording is replaced with evidence-compatible language."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation()

    # Case 1: AI recommendation asserting premium product tier in reason/evidence -> rejected!
    assertive_rec = AIRecommendation(
        recommendation_type="category_review",
        title="Review Premium Category",
        action="Review category assortment.",
        reason="Sales surge was driven by premium product tier purchases.",
        supporting_evidence="Luxury products accounted for major customer spend.",
        expected_objective="Support category growth.",
        confidence="high",
        priority="high",
        risk_level="low",
        tradeoffs=[],
        validation_required=[],
    )

    validated_rejected = service._post_validate_recommendations(
        raw_recs=[assertive_rec],
        anomaly_id=inv.anomaly_id,
        investigation=inv,
        eligible_types=["category_review"],
        driver_confidence_map={},
    )
    assert len(validated_rejected) == 0, "Factual assertions of premium tiers or luxury goods must be rejected"

    # Case 2: Trade-offs or validations with premium SKU mix are sanitized to evidence-compatible phrasing
    tradeoff_rec = AIRecommendation(
        recommendation_type="category_review",
        title="Review Category Dynamics",
        action="Review category mix across catalogs.",
        reason="Category sales exceeded expected baseline.",
        supporting_evidence="Category: Electronics sales rose 33.3%.",
        expected_objective="Validate category baseline.",
        confidence="high",
        priority="high",
        risk_level="low",
        tradeoffs=["Consider whether premium SKU mix shifted demand."],
        validation_required=["Evaluate premium product tier performance."],
    )

    validated_sanitized = service._post_validate_recommendations(
        raw_recs=[tradeoff_rec],
        anomaly_id=inv.anomaly_id,
        investigation=inv,
        eligible_types=["category_review"],
        driver_confidence_map={},
    )
    assert len(validated_sanitized) == 1
    rec = validated_sanitized[0]
    assert "premium sku mix" not in rec.tradeoffs[0].lower()
    assert "product-mix differences among recorded skus" in rec.tradeoffs[0].lower()
    assert "premium product tier" not in rec.validation_required[0].lower()
    assert "Validate product-mix differences among recorded SKUs." in rec.validation_required[0]


def test_unsupported_warehouse_backorder_assertions_rejected():
    """Unsupported operational variables (warehouse capacity, backorders, fulfillment constraints) asserted as facts are rejected; allowed only as validation questions."""
    service = RecommendationService(provider=MockLLMProvider())
    inv = _create_mock_investigation()

    # Case 1: Factual assertion of backorders and warehouse capacity -> rejected!
    assertive_rec = AIRecommendation(
        recommendation_type="inventory_review",
        title="Address Fulfillment Constraints",
        action="Clear warehouse capacity bottleneck and backorders.",
        reason="Fulfillment constraints and inventory shortage caused customer delivery delays.",
        supporting_evidence="Observed high backorders in regional distribution centers.",
        expected_objective="Resolve backorders.",
        confidence="high",
        priority="high",
        risk_level="high",
        tradeoffs=[],
        validation_required=[],
    )

    validated_rejected = service._post_validate_recommendations(
        raw_recs=[assertive_rec],
        anomaly_id=inv.anomaly_id,
        investigation=inv,
        eligible_types=["inventory_review"],
        driver_confidence_map={},
    )
    assert len(validated_rejected) == 0, "Factual assertions of backorders/warehouse capacity must be rejected"

    # Case 2: Framed strictly as human validation questions -> accepted!
    question_rec = AIRecommendation(
        recommendation_type="inventory_review",
        title="Review Inventory Buffer",
        action="Review inventory allocation across recorded categories.",
        reason="Observed demand exceeded historical baseline.",
        supporting_evidence="Category: Electronics sales rose 33.3% relative to reference.",
        expected_objective="Prevent inventory stockouts while evaluating lead times.",
        confidence="high",
        priority="high",
        risk_level="medium",
        tradeoffs=["Validate holding capacity before ordering bulk stock."],
        validation_required=[
            "Does current warehouse capacity support any future inventory adjustment?",
            "Are there fulfillment constraints affecting the leading products?",
        ],
    )

    validated_accepted = service._post_validate_recommendations(
        raw_recs=[question_rec],
        anomaly_id=inv.anomaly_id,
        investigation=inv,
        eligible_types=["inventory_review"],
        driver_confidence_map={},
    )
    assert len(validated_accepted) == 1
    assert "Does current warehouse capacity support any future inventory adjustment?" in validated_accepted[0].validation_required
    assert "Are there fulfillment constraints affecting the leading products?" in validated_accepted[0].validation_required
