from __future__ import annotations

import logging
import re
import time
from typing import Any

from backend.app.llm.gemini_provider import GeminiProvider
from backend.app.llm.prompts import (
    build_recommendation_evidence,
)
from backend.app.llm.provider import (
    LLMConfigurationError,
    LLMProvider,
    LLMProviderError,
)
from backend.app.schemas.ai_reasoning import AIReasoningResponse
from backend.app.schemas.explanations import ExecutiveExplanation
from backend.app.schemas.investigations import InvestigationDriver, InvestigationResponse
from backend.app.schemas.recommendation_ai import (
    AIRecommendation,
    AIRecommendationResponse,
)
from backend.app.schemas.recommendations import (
    ConfidenceLevel,
    PriorityLevel,
    Recommendation,
    RecommendationResponse,
    RecommendationStatus,
    RecommendationType,
    RiskLevel,
)
from backend.app.services.ai_reasoning_service import AIReasoningService
from backend.app.services.anomaly_service import AnomalyDetectionService
from backend.app.services.explanation_service import ExplanationService
from backend.app.services.investigation_service import InvestigationService

logger = logging.getLogger(__name__)

ELIGIBLE_RECOMMENDATION_TYPES: set[str] = {
    "inventory_review",
    "pricing_review",
    "promotion_review",
    "category_review",
    "product_review",
    "demand_monitoring",
    "forecast_review",
    "data_validation",
}

# Regex patterns to catch and sanitize unauthorized numeric directives
NUMERIC_INVENTORY_DIRECTIVE = re.compile(
    r"\b(?:increase|decrease|boost|reduce|order|procure)\s+(?:inventory|stock|units?)\s+(?:by\s+)?(\d+(?:\.\d+)?\s*%|\d+[\d,]*\s*units?)",
    re.IGNORECASE,
)
NUMERIC_PRICE_DIRECTIVE = re.compile(
    r"\b(?:cut|discount|slash|reduce|raise|increase)\s+(?:prices?|pricing)\s+(?:by\s+)?(\d+(?:\.\d+)?\s*%|[₹$]\s*[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)
GUARANTEED_REVENUE_DIRECTIVE = re.compile(
    r"\b(?:will|guaranteed\s+to|definitely)\s+(?:increase|boost|deliver|grow)\s+(?:revenue|sales|profit|margins?)\s+(?:by\s+)?(\d+(?:\.\d+)?\s*%|[₹$]\s*[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)

PROHIBITED_CAUSAL_WORDS = [
    (re.compile(r"\bdefinitely caused by\b", re.I), "strongly associated with"),
    (re.compile(r"\bcaused by\b", re.I), "associated with"),
    (re.compile(r"\bcaused the\b", re.I), "coincided with the"),
    (re.compile(r"\bproves that\b", re.I), "indicates that"),
    (re.compile(r"\bguarantees?\b", re.I), "aims to support"),
]

PROMOTION_CLAIM_PATTERN = re.compile(
    r"\b(?:promot(?:ion|ional|ions?)|marketing\s+campaign|discount\s+campaign|promo\s+event|flash\s+sale)\b",
    re.IGNORECASE,
)

UNSUPPORTED_BUSINESS_CONSEQUENCES = [
    (
        re.compile(
            r"\b(?:over-relying\s+on\s+price\s+realization\s+without\s+volume\s+growth\s+can\s+impair\s+customer\s+retention|can\s+impair\s+customer\s+retention|impair(?:s|ing)?\s+customer\s+retention|impair\s+retention)\b",
            re.I,
        ),
        "validate whether sustained price realization is consistent with customer-retention objectives",
    ),
    (
        re.compile(
            r"\b(?:aggressive\s+discounting\s+erodes\s+gross\s+margin|erode(?:s|ing)?\s+(?:gross\s+)?margins?|gross\s+margin\s+erosion|margin\s+erosion)\b",
            re.I,
        ),
        "validate the margin implications of discount intensity before changing promotional strategy",
    ),
    (
        re.compile(r"\b(?:customer\s+acquisition\s+cost(?:\s+impact)?|cac\s+impact)\b", re.I),
        "customer acquisition cost efficiency",
    ),
    (
        re.compile(r"\bpost-event\s+retention(?:\s+effects)?\b", re.I),
        "post-event customer engagement patterns",
    ),
    (
        re.compile(r"\b(?:profitability\s+improvement|improves?\s+profitability)\b", re.I),
        "commercial margin targets",
    ),
]

PREMIUM_TIER_ASSERTION_PATTERN = re.compile(
    r"\b(?:"
    r"premium\s+(?:product\s+)?(?:lines?|tiers?|purchasing|items?|goods?|catalog|customers?|sku\s+mix)|"
    r"premium\s+tiers?|premium\s+products?|luxury(?:\s+products?|\s+goods?|\s+items?|\s+lines?|\s+segments?|\s+customers?)?|"
    r"higher[- ]tier(?:\s+products?|\s+lines?|\s+items?|\s+goods?)?|high[- ]end\s+products?"
    r")\b",
    re.IGNORECASE,
)

PREMIUM_TIER_REPLACEMENT_PATTERN = re.compile(
    r"\b(?:"
    r"premium\s+sku\s+mix|"
    r"premium\s+product\s+tiers?|"
    r"luxury\s+products?|"
    r"premium\s+product\s+lines?|"
    r"higher[- ]tier\s+products?"
    r")\b",
    re.IGNORECASE,
)

UNSUPPORTED_OPERATIONAL_FACT_PATTERN = re.compile(
    r"\b(?:"
    r"warehouse\s+capacity|"
    r"backorders?|"
    r"fulfillment\s+constraints?|"
    r"inventory\s+shortages?|"
    r"stockout(?:s|\b)|"
    r"out\s+of\s+stock"
    r")\b",
    re.IGNORECASE,
)


class RecommendationService:
    """
    Prescriptive Action Recommendation Engine (Phase 6.5).
    Combines Phase 6.2 investigation, Phase 6.3 explanation, Phase 6.4 AI reasoning,
    and deterministic eligibility rules to formulate advisory recommendations.
    Never executes autonomous actions; human approval is mandatory.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        investigation_service: InvestigationService | None = None,
        explanation_service: ExplanationService | None = None,
        ai_reasoning_service: AIReasoningService | None = None,
        anomaly_service: AnomalyDetectionService | None = None,
    ) -> None:
        self.provider = provider or GeminiProvider()
        self.investigation_service = investigation_service or InvestigationService()
        self.explanation_service = explanation_service or ExplanationService(
            investigation_service=self.investigation_service
        )
        self.ai_reasoning_service = ai_reasoning_service or AIReasoningService(
            investigation_service=self.investigation_service,
            explanation_service=self.explanation_service,
        )
        self.anomaly_service = anomaly_service or AnomalyDetectionService()
        self._cache: dict[str, RecommendationResponse] = {}

    # =========================================================================
    # 1. Deterministic Eligibility Rules
    # =========================================================================

    def determine_eligibility(
        self,
        investigation: InvestigationResponse,
        explanation: ExecutiveExplanation | None = None,
        ai_reasoning: AIReasoningResponse | None = None,
    ) -> list[RecommendationType]:
        """
        Evaluates authoritative empirical evidence against deterministic policy rules
        to establish which recommendation types are eligible.
        """
        eligible: list[RecommendationType] = []

        drivers = investigation.drivers
        driver_types = {d.driver_type for d in drivers}
        direction = (investigation.direction or "").lower()
        dev_pct = abs(investigation.deviation_percent or 0.0)
        severity = (investigation.severity or "medium").lower()

        # Check event context: strictly require active promotional campaign (observed_value > 0)
        promo_driver = next((d for d in drivers if d.driver_type == "promotion"), None)
        promotion_active = bool(
            promo_driver and promo_driver.observed_value is not None and promo_driver.observed_value > 0
        )
        if not promotion_active and promo_driver is None and explanation:
            for s in explanation.sections:
                if s.section_type == "event_context":
                    content_lower = s.content.lower()
                    if "active promotional campaign was present" in content_lower or "active promotion was present" in content_lower:
                        promotion_active = True
                        break

        # 1. Inventory Review:
        # Eligible on positive volume/revenue spike where product or category drivers exist
        has_catalog_evidence = any(d.driver_type in ("category", "product") for d in drivers)
        if (direction == "spike" or investigation.deviation > 0) and has_catalog_evidence:
            eligible.append("inventory_review")

        # 2. Pricing Review:
        # Eligible when unit price realization shifted materially
        if "price" in driver_types or any(
            "price" in d.driver_name.lower() or d.driver_type == "price" for d in drivers
        ):
            eligible.append("pricing_review")

        # 3. Promotion Review:
        # Eligible ONLY when promotional campaign is active (promotion_active is True)
        if promotion_active:
            eligible.append("promotion_review")

        # 4. Category Review:
        # Eligible when a category contributes materially (> 5.0% contribution)
        category_drivers = [d for d in drivers if d.driver_type == "category"]
        if any((d.contribution_score or 0.0) > 5.0 for d in category_drivers):
            eligible.append("category_review")

        # 5. Product Review:
        # Eligible when product-level attribution is present
        if "product" in driver_types:
            eligible.append("product_review")

        # 6. Demand Monitoring:
        # Eligible when the anomaly may represent temporary shock, trend is unstable,
        # or evidence has moderate/low confidence
        low_or_med_confidence = any(d.confidence in ("low", "medium") for d in drivers)
        if low_or_med_confidence or direction == "drop" or "recent_trend" in driver_types:
            eligible.append("demand_monitoring")

        # 7. Forecast Review:
        # Eligible on high/critical magnitude or persistent drift
        if severity in ("high", "critical") or dev_pct >= 10.0 or "baseline_drift" in driver_types or "recent_trend" in driver_types:
            eligible.append("forecast_review")

        # 8. Data Validation:
        # Eligible if confidence is low, abnormal scores exist, or conflicting signals noted
        if any(d.confidence == "low" for d in drivers) or (investigation.anomaly_score or 0.0) >= 4.0:
            eligible.append("data_validation")

        return eligible

    # =========================================================================
    # 2. Deterministic Prioritization & Top-N Capping
    # =========================================================================

    def prioritize_recommendations(
        self,
        candidate_recs: list[Recommendation],
        investigation: InvestigationResponse,
    ) -> list[Recommendation]:
        """
        Applies a deterministic scoring formula based on anomaly severity, driver contribution,
        evidence confidence, and operational risk. Caps at maximum 3 primary recommendations.
        """
        sev_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        conf_map = {"high": 3, "medium": 2, "low": 1}
        risk_map = {"low": 1, "medium": 2, "high": 3}

        sev_score = sev_map.get((investigation.severity or "medium").lower(), 2)
        drivers_by_type = {d.driver_type: d for d in investigation.drivers}

        def score_rec(rec: Recommendation) -> float:
            c_score = conf_map.get(rec.confidence, 2)
            r_score = risk_map.get(rec.risk_level, 1)

            # Driver contribution alignment
            driver_contrib = 0.0
            if rec.recommendation_type in drivers_by_type:
                driver_contrib = drivers_by_type[rec.recommendation_type].contribution_score or 0.0
            elif rec.recommendation_type == "pricing_review" and "price" in drivers_by_type:
                driver_contrib = drivers_by_type["price"].contribution_score or 0.0
            elif rec.recommendation_type == "category_review":
                cat_drivers = [d for d in investigation.drivers if d.driver_type == "category"]
                if cat_drivers:
                    driver_contrib = max((d.contribution_score or 0.0) for d in cat_drivers)

            # Formula: severity (weight 2) + contribution + confidence - risk penalty
            return (sev_score * 2.0) + (driver_contrib * 0.1) + (c_score * 1.5) - (r_score * 0.5)

        sorted_recs = sorted(candidate_recs, key=score_rec, reverse=True)
        return sorted_recs[:3]

    # =========================================================================
    # 3. Deterministic Fallback Engine
    # =========================================================================

    def extract_evidence_for_recommendation_type(
        self,
        rec_type: RecommendationType,
        investigation: InvestigationResponse,
    ) -> tuple[str, ConfidenceLevel]:
        """
        Extracts recommendation-specific supporting evidence and confidence
        strictly aligned to the recommendation type's dimension.
        """
        drivers = investigation.drivers

        if rec_type == "pricing_review":
            price_driver = next(
                (d for d in drivers if d.driver_type == "price" or "price" in d.driver_name.lower()),
                None,
            )
            if price_driver:
                ev = price_driver.evidence.replace("$", "₹")
                return ev, price_driver.confidence
            return (
                f"Average unit price shifted by {abs(investigation.deviation_percent):.1f}% relative to reference levels.",
                "medium",
            )

        elif rec_type == "product_review":
            prod_drivers = [d for d in drivers if d.driver_type == "product"]
            if prod_drivers:
                best_prod = max(prod_drivers, key=lambda d: d.contribution_score or 0.0)
                if not best_prod.evidence.lower().startswith("product"):
                    ev = f"{best_prod.driver_name} — {best_prod.evidence.replace('$', '₹')}"
                else:
                    ev = best_prod.evidence.replace("$", "₹")
                return ev, best_prod.confidence
            return (
                "Individual product-level catalog deviations contributed to overall variance.",
                "medium",
            )

        elif rec_type == "category_review":
            cat_drivers = [d for d in drivers if d.driver_type == "category"]
            if cat_drivers:
                best_cat = max(cat_drivers, key=lambda d: d.contribution_score or 0.0)
                ev = best_cat.evidence.replace("$", "₹")
                return ev, best_cat.confidence
            return (
                "Recorded category-level volume shifts contributed to overall variance.",
                "medium",
            )

        elif rec_type == "promotion_review":
            promo_driver = next((d for d in drivers if d.driver_type == "promotion"), None)
            if promo_driver:
                ev = promo_driver.evidence.replace("$", "₹")
                return ev, promo_driver.confidence
            return ("Promotional campaign activity was recorded during the anomaly window.", "medium")

        elif rec_type == "forecast_review":
            drift_driver = next(
                (d for d in drivers if d.driver_type in ("baseline_drift", "recent_trend")),
                None,
            )
            drift_suffix = f" {drift_driver.evidence.replace('$', '₹')}" if drift_driver else ""
            ev = (
                f"Measured deviation of ₹{investigation.deviation:,.2f} ({investigation.deviation_percent:+.1f}% vs baseline) "
                f"with anomaly score {investigation.anomaly_score:.2f} and severity '{investigation.severity}'.{drift_suffix}"
            )
            conf = drift_driver.confidence if drift_driver else "medium"
            return ev.strip(), conf

        elif rec_type == "demand_monitoring":
            trend_driver = next(
                (d for d in drivers if d.driver_type in ("recent_trend", "baseline_drift")),
                None,
            )
            if trend_driver:
                ev = (
                    f"Empirical trend signals indicate transitional volatility ({trend_driver.evidence.replace('$', '₹')}). "
                    f"Deviation: ₹{investigation.deviation:,.2f} ({investigation.deviation_percent:+.1f}% vs baseline)."
                )
                return ev, trend_driver.confidence
            return (
                f"Measured {investigation.direction} deviation of ₹{investigation.deviation:,.2f} ({investigation.deviation_percent:+.1f}% vs baseline) warrants observational monitoring.",
                "low",
            )

        elif rec_type == "inventory_review":
            cat_driver = next((d for d in drivers if d.driver_type in ("category", "product")), None)
            cat_ev = f" {cat_driver.evidence.replace('$', '₹')}" if cat_driver else ""
            ev = f"Observed positive demand deviation of ₹{investigation.deviation:,.2f} ({investigation.deviation_percent:+.1f}%).{cat_ev}"
            conf = cat_driver.confidence if cat_driver else "medium"
            return ev.strip(), conf

        elif rec_type == "data_validation":
            low_conf = next((d for d in drivers if d.confidence == "low"), None)
            if low_conf:
                ev = f"Anomaly score of {investigation.anomaly_score:.2f} (severity: {investigation.severity}) with data confidence concern in {low_conf.driver_name}."
            else:
                ev = f"Anomaly score of {investigation.anomaly_score:.2f} (severity: {investigation.severity}) warrants transaction-level data consistency verification."
            return ev, "medium"

        return (
            f"Measured deviation of ₹{investigation.deviation:,.2f} ({investigation.deviation_percent:+.1f}% vs baseline).",
            "medium",
        )

    def build_deterministic_fallback(
        self,
        investigation: InvestigationResponse,
        explanation: ExecutiveExplanation | None,
        ai_reasoning: AIReasoningResponse | None,
        eligible_types: list[RecommendationType],
    ) -> RecommendationResponse:
        """
        Produces fully grounded, high-signal recommendations for eligible types
        without calling external LLM providers. Used when Gemini is unavailable.
        """
        anomaly_id = investigation.anomaly_id
        drivers = investigation.drivers
        top_driver = drivers[0] if drivers else None

        fallback_recs: list[Recommendation] = []

        # Strict Promotion Invariance: If promotion is inactive, ensure promotion_review is never eligible
        promo_driver = next((d for d in drivers if d.driver_type == "promotion"), None)
        promotion_active = bool(
            promo_driver and promo_driver.observed_value is not None and promo_driver.observed_value > 0
        )
        if not promotion_active and "promotion_review" in eligible_types:
            eligible_types = [t for t in eligible_types if t != "promotion_review"]

        template_map: dict[RecommendationType, dict[str, Any]] = {
            "inventory_review": {
                "title": "Review Inventory Allocation for Contributing Categories",
                "action": "Review inventory allocation and replenishment schedules across recorded categories showing demand spikes.",
                "reason": "Observed demand exceeded historical baseline, reflecting accelerated sales velocity.",
                "expected_objective": "Prevent inventory stockouts while avoiding premature over-ordering.",
                "tradeoffs": ["Validate whether accelerated replenishment aligns with working capital and holding capacity before placing bulk orders."],
                "validation_required": ["Does current warehouse capacity support any future inventory adjustment?"],
                "risk_level": "medium",
                "priority": "high",
            },
            "pricing_review": {
                "title": "Review Realized Unit Price and Product Mix Behavior",
                "action": "Review realized average unit prices and discount compliance across transaction logs.",
                "reason": "Average unit price realization shifted relative to historical reference levels.",
                "expected_objective": "Validate price realization consistency without assuming cross-price elasticity.",
                "tradeoffs": ["Validate whether sustained price realization is consistent with customer-retention objectives."],
                "validation_required": ["Audit transaction-level discount rates and mix shifts among high-velocity items."],
                "risk_level": "medium",
                "priority": "high",
            },
            "promotion_review": {
                "title": "Review Promotional Campaign Effectiveness and Demand Lift",
                "action": "Review promotional response metrics against non-promotional baseline periods.",
                "reason": "Promotional events were active on or adjacent to the anomaly trading date.",
                "expected_objective": "Assess incremental demand attribution versus standard calendar baseline.",
                "tradeoffs": ["Validate the margin implications of discount intensity before changing promotional strategy."],
                "validation_required": ["Verify discount redemption rates and channel-specific promotional tags."],
                "risk_level": "low",
                "priority": "medium",
            },
            "category_review": {
                "title": "Review Contributing Category Performance Drivers",
                "action": "Review sub-segment demand and catalog participation within the primary contributing category.",
                "reason": f"Category deviations accounted for significant share of overall variance.",
                "expected_objective": "Isolate whether the category movement was broad-based or concentrated.",
                "tradeoffs": ["Validate whether merchandising adjustments risk cannibalizing adjacent product categories."],
                "validation_required": ["Analyze SKU-level sales breakdown within the affected category."],
                "risk_level": "low",
                "priority": "medium",
            },
            "product_review": {
                "title": "Review Leading Product Velocity and Demand Signals",
                "action": "Review individual product velocity trends to determine if deviation is recurring.",
                "reason": "Specific product lines demonstrated substantial deviations from expected levels.",
                "expected_objective": "Determine whether individual SKU demand represents an isolated spike or sustained trend.",
                "tradeoffs": ["Validate whether warehouse holding capacity supports individual SKU safety stock reallocation."],
                "validation_required": ["Are there fulfillment constraints affecting the leading products?"],
                "risk_level": "low",
                "priority": "medium",
            },
            "demand_monitoring": {
                "title": "Monitor Short-Term Demand Trajectory Over Next Cycle",
                "action": "Monitor daily demand over the next 7-day forecast cycle before initiating operational changes.",
                "reason": "Empirical signals indicate possible temporary shock or transitional volatility.",
                "expected_objective": "Avoid premature operational commitments until trend stability is established.",
                "tradeoffs": ["Validate current inventory buffer before maintaining an observational holding period."],
                "validation_required": ["Track daily velocity against 7-day and 30-day moving averages."],
                "risk_level": "low",
                "priority": "low",
            },
            "forecast_review": {
                "title": "Review Upcoming Horizon Production Forecasts for Regime Shift",
                "action": "Review horizon-specific model forecasts (7d, 30d, 90d) for persistent baseline drift.",
                "reason": f"Anomaly magnitude of {abs(investigation.deviation_percent):.1f}% exceeds standard volatility thresholds.",
                "expected_objective": "Ensure production forecasting models adapt appropriately to structural demand shifts.",
                "tradeoffs": ["Validate whether demand spikes represent persistent regime shifts before updating production forecasting baselines."],
                "validation_required": ["Compare rolling origin MAPE against historical holdout performance."],
                "risk_level": "medium",
                "priority": "high",
            },
            "data_validation": {
                "title": "Validate Underlying Transaction Data and Attribution Records",
                "action": "Validate source transaction logs and event calendars to rule out reporting anomalies.",
                "reason": "Statistical anomaly score warrants data consistency verification prior to business action.",
                "expected_objective": "Confirm data integrity and eliminate pipeline logging artifacts.",
                "tradeoffs": ["Validate audit priority against operational deadlines before initiating in-depth transaction reconciliation."],
                "validation_required": ["Check for duplicate batch uploads, cancelled orders, or POS sync delays."],
                "risk_level": "low",
                "priority": "critical" if investigation.severity == "critical" else "medium",
            },
        }

        rec_idx = 1
        for rec_type in eligible_types:
            tmpl = template_map.get(rec_type)
            if not tmpl:
                continue

            supp_ev, conf = self.extract_evidence_for_recommendation_type(rec_type, investigation)

            fallback_recs.append(
                Recommendation(
                    id=f"rec-{anomaly_id}-{rec_idx}",
                    anomaly_id=anomaly_id,
                    recommendation_type=rec_type,
                    title=tmpl["title"],
                    action=tmpl["action"],
                    reason=tmpl["reason"],
                    supporting_evidence=supp_ev.replace("$", "₹"),
                    expected_objective=tmpl["expected_objective"],
                    confidence=conf,
                    priority=tmpl["priority"],
                    risk_level=tmpl["risk_level"],
                    tradeoffs=tmpl["tradeoffs"],
                    validation_required=tmpl["validation_required"],
                    is_actionable=True,
                    human_approval_required=True,
                )
            )
            rec_idx += 1

        prioritized = self.prioritize_recommendations(fallback_recs, investigation)
        # Deterministic Invariant: Every recommendation MUST belong to eligible_types
        prioritized = [r for r in prioritized if r.recommendation_type in eligible_types]

        return RecommendationResponse(
            anomaly_id=anomaly_id,
            recommendation_status="actionable" if prioritized else "no_actionable_recommendation",
            summary=(
                f"Generated {len(prioritized)} evidence-grounded advisory recommendations based on deterministic "
                f"policy rules for {investigation.metric} ({investigation.direction}). Human review is required."
            ),
            recommendations=prioritized,
            limitations=[
                "Deterministic Fallback: Formulated via deterministic policy rules because AI recommendation synthesis was unavailable.",
                "Advisory Guardrail: Recommendations are decision-support suggestions and do not constitute automated operational directives.",
                "Observational Limitations: Recommendations are based on statistical associations and require qualified human review.",
            ],
            human_approval_required=True,
            source="deterministic_fallback",
            cached=False,
        )

    # =========================================================================
    # 4. Post-Validation and Guardrails
    # =========================================================================

    def _is_evidence_dimension_aligned(
        self,
        rec_type: str,
        supporting_evidence: str,
        investigation: InvestigationResponse,
    ) -> bool:
        """
        Validates that supporting evidence matches the specific dimension of the recommendation type,
        and that the underlying investigation actually contains driver evidence for that dimension.
        """
        ev_lower = supporting_evidence.lower()
        drivers = investigation.drivers

        if rec_type == "pricing_review":
            price_terms = ("price", "pricing", "realization", "unit price", "₹/unit", "$/unit")
            price_driver_names = [
                d.driver_name.lower() for d in drivers if d.driver_type == "price" or "price" in d.driver_name.lower()
            ]
            return any(t in ev_lower for t in price_terms) or any(name in ev_lower for name in price_driver_names)

        elif rec_type == "product_review":
            prod_terms = ("product", "sku", "item", "velocity", "catalog deviation")
            prod_driver_names = [
                d.driver_name.lower() for d in drivers if d.driver_type == "product" or "product" in d.driver_name.lower()
            ]
            short_names = [
                re.sub(r"^product:\s*", "", d.driver_name.lower()).split("(")[0].strip()
                for d in drivers if d.driver_type == "product"
            ]
            return (
                any(t in ev_lower for t in prod_terms)
                or any(name in ev_lower for name in prod_driver_names)
                or any(s in ev_lower for s in short_names if len(s) > 2)
            )

        elif rec_type == "category_review":
            cat_terms = ("category", "catalog", "segment", "assortment", "merchandise", "sales", "clothing")
            cat_driver_names = [
                d.driver_name.lower() for d in drivers if d.driver_type == "category" or "category" in d.driver_name.lower()
            ]
            short_cat_names = [
                re.sub(r"^category:\s*", "", d.driver_name.lower()).strip()
                for d in drivers if d.driver_type == "category"
            ]
            return (
                any(t in ev_lower for t in cat_terms)
                or any(name in ev_lower for name in cat_driver_names)
                or any(s in ev_lower for s in short_cat_names if len(s) > 2)
            )

        elif rec_type == "promotion_review":
            promo_driver = next((d for d in drivers if d.driver_type == "promotion"), None)
            promotion_active = bool(
                promo_driver and promo_driver.observed_value is not None and promo_driver.observed_value > 0
            )
            if not promotion_active:
                return False
            promo_terms = ("promot", "discount", "campaign", "lift", "voucher", "coupon", "flash sale")
            return any(t in ev_lower for t in promo_terms)

        elif rec_type == "forecast_review":
            forecast_terms = (
                "forecast", "baseline", "drift", "deviation", "severity", "score",
                "volatility", "magnitude", "regime", "z-score", "sales", "trend"
            )
            return any(t in ev_lower for t in forecast_terms)

        elif rec_type == "demand_monitoring":
            demand_terms = (
                "demand", "deviation", "trend", "drop", "drift", "volatility",
                "trajectory", "shock", "uncertainty", "monitoring", "7-day", "cycle", "baseline", "sales"
            )
            return any(t in ev_lower for t in demand_terms)

        elif rec_type == "inventory_review":
            inv_terms = (
                "inventory", "stock", "volume", "demand", "unit", "sales",
                "category", "product", "sku", "spike", "allocation", "replenishment", "deviation"
            )
            return any(t in ev_lower for t in inv_terms)

        elif rec_type == "data_validation":
            data_terms = (
                "score", "anomaly", "data", "log", "transaction", "confidence",
                "severity", "sample", "metric", "variance", "deviation"
            )
            return any(t in ev_lower for t in data_terms)

        return True

    def _post_validate_recommendations(
        self,
        raw_recs: list[AIRecommendation],
        anomaly_id: str,
        investigation: InvestigationResponse,
        eligible_types: list[RecommendationType],
        driver_confidence_map: dict[str, str],
    ) -> list[Recommendation]:
        """
        Validates AI-generated recommendations against policy rules:
        1. Reject un-eligible recommendation types.
        2. Sanitize numeric target hallucinations (order quantities, price cuts).
        3. Sanitize guaranteed outcome promises.
        4. Sanitize causal language.
        5. Cap confidence against underlying evidence confidence.
        6. Enforce human_approval_required = True.
        """
        validated: list[Recommendation] = []
        eligible_set = set(eligible_types)

        promo_driver = next((d for d in investigation.drivers if d.driver_type == "promotion"), None)
        promotion_active = bool(
            promo_driver and promo_driver.observed_value is not None and promo_driver.observed_value > 0
        )

        for idx, r in enumerate(raw_recs, start=1):
            r_type = r.recommendation_type.lower().strip()

            # 1. Eligibility verification
            if r_type not in eligible_set:
                logger.warning(
                    "Recommendation type '%s' is not in eligible types %s. Rejecting.",
                    r_type,
                    eligible_set,
                )
                continue

            # 1b. Strict promotion invariance: reject promotion_review if promotion is inactive
            if not promotion_active and (r_type == "promotion_review" or "promot" in r_type):
                logger.warning(
                    "Recommendation type '%s' rejected because promotion_active is False.",
                    r_type,
                )
                continue

            # 1c. Strict promotion claim rejection: if promotion is inactive, reject promotional claims in action, reason, or supporting evidence
            if not promotion_active:
                if (
                    PROMOTION_CLAIM_PATTERN.search(r.action)
                    or PROMOTION_CLAIM_PATTERN.search(r.reason)
                    or PROMOTION_CLAIM_PATTERN.search(r.supporting_evidence)
                ):
                    logger.warning(
                        "Recommendation '%s' rejected because it asserts promotional activity while promotion_active is False.",
                        r.title,
                    )
                    continue

            # 1d. Recommendation-Specific Evidence Dimension Alignment
            if not self._is_evidence_dimension_aligned(r_type, r.supporting_evidence, investigation):
                logger.warning(
                    "Recommendation '%s' (type: %s) rejected due to evidence dimension mismatch. Supporting evidence: '%s'",
                    r.title,
                    r_type,
                    r.supporting_evidence,
                )
                continue

            # 1e. Reject unsupported premium-tier claims in reason, supporting evidence, or action
            if (
                PREMIUM_TIER_ASSERTION_PATTERN.search(r.reason)
                or PREMIUM_TIER_ASSERTION_PATTERN.search(r.supporting_evidence)
                or PREMIUM_TIER_ASSERTION_PATTERN.search(r.action)
            ):
                logger.warning(
                    "Recommendation '%s' rejected due to unsupported premium/luxury tier claims.",
                    r.title,
                )
                continue

            # 1f. Reject unsupported operational fact assertions in reason, supporting evidence, or action
            if (
                UNSUPPORTED_OPERATIONAL_FACT_PATTERN.search(r.reason)
                or UNSUPPORTED_OPERATIONAL_FACT_PATTERN.search(r.supporting_evidence)
                or UNSUPPORTED_OPERATIONAL_FACT_PATTERN.search(r.action)
            ):
                logger.warning(
                    "Recommendation '%s' rejected because it asserts unmeasured operational variables as facts.",
                    r.title,
                )
                continue

            # 2. Sanitize numeric target hallucinations
            action_text = r.action
            action_text = NUMERIC_INVENTORY_DIRECTIVE.sub(
                "Review inventory allocation and replenishment schedules based on observed demand",
                action_text,
            )
            action_text = NUMERIC_PRICE_DIRECTIVE.sub(
                "Review realized unit prices and discount compliance against baseline",
                action_text,
            )
            action_text = GUARANTEED_REVENUE_DIRECTIVE.sub(
                "aims to validate demand trajectory and mitigate revenue volatility",
                action_text,
            )

            # Ensure non-autonomous action prefix
            if action_text.lower().startswith(("increase", "decrease", "cut", "order", "buy", "sell")):
                action_text = f"Consider {action_text[0].lower() + action_text[1:]}"

            # 3. Sanitize guaranteed revenue/profit claims in reason and objective
            reason_text = r.reason
            objective_text = r.expected_objective
            reason_text = GUARANTEED_REVENUE_DIRECTIVE.sub(
                "aims to mitigate demand uncertainty", reason_text
            )
            objective_text = GUARANTEED_REVENUE_DIRECTIVE.sub(
                "mitigate revenue volatility and validate baseline alignment", objective_text
            )

            # 4. Sanitize prohibited causal words
            for pat, repl in PROHIBITED_CAUSAL_WORDS:
                action_text = pat.sub(repl, action_text)
                reason_text = pat.sub(repl, reason_text)
                objective_text = pat.sub(repl, objective_text)

            # 4b. Sanitize unsupported business consequence claims
            for pat, repl in UNSUPPORTED_BUSINESS_CONSEQUENCES:
                action_text = pat.sub(repl, action_text)
                reason_text = pat.sub(repl, reason_text)
                objective_text = pat.sub(repl, objective_text)

            # 5. Cap confidence against driver confidence map
            conf = r.confidence.lower()
            if conf not in ("high", "medium", "low"):
                conf = "medium"

            # Check if any driver is cited and cap confidence
            for d_name, d_conf in driver_confidence_map.items():
                if d_name.lower() in r.supporting_evidence.lower():
                    conf_rank = {"low": 1, "medium": 2, "high": 3}
                    max_rank = conf_rank.get(d_conf.lower(), 2)
                    if conf_rank.get(conf, 2) > max_rank:
                        conf = d_conf.lower()
                    break

            # 6. Sanitize and reframe trade-offs and validation questions
            cleaned_tradeoffs: list[str] = []
            for t in r.tradeoffs:
                t_str = t.strip()
                for pat, repl in UNSUPPORTED_BUSINESS_CONSEQUENCES:
                    t_str = pat.sub(repl, t_str)
                for pat, repl in PROHIBITED_CAUSAL_WORDS:
                    t_str = pat.sub(repl, t_str)
                if PREMIUM_TIER_REPLACEMENT_PATTERN.search(t_str):
                    t_str = PREMIUM_TIER_REPLACEMENT_PATTERN.sub("product-mix differences among recorded SKUs", t_str)
                if t_str:
                    cleaned_tradeoffs.append(re.sub(r"\s+", " ", t_str))
            if not cleaned_tradeoffs:
                cleaned_tradeoffs = ["Validate operational adjustments against working capital and labor availability."]

            cleaned_validations: list[str] = []
            for v in r.validation_required:
                v_str = v.strip()
                for pat, repl in UNSUPPORTED_BUSINESS_CONSEQUENCES:
                    v_str = pat.sub(repl, v_str)
                for pat, repl in PROHIBITED_CAUSAL_WORDS:
                    v_str = pat.sub(repl, v_str)
                if PREMIUM_TIER_REPLACEMENT_PATTERN.search(v_str):
                    if re.search(r"\b(?:evaluate|review|check|validate|assess)\b", v_str, re.I):
                        v_str = "Validate product-mix differences among recorded SKUs."
                    else:
                        v_str = PREMIUM_TIER_REPLACEMENT_PATTERN.sub("product-mix differences among recorded SKUs", v_str)
                # Ensure operational variables in validation_required are questions
                if re.search(r"\bwarehouse\s+capacity\b", v_str, re.I) and not v_str.endswith("?"):
                    v_str = "Does current warehouse capacity support any future inventory adjustment?"
                elif re.search(r"\b(?:backorders?|fulfillment\s+constraints?)\b", v_str, re.I) and not v_str.endswith("?"):
                    v_str = "Are there fulfillment constraints affecting the leading products?"
                if v_str:
                    cleaned_validations.append(re.sub(r"\s+", " ", v_str))
            if not cleaned_validations:
                cleaned_validations = ["Validate underlying transaction logs and category stock levels."]

            validated.append(
                Recommendation(
                    id=f"rec-{anomaly_id}-{idx}",
                    anomaly_id=anomaly_id,
                    recommendation_type=r_type,  # type: ignore
                    title=r.title.strip(),
                    action=action_text.strip(),
                    reason=reason_text.strip(),
                    supporting_evidence=r.supporting_evidence.strip().replace("$", "₹"),
                    expected_objective=objective_text.strip(),
                    confidence=conf,  # type: ignore
                    priority=r.priority,  # type: ignore
                    risk_level=r.risk_level,  # type: ignore
                    tradeoffs=cleaned_tradeoffs,
                    validation_required=cleaned_validations,
                    is_actionable=True,
                    human_approval_required=True,
                )
            )

        return self.prioritize_recommendations(validated, investigation)

    # =========================================================================
    # 5. Public Service Entry Point
    # =========================================================================

    def recommend_for_anomaly(
        self,
        anomaly_id: str,
        refresh: bool = False,
        fallback: bool = True,
        user_id: int | None = None,
        db: Any = None,
    ) -> RecommendationResponse:
        """
        Main recommendation pipeline:
        1. Check cache.
        2. Retrieve investigation, explanation, AI reasoning.
        3. Evaluate deterministic eligibility.
        4. Package compact evidence.
        5. Invoke LLM provider or execute deterministic fallback.
        6. Post-validate and cache.
        """
        cache_key = f"{user_id}:{anomaly_id}" if user_id else anomaly_id
        if not refresh and cache_key in self._cache:
            cached_resp = self._cache[cache_key].model_copy(deep=True)
            cached_resp.cached = True
            return cached_resp

        # 1. Retrieve prior pipeline phase artifacts
        investigation = self.investigation_service.investigate_anomaly(
            anomaly_id,
            user_id=user_id,
            db=db,
        )
        explanation = self.explanation_service.generate_explanation(investigation)

        ai_reasoning: AIReasoningResponse | None = None
        try:
            ai_reasoning = self.ai_reasoning_service.reason_about_anomaly(
                anomaly_id,
                refresh=refresh,
                user_id=user_id,
                db=db,
            )
        except Exception as exc:
            logger.info("AI reasoning optional context unavailable for recommendations: %s", exc)

        # 2. Check for zero deviation or insufficient evidence
        if abs(investigation.deviation) == 0 or not investigation.drivers:
            empty_resp = RecommendationResponse(
                anomaly_id=anomaly_id,
                recommendation_status="no_actionable_recommendation",
                summary="Insufficient empirical evidence or zero deviation detected. No actionable operational recommendations identified.",
                recommendations=[],
                limitations=["Insufficient signal depth to justify operational intervention."],
                human_approval_required=True,
                source="deterministic_fallback",
                cached=False,
            )
            self._cache[anomaly_id] = empty_resp
            return empty_resp

        # 3. Determine eligible recommendation categories
        eligible_types = self.determine_eligibility(investigation, explanation, ai_reasoning)
        if not eligible_types:
            no_rec_resp = RecommendationResponse(
                anomaly_id=anomaly_id,
                recommendation_status="no_actionable_recommendation",
                summary="Empirical evidence does not meet policy criteria for any active recommendation category.",
                recommendations=[],
                limitations=["No recommendation categories met the deterministic eligibility criteria."],
                human_approval_required=True,
                source="deterministic_fallback",
                cached=False,
            )
            self._cache[anomaly_id] = no_rec_resp
            return no_rec_resp

        # 4. Build compact evidence package
        evidence_pkg = build_recommendation_evidence(
            investigation=investigation,
            explanation=explanation,
            ai_reasoning=ai_reasoning,
            eligible_recommendation_types=eligible_types,
        )

        driver_confidence_map = {d.driver_name: d.confidence for d in investigation.drivers}

        # 5. Invoke LLM Provider with fallback guard
        start_time = time.perf_counter()
        ai_resp: AIRecommendationResponse | None = None
        provider_error: Exception | None = None

        try:
            ai_resp = self.provider.generate_recommendations(evidence_pkg)
        except (LLMConfigurationError, LLMProviderError, Exception) as exc:
            provider_error = exc
            logger.warning(
                "Gemini recommendation generation failed | anomaly_id=%s | error=%s",
                anomaly_id,
                exc,
            )

        if ai_resp is None:
            if fallback:
                fallback_resp = self.build_deterministic_fallback(
                    investigation=investigation,
                    explanation=explanation,
                    ai_reasoning=ai_reasoning,
                    eligible_types=eligible_types,
                )
                self._cache[anomaly_id] = fallback_resp
                return fallback_resp
            else:
                if isinstance(provider_error, LLMConfigurationError):
                    raise provider_error
                raise LLMProviderError(f"Recommendation generation failed: {provider_error}") from provider_error

        # 6. Post-validate LLM output
        validated_recs = self._post_validate_recommendations(
            raw_recs=ai_resp.recommendations,
            anomaly_id=anomaly_id,
            investigation=investigation,
            eligible_types=eligible_types,
            driver_confidence_map=driver_confidence_map,
        )

        # Deterministic Invariant: Every recommendation MUST belong to eligible_types
        validated_recs = [r for r in validated_recs if r.recommendation_type in eligible_types]

        # If all LLM recommendations were invalid, use deterministic fallback
        if not validated_recs:
            fallback_resp = self.build_deterministic_fallback(
                investigation=investigation,
                explanation=explanation,
                ai_reasoning=ai_reasoning,
                eligible_types=eligible_types,
            )
            self._cache[anomaly_id] = fallback_resp
            return fallback_resp

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Recommendations completed and cached | anomaly_id=%s | count=%d | total_time=%.2fms",
            anomaly_id,
            len(validated_recs),
            elapsed_ms,
        )

        final_resp = RecommendationResponse(
            anomaly_id=anomaly_id,
            recommendation_status="actionable" if validated_recs else "no_actionable_recommendation",
            summary=ai_resp.summary.strip(),
            recommendations=validated_recs,
            limitations=[
                "Advisory Only: Prescriptive recommendations are intended solely for human decision support; autonomous execution is strictly disabled.",
                "Observational Association: Recommended actions reflect empirical correlations and do not guarantee counterfactual outcomes.",
                "Human Authorization: Operational changes require qualified management review and verification of external constraints.",
            ],
            human_approval_required=True,
            source="ai_generated",
            cached=False,
        )

        self._cache[anomaly_id] = final_resp
        return final_resp

    def clear_cache(self, anomaly_id: str | None = None) -> None:
        """Clears recommendation cache for a specific anomaly or entirely."""
        if anomaly_id:
            self._cache.pop(anomaly_id, None)
        else:
            self._cache.clear()
