from __future__ import annotations

import logging
from typing import Literal

from backend.app.schemas.explanations import ExecutiveExplanation, ExplanationSection
from backend.app.schemas.investigations import (
    InvestigationDriver,
    InvestigationResponse,
)
from backend.app.services.investigation_service import InvestigationService

logger = logging.getLogger(__name__)


class ExplanationService:
    """
    Deterministic Executive Narrative Reporting Service for Phase 6.3.
    Transforms structured InvestigationResponse evidence from Phase 6.2
    into executive-ready briefings without using an LLM or non-deterministic generators.
    Strictly observes causal invariance (t < T), observational causality guardrails,
    and user-facing INR (₹) monetary formatting.
    """

    def __init__(
        self, investigation_service: InvestigationService | None = None
    ) -> None:
        self.investigation_service = investigation_service or InvestigationService()

    def explain_anomaly(
        self,
        anomaly_id: str,
        top_n: int = 3,
        user_id: int | None = None,
        db: Any = None,
    ) -> ExecutiveExplanation:
        """
        Locate and investigate anomaly, then generate deterministic executive narrative.
        """
        investigation = self.investigation_service.investigate_anomaly(
            anomaly_id,
            user_id=user_id,
            db=db,
        )
        return self.generate_explanation(investigation=investigation, top_n=top_n)

    def generate_explanation(
        self, investigation: InvestigationResponse, top_n: int = 3
    ) -> ExecutiveExplanation:
        """
        Synthesize deterministic narrative sections strictly from the investigation object.
        """
        top_n = max(1, min(top_n, 10))

        # 1. Headline
        headline = self._build_headline(investigation)

        # 2. What Happened
        what_happened = self._build_what_happened(investigation)

        # 3. Why It Matters
        why_it_matters = self._build_why_it_matters(investigation)

        # 4. Key Contributors Prioritization
        prioritized_drivers = self._prioritize_drivers(investigation.drivers, investigation.direction, top_n)
        key_contributors_text, contributors_narrative = self._build_contributors_narrative(
            prioritized_drivers, investigation
        )

        # 5. Event Context
        event_context = self._build_event_context(investigation.drivers)

        # 6. Trend Context
        trend_context = self._build_trend_context(investigation.drivers)

        # 7. Evidence Quality
        evidence_summary, evidence_narrative, overall_confidence = self._build_evidence_quality(
            investigation.drivers, prioritized_drivers
        )

        # 8. Impact Summary
        impact_summary = investigation.impact_interpretation.replace("$", "₹")

        # 9. Limitations
        limitations = [
            "Observational Attribution: Drivers represent empirical decompositions and statistical associations from observational data; they do not establish counterfactual causality.",
            "Causal Historical Baselines: Reference metrics use strictly causal lookbacks prior to the anomaly date (t < T) to prevent lookahead data leakage.",
            "Pricing Assumptions: Revenue estimations for quantity anomalies utilize baseline realized prices without assuming price elasticity or supply constraints.",
        ]

        # Cleaned price explanation
        cleaned_price_basis = (
            investigation.estimated_impact.price_basis_explanation.replace("$", "₹")
            if investigation.estimated_impact.price_basis_explanation
            else None
        )

        # Assemble Structured Sections
        sections: list[ExplanationSection] = [
            ExplanationSection(
                section_type="headline",
                title="Executive Headline",
                content=headline,
                severity=investigation.severity,
            ),
            ExplanationSection(
                section_type="what_happened",
                title="What Happened",
                content=what_happened,
                severity=investigation.severity,
                evidence=[
                    f"Actual: {self._format_metric_val(investigation.actual_value, investigation.metric)}",
                    f"Baseline: {self._format_metric_val(investigation.expected_baseline, investigation.metric)}",
                    f"Deviation: {investigation.deviation_percent:+.1f}%",
                    f"Anomaly Score: {investigation.anomaly_score:.2f}",
                ],
            ),
            ExplanationSection(
                section_type="why_it_matters",
                title="Why It Matters",
                content=why_it_matters,
                severity=investigation.severity,
                evidence=(
                    [cleaned_price_basis]
                    if cleaned_price_basis
                    else None
                ),
            ),
            ExplanationSection(
                section_type="key_contributors",
                title="Key Contributing Factors",
                content=contributors_narrative,
                confidence=overall_confidence,
                evidence=key_contributors_text,
            ),
            ExplanationSection(
                section_type="event_context",
                title="Event & Calendar Context",
                content=event_context,
            ),
            ExplanationSection(
                section_type="trend_context",
                title="Trend & Drift Context",
                content=trend_context,
            ),
            ExplanationSection(
                section_type="evidence_quality",
                title="Evidence Quality & Confidence",
                content=evidence_narrative,
                confidence=overall_confidence,
            ),
            ExplanationSection(
                section_type="limitations",
                title="Methodological Limitations",
                content=" ".join(limitations),
                evidence=limitations,
            ),
        ]

        return ExecutiveExplanation(
            anomaly_id=investigation.anomaly_id,
            headline=headline,
            what_happened=what_happened,
            why_it_matters=why_it_matters,
            key_contributors=key_contributors_text,
            impact_summary=impact_summary,
            evidence_summary=evidence_summary,
            confidence_summary=(
                f"Overall evidence confidence is rated {overall_confidence} based on pre-anomaly sample depth "
                f"and consistency across decomposed dimensions."
            ),
            limitations=limitations,
            sections=sections,
        )

    # -------------------------------------------------------------------------
    # Helper & Template Builders
    # -------------------------------------------------------------------------

    def _format_metric_val(self, val: float, metric: str) -> str:
        if metric == "quantity":
            return f"{val:,.0f} units"
        return f"₹{val:,.2f}"

    def _build_headline(self, inv: InvestigationResponse) -> str:
        metric_title = "Daily sales revenue" if inv.metric == "sales_amount" else "Daily sales quantity"
        entity_ctx = ""
        if inv.entity_type == "category" and inv.entity_name:
            entity_ctx = f" for {inv.entity_name}"
        elif inv.entity_type == "product" and inv.entity_name:
            entity_ctx = f" for {inv.entity_name}"

        dev_abs = abs(inv.deviation_percent)
        if inv.direction == "spike":
            return f"{metric_title}{entity_ctx} increased {dev_abs:.1f}% above its 28-day baseline on {inv.anomaly_date}."
        else:
            return f"{metric_title}{entity_ctx} decreased {dev_abs:.1f}% below its 28-day baseline on {inv.anomaly_date}."

    def _build_what_happened(self, inv: InvestigationResponse) -> str:
        metric_title = "Daily sales revenue" if inv.metric == "sales_amount" else "Daily sales quantity"
        actual_str = self._format_metric_val(inv.actual_value, inv.metric)
        baseline_str = self._format_metric_val(inv.expected_baseline, inv.metric)
        dev_val_str = (
            f"{inv.deviation:+,.0f} units" if inv.metric == "quantity" else f"₹{inv.deviation:+,.2f}"
        )
        entity_desc = (
            f"across {inv.entity_type} '{inv.entity_name}'"
            if inv.entity_name
            else "at the aggregate system level"
        )

        direction_word = "spike" if inv.direction == "spike" else "drop"
        prep = "above" if inv.direction == "spike" else "below"

        return (
            f"On {inv.anomaly_date}, {metric_title.lower()} {entity_desc} reached {actual_str} "
            f"against an expected 28-day baseline of {baseline_str}. "
            f"This represents an anomalous net deviation of {dev_val_str} ({abs(inv.deviation_percent):.1f}% {prep} baseline), "
            f"triggering a {inv.severity.upper()} severity {direction_word} with an anomaly score of {inv.anomaly_score:.2f}."
        )

    def _build_why_it_matters(self, inv: InvestigationResponse) -> str:
        imp = inv.estimated_impact
        interpretation = imp.interpretation.replace("$", "₹")
        price_note = ""
        if imp.price_basis_explanation:
            price_note = f" Assumption: {imp.price_basis_explanation.replace('$', '₹')}"

        if imp.estimated_revenue_impact is not None and inv.metric == "quantity":
            return (
                f"{interpretation} "
                f"Translating volume shift to financial terms yields an estimated monetary impact of "
                f"₹{imp.estimated_revenue_impact:,.2f}.{price_note}"
            )
        return f"{interpretation}{price_note}"

    def _prioritize_drivers(
        self, drivers: list[InvestigationDriver], direction: Literal["spike", "drop"], top_n: int
    ) -> list[InvestigationDriver]:
        """
        Deterministic selection rules for narrative reporting:
        1. Contribution score > 0 and direction is not neutral
        2. Relevance to anomaly direction (positive for spike, negative for drop)
        3. Confidence weighting (high > medium > low)
        4. Contribution score magnitude
        """
        candidates: list[InvestigationDriver] = []
        for d in drivers:
            # Skip non-contributing or neutral drivers
            if d.contribution_score <= 0.0 or d.direction == "neutral":
                continue
            candidates.append(d)

        def priority_key(d: InvestigationDriver) -> tuple[int, int, float]:
            aligned = 1 if (
                (direction == "spike" and d.direction == "positive")
                or (direction == "drop" and d.direction == "negative")
            ) else 0
            conf_val = {"high": 3, "medium": 2, "low": 1}.get(d.confidence, 1)
            return (aligned, conf_val, d.contribution_score)

        candidates.sort(key=priority_key, reverse=True)

        if not candidates:
            # Fallback to top contribution score regardless of direction
            fallback = [d for d in drivers if d.contribution_score > 0.0]
            fallback.sort(key=lambda d: d.contribution_score, reverse=True)
            return fallback[:top_n]

        return candidates[:top_n]

    def _build_contributors_narrative(
        self,
        prioritized_drivers: list[InvestigationDriver],
        inv: InvestigationResponse,
    ) -> tuple[list[str], str]:
        if not prioritized_drivers:
            empty_msg = (
                "No single primary contributor explained a dominant share of the observed variance; "
                "the deviation reflects diffuse movements across multiple baseline factors."
            )
            return ([], empty_msg)

        bullets: list[str] = []
        paragraphs: list[str] = []

        for d in prioritized_drivers:
            diff_str = (
                f"{d.difference_percent:+.1f}%"
                if d.difference_percent is not None
                else (f"{d.difference:+,.0f}" if d.difference is not None else "N/A")
            )
            obs_str = f"{d.observed_value:,.2f}" if d.observed_value is not None else "N/A"
            ref_str = f"{d.reference_value:,.2f}" if d.reference_value is not None else "N/A"

            cleaned_evidence = d.evidence.replace(" $", " ₹").replace("$", "₹")

            bullet = (
                f"{d.driver_name}: Observed {obs_str} vs historical baseline {ref_str} "
                f"({diff_str} shift), accounting for {d.contribution_score:.1f}% estimated relative "
                f"contribution [Confidence: {d.confidence.upper()}]."
            )
            bullets.append(bullet)

            desc = (
                f"{d.driver_name} showed an observed value of {obs_str} compared to a historical reference of {ref_str}, "
                f"representing a {diff_str} change and an estimated {d.contribution_score:.1f}% contribution "
                f"to the measured deviation (evaluated with {d.confidence} confidence). {cleaned_evidence}"
            )
            paragraphs.append(desc)

        lead = (
            f"Decomposition analysis identified {len(prioritized_drivers)} primary associated "
            f"{'factors' if len(prioritized_drivers) > 1 else 'factor'} contributing to the "
            f"{inv.direction}:"
        )
        full_narrative = f"{lead} {' '.join(paragraphs)}"
        return (bullets, full_narrative)

    def _build_event_context(self, drivers: list[InvestigationDriver]) -> str:
        promo_driver = next((d for d in drivers if d.driver_type == "promotion"), None)
        holiday_driver = next((d for d in drivers if d.driver_type == "holiday"), None)

        events: list[str] = []

        if promo_driver and promo_driver.observed_value and promo_driver.observed_value > 0:
            cleaned_evidence = promo_driver.evidence.replace(" $", " ₹").replace("$", "₹")
            events.append(
                f"Active promotional campaign was present on the anomaly date "
                f"({promo_driver.observed_value:,.0f} products discounted). {cleaned_evidence}"
            )
        else:
            events.append("No active promotional campaign was detected on the anomaly date.")

        if holiday_driver and holiday_driver.observed_value and holiday_driver.observed_value > 0:
            cleaned_evidence = holiday_driver.evidence.replace(" $", " ₹").replace("$", "₹")
            events.append(
                f"The anomaly coincided with a recognized calendar holiday event. {cleaned_evidence}"
            )
        else:
            events.append("The anomaly occurred on a standard non-holiday calendar trading date.")

        return " ".join(events)

    def _build_trend_context(self, drivers: list[InvestigationDriver]) -> str:
        trend_driver = next(
            (d for d in drivers if d.driver_type in ("recent_trend", "baseline_drift")),
            None,
        )
        if not trend_driver or trend_driver.difference_percent is None:
            return "Pre-anomaly trading demonstrated stable baseline momentum without significant drift."

        drift_pct = trend_driver.difference_percent
        if drift_pct >= 2.0:
            trajectory = "rising"
            prep = "above"
        elif drift_pct <= -2.0:
            trajectory = "falling"
            prep = "below"
        else:
            trajectory = "stable"
            prep = "near"

        cleaned_trend_evidence = trend_driver.evidence.replace(" $", " ₹").replace("$", "₹")

        return (
            f"The anomaly followed a {trajectory} recent trajectory, with the 7-day pre-anomaly average "
            f"{abs(drift_pct):.1f}% {prep} the 28-day baseline. {cleaned_trend_evidence}"
        )

    def _build_evidence_quality(
        self,
        all_drivers: list[InvestigationDriver],
        prioritized_drivers: list[InvestigationDriver],
    ) -> tuple[str, str, Literal["low", "medium", "high"]]:
        high_conf = [d.driver_name for d in prioritized_drivers if d.confidence == "high"]
        med_conf = [d.driver_name for d in prioritized_drivers if d.confidence == "medium"]
        low_conf = [d.driver_name for d in prioritized_drivers if d.confidence == "low"]

        if len(high_conf) >= 2 or (len(high_conf) >= 1 and len(med_conf) >= 1):
            overall: Literal["low", "medium", "high"] = "high"
        elif len(high_conf) >= 1 or len(med_conf) >= 1:
            overall = "medium"
        else:
            overall = "low"

        parts: list[str] = []
        if high_conf:
            parts.append(f"Strongest empirical evidence is supported by {', '.join(high_conf)}.")
        if med_conf:
            parts.append(f"Moderate evidence was observed for {', '.join(med_conf)}.")
        if low_conf:
            parts.append(f"Limited historical sample depth was available for {', '.join(low_conf)}.")

        if not parts:
            parts.append("Drivers were derived from standard historical baseline comparison.")

        evidence_narrative = (
            f"Overall evidence quality is assessed as {overall.upper()} confidence. "
            f"{' '.join(parts)} All reference statistics were evaluated strictly prior to the anomaly date."
        )
        evidence_summary = (
            f"{overall.capitalize()} confidence: {len(high_conf)} high, {len(med_conf)} medium, "
            f"and {len(low_conf)} low confidence contributing drivers."
        )

        return (evidence_summary, evidence_narrative, overall)
