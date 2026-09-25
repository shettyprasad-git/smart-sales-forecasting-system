from __future__ import annotations

import logging
import re
import time
from typing import Any

from backend.app.llm.gemini_provider import GeminiProvider
from backend.app.llm.prompts import build_reasoning_evidence
from backend.app.llm.provider import (
    LLMConfigurationError,
    LLMProvider,
    LLMProviderError,
    LLMResponseValidationError,
)
from backend.app.schemas.ai_reasoning import AIInsight, AIReasoningResponse
from backend.app.services.explanation_service import ExplanationService
from backend.app.services.investigation_service import InvestigationService

logger = logging.getLogger(__name__)

STANDARD_CAUSAL_DISCLAIMER = (
    "Observational associations identify statistically correlated contributors; "
    "they do not establish counterfactual causality."
)

PROHIBITED_CAUSAL_REPLACEMENTS = [
    (r"\bdefinitely caused\b", "strongly associated with"),
    (r"\bcaused by\b", "associated with"),
    (r"\bproves that\b", "indicates that"),
    (r"\bproven cause of\b", "primary contributor to"),
    (r"\bdirect cause of\b", "associated factor in"),
    (r"\bcaused the\b", "corresponded with the"),
    (r"\bcaused\b", "associated with"),
    (r"\bcausality\b", "association"),
    (
        r"\bpremium(?:\s+product)?(?:\s+tier)?\s+purchasing\s+(?:contributed|drove|caused)\b",
        "Product-mix differences may be worth validating",
    ),
    (
        r"\bprice\s+realization\s+(?:caused|drove|resulted\s+in)\b",
        "price realization co-occurred with",
    ),
]

HOLIDAY_KEYWORDS = re.compile(
    r"\b(?:during\s+|in\s+)?(?:the\s+)?holiday\s+(?:period|season|shopping|trading|demand|momentum)\b|"
    r"\bholiday(?:-driven|\s+shopping|\s+period|\s+demand|\s+momentum|\s+trading)\b|"
    r"\bchristmas\b|\bpost-christmas\b|\bnew\s+year(?:'s)?\b|\bfestive\b",
    re.IGNORECASE,
)

PROMOTION_KEYWORDS = re.compile(
    r"\bactive\s+promot(?:ion|ional)\b|\bpromotional\s+campaign\b|\bdiscount\s+drive\b|\bmarketing\s+promotion\b",
    re.IGNORECASE,
)

EXTERNAL_FACT_PATTERNS = [
    (re.compile(r"\bcompetitors?\b|\brivals?\b", re.IGNORECASE), "competitor activity"),
    (re.compile(r"\b(?:unrecorded\s+)?marketing(?:\s+activity|\s+campaigns?|\s+spend)?\b|\bemail\s+promotions?\b|\bad\s+campaigns?\b|\bcoupon\s+codes?\b", re.IGNORECASE), "marketing activity"),
    (re.compile(r"\b(?:web|website|visitor|foot)\s+traffic\b|\bsite\s+outages?\b", re.IGNORECASE), "website traffic"),
    (re.compile(r"\binventory\s+(?:shortages?|stockouts?|availability|deficits?)\b|\bstockouts?\b|\bout\s+of\s+stock\b", re.IGNORECASE), "inventory availability"),
    (re.compile(r"\bregional\s+(?:operational\s+issues?|logistics|shipping)\b|\bshipping\s+delays?\b", re.IGNORECASE), "regional operations"),
    (re.compile(r"\b(?:b2b|bulk\s+orders?|wholesale\s+orders?)\b", re.IGNORECASE), "B2B/wholesale orders"),
    (re.compile(r"\bsupply[- ]chains?\s*(?:disruptions?)?\b", re.IGNORECASE), "supply-chain factors"),
    (re.compile(r"\bexternal\s+market\s+events?\b", re.IGNORECASE), "external market events"),
    (
        re.compile(
            r"\b(?:premium\s+(?:product\s+)?(?:lines?|tiers?|purchasing|items?|goods?|catalog|customers?)|"
            r"premium\s+tiers?|premium\s+products?|luxury(?:\s+products?|\s+goods?|\s+items?|\s+lines?|\s+segments?|\s+customers?)?|"
            r"higher[- ]tier(?:\s+products?|\s+lines?|\s+items?|\s+goods?)?|high[- ]end\s+products?)\b",
            re.IGNORECASE,
        ),
        "product tiers / luxury segments",
    ),
]

QUESTION_STARTERS = (
    "did", "was", "were", "is", "are", "could", "would", "should",
    "can", "do", "does", "have", "has", "had", "will", "what",
    "why", "how", "where", "when", "which", "who", "whom", "whose",
)

ALLOWED_DIMENSIONS = {
    "category",
    "product",
    "promotion",
    "holiday",
    "price",
    "discount",
    "drift",
    "recent_trend",
    "baseline_drift",
    "other",
}


class AIReasoningService:
    """
    AI Reasoning Service for Phase 6.4.
    Orchestrates compact evidence packaging from statistical investigation (6.2) and
    executive explanation (6.3), delegates reasoning to the LLMProvider, performs
    deterministic safety post-validation and evidence entailment checking, and provides in-memory caching.
    """

    def __init__(
        self,
        investigation_service: InvestigationService | None = None,
        explanation_service: ExplanationService | None = None,
        provider: LLMProvider | None = None,
    ) -> None:
        self.investigation_service = investigation_service or InvestigationService()
        self.explanation_service = explanation_service or ExplanationService(
            investigation_service=self.investigation_service
        )
        self.provider = provider or GeminiProvider()
        self._cache: dict[str, AIReasoningResponse] = {}

    def reason_about_anomaly(
        self,
        anomaly_id: str,
        refresh: bool = False,
        user_id: int | None = None,
        db: Any = None,
    ) -> AIReasoningResponse:
        """
        Produce or retrieve cached AI reasoning for an anomaly.
        """
        # 1. Check in-memory cache
        cache_key = f"{user_id}:{anomaly_id}" if user_id else anomaly_id
        if not refresh and cache_key in self._cache:
            cached_resp = self._cache[cache_key]
            logger.info("Serving AI reasoning from cache | key=%s", cache_key)
            return cached_resp.model_copy(update={"cached": True})

        start_time = time.perf_counter()

        # 2. Retrieve authoritative statistical outputs
        investigation = self.investigation_service.investigate_anomaly(
            anomaly_id,
            user_id=user_id,
            db=db,
        )
        explanation = self.explanation_service.generate_explanation(investigation)

        # 3. Build compact evidence package (< 2 KB, zero raw CSV data)
        evidence_pkg = build_reasoning_evidence(investigation, explanation)

        # 4. Delegate to LLM provider
        raw_reasoning = self.provider.generate_reasoning(evidence_pkg)

        # 5. Deterministic Safety, Grounding & Evidence Entailment Post-Validation
        validated_reasoning = self._post_validate_reasoning(
            raw_reasoning=raw_reasoning,
            requested_anomaly_id=anomaly_id,
            evidence_pkg=evidence_pkg,
        )

        # 6. Store in cache
        self._cache[anomaly_id] = validated_reasoning

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "AI reasoning completed and cached | anomaly_id=%s | total_time=%.2fms",
            anomaly_id,
            elapsed_ms,
        )

        return validated_reasoning.model_copy(update={"cached": False})

    def _sanitize_causal_text(self, text: str) -> str:
        """Replace prohibited causal certainty assertions with observational language."""
        sanitized = text
        for pattern, replacement in PROHIBITED_CAUSAL_REPLACEMENTS:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        return sanitized

    def _clean_prose(self, text: str) -> str:
        """Clean up double spaces and awkward punctuation after regex edits."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)
        text = re.sub(r"\b(?:during|in)\s+[.,;]", ".", text)
        return text.strip()

    def _normalize_num(self, s: str) -> float | None:
        try:
            clean = re.sub(r"[₹$,%]", "", s).strip()
            return float(clean)
        except (ValueError, TypeError):
            return None

    def _is_close(
        self, val1: float, val2: float, rel_tol: float = 0.02, abs_tol: float = 0.5
    ) -> bool:
        if abs(val1 - val2) <= abs_tol:
            return True
        if val2 != 0 and abs(val1 - val2) / abs(val2) <= rel_tol:
            return True
        return False

    def _post_validate_reasoning(
        self,
        raw_reasoning: AIReasoningResponse,
        requested_anomaly_id: str,
        evidence_pkg: dict[str, Any],
    ) -> AIReasoningResponse:
        """
        Deterministic post-validation & evidence entailment enforcement:
        1. Strict anomaly ID match.
        2. Prohibit unsupported holiday claims if holiday_active is False.
        3. Prohibit unsupported promotion claims if promotion_active is False.
        4. Prevent unsupported external explanations (competitors, inventory, marketing, B2B, supply-chain).
        5. Numerical consistency validation (actual, baseline, deviation %, impact, driver observed/ref/contrib).
        6. Evidence entailment for each AIInsight (dimension, associated driver, confidence cap).
        7. Validation questions verification (must be questions for human review, not declarative claims).
        8. Sanitization of causal language.
        9. Mandatory scientific causal disclaimer.
        """
        anomaly_id = requested_anomaly_id

        anomaly = evidence_pkg.get("anomaly", {})
        actual_val = anomaly.get("actual_value")
        baseline_val = anomaly.get("expected_baseline")
        dev_val = anomaly.get("deviation")
        dev_pct = anomaly.get("deviation_percent")
        metric = anomaly.get("metric", "sales_amount")
        anom_date = anomaly.get("date")

        impact = evidence_pkg.get("impact", {})
        impact_val = impact.get("estimated_revenue_impact") or impact.get("impact_value")

        events = evidence_pkg.get("events", {})
        holiday_active = events.get("holiday_active", False)
        promotion_active = events.get("promotion_active", False)

        top_drivers = evidence_pkg.get("top_drivers", [])
        drivers_by_name = {d["driver_name"]: d for d in top_drivers}
        driver_confidence_map = evidence_pkg.get("driver_confidence_map", {})
        if not driver_confidence_map and top_drivers:
            driver_confidence_map = {
                d["driver_name"]: d.get("confidence", "low") for d in top_drivers
            }

        valid_driver_names = set(driver_confidence_map.keys())

        # 1. Headline & Interpretation Grounding
        headline = self._sanitize_causal_text(raw_reasoning.reasoning_headline)
        interpretation = self._sanitize_causal_text(raw_reasoning.executive_interpretation)
        evidence_assessment = self._sanitize_causal_text(raw_reasoning.evidence_assessment)

        if not holiday_active:
            headline = HOLIDAY_KEYWORDS.sub("", headline)
            interpretation = HOLIDAY_KEYWORDS.sub("", interpretation)

        if not promotion_active:
            headline = PROMOTION_KEYWORDS.sub("", headline)
            interpretation = PROMOTION_KEYWORDS.sub("", interpretation)

        # 1b. Numerical Validation for Headline
        if dev_pct is not None:
            for pm in list(re.finditer(r"([+-]?\d+(?:\.\d+)?)\s*%", headline)):
                val = self._normalize_num(pm.group(1))
                if val is not None and not self._is_close(abs(val), abs(dev_pct), abs_tol=0.3):
                    headline = headline[:pm.start(1)] + f"{abs(dev_pct):.1f}" + headline[pm.end(1):]

        # 1c. Numerical Validation for Interpretation
        if dev_pct is not None:
            for pm in list(re.finditer(r"([+-]?\d+(?:\.\d+)?)\s*%", interpretation)):
                val = self._normalize_num(pm.group(1))
                if val is not None and not self._is_close(abs(val), abs(dev_pct), abs_tol=0.3):
                    interpretation = interpretation[:pm.start(1)] + f"{abs(dev_pct):.1f}" + interpretation[pm.end(1):]

        for m in list(re.finditer(r"[₹$]?([\d,]+(?:\.\d+)?)", interpretation)):
            raw_str = m.group(1)
            if "," in raw_str or (raw_str.replace(".", "").isdigit() and len(raw_str) >= 5):
                num_val = self._normalize_num(raw_str)
                if num_val is not None and num_val > 1000:
                    matches_auth = (
                        (actual_val is not None and self._is_close(num_val, actual_val, rel_tol=0.01)) or
                        (baseline_val is not None and self._is_close(num_val, baseline_val, rel_tol=0.01)) or
                        (dev_val is not None and self._is_close(num_val, abs(dev_val), rel_tol=0.01)) or
                        (impact_val is not None and self._is_close(num_val, abs(impact_val), rel_tol=0.01)) or
                        any(
                            (d.get("observed_value") and self._is_close(num_val, d["observed_value"], rel_tol=0.01)) or
                            (d.get("reference_value") and self._is_close(num_val, d["reference_value"], rel_tol=0.01))
                            for d in top_drivers
                        )
                    )
                    if not matches_auth:
                        preceding = interpretation[:m.start()].lower()
                        if any(w in preceding for w in ["baseline", "expected"]):
                            if baseline_val is not None:
                                interpretation = interpretation.replace(m.group(0), f"₹{baseline_val:,.2f}")
                        elif any(w in preceding for w in ["impact", "excess revenue", "revenue gap", "financial"]):
                            if impact_val is not None:
                                interpretation = interpretation.replace(m.group(0), f"₹{impact_val:,.2f}")
                        elif any(w in preceding for w in ["deviation", "surplus", "deficit"]):
                            if dev_val is not None:
                                interpretation = interpretation.replace(m.group(0), f"₹{abs(dev_val):,.2f}")
                        else:
                            if actual_val is not None:
                                interpretation = interpretation.replace(m.group(0), f"₹{actual_val:,.2f}")

        # Sanitize unsupported product-tier / luxury claims in headline and interpretation
        tier_sub_pattern = re.compile(
            r"\b(?:premium\s+(?:product\s+)?(?:lines?|tiers?|purchasing|items?|goods?|catalog|customers?)|"
            r"premium\s+tiers?|premium\s+products?|luxury(?:\s+products?|\s+goods?|\s+items?|\s+lines?|\s+segments?|\s+customers?)?|"
            r"higher[- ]tier(?:\s+products?|\s+lines?|\s+items?|\s+goods?)?|high[- ]end\s+products?)\b",
            re.IGNORECASE,
        )
        if tier_sub_pattern.search(headline):
            headline = tier_sub_pattern.sub("recorded catalog segments", headline)
        if tier_sub_pattern.search(interpretation):
            interpretation = tier_sub_pattern.sub("recorded catalog segments", interpretation)

        headline = self._clean_prose(headline)
        interpretation = self._clean_prose(interpretation)

        # 2. Key Insights Entailment Validation
        validated_insights: list[AIInsight] = []
        extra_uncertainties: list[str] = []

        for ins in raw_reasoning.key_insights:
            dim = (ins.dimension or "other").lower().strip()
            if dim in ("recent_trend", "baseline_drift"):
                dim = "drift"

            # Check dimension allowed in evidence vocabulary
            if dim not in ALLOWED_DIMENSIONS:
                extra_uncertainties.append(
                    f"{dim.capitalize()}: Dimension not evaluated by the available dataset."
                )
                continue

            # Check false holiday claim
            if not holiday_active:
                if (
                    dim == "holiday"
                    or HOLIDAY_KEYWORDS.search(ins.statement)
                    or HOLIDAY_KEYWORDS.search(ins.supporting_evidence)
                ):
                    extra_uncertainties.append(
                        "Holiday factors: Not evaluated / no calendar holiday active on this date."
                    )
                    continue

            # Check false promotion claim
            if not promotion_active:
                if (
                    dim == "promotion"
                    or PROMOTION_KEYWORDS.search(ins.statement)
                    or PROMOTION_KEYWORDS.search(ins.supporting_evidence)
                ):
                    extra_uncertainties.append(
                        "Promotion factors: Not evaluated / no active promotional campaign on this date."
                    )
                    continue

            # Check driver matching
            driver_name = ins.associated_driver or ins.related_driver
            matched_driver: str | None = None
            if driver_name:
                driver_clean = driver_name.strip()
                # Direct or case-insensitive match
                for vn in valid_driver_names:
                    if driver_clean.lower() == vn.lower():
                        matched_driver = vn
                        break
                    elif (
                        len(driver_clean) > 3
                        and (driver_clean.lower() in vn.lower() or vn.lower() in driver_clean.lower())
                    ):
                        matched_driver = vn
                        break

                if not matched_driver:
                    # Mismatch: Driver not present in evidence package
                    extra_uncertainties.append(
                        f"Driver '{driver_name}': Mismatched to evaluated investigation drivers."
                    )
                    continue

            # Check unsupported external facts in insight statement or evidence
            has_external_fact = False
            for pattern, label in EXTERNAL_FACT_PATTERNS:
                if pattern.search(ins.statement) or pattern.search(ins.supporting_evidence):
                    extra_uncertainties.append(
                        f"{label.capitalize()}: Not evaluated by the available dataset."
                    )
                    has_external_fact = True
                    break

            if has_external_fact:
                continue

            # Check confidence cap against evidence driver confidence
            conf = ins.confidence.lower()
            if conf not in ("high", "medium", "low"):
                conf = "medium"

            if matched_driver and matched_driver in driver_confidence_map:
                max_conf = driver_confidence_map[matched_driver].lower()
                conf_rank = {"low": 1, "medium": 2, "high": 3}
                if conf_rank.get(conf, 2) > conf_rank.get(max_conf, 2):
                    conf = max_conf

            stmt = self._clean_prose(self._sanitize_causal_text(ins.statement))
            supp = self._clean_prose(self._sanitize_causal_text(ins.supporting_evidence))

            # Numerical validation against authoritative driver figures
            if matched_driver and matched_driver in drivers_by_name:
                d = drivers_by_name[matched_driver]
                d_obs = d.get("observed_value")
                d_ref = d.get("reference_value")
                d_diff = d.get("difference")
                d_diff_pct = d.get("difference_percent")
                d_contrib = d.get("contribution_score")
                d_ev = d.get("evidence", "").replace("$", "₹")

                combined_text = f"{stmt} {supp}"
                has_num_contradiction = False

                # 1. Contribution % check
                for cm in re.finditer(
                    r"(\d+(?:\.\d+)?)\s*%\s*(?:contribution|share|of\s+total|relative\s+share)",
                    combined_text,
                    re.IGNORECASE,
                ):
                    c_val = self._normalize_num(cm.group(1))
                    if c_val is not None and d_contrib is not None:
                        if not self._is_close(c_val, d_contrib, abs_tol=0.5):
                            has_num_contradiction = True
                            break

                # 2. Difference % check
                if not has_num_contradiction:
                    for dm in re.finditer(
                        r"([+-]?\d+(?:\.\d+)?)\s*%\s*(?:shift|lift|vs\s+baseline|above|below)",
                        combined_text,
                        re.IGNORECASE,
                    ):
                        dm_val = self._normalize_num(dm.group(1))
                        if dm_val is not None and d_diff_pct is not None:
                            if not self._is_close(abs(dm_val), abs(d_diff_pct), abs_tol=0.5):
                                has_num_contradiction = True
                                break

                # 3. Currency / number check
                if not has_num_contradiction:
                    for nm in re.finditer(r"[₹$]?([\d,]+(?:\.\d+)?)", combined_text):
                        raw_n = nm.group(1)
                        if "," in raw_n or (raw_n.replace(".", "").isdigit() and len(raw_n) >= 4):
                            n_val = self._normalize_num(raw_n)
                            if n_val is not None and n_val > 100:
                                matches_d = (
                                    (d_obs is not None and self._is_close(n_val, d_obs, rel_tol=0.01))
                                    or (d_ref is not None and self._is_close(n_val, d_ref, rel_tol=0.01))
                                    or (d_diff is not None and self._is_close(n_val, abs(d_diff), rel_tol=0.01))
                                    or (d_contrib is not None and self._is_close(n_val, d_contrib, abs_tol=0.5))
                                )
                                if not matches_d:
                                    has_num_contradiction = True
                                    break

                if has_num_contradiction:
                    if d.get("driver_type") == "price":
                        stmt = f"Average realized unit price was ₹{d_obs:,.2f} vs reference baseline of ₹{d_ref:,.2f} ({d_diff_pct:+.1f}% shift)."
                        supp = f"Average unit price on anomaly date was ₹{d_obs:,.2f} vs 28-day historical reference of ₹{d_ref:,.2f} ({d_diff_pct:+.1f}% shift)."
                    else:
                        stmt = f"{matched_driver} showed an observed value of ₹{d_obs:,.2f} vs reference baseline of ₹{d_ref:,.2f} ({d_diff_pct:+.1f}% shift), contributing {d_contrib:.1f}% to total variance."
                        supp = d_ev

            validated_insights.append(
                AIInsight(
                    dimension=dim,
                    statement=stmt,
                    supporting_evidence=supp,
                    confidence=conf,  # type: ignore
                    associated_driver=matched_driver,
                    related_driver=matched_driver,
                )
            )

        # Fallback if all insights were filtered out: provide a strictly grounded default
        if not validated_insights and top_drivers:
            first_driver = top_drivers[0]
            first_name = first_driver["driver_name"]
            validated_insights.append(
                AIInsight(
                    dimension=first_driver.get("driver_type", "category"),
                    statement=f"{first_name} was the primary associated driver evaluated.",
                    supporting_evidence=first_driver.get(
                        "evidence",
                        f"Observed value was {first_driver.get('observed_value', 'N/A')}.",
                    ),
                    confidence=first_driver.get("confidence", "medium"),
                    associated_driver=first_name,
                    related_driver=first_name,
                )
            )

        # 3. Alternative Explanations Grounding
        validated_alt_explanations: list[str] = []
        for alt in raw_reasoning.alternative_explanations:
            if not holiday_active and HOLIDAY_KEYWORDS.search(alt):
                extra_uncertainties.append(
                    "Calendar factors: No recognized holiday active on this date; holiday hypotheses not evaluated."
                )
                continue

            if not promotion_active and PROMOTION_KEYWORDS.search(alt):
                extra_uncertainties.append(
                    "Promotion factors: No promotional campaign active on this date; unrecorded promotions not evaluated."
                )
                continue

            has_external = False
            for pattern, label in EXTERNAL_FACT_PATTERNS:
                if pattern.search(alt):
                    extra_uncertainties.append(
                        f"{label.capitalize()}: Not evaluated by the available dataset."
                    )
                    has_external = True
                    break

            if has_external:
                continue

            sanitized_alt = self._clean_prose(self._sanitize_causal_text(alt))
            if sanitized_alt:
                validated_alt_explanations.append(sanitized_alt)

        if not validated_alt_explanations:
            validated_alt_explanations.append(
                "Multiple recorded categories showed positive deviations, indicating that the observed revenue increase was distributed across several catalog segments."
            )

        # 4. Validation Questions Verification (must be explicit questions)
        validated_questions: list[str] = []
        for q in raw_reasoning.validation_questions:
            q_clean = q.strip()
            if not q_clean:
                continue

            # Convert product-mix hypothesis statements or legacy tier references to explicit questions
            if (
                re.search(r"product[- ]mix.*(?:worth validating|validate|hypothesis)", q_clean, re.IGNORECASE)
                or re.search(r"premium.*(?:tier|product|purchasing).*(?:contribute|unit price|price)", q_clean, re.IGNORECASE)
            ):
                converted_q = "Could product-mix differences within recorded categories contribute to the higher realized unit price?"
                if converted_q not in validated_questions:
                    validated_questions.append(converted_q)
                continue

            # Reject declarative assertions masquerading as questions
            if (
                any(
                    w in q_clean.lower()
                    for w in [
                        "contributed to",
                        "caused by",
                        "caused the",
                        "resulted in",
                        "increased revenue",
                        "purchasing increased",
                        "may be worth validating",
                        "contributed",
                    ]
                )
                and not q_clean.lower().startswith(QUESTION_STARTERS)
            ):
                continue

            # Reject unsupported tier assertions if not already converted
            if re.search(r"\b(?:premium|luxury|higher[- ]tier)\b", q_clean, re.IGNORECASE):
                continue

            # If it ends with ? and starts with a recognized question starter, accept it
            if q_clean.endswith("?") and q_clean.lower().startswith(QUESTION_STARTERS):
                sanitized_q = self._sanitize_causal_text(q_clean)
                if sanitized_q not in validated_questions:
                    validated_questions.append(sanitized_q)
            elif q_clean.endswith("?"):
                # Ends with ?, verify it is not a declarative sentence with a question mark
                if any(
                    w in q_clean.lower()
                    for w in [
                        "contributed to",
                        "caused by",
                        "increased revenue",
                        "drove the",
                    ]
                ):
                    continue
                sanitized_q = self._sanitize_causal_text(q_clean)
                if sanitized_q not in validated_questions:
                    validated_questions.append(sanitized_q)
            elif q_clean.lower().startswith(QUESTION_STARTERS):
                # Starts with question word but missing trailing ?
                sanitized_q = f"{self._sanitize_causal_text(q_clean)}?"
                if sanitized_q not in validated_questions:
                    validated_questions.append(sanitized_q)
            else:
                # Any other declarative statement without question structure: reject!
                continue

        if not validated_questions:
            validated_questions.append(
                "Could product-mix differences within recorded categories contribute to the higher realized unit price?"
            )

        # 5. Uncertainties Synthesis & Deduplication
        all_uncertainties = list(raw_reasoning.uncertainties) + extra_uncertainties
        deduped_uncertainties: list[str] = []
        seen_unc: set[str] = set()
        for u in all_uncertainties:
            u_clean = self._clean_prose(self._sanitize_causal_text(u))
            if u_clean and u_clean not in seen_unc:
                seen_unc.add(u_clean)
                deduped_uncertainties.append(u_clean)

        # 6. Always enforce standard causal disclaimer
        causal_disclaimer = STANDARD_CAUSAL_DISCLAIMER

        return AIReasoningResponse(
            anomaly_id=anomaly_id,
            reasoning_headline=headline,
            executive_interpretation=interpretation,
            key_insights=validated_insights,
            alternative_explanations=validated_alt_explanations,
            evidence_assessment=evidence_assessment,
            uncertainties=deduped_uncertainties,
            validation_questions=validated_questions,
            risk_flags=[self._sanitize_causal_text(rf) for rf in raw_reasoning.risk_flags],
            causal_disclaimer=causal_disclaimer,
            cached=False,
        )

    def clear_cache(self, anomaly_id: str | None = None) -> None:
        """Clear reasoning cache for a specific anomaly or entirely."""
        if anomaly_id:
            self._cache.pop(anomaly_id, None)
        else:
            self._cache.clear()
