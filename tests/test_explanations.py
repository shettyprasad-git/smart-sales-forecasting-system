import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.schemas.explanations import ExecutiveExplanation
from backend.app.services.anomaly_service import AnomalyDetectionService
from backend.app.services.explanation_service import ExplanationService
from backend.app.services.investigation_service import InvestigationService


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def investigation_service():
    return InvestigationService()


@pytest.fixture
def explanation_service(investigation_service):
    return ExplanationService(investigation_service=investigation_service)


@pytest.fixture
def sample_anomaly_id(investigation_service):
    """Retrieve a real anomaly ID from the detection service."""
    anomalies = investigation_service.anomaly_service.get_anomalies(limit=5)
    assert len(anomalies.items) > 0
    return anomalies.items[0].id


@pytest.fixture
def spike_and_drop_anomaly_ids(investigation_service):
    spikes = investigation_service.anomaly_service.get_anomalies(direction="spike", limit=3)
    drops = investigation_service.anomaly_service.get_anomalies(direction="drop", limit=3)
    assert len(spikes.items) > 0
    assert len(drops.items) > 0
    return spikes.items[0].id, drops.items[0].id


# =====================================================================
# Narrative Section Generation Tests
# =====================================================================


def test_headline_generation(explanation_service, sample_anomaly_id):
    """Verify headline is a concise factual sentence containing metric, deviation %, and date."""
    expl = explanation_service.explain_anomaly(sample_anomaly_id)
    assert expl.headline is not None
    assert len(expl.headline) > 10
    assert str(expl.sections[0].content) == expl.headline
    assert "baseline on" in expl.headline.lower()
    assert any(term in expl.headline.lower() for term in ("increased", "decreased"))


def test_spike_and_drop_narratives(explanation_service, spike_and_drop_anomaly_ids):
    """Verify distinct narrative phrasing for spikes vs drops in 'what_happened'."""
    spike_id, drop_id = spike_and_drop_anomaly_ids

    spike_expl = explanation_service.explain_anomaly(spike_id)
    assert "spike" in spike_expl.what_happened.lower()
    assert "above" in spike_expl.what_happened.lower()
    assert spike_expl.sections[0].severity == "critical" or spike_expl.sections[0].severity in (
        "high", "medium", "low"
    )

    drop_expl = explanation_service.explain_anomaly(drop_id)
    assert "drop" in drop_expl.what_happened.lower()
    assert "below" in drop_expl.what_happened.lower()


def test_impact_narrative(explanation_service, sample_anomaly_id):
    """Verify 'why_it_matters' and 'impact_summary' contain quantified impact and pricing notes."""
    expl = explanation_service.explain_anomaly(sample_anomaly_id)
    assert expl.why_it_matters is not None
    assert len(expl.why_it_matters) > 10
    assert expl.impact_summary is not None
    assert any(term in expl.why_it_matters.lower() for term in ("estimated", "potential", "excess", "gap", "deficit"))


def test_currency_formatting_uses_inr(explanation_service, sample_anomaly_id):
    """Verify narrative output formats currency using Indian Rupees (₹) and does not leak USD ($)."""
    expl = explanation_service.explain_anomaly(sample_anomaly_id)
    assert "$" not in expl.headline
    assert "$" not in expl.what_happened
    assert "$" not in expl.why_it_matters
    assert "$" not in expl.impact_summary
    for contrib in expl.key_contributors:
        assert "$" not in contrib
    for section in expl.sections:
        assert "$" not in section.content

    inv = explanation_service.investigation_service.investigate_anomaly(sample_anomaly_id)
    if inv.metric == "sales_amount":
        assert "₹" in expl.what_happened


def test_contributor_ranking_and_top_n(explanation_service, sample_anomaly_id):
    """Verify key contributors prioritize directional relevance and strictly observe top_n cap."""
    expl_3 = explanation_service.explain_anomaly(sample_anomaly_id, top_n=3)
    assert len(expl_3.key_contributors) <= 3

    expl_1 = explanation_service.explain_anomaly(sample_anomaly_id, top_n=1)
    assert len(expl_1.key_contributors) <= 1

    # Section for key contributors
    contrib_section = next(
        s for s in expl_3.sections if s.section_type == "key_contributors"
    )
    assert contrib_section is not None
    assert "contributing" in contrib_section.content.lower() or "contributor" in contrib_section.content.lower()


def test_event_context_promotion_and_holiday(explanation_service, sample_anomaly_id):
    """Verify event context section handles promotional and holiday calendar factors deterministically."""
    expl = explanation_service.explain_anomaly(sample_anomaly_id)
    event_section = next(
        s for s in expl.sections if s.section_type == "event_context"
    )
    assert event_section is not None
    assert len(event_section.content) > 10
    # Must mention promotion and holiday status
    assert "promotional" in event_section.content.lower() or "promotion" in event_section.content.lower()
    assert "holiday" in event_section.content.lower()


def test_trend_context_narrative(explanation_service, sample_anomaly_id):
    """Verify trend context analyzes pre-anomaly trajectory (rising, falling, or stable)."""
    expl = explanation_service.explain_anomaly(sample_anomaly_id)
    trend_section = next(
        s for s in expl.sections if s.section_type == "trend_context"
    )
    assert trend_section is not None
    assert any(
        traj in trend_section.content.lower()
        for traj in ("rising", "falling", "stable", "trajectory", "baseline")
    )


def test_evidence_quality_and_confidence_summary(explanation_service, sample_anomaly_id):
    """Verify evidence quality synthesizes driver confidences into an overall rating."""
    expl = explanation_service.explain_anomaly(sample_anomaly_id)
    assert expl.evidence_summary is not None
    assert expl.confidence_summary is not None
    assert any(c in expl.confidence_summary.lower() for c in ("high", "medium", "low"))


def test_methodological_limitations_included(explanation_service, sample_anomaly_id):
    """Verify limitations contain clear disclaimers regarding observational data and causality."""
    expl = explanation_service.explain_anomaly(sample_anomaly_id)
    assert len(expl.limitations) >= 3
    joined = " ".join(expl.limitations).lower()
    assert "observational" in joined
    assert "causality" in joined or "counterfactual" in joined
    assert "leakage" in joined or "lookback" in joined


# =====================================================================
# Determinism & Leakage Invariance Tests
# =====================================================================


def test_deterministic_output(explanation_service, sample_anomaly_id):
    """Verify calling explain_anomaly multiple times produces byte-for-byte identical output."""
    expl1 = explanation_service.explain_anomaly(sample_anomaly_id, top_n=3)
    expl2 = explanation_service.explain_anomaly(sample_anomaly_id, top_n=3)

    assert expl1.model_dump() == expl2.model_dump()
    assert expl1.headline == expl2.headline
    assert expl1.what_happened == expl2.what_happened
    assert expl1.why_it_matters == expl2.why_it_matters
    assert expl1.key_contributors == expl2.key_contributors


def test_no_future_data_influence(explanation_service, sample_anomaly_id):
    """Verify modifying future records in the dataset has ZERO effect on explanation for date T."""
    expl_orig = explanation_service.explain_anomaly(sample_anomaly_id)

    # Check underlying investigation date
    inv = explanation_service.investigation_service.investigate_anomaly(sample_anomaly_id)
    target_date = inv.anomaly_date

    # Mutate data after target_date
    df = explanation_service.investigation_service._load_product_data()
    future_mask = df["Date"] > pd.Timestamp(target_date)

    if future_mask.sum() > 0:
        original_values = df.loc[future_mask, "Quantity"].copy()
        try:
            df.loc[future_mask, "Quantity"] = 999999.0

            # Generate explanation with tampered future data
            expl_tampered = explanation_service.explain_anomaly(sample_anomaly_id)
            assert expl_orig.model_dump() == expl_tampered.model_dump()
        finally:
            df.loc[future_mask, "Quantity"] = original_values


# =====================================================================
# REST API Endpoint Tests
# =====================================================================


def test_api_get_explanation_success(client, sample_anomaly_id):
    """GET /api/explanations/{anomaly_id} returns 200 with valid ExecutiveExplanation schema."""
    resp = client.get(f"/api/explanations/{sample_anomaly_id}?top_n=3&format=json")
    assert resp.status_code == 200
    data = resp.json()

    # Validate against schema
    obj = ExecutiveExplanation.model_validate(data)
    assert obj.anomaly_id == sample_anomaly_id
    assert len(obj.sections) >= 6
    assert len(obj.headline) > 0


def test_api_get_explanation_404(client):
    """GET /api/explanations/{unknown_id} returns 404."""
    resp = client.get("/api/explanations/anom-19900101-sal-agg")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_api_get_explanation_422_invalid_top_n(client, sample_anomaly_id):
    """GET /api/explanations/{anomaly_id}?top_n=0 or top_n=99 returns 422."""
    resp_zero = client.get(f"/api/explanations/{sample_anomaly_id}?top_n=0")
    assert resp_zero.status_code == 422

    resp_too_large = client.get(f"/api/explanations/{sample_anomaly_id}?top_n=99")
    assert resp_too_large.status_code == 422


def test_api_get_explanation_requires_auth_in_production(client, sample_anomaly_id):
    """Unauthenticated calls in production return 401."""
    from unittest.mock import patch
    from backend.app.core.config import settings

    with patch.object(settings, "environment", "production"):
        resp = client.get(f"/api/explanations/{sample_anomaly_id}")
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Authentication required."
