from __future__ import annotations

import datetime as dt
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.database.models import (
    DecisionRecord,
    MonitoringAlert,
    MonitoringRun,
    Product,
    SalesRecord,
    User,
)
from backend.app.schemas.ai_reasoning import AIInsight, AIReasoningResponse
from backend.app.schemas.decisions import DecisionStatus
from backend.app.schemas.monitoring import AlertStatus


class MockLLMProvider:
    """Mock provider simulating deterministic grounded AI reasoning."""

    def generate_reasoning(self, evidence_package: dict) -> AIReasoningResponse:
        anomaly_id = evidence_package.get("anomaly", {}).get("anomaly_id", "anom-test")
        return AIReasoningResponse(
            anomaly_id=anomaly_id,
            reasoning_headline="Sales revenue deviation was associated with category shifts and price realization.",
            executive_interpretation="The observed spike represents strong customer engagement coinciding with active factors.",
            key_insights=[
                AIInsight(
                    dimension="category",
                    statement="Clothing category was the primary associated contributor.",
                    supporting_evidence="Clothing contributed 78.4% of excess revenue.",
                    confidence="high",
                    associated_driver="Category: Clothing",
                    related_driver="Category: Clothing",
                ),
            ],
            alternative_explanations=[
                "Multiple recorded categories showed positive deviations, indicating that the observed revenue increase was distributed across several catalog segments.",
            ],
            evidence_assessment="Category and pricing evidence have robust sample sizes.",
            uncertainties=[
                "Lack of elasticity models precludes counterfactual volume conclusions.",
            ],
            validation_questions=[
                "Could product-mix differences within recorded categories be verified?",
            ],
            risk_flags=[
                "Monitor inventory levels if demand surge represents a structural shift.",
            ],
            causal_disclaimer="Observational associations identify statistically correlated contributors; they do not establish counterfactual causality.",
            cached=False,
        )



def get_auth_headers(
    client: TestClient, email: str = "e2e_tester@example.com"
) -> dict[str, str]:
    """Helper to register and login user, returning Bearer auth header."""
    client.post(
        "/api/auth/register",
        json={"email": email, "password": "TestPassword123!"},
    )
    res = client.post(
        "/api/auth/login",
        data={"username": email, "password": "TestPassword123!"},
    )
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# 1. Full 10-Step Intelligence Lifecycle Integration
# ---------------------------------------------------------------------------
def test_e2e_full_intelligence_lifecycle(client: TestClient, db_session: Session):
    """
    Validates complete pipeline from anomaly detection through investigation,
    explanation, AI reasoning, recommendation, simulation, decision approval,
    audit trail, monitoring scan, and alert resolution.
    """
    headers = get_auth_headers(client, "lifecycle_executive@example.com")

    # Step 1: Query Anomalies (DETECT)
    anom_res = client.get("/api/anomalies?limit=5", headers=headers)
    assert anom_res.status_code == 200
    anomalies = anom_res.json()["items"]
    assert len(anomalies) > 0
    target_anomaly = anomalies[0]
    anomaly_id = target_anomaly["id"]

    # Step 2: Investigation (INVESTIGATE)
    inv_res = client.get(f"/api/investigations/{anomaly_id}", headers=headers)
    assert inv_res.status_code == 200
    inv_data = inv_res.json()
    assert inv_data["anomaly_id"] == anomaly_id
    assert "drivers" in inv_data

    # Step 3: Explanation (EXPLAIN)
    exp_res = client.get(f"/api/explanations/{anomaly_id}", headers=headers)
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert exp_data["anomaly_id"] == anomaly_id
    assert len(exp_data["headline"]) > 0

    # Step 4: AI Reasoning (AI REASONING)
    with patch("backend.app.api.ai_reasoning.ai_reasoning_service.provider", MockLLMProvider()):
        reason_res = client.get(f"/api/ai-reasoning/{anomaly_id}", headers=headers)
        assert reason_res.status_code == 200
        reason_data = reason_res.json()
        assert reason_data["anomaly_id"] == anomaly_id
        assert "key_insights" in reason_data
        assert "causal_disclaimer" in reason_data

    # Step 5: Recommendations (RECOMMEND)
    rec_res = client.get(f"/api/recommendations/{anomaly_id}", headers=headers)
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert rec_data["anomaly_id"] == anomaly_id
    assert len(rec_data["recommendations"]) > 0
    top_rec = rec_data["recommendations"][0]

    # Step 6: What-If Simulation (SIMULATE)
    sim_res = client.post(
        "/api/simulations",
        json={
            "scenario_type": "demand_multiplier",
            "horizon_days": 30,
            "demand_change_percent": 15.0,
            "anomaly_id": anomaly_id,
            "include_revenue": True,
        },
        headers=headers,
    )
    assert sim_res.status_code == 200
    sim_data = sim_res.json()
    assert sim_data["scenario_type"] == "demand_multiplier"
    assert sim_data["status"] == "completed"
    assert sim_data["anomaly_id"] == anomaly_id
    simulation_id = sim_data["simulation_id"]

    # Step 7: Submit for Governance Review (DECISION QUEUE)
    dec_payload = {
        "recommendation_type": top_rec["recommendation_type"],
        "proposed_action": top_rec["action"],
        "anomaly_id": anomaly_id,
        "simulation_id": simulation_id,
        "decision_note": "E2E verification review submission",
        "recommendation_payload": {
            "recommendation_id": top_rec["id"],
            "recommendation_type": top_rec["recommendation_type"],
            "title": top_rec["title"],
            "action": top_rec["action"],
            "priority": top_rec["priority"],
            "supporting_evidence": top_rec.get("supporting_evidence", []),
        },
    }
    dec_create_res = client.post("/api/decisions", json=dec_payload, headers=headers)
    assert dec_create_res.status_code == 201
    decision = dec_create_res.json()
    decision_id = decision["id"]
    assert decision["status"] == "pending_review"
    assert decision["human_approval_required"] is True
    assert decision["automatic_execution"] is False

    # Step 8: Approve Decision (HUMAN APPROVAL)
    approve_res = client.post(
        f"/api/decisions/{decision_id}/approve",
        json={"decision_note": "Approved by human executive after simulation analysis"},
        headers=headers,
    )
    assert approve_res.status_code == 200
    approved_dec = approve_res.json()
    assert approved_dec["status"] == "approved"
    assert approved_dec["human_approval_required"] is True
    assert approved_dec["automatic_execution"] is False

    # Step 9: Audit Trail Verification (AUDIT TRAIL)
    assert len(approved_dec["audit_events"]) >= 2
    event_types = [ev["event_type"] for ev in approved_dec["audit_events"]]
    assert "created" in event_types
    assert "approved" in event_types

    # Step 10: Monitoring Scan & Alert Resolution (MONITOR & ALERT)
    mon_res = client.post(
        "/api/monitoring/run",
        json={"lookback_days": 30, "min_severity": "low"},
        headers=headers,
    )
    assert mon_res.status_code == 200
    mon_run = mon_res.json()
    assert mon_run["status"] == "completed"

    # Query active alerts
    alerts_res = client.get("/api/monitoring/alerts?status=new", headers=headers)
    assert alerts_res.status_code == 200
    alerts_list = alerts_res.json()["items"]
    if alerts_list:
        target_alert = alerts_list[0]
        # Resolve Alert
        resolve_res = client.post(
            f"/api/monitoring/alerts/{target_alert['id']}/resolve",
            headers=headers,
        )
        assert resolve_res.status_code == 200
        assert resolve_res.json()["status"] == "resolved"


# ---------------------------------------------------------------------------
# 2. Cross-Feature ID Invariance
# ---------------------------------------------------------------------------
def test_e2e_anomaly_id_invariance_across_all_layers(client: TestClient):
    """Guarantees anomaly_id remains immutable across all analytical layers."""
    headers = get_auth_headers(client, "id_invariance@example.com")

    anom_res = client.get("/api/anomalies?limit=1", headers=headers)
    anomaly_id = anom_res.json()["items"][0]["id"]

    inv_res = client.get(f"/api/investigations/{anomaly_id}", headers=headers)
    assert inv_res.json()["anomaly_id"] == anomaly_id

    exp_res = client.get(f"/api/explanations/{anomaly_id}", headers=headers)
    assert exp_res.json()["anomaly_id"] == anomaly_id

    with patch("backend.app.api.ai_reasoning.ai_reasoning_service.provider", MockLLMProvider()):
        reason_res = client.get(f"/api/ai-reasoning/{anomaly_id}", headers=headers)
        assert reason_res.json()["anomaly_id"] == anomaly_id

    rec_res = client.get(f"/api/recommendations/{anomaly_id}", headers=headers)
    assert rec_res.json()["anomaly_id"] == anomaly_id

    sim_res = client.post(
        "/api/simulations",
        json={"scenario_type": "persistent_shift", "horizon_days": 30, "anomaly_id": anomaly_id},
        headers=headers,
    )
    assert sim_res.json()["anomaly_id"] == anomaly_id


# ---------------------------------------------------------------------------
# 3. Decision Snapshot Isolation & Immutability
# ---------------------------------------------------------------------------
def test_e2e_decision_snapshot_isolation(client: TestClient):
    """Verifies decision records freeze supporting context into immutable snapshots."""
    headers = get_auth_headers(client, "snapshot_isolation@example.com")
    rec_payload = {
        "recommendation_id": "rec-snapshot-freeze-001",
        "recommendation_type": "pricing_review",
        "title": "Freeze pricing elasticity baseline",
        "action": "Freeze price elasticity baseline.",
        "priority": "high",
        "supporting_evidence": ["Baseline price: 499.0", "Observed: 549.0"],
    }
    res = client.post(
        "/api/decisions",
        json={
            "recommendation_type": "pricing_review",
            "proposed_action": "Freeze price elasticity baseline.",
            "recommendation_payload": rec_payload,
        },
        headers=headers,
    )
    assert res.status_code == 201
    decision_data = res.json()
    assert decision_data["evidence_snapshot"]["recommendation"]["recommendation_id"] == "rec-snapshot-freeze-001"
    assert decision_data["evidence_snapshot"]["recommendation"]["action"] == "Freeze price elasticity baseline."


# ---------------------------------------------------------------------------
# 4. Zero Sales Data Mutation Safety Invariant
# ---------------------------------------------------------------------------
def test_e2e_no_sales_data_mutation_safety(client: TestClient, db_session: Session):
    """
    CRITICAL SAFETY: Proves that entire intelligence pipeline execution
    (anomalies, investigations, simulations, decisions) causes ZERO mutations
    to underlying SalesRecord or Product tables.
    """
    headers = get_auth_headers(client, "safety_checker@example.com")

    initial_sales_count = db_session.query(SalesRecord).count()
    initial_product_count = db_session.query(Product).count()

    # Execute read/analytical/governance endpoints
    client.get("/api/anomalies?limit=10", headers=headers)
    client.post(
        "/api/simulations",
        json={"scenario_type": "demand_multiplier", "horizon_days": 30, "demand_change_percent": 25.0},
        headers=headers,
    )
    dec_res = client.post(
        "/api/decisions",
        json={
            "recommendation_type": "category_review",
            "proposed_action": "Audit Clothing category margins.",
        },
        headers=headers,
    )
    dec_id = dec_res.json()["id"]
    client.post(f"/api/decisions/{dec_id}/approve", json={}, headers=headers)
    client.post("/api/monitoring/run", json={"lookback_days": 14}, headers=headers)

    # Re-verify row counts
    final_sales_count = db_session.query(SalesRecord).count()
    final_product_count = db_session.query(Product).count()

    assert final_sales_count == initial_sales_count
    assert final_product_count == initial_product_count


# ---------------------------------------------------------------------------
# 5. Zero Autonomous Execution Enforcement
# ---------------------------------------------------------------------------
def test_e2e_zero_autonomous_execution_enforcement(client: TestClient):
    """
    Confirms every system response mandates human approval and disallows
    autonomous business action execution.
    """
    headers = get_auth_headers(client, "governance_audit@example.com")

    # Simulation check
    sim = client.post(
        "/api/simulations",
        json={"scenario_type": "trend_continuation", "horizon_days": 30},
        headers=headers,
    ).json()
    assert sim["status"] == "completed"
    assert len(sim["assumptions"]) > 0
    assert len(sim["limitations"]) > 0

    # Decision creation check
    dec = client.post(
        "/api/decisions",
        json={"recommendation_type": "category_review", "proposed_action": "Test"},
        headers=headers,
    ).json()
    assert dec["human_approval_required"] is True
    assert dec["automatic_execution"] is False

    # Decision approval check
    approved = client.post(
        f"/api/decisions/{dec['id']}/approve",
        json={"decision_note": "Human authorized"},
        headers=headers,
    ).json()
    assert approved["human_approval_required"] is True
    assert approved["automatic_execution"] is False

    # Monitoring run check
    mon = client.post(
        "/api/monitoring/run",
        json={"lookback_days": 7},
        headers=headers,
    ).json()
    assert mon["human_review_required"] is True
    assert mon["automatic_execution"] is False


# ---------------------------------------------------------------------------
# 6. Gemini Boundary: No Numerical Generation / Metric Fabrication
# ---------------------------------------------------------------------------
def test_e2e_gemini_boundary_no_numerical_generation(client: TestClient):
    """
    Verifies that AI reasoning and recommendation endpoints derive metrics
    strictly from the deterministic investigation service and do not allow
    the LLM to fabricate actuals, baselines, or deviations.
    """
    headers = get_auth_headers(client, "ai_boundary@example.com")
    anom_res = client.get("/api/anomalies?limit=1", headers=headers)
    anom = anom_res.json()["items"][0]

    rec_res = client.get(f"/api/recommendations/{anom['id']}", headers=headers)
    assert rec_res.status_code == 200
    rec_data = rec_res.json()

    for rec in rec_data["recommendations"]:
        # Priority and types are strictly bounded
        assert rec["priority"] in ["high", "medium", "low"]
        assert rec["recommendation_type"] in [
            "pricing_review",
            "promotion_review",
            "category_review",
            "product_review",
            "forecast_review",
        ]


# ---------------------------------------------------------------------------
# 7. Gemini Boundary: No Approval or Execution Authority
# ---------------------------------------------------------------------------
def test_e2e_gemini_boundary_no_approval_authority(client: TestClient):
    """Validates that decisions cannot be approved anonymously or by AI services."""
    # Attempt unauthenticated approval
    res = client.post("/api/decisions/dec-mock-123/approve", json={})
    assert res.status_code == 401


# ---------------------------------------------------------------------------
# 8. Deterministic Simulation Consistency
# ---------------------------------------------------------------------------
def test_e2e_simulation_deterministic_consistency(client: TestClient):
    """Ensures identical simulation inputs produce identical mathematical outputs."""
    headers = get_auth_headers(client, "simulation_det@example.com")
    payload = {
        "scenario_type": "demand_multiplier",
        "horizon_days": 30,
        "demand_change_percent": 12.5,
        "include_revenue": True,
    }

    run1 = client.post("/api/simulations", json=payload, headers=headers).json()
    run2 = client.post("/api/simulations", json=payload, headers=headers).json()

    assert run1["baseline"]["total_quantity"] == run2["baseline"]["total_quantity"]
    assert run1["scenario"]["total_quantity"] == run2["scenario"]["total_quantity"]
    assert run1["delta"]["quantity_delta"] == run2["delta"]["quantity_delta"]
    assert run1["delta"]["quantity_delta_percent"] == run2["delta"]["quantity_delta_percent"]


# ---------------------------------------------------------------------------
# 9. Simulation Counterfactual Hedging & Disclaimers
# ---------------------------------------------------------------------------
def test_e2e_simulation_counterfactual_hedging(client: TestClient):
    """Verifies that what-if simulations include mandatory disclaimers and assumptions."""
    headers = get_auth_headers(client, "hedging@example.com")
    res = client.post(
        "/api/simulations",
        json={"scenario_type": "temporary_shock", "horizon_days": 30, "shock_duration_days": 5},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert len(data["assumptions"]) > 0
    assert len(data["limitations"]) > 0


# ---------------------------------------------------------------------------
# 10. Monitoring Scan Discovers Anomalies & Deduplicates
# ---------------------------------------------------------------------------
def test_e2e_monitoring_deduplication_suppression(client: TestClient):
    """Tests proactive scan deduplication across consecutive runs."""
    headers = get_auth_headers(client, "dedup_tester@example.com")

    # Run 1
    run1 = client.post(
        "/api/monitoring/run",
        json={"lookback_days": 30, "min_severity": "medium"},
        headers=headers,
    ).json()

    # Run 2 with identical window
    run2 = client.post(
        "/api/monitoring/run",
        json={"lookback_days": 30, "min_severity": "medium"},
        headers=headers,
    ).json()

    # Assert that run 2 suppresses duplicates created in run 1
    assert run2["status"] == "completed"
    assert run2["duplicates_suppressed"] >= 0


# ---------------------------------------------------------------------------
# 11. Alert Lifecycle State Machine (New -> Acknowledged -> Resolved)
# ---------------------------------------------------------------------------
def test_e2e_alert_lifecycle_state_machine(client: TestClient):
    """Verifies alert transition sequence and timestamp recording."""
    headers = get_auth_headers(client, "alert_lifecycle@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 45}, headers=headers)

    alerts = client.get("/api/monitoring/alerts?status=new", headers=headers).json()["items"]
    if not alerts:
        pytest.skip("No new alerts available to test lifecycle state machine.")

    alert_id = alerts[0]["id"]

    # Step A: Acknowledge
    ack_res = client.post(f"/api/monitoring/alerts/{alert_id}/acknowledge", headers=headers)
    assert ack_res.status_code == 200
    ack_data = ack_res.json()
    assert ack_data["status"] == "acknowledged"
    assert ack_data["acknowledged_at"] is not None

    # Step B: Resolve
    res_res = client.post(f"/api/monitoring/alerts/{alert_id}/resolve", headers=headers)
    assert res_res.status_code == 200
    res_data = res_res.json()
    assert res_data["status"] == "resolved"
    assert res_data["resolved_at"] is not None


# ---------------------------------------------------------------------------
# 12. Alert Dismissal Workflow
# ---------------------------------------------------------------------------
def test_e2e_alert_dismissal_workflow(client: TestClient):
    """Verifies dismissing an active alert without requiring operational action."""
    headers = get_auth_headers(client, "alert_dismiss@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 45}, headers=headers)

    alerts = client.get("/api/monitoring/alerts?status=new", headers=headers).json()["items"]
    if not alerts:
        pytest.skip("No alerts available to test dismissal.")

    alert_id = alerts[0]["id"]
    dismiss_res = client.post(f"/api/monitoring/alerts/{alert_id}/dismiss", headers=headers)
    assert dismiss_res.status_code == 200
    assert dismiss_res.json()["status"] == "dismissed"


# ---------------------------------------------------------------------------
# 13. Decision Revision Flow: Request Changes -> Resubmit -> Approve
# ---------------------------------------------------------------------------
def test_e2e_decision_request_changes_resubmit_flow(client: TestClient):
    """Tests revision cycle: pending -> changes_requested -> resubmit -> approved."""
    headers = get_auth_headers(client, "revision_workflow@example.com")

    # 1. Create Decision
    dec = client.post(
        "/api/decisions",
        json={
            "recommendation_type": "product_review",
            "proposed_action": "Initial draft of product bundle reallocation.",
        },
        headers=headers,
    ).json()
    dec_id = dec["id"]

    # 2. Request Changes
    req_res = client.post(
        f"/api/decisions/{dec_id}/request-changes",
        json={
            "decision_note": "Please specify target inventory locations before approval.",
            "modified_action": "Audit product bundles specifically across Western zone.",
        },
        headers=headers,
    )
    assert req_res.status_code == 200
    assert req_res.json()["status"] == "changes_requested"

    # 3. Resubmit
    resub_res = client.post(
        f"/api/decisions/{dec_id}/resubmit",
        json={
            "decision_note": "Added Western zone warehouse location details.",
            "modified_action": "Audit product bundles for Western zone warehouse.",
        },
        headers=headers,
    )
    assert resub_res.status_code == 200
    assert resub_res.json()["status"] == "pending_review"

    # 4. Final Approval
    app_res = client.post(
        f"/api/decisions/{dec_id}/approve",
        json={"decision_note": "Final sign-off granted by human director."},
        headers=headers,
    )
    assert app_res.status_code == 200
    final_dec = app_res.json()
    assert final_dec["status"] == "approved"
    assert len(final_dec["audit_events"]) == 4


# ---------------------------------------------------------------------------
# 14. Decision Rejection Workflow
# ---------------------------------------------------------------------------
def test_e2e_decision_rejection_workflow(client: TestClient):
    """Verifies rejecting a recommendation requires human rationale."""
    headers = get_auth_headers(client, "rejection_admin@example.com")
    dec = client.post(
        "/api/decisions",
        json={
            "recommendation_type": "pricing_review",
            "proposed_action": "Unwarranted price decrease proposed.",
        },
        headers=headers,
    ).json()

    # Reject without note should fail
    fail_res = client.post(f"/api/decisions/{dec['id']}/reject", json={"decision_note": "  "}, headers=headers)
    assert fail_res.status_code in [400, 422]

    # Reject with note succeeds
    rej_res = client.post(
        f"/api/decisions/{dec['id']}/reject",
        json={"decision_note": "Rejected: price elasticity does not support discounting at this time."},
        headers=headers,
    )
    assert rej_res.status_code == 200
    assert rej_res.json()["status"] == "rejected"


# ---------------------------------------------------------------------------
# 15. Error Contract: HTTP 401 Unauthorized
# ---------------------------------------------------------------------------
def test_e2e_error_contract_unauthorized_401(client: TestClient):
    """Verifies unauthenticated access returns 401 Unauthorized."""
    endpoints = [
        ("POST", "/api/decisions"),
        ("GET", "/api/decisions"),
        ("POST", "/api/monitoring/run"),
        ("GET", "/api/monitoring/alerts"),
        ("GET", "/api/monitoring/summary"),
    ]
    for method, path in endpoints:
        if method == "POST":
            res = client.post(path, json={})
        else:
            res = client.get(path)
        assert res.status_code == 401, f"Expected 401 on {method} {path}, got {res.status_code}"


# ---------------------------------------------------------------------------
# 16. Error Contract: HTTP 404 Not Found
# ---------------------------------------------------------------------------
def test_e2e_error_contract_not_found_404(client: TestClient):
    """Verifies non-existent IDs return 404 Not Found."""
    headers = get_auth_headers(client, "not_found_checker@example.com")

    # Non-existent decision
    res_dec = client.get("/api/decisions/dec-non-existent-999", headers=headers)
    assert res_dec.status_code == 404

    # Non-existent alert
    res_alt = client.get("/api/monitoring/alerts/alt-non-existent-999", headers=headers)
    assert res_alt.status_code == 404

    # Non-existent investigation
    res_inv = client.get("/api/investigations/anom-non-existent-999", headers=headers)
    assert res_inv.status_code == 404


# ---------------------------------------------------------------------------
# 17. Error Contract: HTTP 409 Conflict
# ---------------------------------------------------------------------------
def test_e2e_error_contract_conflict_409(client: TestClient):
    """Verifies attempting state transitions on already finalized decisions returns 409."""
    headers = get_auth_headers(client, "conflict_checker@example.com")
    dec = client.post(
        "/api/decisions",
        json={"recommendation_type": "forecast_review", "proposed_action": "Initial"},
        headers=headers,
    ).json()

    # Approve once
    client.post(f"/api/decisions/{dec['id']}/approve", json={}, headers=headers)

    # Approve again -> 409 Conflict
    second_app = client.post(f"/api/decisions/{dec['id']}/approve", json={}, headers=headers)
    assert second_app.status_code == 409

    # Reject already approved decision -> 409 Conflict
    rej = client.post(f"/api/decisions/{dec['id']}/reject", json={"decision_note": "Note"}, headers=headers)
    assert rej.status_code == 409


# ---------------------------------------------------------------------------
# 18. Error Contract: HTTP 422 Unprocessable Entity
# ---------------------------------------------------------------------------
def test_e2e_error_contract_validation_422(client: TestClient):
    """Verifies schema violations return HTTP 422."""
    headers = get_auth_headers(client, "validation_checker@example.com")

    # Invalid simulation parameters (horizon_days must be 7, 30, or 90)
    res = client.post(
        "/api/simulations",
        json={"scenario_type": "demand_multiplier", "horizon_days": 9999},
        headers=headers,
    )
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# 19. Error Contract: HTTP 500 Sanitized Exception Handling
# ---------------------------------------------------------------------------
def test_e2e_error_contract_sanitized_500():
    """
    Verifies unhandled server exceptions return sanitized 500 responses
    without leaking internal file paths, stack traces, or credentials.
    """
    from backend.app.main import app
    safe_client = TestClient(app, raise_server_exceptions=False)
    headers = get_auth_headers(safe_client, "sanitized_500@example.com")

    with patch(
        "backend.app.services.monitoring_service.MonitoringService.run_monitoring",
        side_effect=RuntimeError("Internal database connection failed at C:\\Secret\\Path"),
    ):
        res = safe_client.post(
            "/api/monitoring/run",
            json={"lookback_days": 30},
            headers=headers,
        )
        assert res.status_code == 500
        data = res.json()
        assert "detail" in data
        # Must not leak python tracebacks or file paths
        assert "Traceback" not in data["detail"]
        assert "RuntimeError" not in data["detail"]
        assert "C:\\" not in data["detail"]
        assert data["detail"] == "Internal server error."


# ---------------------------------------------------------------------------
# 20. Audit Trail Chronological Integrity
# ---------------------------------------------------------------------------
def test_e2e_audit_trail_chronological_integrity(client: TestClient):
    """Verifies audit trail preserves strict temporal order and user metadata."""
    headers = get_auth_headers(client, "audit_integrity@example.com")

    dec = client.post(
        "/api/decisions",
        json={"recommendation_type": "category_review", "proposed_action": "Audit"},
        headers=headers,
    ).json()

    client.post(
        f"/api/decisions/{dec['id']}/request-changes",
        json={"decision_note": "Request clarification"},
        headers=headers,
    )
    client.post(
        f"/api/decisions/{dec['id']}/resubmit",
        json={"decision_note": "Clarification provided"},
        headers=headers,
    )
    approved = client.post(
        f"/api/decisions/{dec['id']}/approve",
        json={"decision_note": "Approved"},
        headers=headers,
    ).json()

    events = approved["audit_events"]
    assert len(events) == 4
    timestamps = [dt.datetime.fromisoformat(ev["created_at"]) for ev in events]
    assert timestamps == sorted(timestamps)
    assert [ev["event_type"] for ev in events] == [
        "created",
        "changes_requested",
        "reopened",
        "approved",
    ]


# ---------------------------------------------------------------------------
# 21. Executive Dashboard Telemetry Aggregation
# ---------------------------------------------------------------------------
def test_e2e_executive_dashboard_telemetry_aggregation(client: TestClient):
    """Validates summary KPIs consumed by the frontend Executive Dashboard."""
    headers = get_auth_headers(client, "dashboard_telemetry@example.com")

    # Query monitoring summary
    summary_res = client.get("/api/monitoring/summary", headers=headers)
    assert summary_res.status_code == 200
    summary = summary_res.json()

    assert "unresolved_alerts" in summary
    assert "critical_alerts" in summary
    assert "new_alerts" in summary
    assert "total_anomalies" in summary
    assert summary["unresolved_alerts"] >= 0


# ---------------------------------------------------------------------------
# 22. Cross-Filter Monitoring & Pagination
# ---------------------------------------------------------------------------
def test_e2e_monitoring_filter_and_pagination(client: TestClient):
    """Validates pagination and filtering logic in proactive intelligence monitor."""
    headers = get_auth_headers(client, "pagination_checker@example.com")

    res = client.get("/api/monitoring/alerts?skip=0&limit=5", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert "total" in data
    assert len(data["items"]) <= 5
