from __future__ import annotations

import datetime as dt
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.database.models import DecisionAuditEvent, DecisionRecord, Product, SalesRecord
from backend.app.schemas.decisions import DecisionStatus


def get_auth_headers(client: TestClient, email: str = "decision_tester@example.com") -> dict[str, str]:
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
# 1. Create Decision Record
# ---------------------------------------------------------------------------
def test_create_decision(client: TestClient):
    headers = get_auth_headers(client, "dec_create_user@example.com")
    payload = {
        "recommendation_type": "pricing_review",
        "proposed_action": "Review unit pricing for Clothing category.",
        "anomaly_id": "anom-20251228-sal-agg",
        "decision_note": "Initial review submission",
    }
    res = client.post("/api/decisions", json=payload, headers=headers)
    assert res.status_code == 201
    data = res.json()
    assert data["status"] == "pending_review"
    assert data["proposed_action"] == payload["proposed_action"]
    assert data["recommendation_type"] == "pricing_review"
    assert data["human_approval_required"] is True
    assert data["automatic_execution"] is False
    assert len(data["audit_events"]) == 1
    assert data["audit_events"][0]["event_type"] == "created"


# ---------------------------------------------------------------------------
# 2. Create Requires Authentication
# ---------------------------------------------------------------------------
def test_create_requires_authentication(client: TestClient):
    payload = {
        "recommendation_type": "pricing_review",
        "proposed_action": "Review unit pricing.",
    }
    res = client.post("/api/decisions", json=payload)
    assert res.status_code == 401


# ---------------------------------------------------------------------------
# 3. Get Decision by ID
# ---------------------------------------------------------------------------
def test_get_decision(client: TestClient):
    headers = get_auth_headers(client, "dec_get_user@example.com")
    payload = {
        "recommendation_type": "inventory_review",
        "proposed_action": "Verify safety stock levels for key SKUs.",
    }
    create_res = client.post("/api/decisions", json=payload, headers=headers)
    assert create_res.status_code == 201
    dec_id = create_res.json()["id"]

    get_res = client.get(f"/api/decisions/{dec_id}", headers=headers)
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["id"] == dec_id
    assert data["status"] == "pending_review"
    assert "evidence_snapshot" in data


# ---------------------------------------------------------------------------
# 4. List Decisions
# ---------------------------------------------------------------------------
def test_list_decisions(client: TestClient):
    headers = get_auth_headers(client, "dec_list_user@example.com")
    for i in range(3):
        client.post(
            "/api/decisions",
            json={
                "recommendation_type": "product_review",
                "proposed_action": f"Review product SKU {i}",
            },
            headers=headers,
        )

    res = client.get("/api/decisions", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 3
    # Check ordering (newest first)
    assert data[0]["created_at"] >= data[1]["created_at"]


# ---------------------------------------------------------------------------
# 5. Approve Decision
# ---------------------------------------------------------------------------
def test_approve_decision(client: TestClient):
    headers = get_auth_headers(client, "dec_approve_user@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Review prices"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]

    app_res = client.post(
        f"/api/decisions/{dec_id}/approve",
        json={"decision_note": "Approved for pricing committee review"},
        headers=headers,
    )
    assert app_res.status_code == 200
    data = app_res.json()
    assert data["status"] == "approved"
    assert data["reviewed_at"] is not None
    assert data["decision_note"] == "Approved for pricing committee review"


# ---------------------------------------------------------------------------
# 6. Reject Decision
# ---------------------------------------------------------------------------
def test_reject_decision(client: TestClient):
    headers = get_auth_headers(client, "dec_reject_user@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "promotion_review", "proposed_action": "Pause promo"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]

    rej_res = client.post(
        f"/api/decisions/{dec_id}/reject",
        json={"decision_note": "Promo contract is non-cancellable"},
        headers=headers,
    )
    assert rej_res.status_code == 200
    data = rej_res.json()
    assert data["status"] == "rejected"
    assert data["rationale"] == "Promo contract is non-cancellable"


# ---------------------------------------------------------------------------
# 7. Request Changes
# ---------------------------------------------------------------------------
def test_request_changes(client: TestClient):
    headers = get_auth_headers(client, "dec_changes_user@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "category_review", "proposed_action": "Reallocate budget"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]

    chg_res = client.post(
        f"/api/decisions/{dec_id}/request-changes",
        json={"decision_note": "Please specify which sub-categories should be included"},
        headers=headers,
    )
    assert chg_res.status_code == 200
    data = chg_res.json()
    assert data["status"] == "changes_requested"
    assert data["decision_note"] == "Please specify which sub-categories should be included"


# ---------------------------------------------------------------------------
# 8. Approval Note Recorded
# ---------------------------------------------------------------------------
def test_approval_note(client: TestClient):
    headers = get_auth_headers(client, "dec_appnote_user@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "forecast_review", "proposed_action": "Re-evaluate model"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]

    app_res = client.post(
        f"/api/decisions/{dec_id}/approve",
        json={"decision_note": "Executive sign-off given for Q1 planning"},
        headers=headers,
    )
    assert app_res.status_code == 200
    assert app_res.json()["decision_note"] == "Executive sign-off given for Q1 planning"


# ---------------------------------------------------------------------------
# 9. Rejection Requires Rationale (Missing/empty note returns 400)
# ---------------------------------------------------------------------------
def test_rejection_requires_rationale(client: TestClient):
    headers = get_auth_headers(client, "dec_rej_req_user@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "product_review", "proposed_action": "Delist SKU"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]

    # Attempt rejection with empty note
    rej_res = client.post(
        f"/api/decisions/{dec_id}/reject",
        json={"decision_note": "   "},
        headers=headers,
    )
    assert rej_res.status_code == 400


# ---------------------------------------------------------------------------
# 10. Changes Request Requires Rationale
# ---------------------------------------------------------------------------
def test_changes_request_requires_rationale(client: TestClient):
    headers = get_auth_headers(client, "dec_chg_req_user@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "demand_monitoring", "proposed_action": "Monitor daily"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]

    chg_res = client.post(
        f"/api/decisions/{dec_id}/request-changes",
        json={"decision_note": ""},
        headers=headers,
    )
    assert chg_res.status_code == 400


# ---------------------------------------------------------------------------
# 11. Modified Recommendation Preserved
# ---------------------------------------------------------------------------
def test_modified_recommendation_preserved(client: TestClient):
    headers = get_auth_headers(client, "dec_mod_user@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Original pricing text"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]

    app_res = client.post(
        f"/api/decisions/{dec_id}/approve",
        json={
            "modified_action": "Review pricing for Clothing category before next planning cycle",
            "decision_note": "Approved with refined scope",
        },
        headers=headers,
    )
    assert app_res.status_code == 200
    data = app_res.json()
    assert data["modified_action"] == "Review pricing for Clothing category before next planning cycle"
    assert data["proposed_action"] == "Original pricing text"


# ---------------------------------------------------------------------------
# 12. Original Recommendation Immutable
# ---------------------------------------------------------------------------
def test_original_recommendation_immutable(client: TestClient):
    headers = get_auth_headers(client, "dec_imm_user@example.com")
    original = "Immutable original text that must never change"
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "category_review", "proposed_action": original},
        headers=headers,
    )
    dec_id = create_res.json()["id"]

    client.post(
        f"/api/decisions/{dec_id}/approve",
        json={"modified_action": "Completely new wording"},
        headers=headers,
    )

    get_res = client.get(f"/api/decisions/{dec_id}", headers=headers)
    assert get_res.json()["proposed_action"] == original


# ---------------------------------------------------------------------------
# 13. Evidence Snapshot Created
# ---------------------------------------------------------------------------
def test_evidence_snapshot_created(client: TestClient):
    headers = get_auth_headers(client, "dec_ev_snap_user@example.com")
    payload = {
        "recommendation_type": "pricing_review",
        "proposed_action": "Check price shift",
        "recommendation_payload": {
            "recommendation_type": "pricing_review",
            "title": "Pricing Audit",
            "action": "Check price shift",
            "supporting_evidence": ["Unit price shifted +11.1%"],
            "assumptions": ["Constant demand"],
            "tradeoffs": ["Volume risk"],
            "limitations": ["No elasticity curve"],
        },
    }
    res = client.post("/api/decisions", json=payload, headers=headers)
    assert res.status_code == 201
    snap = res.json()["evidence_snapshot"]
    assert "recommendation" in snap
    assert snap["recommendation"]["title"] == "Pricing Audit"


# ---------------------------------------------------------------------------
# 14. Simulation Snapshot Created
# ---------------------------------------------------------------------------
def test_simulation_snapshot_created(client: TestClient):
    headers = get_auth_headers(client, "dec_sim_snap_user@example.com")
    # Run a simulation first
    sim_res = client.post(
        "/api/simulations",
        json={"scenario_type": "demand_multiplier", "demand_change_percent": 10.0, "horizon_days": 7},
        headers=headers,
    )
    sim_id = sim_res.json()["simulation_id"]

    # Now create decision referencing this simulation
    dec_res = client.post(
        "/api/decisions",
        json={
            "recommendation_type": "demand_monitoring",
            "proposed_action": "Monitor +10% lift",
            "simulation_id": sim_id,
        },
        headers=headers,
    )
    assert dec_res.status_code == 201
    sim_snap = dec_res.json()["simulation_snapshot"]
    assert sim_snap is not None
    assert sim_snap["simulation_id"] == sim_id
    assert sim_snap["scenario_type"] == "demand_multiplier"


# ---------------------------------------------------------------------------
# 15. Anomaly Snapshot Created
# ---------------------------------------------------------------------------
def test_anomaly_snapshot_created(client: TestClient):
    headers = get_auth_headers(client, "dec_anom_snap_user@example.com")
    dec_res = client.post(
        "/api/decisions",
        json={
            "recommendation_type": "pricing_review",
            "proposed_action": "Investigate pricing anomaly",
            "anomaly_id": "anom-20251228-sal-agg",
        },
        headers=headers,
    )
    assert dec_res.status_code == 201
    snap = dec_res.json()["evidence_snapshot"]
    assert snap["anomaly_context"] is not None
    assert snap["anomaly_context"]["anomaly_id"] == "anom-20251228-sal-agg"


# ---------------------------------------------------------------------------
# 16. Audit Event on Creation
# ---------------------------------------------------------------------------
def test_audit_event_on_creation(client: TestClient):
    headers = get_auth_headers(client, "dec_aud_create_user@example.com")
    res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Action"},
        headers=headers,
    )
    events = res.json()["audit_events"]
    assert len(events) == 1
    assert events[0]["event_type"] == "created"
    assert events[0]["new_status"] == "pending_review"


# ---------------------------------------------------------------------------
# 17. Audit Event on Approval
# ---------------------------------------------------------------------------
def test_audit_event_on_approval(client: TestClient):
    headers = get_auth_headers(client, "dec_aud_app_user@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Action"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]
    app_res = client.post(f"/api/decisions/{dec_id}/approve", headers=headers)

    events = app_res.json()["audit_events"]
    assert len(events) == 2
    assert events[1]["event_type"] == "approved"
    assert events[1]["previous_status"] == "pending_review"
    assert events[1]["new_status"] == "approved"


# ---------------------------------------------------------------------------
# 18. Audit Event on Rejection
# ---------------------------------------------------------------------------
def test_audit_event_on_rejection(client: TestClient):
    headers = get_auth_headers(client, "dec_aud_rej_user@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Action"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]
    rej_res = client.post(
        f"/api/decisions/{dec_id}/reject",
        json={"decision_note": "Rationale for rejection"},
        headers=headers,
    )

    events = rej_res.json()["audit_events"]
    assert len(events) == 2
    assert events[1]["event_type"] == "rejected"
    assert events[1]["new_status"] == "rejected"


# ---------------------------------------------------------------------------
# 19. Audit Event on Request Changes
# ---------------------------------------------------------------------------
def test_audit_event_on_request_changes(client: TestClient):
    headers = get_auth_headers(client, "dec_aud_chg_user@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Action"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]
    chg_res = client.post(
        f"/api/decisions/{dec_id}/request-changes",
        json={"decision_note": "Please update wording"},
        headers=headers,
    )

    events = chg_res.json()["audit_events"]
    assert len(events) == 2
    assert events[1]["event_type"] == "changes_requested"
    assert events[1]["new_status"] == "changes_requested"


# ---------------------------------------------------------------------------
# 20. Finalized Decision Cannot Be Mutated
# ---------------------------------------------------------------------------
def test_finalized_decision_cannot_be_mutated(client: TestClient):
    headers = get_auth_headers(client, "dec_fin_imm_user@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Action"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]
    client.post(f"/api/decisions/{dec_id}/approve", headers=headers)

    # Attempt to reject an already approved decision
    rej_res = client.post(
        f"/api/decisions/{dec_id}/reject",
        json={"decision_note": "Attempt reject"},
        headers=headers,
    )
    assert rej_res.status_code == 409


# ---------------------------------------------------------------------------
# 21. Finalized Decision Returns 409 Conflict
# ---------------------------------------------------------------------------
def test_finalized_decision_returns_409(client: TestClient):
    headers = get_auth_headers(client, "dec_fin_409_user@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "product_review", "proposed_action": "Action"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]
    client.post(
        f"/api/decisions/{dec_id}/reject",
        json={"decision_note": "Rejected initially"},
        headers=headers,
    )

    # Attempt to approve
    app_res = client.post(f"/api/decisions/{dec_id}/approve", headers=headers)
    assert app_res.status_code == 409


# ---------------------------------------------------------------------------
# 22. Unauthorized Decision Access Blocked (403/404)
# ---------------------------------------------------------------------------
def test_unauthorized_decision_access_blocked(client: TestClient):
    user1_headers = get_auth_headers(client, "dec_owner1@example.com")
    user2_headers = get_auth_headers(client, "dec_owner2@example.com")

    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Secret action"},
        headers=user1_headers,
    )
    dec_id = create_res.json()["id"]

    # User 2 tries to read user 1's decision
    bad_get = client.get(f"/api/decisions/{dec_id}", headers=user2_headers)
    assert bad_get.status_code in (403, 404)


# ---------------------------------------------------------------------------
# 23. Another User Cannot Modify Decision
# ---------------------------------------------------------------------------
def test_another_user_cannot_modify_decision(client: TestClient):
    user1_headers = get_auth_headers(client, "dec_mod_own1@example.com")
    user2_headers = get_auth_headers(client, "dec_mod_own2@example.com")

    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Action"},
        headers=user1_headers,
    )
    dec_id = create_res.json()["id"]

    # User 2 tries to approve user 1's decision
    bad_app = client.post(f"/api/decisions/{dec_id}/approve", headers=user2_headers)
    assert bad_app.status_code in (403, 404)


# ---------------------------------------------------------------------------
# 24. No SalesRecords Mutation
# ---------------------------------------------------------------------------
def test_no_sales_records_mutation(client: TestClient, db_session: Session):
    headers = get_auth_headers(client, "dec_no_sales_user@example.com")
    sales_count_before = db_session.query(SalesRecord).count()

    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "inventory_review", "proposed_action": "Action"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]
    client.post(f"/api/decisions/{dec_id}/approve", headers=headers)

    sales_count_after = db_session.query(SalesRecord).count()
    assert sales_count_before == sales_count_after


# ---------------------------------------------------------------------------
# 25. No Product Mutation
# ---------------------------------------------------------------------------
def test_no_product_mutation(client: TestClient, db_session: Session):
    headers = get_auth_headers(client, "dec_no_prod_user@example.com")
    prod_count_before = db_session.query(Product).count()

    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Change price"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]
    client.post(f"/api/decisions/{dec_id}/approve", headers=headers)

    prod_count_after = db_session.query(Product).count()
    assert prod_count_before == prod_count_after


# ---------------------------------------------------------------------------
# 26. No Automatic Business Execution
# ---------------------------------------------------------------------------
def test_no_automatic_business_execution(client: TestClient):
    headers = get_auth_headers(client, "dec_no_auto_exec@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Execute order"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]
    app_res = client.post(f"/api/decisions/{dec_id}/approve", headers=headers)

    assert app_res.json()["automatic_execution"] is False
    assert app_res.json()["human_approval_required"] is True


# ---------------------------------------------------------------------------
# 27. Gemini Cannot Change Approval Status
# ---------------------------------------------------------------------------
def test_gemini_cannot_change_approval_status(client: TestClient):
    headers = get_auth_headers(client, "dec_no_llm_status@example.com")
    # Verify that decision status can only change through the dedicated human API
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "forecast_review", "proposed_action": "Review"},
        headers=headers,
    )
    data = create_res.json()
    assert data["status"] == "pending_review"


# ---------------------------------------------------------------------------
# 28. Audit Trail Strictly Chronological
# ---------------------------------------------------------------------------
def test_audit_trail_chronological(client: TestClient):
    headers = get_auth_headers(client, "dec_chrono_user@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Action"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]

    client.post(
        f"/api/decisions/{dec_id}/request-changes",
        json={"decision_note": "Please update scope"},
        headers=headers,
    )
    client.post(
        f"/api/decisions/{dec_id}/resubmit",
        json={"modified_action": "Updated action"},
        headers=headers,
    )
    app_res = client.post(f"/api/decisions/{dec_id}/approve", headers=headers)

    events = app_res.json()["audit_events"]
    assert len(events) == 4
    for i in range(1, len(events)):
        assert events[i]["created_at"] >= events[i - 1]["created_at"]


# ---------------------------------------------------------------------------
# 29. State Transition Validation
# ---------------------------------------------------------------------------
def test_decision_state_transition_validation(client: TestClient):
    headers = get_auth_headers(client, "dec_trans_val_user@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "category_review", "proposed_action": "Action"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]

    # Trying to resubmit while still in pending_review
    resubmit_res = client.post(f"/api/decisions/{dec_id}/resubmit", headers=headers)
    assert resubmit_res.status_code == 409


# ---------------------------------------------------------------------------
# 30. Human Approval Flag Always True
# ---------------------------------------------------------------------------
def test_human_approval_flag_always_true(client: TestClient):
    headers = get_auth_headers(client, "dec_flag_true@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Action"},
        headers=headers,
    )
    assert create_res.json()["human_approval_required"] is True

    dec_id = create_res.json()["id"]
    app_res = client.post(f"/api/decisions/{dec_id}/approve", headers=headers)
    assert app_res.json()["human_approval_required"] is True


# ---------------------------------------------------------------------------
# 31. Automatic Execution Always False
# ---------------------------------------------------------------------------
def test_automatic_execution_always_false(client: TestClient):
    headers = get_auth_headers(client, "dec_auto_false@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Action"},
        headers=headers,
    )
    assert create_res.json()["automatic_execution"] is False

    dec_id = create_res.json()["id"]
    app_res = client.post(f"/api/decisions/{dec_id}/approve", headers=headers)
    assert app_res.json()["automatic_execution"] is False


# ---------------------------------------------------------------------------
# 32. Modified Action Length Validation (> 2000 chars rejected)
# ---------------------------------------------------------------------------
def test_modified_action_length_validation(client: TestClient):
    headers = get_auth_headers(client, "dec_len_mod@example.com")
    oversized = "A" * 2005
    res = client.post(
        "/api/decisions",
        json={
            "recommendation_type": "pricing_review",
            "proposed_action": "Valid action",
            "modified_action": oversized,
        },
        headers=headers,
    )
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# 33. Decision Note Length Validation (> 2000 chars rejected)
# ---------------------------------------------------------------------------
def test_decision_note_length_validation(client: TestClient):
    headers = get_auth_headers(client, "dec_len_note@example.com")
    oversized = "B" * 2005
    res = client.post(
        "/api/decisions",
        json={
            "recommendation_type": "pricing_review",
            "proposed_action": "Valid action",
            "decision_note": oversized,
        },
        headers=headers,
    )
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# 34. Decision Snapshot Consistency
# ---------------------------------------------------------------------------
def test_decision_snapshot_consistency(client: TestClient):
    headers = get_auth_headers(client, "dec_snap_cons@example.com")
    payload = {
        "recommendation_type": "pricing_review",
        "proposed_action": "Review action text",
        "recommendation_id": "rec-test-1234",
    }
    create_res = client.post("/api/decisions", json=payload, headers=headers)
    assert create_res.status_code == 201
    snap = create_res.json()["evidence_snapshot"]
    assert snap["recommendation"]["recommendation_id"] == "rec-test-1234"
    assert snap["recommendation"]["action"] == "Review action text"


# ---------------------------------------------------------------------------
# 35. Repeated Approval Handled Safely (Returns 409)
# ---------------------------------------------------------------------------
def test_repeated_approval_handled_safely(client: TestClient):
    headers = get_auth_headers(client, "dec_repeat_app@example.com")
    create_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Action"},
        headers=headers,
    )
    dec_id = create_res.json()["id"]

    app1 = client.post(f"/api/decisions/{dec_id}/approve", headers=headers)
    assert app1.status_code == 200

    # Second approval must return 409
    app2 = client.post(f"/api/decisions/{dec_id}/approve", headers=headers)
    assert app2.status_code == 409
