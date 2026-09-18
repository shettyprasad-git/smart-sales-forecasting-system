from __future__ import annotations

import datetime as dt
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.database.models import (
    DecisionRecord,
    MonitoringAlert,
    MonitoringRun,
    Product,
    SalesRecord,
)
from backend.app.schemas.anomalies import AnomalyItem
from backend.app.schemas.monitoring import AlertStatus, AlertType


def get_auth_headers(
    client: TestClient, email: str = "monitoring_tester@example.com"
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
# 1. Monitoring Run Succeeds
# ---------------------------------------------------------------------------
def test_monitoring_run_succeeds(client: TestClient):
    headers = get_auth_headers(client, "mon_run_user@example.com")
    res = client.post(
        "/api/monitoring/run",
        json={"lookback_days": 30, "min_severity": "medium"},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert data["run_id"].startswith("run-")
    assert data["human_review_required"] is True
    assert data["automatic_execution"] is False
    assert "summary" in data
    assert data["summary"]["unresolved_alerts"] >= 0


# ---------------------------------------------------------------------------
# 2. Monitoring Requires Authentication
# ---------------------------------------------------------------------------
def test_monitoring_requires_authentication(client: TestClient):
    res = client.post("/api/monitoring/run", json={})
    assert res.status_code == 401


# ---------------------------------------------------------------------------
# 3. Anomaly Becomes Alert When Material
# ---------------------------------------------------------------------------
def test_anomaly_becomes_alert_when_material(client: TestClient):
    headers = get_auth_headers(client, "material_user@example.com")
    res = client.post(
        "/api/monitoring/run",
        json={"lookback_days": 30, "min_severity": "medium"},
        headers=headers,
    )
    assert res.status_code == 200
    data = res.json()
    if data["alerts_created"] > 0:
        list_res = client.get("/api/monitoring/alerts", headers=headers)
        alerts = list_res.json()["items"]
        assert len(alerts) > 0
        for a in alerts:
            if a["alert_type"] == "repeated_anomaly":
                assert a["priority"] in ("urgent", "high", "medium")
            else:
                assert a["severity"] in ("critical", "high", "medium")
                assert a["priority"] in ("urgent", "high", "medium")


# ---------------------------------------------------------------------------
# 4. Low Severity Threshold Behavior
# ---------------------------------------------------------------------------
def test_low_severity_threshold_behavior(client: TestClient):
    headers = get_auth_headers(client, "low_thresh_user@example.com")
    # Scan with high severity filter
    res = client.post(
        "/api/monitoring/run",
        json={"lookback_days": 30, "min_severity": "high"},
        headers=headers,
    )
    assert res.status_code == 200
    list_res = client.get("/api/monitoring/alerts", headers=headers)
    alerts = list_res.json()["items"]
    for a in alerts:
        if a["alert_type"] != "repeated_anomaly":
            assert a["severity"] in ("high", "critical")


# ---------------------------------------------------------------------------
# 5. Critical Alert Creation
# ---------------------------------------------------------------------------
def test_critical_alert_creation(client: TestClient, db_session: Session):
    headers = get_auth_headers(client, "crit_user@example.com")
    # Run scan
    client.post("/api/monitoring/run", json={"lookback_days": 60}, headers=headers)
    list_res = client.get(
        "/api/monitoring/alerts?severity=critical", headers=headers
    )
    assert list_res.status_code == 200
    crits = list_res.json()["items"]
    for c in crits:
        assert c["severity"] == "critical"
        assert c["priority"] == "urgent"


# ---------------------------------------------------------------------------
# 6. Duplicate Suppression
# ---------------------------------------------------------------------------
def test_duplicate_suppression(client: TestClient):
    headers = get_auth_headers(client, "dup_user@example.com")
    # First run
    res1 = client.post(
        "/api/monitoring/run",
        json={"lookback_days": 30, "min_severity": "medium"},
        headers=headers,
    )
    assert res1.status_code == 200
    created1 = res1.json()["alerts_created"]

    # Second run immediately after
    res2 = client.post(
        "/api/monitoring/run",
        json={"lookback_days": 30, "min_severity": "medium"},
        headers=headers,
    )
    assert res2.status_code == 200
    created2 = res2.json()["alerts_created"]
    suppressed2 = res2.json()["duplicates_suppressed"]

    assert created2 == 0
    assert suppressed2 >= created1


# ---------------------------------------------------------------------------
# 7. Repeated Anomaly Handling
# ---------------------------------------------------------------------------
def test_repeated_anomaly_handling(client: TestClient):
    headers = get_auth_headers(client, "repeated_user@example.com")
    res = client.post(
        "/api/monitoring/run",
        json={"lookback_days": 90, "min_severity": "medium"},
        headers=headers,
    )
    assert res.status_code == 200
    list_res = client.get(
        "/api/monitoring/alerts?alert_type=repeated_anomaly", headers=headers
    )
    assert list_res.status_code == 200
    repeated = list_res.json()["items"]
    for r in repeated:
        assert r["alert_type"] == "repeated_anomaly"
        assert "Repeated Anomaly" in r["title"]


# ---------------------------------------------------------------------------
# 8. Alert Listing
# ---------------------------------------------------------------------------
def test_alert_listing(client: TestClient):
    headers = get_auth_headers(client, "listing_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    res = client.get("/api/monitoring/alerts", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert "total" in data
    assert isinstance(data["items"], list)


# ---------------------------------------------------------------------------
# 9. Status Filtering
# ---------------------------------------------------------------------------
def test_status_filtering(client: TestClient):
    headers = get_auth_headers(client, "status_filter_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    res = client.get("/api/monitoring/alerts?status=new", headers=headers)
    assert res.status_code == 200
    for a in res.json()["items"]:
        assert a["status"] == "new"


# ---------------------------------------------------------------------------
# 10. Severity Filtering
# ---------------------------------------------------------------------------
def test_severity_filtering(client: TestClient):
    headers = get_auth_headers(client, "sev_filter_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    res = client.get("/api/monitoring/alerts?severity=high", headers=headers)
    assert res.status_code == 200
    for a in res.json()["items"]:
        assert a["severity"] == "high"


# ---------------------------------------------------------------------------
# 11. Alert Retrieval
# ---------------------------------------------------------------------------
def test_alert_retrieval(client: TestClient):
    headers = get_auth_headers(client, "retrieve_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    alerts = client.get("/api/monitoring/alerts", headers=headers).json()["items"]
    assert len(alerts) > 0
    alert_id = alerts[0]["id"]

    res = client.get(f"/api/monitoring/alerts/{alert_id}", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == alert_id
    assert "evidence_snapshot" in data
    assert data["human_review_required"] is True
    assert data["automatic_execution"] is False


# ---------------------------------------------------------------------------
# 12. Ownership Enforcement
# ---------------------------------------------------------------------------
def test_ownership_enforcement(client: TestClient):
    headers_a = get_auth_headers(client, "owner_a@example.com")
    headers_b = get_auth_headers(client, "owner_b@example.com")

    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers_a)
    alerts_a = client.get("/api/monitoring/alerts", headers=headers_a).json()["items"]
    assert len(alerts_a) > 0
    alert_id = alerts_a[0]["id"]

    # User B tries to retrieve User A's alert
    res_b = client.get(f"/api/monitoring/alerts/{alert_id}", headers=headers_b)
    assert res_b.status_code == 403


# ---------------------------------------------------------------------------
# 13. Acknowledge Transition
# ---------------------------------------------------------------------------
def test_acknowledge_transition(client: TestClient):
    headers = get_auth_headers(client, "ack_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    alerts = client.get("/api/monitoring/alerts?status=new", headers=headers).json()["items"]
    assert len(alerts) > 0
    alert_id = alerts[0]["id"]

    res = client.post(f"/api/monitoring/alerts/{alert_id}/acknowledge", headers=headers)
    assert res.status_code == 200
    assert res.json()["status"] == "acknowledged"
    assert res.json()["acknowledged_at"] is not None


# ---------------------------------------------------------------------------
# 14. Resolve Transition
# ---------------------------------------------------------------------------
def test_resolve_transition(client: TestClient):
    headers = get_auth_headers(client, "res_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    alerts = client.get("/api/monitoring/alerts?status=new", headers=headers).json()["items"]
    assert len(alerts) > 0
    alert_id = alerts[0]["id"]

    res = client.post(f"/api/monitoring/alerts/{alert_id}/resolve", headers=headers)
    assert res.status_code == 200
    assert res.json()["status"] == "resolved"
    assert res.json()["resolved_at"] is not None


# ---------------------------------------------------------------------------
# 15. Dismiss Transition
# ---------------------------------------------------------------------------
def test_dismiss_transition(client: TestClient):
    headers = get_auth_headers(client, "dism_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    alerts = client.get("/api/monitoring/alerts?status=new", headers=headers).json()["items"]
    assert len(alerts) > 0
    alert_id = alerts[0]["id"]

    res = client.post(f"/api/monitoring/alerts/{alert_id}/dismiss", headers=headers)
    assert res.status_code == 200
    assert res.json()["status"] == "dismissed"


# ---------------------------------------------------------------------------
# 16. Invalid Alert Transition
# ---------------------------------------------------------------------------
def test_invalid_alert_transition(client: TestClient):
    headers = get_auth_headers(client, "invalid_trans_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    alerts = client.get("/api/monitoring/alerts?status=new", headers=headers).json()["items"]
    assert len(alerts) > 0
    alert_id = alerts[0]["id"]

    # First resolve the alert
    client.post(f"/api/monitoring/alerts/{alert_id}/resolve", headers=headers)

    # Attempt to acknowledge an already resolved alert -> 409 Conflict
    res = client.post(f"/api/monitoring/alerts/{alert_id}/acknowledge", headers=headers)
    assert res.status_code == 409


# ---------------------------------------------------------------------------
# 17. Monitoring Run Record Created
# ---------------------------------------------------------------------------
def test_monitoring_run_record_created(client: TestClient):
    headers = get_auth_headers(client, "run_record_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 15}, headers=headers)
    res = client.get("/api/monitoring/runs", headers=headers)
    assert res.status_code == 200
    runs = res.json()
    assert len(runs) >= 1
    assert runs[0]["status"] == "completed"
    assert runs[0]["total_records_evaluated"] > 0


# ---------------------------------------------------------------------------
# 18. Failed Run Handled Safely
# ---------------------------------------------------------------------------
def test_failed_run_handled_safely(client: TestClient):
    headers = get_auth_headers(client, "failed_run_user@example.com")
    with patch("backend.app.services.anomaly_service.AnomalyDetectionService.get_anomalies", side_effect=RuntimeError("Data read failure")):
        res = client.post("/api/monitoring/run", json={}, headers=headers)
        assert res.status_code == 500

    runs = client.get("/api/monitoring/runs", headers=headers).json()
    assert len(runs) >= 1
    assert runs[0]["status"] == "failed"
    assert "Data read failure" in runs[0]["error_message"]


# ---------------------------------------------------------------------------
# 19. Evidence Snapshot Preserved
# ---------------------------------------------------------------------------
def test_evidence_snapshot_preserved(client: TestClient):
    headers = get_auth_headers(client, "snap_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    alerts = client.get("/api/monitoring/alerts", headers=headers).json()["items"]
    assert len(alerts) > 0
    snap = alerts[0]["evidence_snapshot"]
    assert isinstance(snap, dict)
    assert "anomaly_id" in snap
    assert "actual_value" in snap
    assert "expected_value" in snap


# ---------------------------------------------------------------------------
# 20. Anomaly Data Not Mutated
# ---------------------------------------------------------------------------
def test_anomaly_data_not_mutated(client: TestClient):
    headers = get_auth_headers(client, "anom_nomut_user@example.com")
    anoms_before = client.get("/api/anomalies?limit=10").json()["items"]
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    anoms_after = client.get("/api/anomalies?limit=10").json()["items"]
    assert anoms_before == anoms_after


# ---------------------------------------------------------------------------
# 21. Sales Data Not Mutated
# ---------------------------------------------------------------------------
def test_sales_data_not_mutated(client: TestClient, db_session: Session):
    headers = get_auth_headers(client, "sales_nomut_user@example.com")
    sales_count_before = db_session.query(SalesRecord).count()
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    sales_count_after = db_session.query(SalesRecord).count()
    assert sales_count_before == sales_count_after


# ---------------------------------------------------------------------------
# 22. Product Data Not Mutated
# ---------------------------------------------------------------------------
def test_product_data_not_mutated(client: TestClient, db_session: Session):
    headers = get_auth_headers(client, "prod_nomut_user@example.com")
    prod_count_before = db_session.query(Product).count()
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    prod_count_after = db_session.query(Product).count()
    assert prod_count_before == prod_count_after


# ---------------------------------------------------------------------------
# 23. Monitoring Does Not Approve Decisions
# ---------------------------------------------------------------------------
def test_monitoring_does_not_approve_decisions(client: TestClient, db_session: Session):
    headers = get_auth_headers(client, "no_dec_approve_user@example.com")
    # Create a pending decision record
    dec_res = client.post(
        "/api/decisions",
        json={"recommendation_type": "pricing_review", "proposed_action": "Review pricing."},
        headers=headers,
    )
    dec_id = dec_res.json()["id"]

    # Run monitoring
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)

    # Check decision is still pending_review
    dec_check = client.get(f"/api/decisions/{dec_id}", headers=headers).json()
    assert dec_check["status"] == "pending_review"


# ---------------------------------------------------------------------------
# 24. Monitoring Does Not Execute Business Operations
# ---------------------------------------------------------------------------
def test_monitoring_does_not_execute_business_operations(client: TestClient):
    headers = get_auth_headers(client, "no_ops_user@example.com")
    res = client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    data = res.json()
    assert data["automatic_execution"] is False
    assert data["human_review_required"] is True

    alerts = client.get("/api/monitoring/alerts", headers=headers).json()["items"]
    for a in alerts:
        assert a["automatic_execution"] is False
        assert a["human_review_required"] is True


# ---------------------------------------------------------------------------
# 25. Gemini Cannot Change Severity
# ---------------------------------------------------------------------------
def test_gemini_cannot_change_severity(client: TestClient):
    # Verify severity is derived strictly from statistical thresholds in AnomalyDetectionService
    headers = get_auth_headers(client, "no_gemini_sev@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    alerts = client.get("/api/monitoring/alerts", headers=headers).json()["items"]
    for a in alerts:
        score = a["evidence_snapshot"].get("anomaly_score", 0.0)
        if score >= 4.0:
            assert a["severity"] == "critical"
        elif score >= 3.0:
            assert a["severity"] == "high"
        elif score >= 2.5:
            assert a["severity"] == "medium"


# ---------------------------------------------------------------------------
# 26. Gemini Cannot Change Priority
# ---------------------------------------------------------------------------
def test_gemini_cannot_change_priority(client: TestClient):
    headers = get_auth_headers(client, "no_gemini_prio@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    alerts = client.get("/api/monitoring/alerts", headers=headers).json()["items"]
    for a in alerts:
        if a["severity"] == "critical":
            assert a["priority"] == "urgent"
        elif a["severity"] == "high":
            assert a["priority"] == "high"


# ---------------------------------------------------------------------------
# 27. Recommendation Link Integration
# ---------------------------------------------------------------------------
def test_recommendation_link_integration(client: TestClient):
    headers = get_auth_headers(client, "rec_link_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    alerts = client.get("/api/monitoring/alerts", headers=headers).json()["items"]
    assert len(alerts) > 0
    detail = client.get(f"/api/monitoring/alerts/{alerts[0]['id']}", headers=headers).json()
    assert "has_recommendation" in detail
    assert isinstance(detail["has_recommendation"], bool)


# ---------------------------------------------------------------------------
# 28. Simulation Link Integration
# ---------------------------------------------------------------------------
def test_simulation_link_integration(client: TestClient):
    headers = get_auth_headers(client, "sim_link_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    alerts = client.get("/api/monitoring/alerts", headers=headers).json()["items"]
    assert len(alerts) > 0
    detail = client.get(f"/api/monitoring/alerts/{alerts[0]['id']}", headers=headers).json()
    assert detail["simulation_available"] is True
    assert detail["recommended_scenario"] in ("demand_multiplier", "temporary_shock")


# ---------------------------------------------------------------------------
# 29. Repeated Monitoring Does Not Create Duplicate Alerts
# ---------------------------------------------------------------------------
def test_repeated_monitoring_does_not_create_duplicate_alerts(client: TestClient):
    headers = get_auth_headers(client, "rep_mon_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    count1 = client.get("/api/monitoring/alerts", headers=headers).json()["total"]

    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    count2 = client.get("/api/monitoring/alerts", headers=headers).json()["total"]

    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    count3 = client.get("/api/monitoring/alerts", headers=headers).json()["total"]

    assert count1 == count2 == count3


# ---------------------------------------------------------------------------
# 30. Chronological Alert Ordering
# ---------------------------------------------------------------------------
def test_chronological_alert_ordering(client: TestClient):
    headers = get_auth_headers(client, "chrono_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 60}, headers=headers)
    alerts = client.get("/api/monitoring/alerts", headers=headers).json()["items"]
    if len(alerts) > 1:
        for i in range(len(alerts) - 1):
            assert alerts[i]["event_date"] >= alerts[i + 1]["event_date"]


# ---------------------------------------------------------------------------
# 31. Pagination Behavior
# ---------------------------------------------------------------------------
def test_pagination_behavior(client: TestClient):
    headers = get_auth_headers(client, "page_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 60}, headers=headers)
    all_res = client.get("/api/monitoring/alerts?limit=100", headers=headers).json()
    total = all_res["total"]

    if total >= 2:
        p1 = client.get("/api/monitoring/alerts?skip=0&limit=1", headers=headers).json()["items"]
        p2 = client.get("/api/monitoring/alerts?skip=1&limit=1", headers=headers).json()["items"]
        assert len(p1) == 1
        assert len(p2) == 1
        assert p1[0]["id"] != p2[0]["id"]


# ---------------------------------------------------------------------------
# 32. Unsupported Notification Channels Not Executed
# ---------------------------------------------------------------------------
def test_unsupported_notification_channels_not_executed(client: TestClient):
    headers = get_auth_headers(client, "no_notify_user@example.com")
    # Verify no external notifications (email, sms, whatsapp, webhooks) occur
    with patch("urllib.request.urlopen") as mock_url:
        res = client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
        assert res.status_code == 200
        mock_url.assert_not_called()


# ---------------------------------------------------------------------------
# 33. Monitoring Summary Counts Consistent
# ---------------------------------------------------------------------------
def test_monitoring_summary_counts_consistent(client: TestClient):
    headers = get_auth_headers(client, "summary_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    summary = client.get("/api/monitoring/summary", headers=headers).json()

    alerts = client.get("/api/monitoring/alerts?limit=200", headers=headers).json()["items"]
    new_alerts = [a for a in alerts if a["status"] == "new"]
    crit_alerts = [a for a in alerts if a["severity"] == "critical"]
    high_alerts = [a for a in alerts if a["severity"] == "high"]
    med_alerts = [a for a in alerts if a["severity"] == "medium"]

    assert summary["new_alerts"] == len(new_alerts)
    assert summary["critical_alerts"] == len(crit_alerts)
    assert summary["high_alerts"] == len(high_alerts)
    assert summary["medium_alerts"] == len(med_alerts)


# ---------------------------------------------------------------------------
# 34. Alert Fingerprint Deterministic
# ---------------------------------------------------------------------------
def test_alert_fingerprint_deterministic(client: TestClient):
    headers = get_auth_headers(client, "fp_user@example.com")
    client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
    alerts = client.get("/api/monitoring/alerts", headers=headers).json()["items"]
    for a in alerts:
        fp = a["fingerprint"]
        assert a["event_date"].replace("-", "") in fp
        assert a["metric"] in fp
        assert a["alert_type"] in fp


# ---------------------------------------------------------------------------
# 35. Completed Run Status Correct
# ---------------------------------------------------------------------------
def test_completed_run_status_correct(client: TestClient):
    headers = get_auth_headers(client, "status_correct_user@example.com")
    res = client.post("/api/monitoring/run", json={"lookback_days": 20}, headers=headers)
    data = res.json()
    assert data["status"] == "completed"
    assert data["completed_at"] is not None
    assert data["started_at"] <= data["completed_at"]
