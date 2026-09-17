from __future__ import annotations

import datetime as dt
import math
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.llm.provider import LLMProviderError
from backend.app.schemas.simulations import (
    ScenarioType,
    SimulationRequest,
    SimulationResponse,
    SimulationStatus,
)
from backend.app.services.simulation_service import (
    SimulationService,
    validate_simulation_result,
)


def get_auth_headers(client: TestClient, email: str = "sim_tester@example.com") -> dict[str, str]:
    """Register and login helper to obtain Bearer authorization header."""
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
# 1. Positive Demand Multiplier (+10%)
# ---------------------------------------------------------------------------
def test_simulation_demand_multiplier_positive():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.DEMAND_MULTIPLIER,
        demand_change_percent=10.0,
        horizon_days=7,
        include_revenue=True,
    )
    res = service.run_simulation(req)

    assert res.status == SimulationStatus.COMPLETED
    assert res.horizon_days == 7
    assert res.baseline is not None
    assert res.scenario is not None
    assert res.delta is not None
    assert res.delta.quantity_delta_percent == pytest.approx(10.0, rel=1e-2)
    assert res.delta.quantity_delta > 0
    assert len(res.daily_results) == 7
    for d in res.daily_results:
        assert d.scenario_quantity >= d.baseline_quantity
        assert d.delta_percent == pytest.approx(10.0, rel=1e-2)


# ---------------------------------------------------------------------------
# 2. Negative Demand Multiplier (-15%)
# ---------------------------------------------------------------------------
def test_simulation_demand_multiplier_negative():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.DEMAND_MULTIPLIER,
        demand_change_percent=-15.0,
        horizon_days=7,
        include_revenue=True,
    )
    res = service.run_simulation(req)

    assert res.status == SimulationStatus.COMPLETED
    assert res.delta.quantity_delta_percent == pytest.approx(-15.0, rel=1e-2)
    assert res.delta.quantity_delta < 0
    for d in res.daily_results:
        assert d.scenario_quantity <= d.baseline_quantity
        assert d.delta_percent == pytest.approx(-15.0, rel=1e-2)


# ---------------------------------------------------------------------------
# 3. Temporary Shock (Shock applies for duration, baseline thereafter)
# ---------------------------------------------------------------------------
def test_simulation_temporary_shock():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.TEMPORARY_SHOCK,
        demand_change_percent=-20.0,
        shock_duration_days=3,
        horizon_days=7,
    )
    res = service.run_simulation(req)

    assert res.status == SimulationStatus.COMPLETED
    # First 3 days should have shock applied
    for d in res.daily_results[:3]:
        assert d.delta_percent == pytest.approx(-20.0, rel=1e-2)
    # Remaining 4 days should revert to baseline (delta = 0)
    for d in res.daily_results[3:]:
        assert d.delta_quantity == 0.0
        assert d.delta_percent == 0.0
        assert d.scenario_quantity == d.baseline_quantity


# ---------------------------------------------------------------------------
# 4. Persistent Shift (Applies across full horizon)
# ---------------------------------------------------------------------------
def test_simulation_persistent_shift():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.PERSISTENT_SHIFT,
        demand_change_percent=12.5,
        horizon_days=30,
    )
    res = service.run_simulation(req)

    assert res.status == SimulationStatus.COMPLETED
    assert len(res.daily_results) == 30
    assert res.delta.quantity_delta_percent == pytest.approx(12.5, rel=1e-2)
    for d in res.daily_results:
        assert d.delta_percent == pytest.approx(12.5, rel=1e-2)


# ---------------------------------------------------------------------------
# 5. Trend Continuation (Historical linear slope continuation)
# ---------------------------------------------------------------------------
def test_simulation_trend_continuation():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.TREND_CONTINUATION,
        trend_window_days=28,
        horizon_days=7,
    )
    res = service.run_simulation(req)

    assert res.status == SimulationStatus.COMPLETED
    assert len(res.daily_results) == 7
    assert res.confidence == "medium"
    assert any("linear trend" in a.lower() for a in res.assumptions)


# ---------------------------------------------------------------------------
# 6. Promotion Scenario (Simulates ML feature lift)
# ---------------------------------------------------------------------------
def test_simulation_promotion_scenario():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.PROMOTION_SCENARIO,
        promotion_active=True,
        horizon_days=7,
    )
    res = service.run_simulation(req)

    assert res.status == SimulationStatus.COMPLETED
    assert len(res.daily_results) == 7
    assert any("promotional campaign" in a.lower() for a in res.assumptions)


# ---------------------------------------------------------------------------
# 7. Holiday Scenario (Simulates ML holiday lift)
# ---------------------------------------------------------------------------
def test_simulation_holiday_scenario():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.HOLIDAY_SCENARIO,
        holiday_active=True,
        horizon_days=7,
    )
    res = service.run_simulation(req)

    assert res.status == SimulationStatus.COMPLETED
    assert len(res.daily_results) == 7
    assert any("holiday trading" in a.lower() for a in res.assumptions)


# ---------------------------------------------------------------------------
# 8. Price Change (Requires Econometric Model, returns requires_model)
# ---------------------------------------------------------------------------
def test_simulation_price_change_requires_model():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.PRICE_CHANGE,
        horizon_days=30,
    )
    res = service.run_simulation(req)

    assert res.status == SimulationStatus.REQUIRES_MODEL
    assert res.confidence == "not_available"
    assert res.required_model == "econometric_price_elasticity"
    assert "price" in res.reason.lower()
    assert res.baseline is None
    assert res.scenario is None
    assert res.delta is None
    assert len(res.daily_results) == 0


# ---------------------------------------------------------------------------
# 9. Discount Change (Requires Discount Model, returns requires_model)
# ---------------------------------------------------------------------------
def test_simulation_discount_change_requires_model():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.DISCOUNT_CHANGE,
        horizon_days=30,
    )
    res = service.run_simulation(req)

    assert res.status == SimulationStatus.REQUIRES_MODEL
    assert res.confidence == "not_available"
    assert res.required_model == "discount_elasticity_model"
    assert "discount" in res.reason.lower()
    assert res.baseline is None
    assert res.scenario is None
    assert res.delta is None
    assert len(res.daily_results) == 0


# ---------------------------------------------------------------------------
# 10. Invariant: Sum of baseline daily quantities equals summary total
# ---------------------------------------------------------------------------
def test_simulation_invariant_baseline_sum():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.DEMAND_MULTIPLIER,
        demand_change_percent=5.0,
        horizon_days=30,
    )
    res = service.run_simulation(req)

    daily_sum = sum(d.baseline_quantity for d in res.daily_results)
    assert daily_sum == pytest.approx(res.baseline.total_quantity, abs=0.05)


# ---------------------------------------------------------------------------
# 11. Invariant: Sum of scenario daily quantities equals summary total
# ---------------------------------------------------------------------------
def test_simulation_invariant_scenario_sum():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.DEMAND_MULTIPLIER,
        demand_change_percent=15.0,
        horizon_days=30,
    )
    res = service.run_simulation(req)

    daily_sum = sum(d.scenario_quantity for d in res.daily_results)
    assert daily_sum == pytest.approx(res.scenario.total_quantity, abs=0.05)


# ---------------------------------------------------------------------------
# 12. Invariant: Delta math is consistent
# ---------------------------------------------------------------------------
def test_simulation_invariant_delta_math():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.DEMAND_MULTIPLIER,
        demand_change_percent=-10.0,
        horizon_days=7,
        include_revenue=True,
    )
    res = service.run_simulation(req)

    expected_qty_delta = res.scenario.total_quantity - res.baseline.total_quantity
    assert res.delta.quantity_delta == pytest.approx(expected_qty_delta, abs=0.05)

    expected_qty_pct = (expected_qty_delta / res.baseline.total_quantity) * 100.0
    assert res.delta.quantity_delta_percent == pytest.approx(expected_qty_pct, abs=0.1)

    expected_rev_delta = res.scenario.total_revenue - res.baseline.total_revenue
    assert res.delta.revenue_delta == pytest.approx(expected_rev_delta, abs=0.05)


# ---------------------------------------------------------------------------
# 13. Invariant: No negative quantities
# ---------------------------------------------------------------------------
def test_simulation_invariant_no_negative_values():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.DEMAND_MULTIPLIER,
        demand_change_percent=-50.0,
        horizon_days=7,
    )
    res = service.run_simulation(req)

    for d in res.daily_results:
        assert d.baseline_quantity >= 0.0
        assert d.scenario_quantity >= 0.0


# ---------------------------------------------------------------------------
# 14. Invariant: No NaN or Infinite values
# ---------------------------------------------------------------------------
def test_simulation_invariant_no_nan_or_inf():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.DEMAND_MULTIPLIER,
        demand_change_percent=20.0,
        horizon_days=7,
        include_revenue=True,
    )
    res = service.run_simulation(req)

    for d in res.daily_results:
        assert not math.isnan(d.baseline_quantity) and not math.isinf(d.baseline_quantity)
        assert not math.isnan(d.scenario_quantity) and not math.isinf(d.scenario_quantity)
        assert not math.isnan(d.delta_quantity) and not math.isinf(d.delta_quantity)
        assert not math.isnan(d.delta_percent) and not math.isinf(d.delta_percent)
        if d.baseline_revenue is not None:
            assert not math.isnan(d.baseline_revenue) and not math.isinf(d.baseline_revenue)
            assert not math.isnan(d.scenario_revenue) and not math.isinf(d.scenario_revenue)


# ---------------------------------------------------------------------------
# 15. Invariant: Continuous chronological dates matching horizon
# ---------------------------------------------------------------------------
def test_simulation_invariant_continuous_dates():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.DEMAND_MULTIPLIER,
        demand_change_percent=5.0,
        horizon_days=30,
    )
    res = service.run_simulation(req)

    assert len(res.daily_results) == 30
    for i in range(1, len(res.daily_results)):
        prev_date = res.daily_results[i - 1].date
        curr_date = res.daily_results[i].date
        assert curr_date == prev_date + dt.timedelta(days=1)


# ---------------------------------------------------------------------------
# 16. Revenue Calculation with Baseline Realized Unit Price
# ---------------------------------------------------------------------------
def test_simulation_revenue_calculation():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.DEMAND_MULTIPLIER,
        demand_change_percent=10.0,
        horizon_days=7,
        include_revenue=True,
    )
    res = service.run_simulation(req)

    assert res.baseline.total_revenue is not None
    assert res.scenario.total_revenue is not None
    assert res.delta.revenue_delta is not None
    assert res.delta.revenue_delta > 0
    # Revenue sum matches sum of daily revenues
    daily_b_rev = sum(d.baseline_revenue for d in res.daily_results)
    assert daily_b_rev == pytest.approx(res.baseline.total_revenue, abs=0.05)


# ---------------------------------------------------------------------------
# 17. Revenue Excluded when requested
# ---------------------------------------------------------------------------
def test_simulation_revenue_excluded():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.DEMAND_MULTIPLIER,
        demand_change_percent=10.0,
        horizon_days=7,
        include_revenue=False,
    )
    res = service.run_simulation(req)

    assert res.baseline.total_revenue is None
    assert res.scenario.total_revenue is None
    assert res.delta.revenue_delta is None
    for d in res.daily_results:
        assert d.baseline_revenue is None
        assert d.scenario_revenue is None
        assert d.delta_revenue is None


# ---------------------------------------------------------------------------
# 18. Anchored Anomaly Context
# ---------------------------------------------------------------------------
def test_simulation_anchored_anomaly():
    service = SimulationService()
    req = SimulationRequest(
        scenario_type=ScenarioType.DEMAND_MULTIPLIER,
        demand_change_percent=10.0,
        horizon_days=7,
        anomaly_id="anom-20251228-sal-agg",
    )
    res = service.run_simulation(req)

    assert res.status == SimulationStatus.COMPLETED
    assert res.anomaly_id == "anom-20251228-sal-agg"
    assert res.anomaly_context is not None
    assert res.anomaly_context.anomaly_id == "anom-20251228-sal-agg"
    assert res.anomaly_context.deviation_percent != 0.0


# ---------------------------------------------------------------------------
# 19. API POST /api/simulations (Authenticated - 200 OK)
# ---------------------------------------------------------------------------
def test_simulation_api_post_authenticated(client: TestClient):
    headers = get_auth_headers(client, email="sim_user_auth@example.com")
    payload = {
        "scenario_type": "demand_multiplier",
        "demand_change_percent": 8.0,
        "horizon_days": 7,
        "include_revenue": True,
        "include_explanation": True,
    }
    response = client.post("/api/simulations", json=payload, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["scenario_type"] == "demand_multiplier"
    assert data["delta"]["quantity_delta_percent"] == pytest.approx(8.0, rel=1e-2)
    assert data["explanation"] is not None


# ---------------------------------------------------------------------------
# 20. API POST /api/simulations (Unauthenticated - 401 Unauthorized)
# ---------------------------------------------------------------------------
def test_simulation_api_post_unauthenticated(client: TestClient):
    payload = {
        "scenario_type": "demand_multiplier",
        "demand_change_percent": 10.0,
        "horizon_days": 7,
    }
    response = client.post("/api/simulations", json=payload)
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# 21. API GET /api/simulations/{id} (Cached retrieval and 404 for unknown)
# ---------------------------------------------------------------------------
def test_simulation_api_get_cached(client: TestClient):
    headers = get_auth_headers(client, email="sim_cache_user@example.com")
    payload = {
        "scenario_type": "demand_multiplier",
        "demand_change_percent": 5.0,
        "horizon_days": 7,
    }
    post_res = client.post("/api/simulations", json=payload, headers=headers)
    assert post_res.status_code == 200
    sim_id = post_res.json()["simulation_id"]

    # Retrieve cached simulation
    get_res = client.get(f"/api/simulations/{sim_id}", headers=headers)
    assert get_res.status_code == 200
    assert get_res.json()["simulation_id"] == sim_id

    # Retrieve nonexistent simulation
    bad_res = client.get("/api/simulations/sim-nonexistent999", headers=headers)
    assert bad_res.status_code == 404


# ---------------------------------------------------------------------------
# 22. Narrative Explanation: Gemini Explanation & Deterministic Fallback
# ---------------------------------------------------------------------------
def test_simulation_llm_explanation_and_fallback():
    mock_llm = MagicMock()
    mock_llm.api_key = "fake_key"
    mock_llm.generate_simulation_explanation.return_value = (
        "Under a hypothetical 10.0% demand multiplier over 7 days, projected volume increases by 3,796 units. "
        "Revenue expands assuming constant baseline pricing."
    )

    # With mocked GeminiProvider
    from backend.app.llm.gemini_provider import GeminiProvider

    with patch.object(GeminiProvider, "__init__", lambda self, *args, **kwargs: None):
        service = SimulationService(llm_provider=mock_llm)
        service.llm_provider.__class__ = GeminiProvider
        req = SimulationRequest(
            scenario_type=ScenarioType.DEMAND_MULTIPLIER,
            demand_change_percent=10.0,
            horizon_days=7,
            include_explanation=True,
        )
        res = service.run_simulation(req)

        assert res.source == "gemini_explanation"
        assert "projected volume increases" in res.explanation

        # Now test fallback when LLM fails
        mock_llm.generate_simulation_explanation.side_effect = LLMProviderError("Quota exceeded")
        res_fallback = service.run_simulation(req)
        assert res_fallback.source == "deterministic"
        assert "Under a hypothetical Demand Multiplier" in res_fallback.explanation
        assert "₹" in res_fallback.explanation
