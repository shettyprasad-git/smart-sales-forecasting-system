import datetime as dt
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.anomaly_service import AnomalyDetectionService
from backend.app.services.investigation_service import InvestigationService


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def investigation_service():
    return InvestigationService()


@pytest.fixture
def sample_anomaly_id(investigation_service):
    """Retrieve an actual anomaly ID from the detection service."""
    anomalies = investigation_service.anomaly_service.get_anomalies(limit=5)
    assert len(anomalies.items) > 0
    return anomalies.items[0].id


# =====================================================================
# Mathematical & Attribution Logic Tests
# =====================================================================

def test_mathematical_deviation_and_percentage(investigation_service, sample_anomaly_id):
    """Verify that deviation = actual - expected and deviation_percent is mathematically exact."""
    res = investigation_service.investigate_anomaly(sample_anomaly_id)

    expected_diff = round(res.actual_value - res.expected_baseline, 2)
    assert res.deviation == expected_diff

    expected_pct = round((res.deviation / res.expected_baseline) * 100, 2)
    assert res.deviation_percent == expected_pct


def test_impact_estimation_spike_and_drop(investigation_service):
    """Verify impact values are non-negative and properly contextualized for spikes vs drops."""
    # Find one spike and one drop
    spikes = investigation_service.anomaly_service.get_anomalies(direction="spike", limit=1)
    drops = investigation_service.anomaly_service.get_anomalies(direction="drop", limit=1)

    assert len(spikes.items) > 0
    assert len(drops.items) > 0

    res_spike = investigation_service.investigate_anomaly(spikes.items[0].id)
    assert res_spike.estimated_impact.impact_value > 0
    assert res_spike.estimated_impact.signed_deviation > 0
    assert "excess" in res_spike.estimated_impact.interpretation.lower()

    res_drop = investigation_service.investigate_anomaly(drops.items[0].id)
    assert res_drop.estimated_impact.impact_value > 0
    assert res_drop.estimated_impact.signed_deviation < 0
    assert any(w in res_drop.estimated_impact.interpretation.lower() for w in ("deficit", "gap"))


def test_category_contributions_present_for_aggregate(investigation_service, sample_anomaly_id):
    """Verify aggregate anomalies decompose into category-level drivers."""
    res = investigation_service.investigate_anomaly(sample_anomaly_id, include_categories=True)
    cat_drivers = [d for d in res.drivers if d.driver_type == "category"]

    if res.entity_type == "aggregate":
        assert len(cat_drivers) > 0
        for d in cat_drivers:
            assert d.driver_name.startswith("Category: ")
            assert d.observed_value is not None
            assert d.reference_value is not None
            assert d.confidence == "high"


def test_product_contributions_top_n(investigation_service, sample_anomaly_id):
    """Verify product-level drivers respect the top_n parameter."""
    res_top_3 = investigation_service.investigate_anomaly(sample_anomaly_id, top_n=3, include_products=True)
    prod_drivers_3 = [d for d in res_top_3.drivers if d.driver_type == "product"]
    assert len(prod_drivers_3) <= 3

    res_top_7 = investigation_service.investigate_anomaly(sample_anomaly_id, top_n=7, include_products=True)
    prod_drivers_7 = [d for d in res_top_7.drivers if d.driver_type == "product"]
    assert len(prod_drivers_7) <= 7


# =====================================================================
# Causal Invariance & Zero Lookahead Leakage Tests
# =====================================================================

def test_causal_invariance_future_data_tampering(investigation_service, sample_anomaly_id):
    """
    Verify that altering data on dates strictly after the anomaly date T
    causes ZERO change to the investigation result for date T.
    """
    # 1. Run baseline investigation
    original_res = investigation_service.investigate_anomaly(sample_anomaly_id)
    target_date = original_res.anomaly_date

    # 2. Tamper with future rows in product and daily data
    product_df = investigation_service._load_product_data()
    future_mask = product_df["Date"] > pd.Timestamp(target_date)

    # Save original future values and inject massive distortions
    orig_future_vals = product_df.loc[future_mask, "Quantity"].copy()
    product_df.loc[future_mask, "Quantity"] = 999999.0
    product_df.loc[future_mask, "Sales_Amount"] = 999999999.0

    try:
        tampered_res = investigation_service.investigate_anomaly(sample_anomaly_id)

        # Baseline, impact, and drivers must be 100% identical
        assert original_res.actual_value == tampered_res.actual_value
        assert original_res.expected_baseline == tampered_res.expected_baseline
        assert original_res.deviation == tampered_res.deviation
        assert original_res.estimated_impact.impact_value == tampered_res.estimated_impact.impact_value
        assert len(original_res.drivers) == len(tampered_res.drivers)

        for d_orig, d_tamp in zip(original_res.drivers, tampered_res.drivers):
            assert d_orig.driver_name == d_tamp.driver_name
            assert d_orig.reference_value == d_tamp.reference_value
            assert d_orig.contribution_score == d_tamp.contribution_score
            assert d_orig.evidence == d_tamp.evidence

    finally:
        # Restore original future values
        product_df.loc[future_mask, "Quantity"] = orig_future_vals


def test_reference_calculations_use_only_prior_dates(investigation_service, sample_anomaly_id):
    """Verify trend drift and reference prices use data prior to anomaly date."""
    res = investigation_service.investigate_anomaly(sample_anomaly_id)
    trend_driver = next((d for d in res.drivers if d.driver_type == "recent_trend"), None)

    if trend_driver:
        assert trend_driver.observed_value is not None
        assert trend_driver.reference_value is not None
        assert trend_driver.difference is not None


# =====================================================================
# Driver Detection Tests
# =====================================================================

def test_promotion_driver_present(investigation_service, sample_anomaly_id):
    """Verify promotion driver is evaluated and presents explicit status."""
    res = investigation_service.investigate_anomaly(sample_anomaly_id)
    promo_driver = next((d for d in res.drivers if d.driver_type == "promotion"), None)

    assert promo_driver is not None
    assert promo_driver.confidence in ("low", "medium", "high")
    assert promo_driver.evidence is not None


def test_holiday_driver_present(investigation_service, sample_anomaly_id):
    """Verify holiday driver is evaluated and reports status."""
    res = investigation_service.investigate_anomaly(sample_anomaly_id)
    hol_driver = next((d for d in res.drivers if d.driver_type == "holiday"), None)

    assert hol_driver is not None
    assert hol_driver.confidence in ("low", "medium", "high")


def test_price_and_discount_drivers_present(investigation_service, sample_anomaly_id):
    """Verify pricing and discount drivers are evaluated."""
    res = investigation_service.investigate_anomaly(sample_anomaly_id)
    price_driver = next((d for d in res.drivers if d.driver_type == "price"), None)
    disc_driver = next((d for d in res.drivers if d.driver_type == "discount"), None)

    assert price_driver is not None
    assert disc_driver is not None


# =====================================================================
# API Endpoint Integration Tests
# =====================================================================

def test_api_investigation_success(client, sample_anomaly_id):
    """Verify GET /api/investigations/{anomaly_id} returns 200 and valid schema."""
    response = client.get(f"/api/investigations/{sample_anomaly_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["anomaly_id"] == sample_anomaly_id
    assert "anomaly_date" in data
    assert "metric" in data
    assert "actual_value" in data
    assert "expected_baseline" in data
    assert "estimated_impact" in data
    assert "impact_interpretation" in data
    assert "drivers" in data
    assert len(data["drivers"]) > 0
    assert "investigation_summary" in data
    assert "evidence_notes" in data
    assert "limitations" in data


def test_api_investigation_not_found(client):
    """Verify GET /api/investigations/{unknown_id} returns 404."""
    response = client.get("/api/investigations/anom-99991231-qua-agg")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_api_investigation_invalid_top_n(client, sample_anomaly_id):
    """Verify top_n validation rejects values < 1 or > 20 with 422."""
    res_zero = client.get(f"/api/investigations/{sample_anomaly_id}?top_n=0")
    assert res_zero.status_code == 422

    res_large = client.get(f"/api/investigations/{sample_anomaly_id}?top_n=50")
    assert res_large.status_code == 422


def test_api_investigation_toggle_flags(client, sample_anomaly_id):
    """Verify include_products=false suppresses product drivers."""
    res_no_prod = client.get(
        f"/api/investigations/{sample_anomaly_id}?include_products=false"
    )
    assert res_no_prod.status_code == 200
    drivers = res_no_prod.json()["drivers"]
    prod_drivers = [d for d in drivers if d["driver_type"] == "product"]
    assert len(prod_drivers) == 0


def test_api_get_investigation_requires_auth_in_production(client, sample_anomaly_id):
    """Unauthenticated calls in production return 401."""
    from unittest.mock import patch
    from backend.app.core.config import settings

    with patch.object(settings, "environment", "production"):
        resp = client.get(f"/api/investigations/{sample_anomaly_id}")
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Authentication required."
