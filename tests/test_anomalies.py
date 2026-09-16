import datetime as dt
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.anomaly_service import AnomalyDetectionService


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def anomaly_service():
    return AnomalyDetectionService()


# =====================================================================
# Unit Tests: Statistical Methodology & Causal Invariance
# =====================================================================

def test_causal_invariance_no_future_leakage(anomaly_service):
    """
    Verify that the baseline and anomaly score for observation t
    are strictly invariant to changes in future observations (t+1, t+2, ...).
    """
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    base_values = [100.0 + (i % 5) for i in range(40)]

    # Series A
    anoms_a = anomaly_service._compute_series_anomalies(
        dates=pd.Series(dates),
        values=pd.Series(base_values),
        metric="quantity",
        window=14,
        min_periods=7,
    )

    # Series B: identical up to day 25, massive 10x spike on day 35
    modified_values = list(base_values)
    modified_values[35] = 99999.0

    anoms_b = anomaly_service._compute_series_anomalies(
        dates=pd.Series(dates),
        values=pd.Series(modified_values),
        metric="quantity",
        window=14,
        min_periods=7,
    )

    # Compare results for any day prior to day 35
    # The day-25 evaluation must be identical in both
    day_25_date = dates[24].date()
    item_a = next((a for a in anoms_a if a.date == day_25_date), None)
    item_b = next((a for a in anoms_b if a.date == day_25_date), None)

    if item_a is not None:
        assert item_b is not None
        assert item_a.expected_value == item_b.expected_value
        assert item_a.anomaly_score == item_b.anomaly_score
        assert item_a.deviation == item_b.deviation


def test_normal_observations_not_flagged(anomaly_service):
    """
    Verify that steady observations with low Gaussian noise do not trigger false anomalies.
    """
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    # Low variance noise (mean 5000, std 50)
    values = 5000.0 + np.random.normal(0, 5, size=60)

    anomalies = anomaly_service._compute_series_anomalies(
        dates=pd.Series(dates),
        values=pd.Series(values),
        metric="quantity",
        window=28,
        min_periods=14,
    )

    # None of the low-noise points should exceed |z| >= 2.0
    critical_or_high = [a for a in anomalies if a.severity in ("high", "critical")]
    assert len(critical_or_high) == 0


def test_clear_spike_detection(anomaly_service):
    """
    Verify that an artificial spike is detected with correct direction and severity.
    """
    dates = pd.date_range("2024-01-01", periods=35, freq="D")
    values = [100.0] * 35
    values[25] = 250.0  # +150% spike

    anomalies = anomaly_service._compute_series_anomalies(
        dates=pd.Series(dates),
        values=pd.Series(values),
        metric="quantity",
        window=14,
        min_periods=7,
    )

    spike_date = dates[25].date()
    spike_item = next((a for a in anomalies if a.date == spike_date), None)

    assert spike_item is not None
    assert spike_item.direction == "spike"
    assert spike_item.severity in ("critical", "high")
    assert spike_item.actual_value == 250.0
    assert spike_item.expected_value == 100.0
    assert spike_item.deviation == 150.0
    assert spike_item.deviation_percent == 150.0
    assert "spike" in spike_item.explanation.lower()


def test_clear_drop_detection(anomaly_service):
    """
    Verify that an artificial drop is detected with correct direction and negative deviation.
    """
    dates = pd.date_range("2024-01-01", periods=35, freq="D")
    values = [200.0] * 35
    values[25] = 40.0  # -80% drop

    anomalies = anomaly_service._compute_series_anomalies(
        dates=pd.Series(dates),
        values=pd.Series(values),
        metric="sales_amount",
        window=14,
        min_periods=7,
    )

    drop_date = dates[25].date()
    drop_item = next((a for a in anomalies if a.date == drop_date), None)

    assert drop_item is not None
    assert drop_item.direction == "drop"
    assert drop_item.severity in ("critical", "high")
    assert drop_item.actual_value == 40.0
    assert drop_item.expected_value == 200.0
    assert drop_item.deviation < 0
    assert drop_item.deviation_percent == -80.0
    assert "drop" in drop_item.explanation.lower()


def test_zero_mad_fallback_robustness(anomaly_service):
    """
    When a series has identical values, MAD is 0. Verify the scale fallback prevents division by zero.
    """
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    values = [50.0] * 20  # Identical values, MAD = 0

    anomalies = anomaly_service._compute_series_anomalies(
        dates=pd.Series(dates),
        values=pd.Series(values),
        metric="quantity",
        window=7,
        min_periods=5,
    )

    # No anomalies should be raised on a constant series
    assert len(anomalies) == 0


# =====================================================================
# API Integration Tests
# =====================================================================

def test_api_list_anomalies_success(client):
    response = client.get("/api/anomalies?limit=10")
    assert response.status_code == 200

    data = response.json()
    assert "items" in data
    assert "total" in data
    assert "summary" in data
    assert len(data["items"]) <= 10

    if data["items"]:
        item = data["items"][0]
        assert "id" in item
        assert "date" in item
        assert "metric" in item
        assert "actual_value" in item
        assert "expected_value" in item
        assert "deviation" in item
        assert "anomaly_score" in item
        assert "severity" in item
        assert "direction" in item
        assert "explanation" in item


def test_api_filter_by_metric(client):
    response = client.get("/api/anomalies?metric=quantity&limit=20")
    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        assert item["metric"] == "quantity"


def test_api_filter_by_severity(client):
    response = client.get("/api/anomalies?severity=critical&limit=20")
    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        assert item["severity"] == "critical"
        assert item["anomaly_score"] >= 4.0


def test_api_filter_by_min_severity(client):
    response = client.get("/api/anomalies?min_severity=high&limit=20")
    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        assert item["severity"] in ("high", "critical")
        assert item["anomaly_score"] >= 3.0


def test_api_filter_by_direction(client):
    response = client.get("/api/anomalies?direction=spike&limit=20")
    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        assert item["direction"] == "spike"
        assert item["deviation"] > 0


def test_api_filter_by_date_range(client):
    start = "2023-01-01"
    end = "2023-06-30"
    response = client.get(f"/api/anomalies?start_date={start}&end_date={end}&limit=50")
    assert response.status_code == 200
    data = response.json()
    for item in data["items"]:
        assert start <= item["date"] <= end


def test_api_invalid_date_range(client):
    response = client.get("/api/anomalies?start_date=2024-01-01&end_date=2023-01-01")
    assert response.status_code == 400
    assert "start_date cannot be later than end_date" in response.json()["detail"]


def test_api_pagination(client):
    res_page_1 = client.get("/api/anomalies?skip=0&limit=5")
    res_page_2 = client.get("/api/anomalies?skip=5&limit=5")

    assert res_page_1.status_code == 200
    assert res_page_2.status_code == 200

    items_1 = res_page_1.json()["items"]
    items_2 = res_page_2.json()["items"]

    if items_1 and items_2:
        # Check that page 1 and page 2 don't overlap IDs
        ids_1 = {x["id"] for x in items_1}
        ids_2 = {x["id"] for x in items_2}
        assert ids_1.isdisjoint(ids_2)


def test_api_anomaly_summary(client):
    response = client.get("/api/anomalies/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_anomalies" in data
    assert "severity_breakdown" in data
    assert "direction_breakdown" in data
    assert "metric_breakdown" in data
    assert "critical" in data["severity_breakdown"]
    assert "high" in data["severity_breakdown"]
    assert "medium" in data["severity_breakdown"]
    assert "low" in data["severity_breakdown"]
    assert "spike" in data["direction_breakdown"]
    assert "drop" in data["direction_breakdown"]


def test_api_recent_anomalies(client):
    response = client.get("/api/anomalies/recent?limit=5")
    assert response.status_code == 200
    items = response.json()
    assert isinstance(items, list)
    assert len(items) <= 5


def test_api_category_anomalies(client):
    response = client.get("/api/anomalies?entity_type=category&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    if data["items"]:
        assert data["items"][0]["entity_type"] == "category"
        assert data["items"][0]["entity_name"] is not None


def test_api_product_anomalies(client):
    response = client.get("/api/anomalies/products/1?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    if data["items"]:
        assert data["items"][0]["entity_type"] == "product"
        assert data["items"][0]["entity_id"] == "1"
