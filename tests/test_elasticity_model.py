from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.database.models import CompanyElasticityModel, DatasetUpload, User
from tests.conftest import TestingSessionLocal
from backend.app.schemas.simulations import (
    ScenarioType,
    SimulationRequest,
    SimulationResponse,
    SimulationStatus,
)
from backend.app.services.simulation_service import (
    SimulationService,
    UnsupportedScenarioError,
)
from ml.training.elasticity_trainer import train_company_elasticity


def get_auth_headers(client: TestClient, email: str = "elasticity_tester@example.com") -> dict[str, str]:
    """Helper to register/login and retrieve auth bearer token."""
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
# 1. Elasticity Trainer Unit Tests
# ---------------------------------------------------------------------------

def test_elasticity_trainer_ready():
    """Train on 100 observations with valid variation in price and discount."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    prices = np.random.uniform(50.0, 150.0, size=n)
    discounts = np.random.uniform(0.0, 30.0, size=n)
    # Log-log relation: log(Q) = 6 - 0.6 * log(P) + 0.015 * D + noise
    log_q = 6.0 - 0.6 * np.log(prices) + 0.015 * discounts + np.random.normal(0, 0.05, size=n)
    quantities = np.exp(log_q)

    df = pd.DataFrame({
        "date": dates,
        "price": prices,
        "discount": discounts,
        "quantity": quantities,
        "category": "Electronics",
    })

    result = train_company_elasticity(df)

    assert result["status"] == "ready"
    assert result["price_supported"] is True
    assert result["discount_supported"] is True
    assert result["price_elasticity"] is not None
    assert result["discount_sensitivity"] is not None
    # Price elasticity should be negative and reasonably close to -0.6
    assert result["price_elasticity"] < 0
    assert pytest.approx(-0.6, abs=0.4) == result["price_elasticity"]
    # Discount sensitivity should be positive and reasonably close to 1.5
    assert result["discount_sensitivity"] > 0
    assert result["artifact_bytes"] is not None
    assert len(result["artifact_bytes"]) > 0
    assert result["r2_score"] is not None
    assert result["mae"] is not None
    assert result["rmse"] is not None


def test_elasticity_trainer_insufficient_data():
    """Train on fewer than 30 observations -> status insufficient_data."""
    n = 20
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "price": np.random.uniform(50.0, 100.0, size=n),
        "discount": np.random.uniform(0.0, 20.0, size=n),
        "quantity": np.random.uniform(10.0, 50.0, size=n),
    })

    result = train_company_elasticity(df)

    assert result["status"] == "insufficient_data"
    assert result["price_supported"] is False
    assert result["discount_supported"] is False
    assert result["artifact_bytes"] is None
    assert "minimum 30" in (result["status_message"] or "").lower()


def test_elasticity_trainer_constant_price():
    """Train on data with constant price -> price unsupported, discount supported."""
    np.random.seed(42)
    n = 60
    prices = np.full(n, 100.0)
    discounts = np.random.uniform(0.0, 30.0, size=n)
    quantities = 50.0 + 1.2 * discounts + np.random.normal(0, 1.0, size=n)

    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "price": prices,
        "discount": discounts,
        "quantity": quantities,
    })

    result = train_company_elasticity(df)

    assert result["status"] == "ready"
    assert result["price_supported"] is False
    assert result["price_elasticity"] is None
    assert "price variation" in (result["price_reason"] or "").lower()
    assert result["discount_supported"] is True
    assert result["discount_sensitivity"] is not None
    assert result["artifact_bytes"] is not None


def test_elasticity_trainer_constant_discount():
    """Train on data with constant zero discount -> discount unsupported, price supported."""
    np.random.seed(42)
    n = 60
    prices = np.random.uniform(50.0, 150.0, size=n)
    discounts = np.zeros(n)
    quantities = 200.0 - 0.8 * prices + np.random.normal(0, 2.0, size=n)

    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "price": prices,
        "discount": discounts,
        "quantity": np.maximum(quantities, 1.0),
    })

    result = train_company_elasticity(df)

    assert result["status"] == "ready"
    assert result["discount_supported"] is False
    assert result["discount_sensitivity"] is None
    assert "discount variation" in (result["discount_reason"] or "").lower()
    assert result["price_supported"] is True
    assert result["price_elasticity"] is not None
    assert result["artifact_bytes"] is not None


# ---------------------------------------------------------------------------
# 2. Simulation Service with Elasticity Model
# ---------------------------------------------------------------------------

def test_simulation_price_change_with_model():
    """Simulate a price change scenario with a ready elasticity model."""
    mock_model = CompanyElasticityModel(
        id="el-test-1",
        user_id=1,
        dataset_id="ds-1",
        model_version=1,
        status="ready",
        price_elasticity=-0.5,
        discount_sensitivity=1.2,
        diagnostics={"price_supported": True, "discount_supported": True},
        artifact_path="mock/path.joblib",
        is_active=True,
    )

    service = SimulationService(elasticity_model=mock_model)
    req = SimulationRequest(
        scenario_type=ScenarioType.PRICE_CHANGE,
        price_change_percent=10.0,
        horizon_days=7,
        include_revenue=True,
    )
    res = service.run_simulation(req)

    assert res.status == SimulationStatus.COMPLETED
    assert res.elasticity_model_version == 1
    assert res.price_elasticity == -0.5
    assert res.discount_sensitivity == 1.2
    assert res.baseline is not None
    assert res.scenario is not None
    assert res.delta is not None
    # Because price elasticity is negative and price rose by 10%, scenario volume should be lower
    assert res.scenario.total_quantity < res.baseline.total_quantity
    # Verify non-negativity and daily consistency
    assert len(res.daily_results) == 7
    total_rev = 0.0
    for d in res.daily_results:
        assert d.scenario_quantity >= 0
        assert d.scenario_revenue is not None
        assert d.scenario_revenue >= 0
        total_rev += d.scenario_revenue
    assert res.scenario.total_revenue == pytest.approx(total_rev, rel=1e-4)


def test_simulation_discount_change_with_model():
    """Simulate a discount depth scenario with a ready elasticity model."""
    mock_model = CompanyElasticityModel(
        id="el-test-2",
        user_id=1,
        dataset_id="ds-1",
        model_version=2,
        status="ready",
        price_elasticity=-0.4,
        discount_sensitivity=0.8,
        diagnostics={"price_supported": True, "discount_supported": True},
        artifact_path="mock/path.joblib",
        is_active=True,
    )

    service = SimulationService(elasticity_model=mock_model)
    req = SimulationRequest(
        scenario_type=ScenarioType.DISCOUNT_CHANGE,
        discount_change_percent=15.0,
        horizon_days=30,
        include_revenue=True,
    )
    res = service.run_simulation(req)

    assert res.status == SimulationStatus.COMPLETED
    assert res.elasticity_model_version == 2
    assert res.discount_sensitivity == 0.8
    assert res.baseline is not None
    assert res.scenario is not None
    assert res.delta is not None
    # Because discount sensitivity is positive and discount depth increased by 15%, volume should expand
    assert res.scenario.total_quantity > res.baseline.total_quantity
    # Verify non-negativity and daily consistency
    assert len(res.daily_results) == 30
    total_rev = 0.0
    for d in res.daily_results:
        assert d.scenario_quantity >= 0
        assert d.scenario_revenue is not None
        assert d.scenario_revenue >= 0
        total_rev += d.scenario_revenue
    assert res.scenario.total_revenue == pytest.approx(total_rev, rel=1e-4)


def test_simulation_price_change_rejected_when_price_not_supported():
    """Price change must raise UnsupportedScenarioError when price_supported is False."""
    mock_model = CompanyElasticityModel(
        id="el-test-3",
        user_id=1,
        dataset_id="ds-1",
        model_version=1,
        status="ready",
        price_elasticity=None,
        discount_sensitivity=1.2,
        diagnostics={
            "price_supported": False,
            "price_reason": "No price variation in historical sales.",
            "discount_supported": True,
        },
        artifact_path="mock/path.joblib",
        is_active=True,
    )

    service = SimulationService(elasticity_model=mock_model)
    req = SimulationRequest(
        scenario_type=ScenarioType.PRICE_CHANGE,
        price_change_percent=5.0,
        horizon_days=7,
    )
    with pytest.raises(UnsupportedScenarioError) as exc_info:
        service.run_simulation(req)
    assert "No price variation in historical sales." in str(exc_info.value)


def test_simulation_discount_change_rejected_when_discount_not_supported():
    """Discount change must raise UnsupportedScenarioError when discount_supported is False."""
    mock_model = CompanyElasticityModel(
        id="el-test-4",
        user_id=1,
        dataset_id="ds-1",
        model_version=1,
        status="ready",
        price_elasticity=-0.5,
        discount_sensitivity=None,
        diagnostics={
            "price_supported": True,
            "discount_supported": False,
            "discount_reason": "No discount markdown records found in dataset.",
        },
        artifact_path="mock/path.joblib",
        is_active=True,
    )

    service = SimulationService(elasticity_model=mock_model)
    req = SimulationRequest(
        scenario_type=ScenarioType.DISCOUNT_CHANGE,
        discount_change_percent=10.0,
        horizon_days=7,
    )
    with pytest.raises(UnsupportedScenarioError) as exc_info:
        service.run_simulation(req)
    assert "No discount markdown records found in dataset." in str(exc_info.value)


# ---------------------------------------------------------------------------
# 3. API & Tenant Isolation Tests
# ---------------------------------------------------------------------------

def test_api_elasticity_model_endpoint_and_isolation(client: TestClient):
    """Verify GET /api/models/elasticity enforces strict tenant isolation."""
    headers_a = get_auth_headers(client, email="user_a_el@example.com")
    headers_b = get_auth_headers(client, email="user_b_el@example.com")

    # User A initially has no active model -> 404
    res_a_init = client.get("/api/models/elasticity", headers=headers_a)
    assert res_a_init.status_code == 404

    # Direct DB setup: Insert active dataset and elasticity model for User A
    db = TestingSessionLocal()
    try:
        user_a = db.query(User).filter(User.email == "user_a_el@example.com").first()
        assert user_a is not None

        # Create active dataset upload for User A
        ds_a = DatasetUpload(
            id="ds-test-tenant-a",
            user_id=user_a.id,
            original_filename="sales_a.csv",
            dataset_key="sales_a_key",
            row_count=100,
            status="active",
        )
        db.add(ds_a)
        db.flush()

        model_a = CompanyElasticityModel(
            id="el-model-tenant-a",
            user_id=user_a.id,
            dataset_id=ds_a.id,
            model_version=1,
            status="ready",
            price_elasticity=-0.72,
            discount_sensitivity=1.45,
            artifact_path="company-models/comp_a_el_v1.pkl",
            r2_score=0.82,
            mae=12.4,
            rmse=15.8,
            training_rows=100,
            diagnostics={
                "price_supported": True,
                "discount_supported": True,
            },
            is_active=True,
        )
        db.add(model_a)
        db.commit()
    finally:
        db.close()

    # User A now retrieves their elasticity model -> 200
    res_a = client.get("/api/models/elasticity", headers=headers_a)
    assert res_a.status_code == 200
    data_a = res_a.json()
    assert data_a["status"] == "ready"
    assert data_a["price_elasticity"] == pytest.approx(-0.72)
    assert data_a["discount_sensitivity"] == pytest.approx(1.45)
    assert data_a["price_supported"] is True
    assert data_a["discount_supported"] is True

    # User B retrieves elasticity model -> 404 (strictly isolated)
    res_b = client.get("/api/models/elasticity", headers=headers_b)
    assert res_b.status_code == 404
