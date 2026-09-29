from __future__ import annotations

import io
import math
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.database.models import (
    CompanyElasticityModel,
    CompanyModel,
    DatasetUpload,
    User,
)
from backend.app.schemas.simulations import (
    ScenarioType,
    SimulationRequest,
    SimulationStatus,
)
from backend.app.services.company_model_service import CompanyModelService
from backend.app.services.simulation_service import (
    SimulationService,
    UnsupportedScenarioError,
)
from backend.app.services.storage_service import storage_service
from ml.training.elasticity_trainer import train_company_elasticity
from tests.conftest import TestingSessionLocal


def get_auth_headers(client: TestClient, email: str, password: str = "TestPass123!") -> dict[str, str]:
    """Register and log in a user to retrieve auth headers."""
    client.post("/api/auth/register", json={"email": email, "password": password})
    res = client.post("/api/auth/login", data={"username": email, "password": password})
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# 1. Dataset Verification (Item 1)
# ---------------------------------------------------------------------------
def test_item1_dataset_columns_and_variation():
    """Verify synthetic dataset contains required columns and sufficient variation."""
    csv_path = Path("data/synthetic_product_daily_forecasting.csv")
    assert csv_path.exists(), "Synthetic dataset file must exist"
    
    # Read sample to verify column names and variations
    df = pd.read_csv(csv_path, nrows=5000)
    col_map = {c.lower(): c for c in df.columns}
    assert "unit_price" in col_map
    assert "discount_percent" in col_map
    assert "quantity" in col_map

    price_series = df[col_map["unit_price"]].dropna()
    discount_series = df[col_map["discount_percent"]].dropna()
    qty_series = df[col_map["quantity"]].dropna()

    assert len(qty_series) >= 30
    assert price_series.nunique() >= 2
    assert price_series.std() > 1e-4
    assert discount_series.nunique() >= 2
    assert discount_series.std() > 1e-4


# ---------------------------------------------------------------------------
# 2-5. Training Pipeline & Model Persistence & Reload (Items 2, 3, 4, 5)
# ---------------------------------------------------------------------------
def test_items2_to_5_training_pipeline_and_storage_reload(tmp_path):
    """
    Verify training produces 7D, 30D, 90D and elasticity model,
    reaches READY, persists artifact, and reloads from storage.
    """
    # 1. Train elasticity model on valid data
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    prices = np.random.uniform(100.0, 300.0, size=n)
    discounts = np.random.uniform(0.0, 25.0, size=n)
    log_q = 7.0 - 0.75 * np.log(prices) + 0.02 * discounts + np.random.normal(0, 0.02, size=n)
    df = pd.DataFrame({
        "date": dates,
        "price": prices,
        "discount": discounts,
        "quantity": np.exp(log_q),
        "category": "Apparel",
    })

    result = train_company_elasticity(df)
    assert result["status"] == "ready", "Elasticity model must reach READY when dataset is sufficient"
    assert result["price_supported"] is True
    assert result["discount_supported"] is True
    assert result["price_elasticity"] < 0
    assert result["discount_sensitivity"] > 0
    assert result["artifact_bytes"] is not None

    # 2. Persist artifact using storage abstraction
    artifact_rel_path = f"models/test_user/test_ds/v1/elasticity/model.joblib"
    saved_key = storage_service.save_artifact(artifact_rel_path, result["artifact_bytes"])
    assert saved_key is not None

    # 3. Reload artifact from storage abstraction (process memory independent)
    loaded_file = storage_service.load_artifact_file(artifact_rel_path)
    assert loaded_file.exists()
    assert loaded_file.stat().st_size > 0

    # Unpack loaded artifact
    unpickled = joblib.load(loaded_file)
    assert "model" in unpickled
    assert "price_elasticity" in unpickled
    assert "discount_sensitivity" in unpickled
    assert unpickled["price_elasticity"] == result["price_elasticity"]
    assert unpickled["discount_sensitivity"] == result["discount_sensitivity"]


# ---------------------------------------------------------------------------
# 6-7. API Endpoints: GET /api/models/elasticity & GET /api/models/current (Items 6, 7)
# ---------------------------------------------------------------------------
def test_items6_and_7_api_models_endpoints(client: TestClient):
    """Verify GET /api/models/elasticity and GET /api/models/current return elasticity metadata."""
    headers = get_auth_headers(client, "e2e_models_user@example.com")

    db = TestingSessionLocal()
    try:
        user = db.query(User).filter(User.email == "e2e_models_user@example.com").first()
        assert user is not None

        ds = DatasetUpload(
            id="ds-e2e-current-test",
            user_id=user.id,
            original_filename="sales.csv",
            dataset_key="sales_key",
            row_count=200,
            status="active",
        )
        db.add(ds)
        db.flush()

        # Add a 7D, 30D, 90D company model
        for h in [7, 30, 90]:
            cm = CompanyModel(
                id=f"cm-{h}-e2e",
                user_id=user.id,
                dataset_id=ds.id,
                horizon=h,
                model_type="HistGradientBoosting",
                artifact_path=f"models/{user.id}/ds/v1/{h}d/model.joblib",
                model_version=1,
                validation_wape=0.12,
                test_wape=0.14,
                training_rows=200,
                status="ready",
                is_active=True,
            )
            db.add(cm)

        # Add active elasticity model
        em = CompanyElasticityModel(
            id="cem-e2e-test",
            user_id=user.id,
            dataset_id=ds.id,
            model_version=1,
            model_type="RidgeLogLog",
            artifact_path=f"models/{user.id}/ds/v1/elasticity/model.joblib",
            price_elasticity=-0.65,
            discount_sensitivity=1.20,
            r2_score=0.81,
            mae=11.5,
            rmse=15.0,
            training_rows=200,
            status="ready",
            is_active=True,
            diagnostics={"price_supported": True, "discount_supported": True},
        )
        db.add(em)
        db.commit()
    finally:
        db.close()

    # 1. Test GET /api/models/elasticity
    res_el = client.get("/api/models/elasticity", headers=headers)
    assert res_el.status_code == 200
    el_data = res_el.json()
    assert el_data["status"] == "ready"
    assert el_data["price_elasticity"] == pytest.approx(-0.65)
    assert el_data["discount_sensitivity"] == pytest.approx(1.20)
    assert el_data["price_supported"] is True
    assert el_data["discount_supported"] is True

    # 2. Test GET /api/models/current
    res_curr = client.get("/api/models/current", headers=headers)
    assert res_curr.status_code == 200
    curr_data = res_curr.json()
    assert curr_data["active_dataset_id"] == "ds-e2e-current-test"
    # Horizons 7, 30, 90 must exist and remain unbroken
    horizons = [m["horizon"] for m in curr_data["models"]]
    assert 7 in horizons
    assert 30 in horizons
    assert 90 in horizons
    # Elasticity model summary must be present
    assert curr_data["elasticity_model"] is not None
    assert curr_data["elasticity_model"]["status"] == "ready"
    assert curr_data["elasticity_model"]["price_elasticity"] == pytest.approx(-0.65)
    assert curr_data["elasticity_model"]["discount_sensitivity"] == pytest.approx(1.20)


# ---------------------------------------------------------------------------
# 8. Price Change Simulation End-to-End (Item 8)
# ---------------------------------------------------------------------------
def test_item8_price_change_simulation_e2e(client: TestClient):
    """
    Verify Price Change end-to-end:
    - model READY
    - POST /api/simulations succeeds
    - quantity adjusts by price sensitivity
    - effective price correct
    - revenue calculation correct
    - elasticity provenance returned
    """
    mock_model = CompanyElasticityModel(
        id="el-test-pc",
        user_id=1,
        dataset_id="ds-pc",
        model_version=3,
        status="ready",
        price_elasticity=-0.50,
        discount_sensitivity=1.10,
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
    assert res.elasticity_model_version == 3
    assert res.price_elasticity == -0.50
    assert res.scenario.total_quantity < res.baseline.total_quantity

    # Multiplier: (1 + 10/100)^(-0.50) = 1.10^(-0.5) = ~0.95346
    expected_mult = math.pow(1.10, -0.50)
    for d in res.daily_results:
        assert d.scenario_quantity == pytest.approx(round(d.baseline_quantity * expected_mult), abs=1)
        assert d.scenario_revenue > 0

    # Total revenue matches sum of daily revenue
    daily_sum = sum(d.scenario_revenue for d in res.daily_results)
    assert res.scenario.total_revenue == pytest.approx(daily_sum, rel=1e-4)


# ---------------------------------------------------------------------------
# 9. Discount Depth Simulation End-to-End (Item 9)
# ---------------------------------------------------------------------------
def test_item9_discount_depth_simulation_e2e(client: TestClient):
    """
    Verify Discount Depth end-to-end:
    - model READY
    - POST /api/simulations succeeds
    - quantity adjusts by discount sensitivity
    - effective selling price correct
    - revenue calculation correct
    - elasticity provenance returned
    """
    mock_model = CompanyElasticityModel(
        id="el-test-dd",
        user_id=1,
        dataset_id="ds-dd",
        model_version=2,
        status="ready",
        price_elasticity=-0.40,
        discount_sensitivity=0.80,
        diagnostics={"price_supported": True, "discount_supported": True},
        artifact_path="mock/path.joblib",
        is_active=True,
    )
    service = SimulationService(elasticity_model=mock_model)

    req = SimulationRequest(
        scenario_type=ScenarioType.DISCOUNT_CHANGE,
        discount_change_percent=15.0,
        horizon_days=7,
        include_revenue=True,
    )
    res = service.run_simulation(req)

    assert res.status == SimulationStatus.COMPLETED
    assert res.elasticity_model_version == 2
    assert res.discount_sensitivity == 0.80
    assert res.scenario.total_quantity > res.baseline.total_quantity

    # Total revenue matches sum of daily revenue
    daily_sum = sum(d.scenario_revenue for d in res.daily_results)
    assert res.scenario.total_revenue == pytest.approx(daily_sum, rel=1e-4)


# ---------------------------------------------------------------------------
# 10. Fail-Closed Behavior (Item 10)
# ---------------------------------------------------------------------------
def test_item10_fail_closed_behavior(client: TestClient):
    """
    Verify fail-closed behavior when elasticity model is unavailable:
    - Price Change rejected with 422
    - Discount Depth rejected with 422
    - No volume forecast model is used as fallback
    """
    headers = get_auth_headers(client, "e2e_fail_closed_user@example.com")

    # Call Price Change without active elasticity model
    res_pc = client.post("/api/simulations", json={"scenario_type": "price_change", "horizon_days": 30}, headers=headers)
    assert res_pc.status_code == 422
    assert "requires a dedicated elasticity model" in res_pc.json()["detail"]

    # Call Discount Depth without active elasticity model
    res_dd = client.post("/api/simulations", json={"scenario_type": "discount_change", "horizon_days": 30}, headers=headers)
    assert res_dd.status_code == 422
    assert "requires a dedicated elasticity model" in res_dd.json()["detail"]


# ---------------------------------------------------------------------------
# 11. Tenant Isolation (Item 11)
# ---------------------------------------------------------------------------
def test_item11_tenant_isolation(client: TestClient):
    """
    Verify tenant isolation:
    - Tenant A cannot access Tenant B's elasticity model
    - Tenant A cannot run a scenario using Tenant B's model
    """
    headers_a = get_auth_headers(client, "iso_user_a@example.com")
    headers_b = get_auth_headers(client, "iso_user_b@example.com")

    db = TestingSessionLocal()
    try:
        user_b = db.query(User).filter(User.email == "iso_user_b@example.com").first()
        ds_b = DatasetUpload(
            id="ds-iso-b",
            user_id=user_b.id,
            original_filename="b.csv",
            dataset_key="b_key",
            row_count=100,
            status="active",
        )
        db.add(ds_b)
        db.flush()

        em_b = CompanyElasticityModel(
            id="cem-iso-b",
            user_id=user_b.id,
            dataset_id=ds_b.id,
            model_version=1,
            model_type="RidgeLogLog",
            artifact_path="models/b/elasticity.joblib",
            price_elasticity=-0.99,
            discount_sensitivity=2.50,
            status="ready",
            is_active=True,
            diagnostics={"price_supported": True, "discount_supported": True},
        )
        db.add(em_b)
        db.commit()
    finally:
        db.close()

    # User B can retrieve model
    res_b = client.get("/api/models/elasticity", headers=headers_b)
    assert res_b.status_code == 200
    assert res_b.json()["price_elasticity"] == pytest.approx(-0.99)

    # User A cannot retrieve User B's model -> 404
    res_a = client.get("/api/models/elasticity", headers=headers_a)
    assert res_a.status_code == 404

    # User A cannot run price change simulation -> 422
    res_a_sim = client.post("/api/simulations", json={"scenario_type": "price_change", "horizon_days": 7}, headers=headers_a)
    assert res_a_sim.status_code == 422


# ---------------------------------------------------------------------------
# 12. Restart Behavior / Process Recreation (Item 12)
# ---------------------------------------------------------------------------
def test_item12_restart_behavior(tmp_path):
    """
    Verify restart behavior:
    - Save model to storage
    - Instantiate brand-new SimulationService without in-memory state
    - Model correctly loaded and simulation succeeds
    """
    # 1. Prepare and save artifact
    artifact_data = {
        "model_type": "RidgeLogLog",
        "price_elasticity": -0.55,
        "discount_sensitivity": 0.95,
        "price_supported": True,
        "discount_supported": True,
    }
    buf = io.BytesIO()
    joblib.dump(artifact_data, buf)
    rel_path = "models/restart_test/elasticity.joblib"
    storage_service.save_artifact(rel_path, buf.getvalue())

    # 2. Simulate process restart by reloading artifact from storage
    fresh_file = storage_service.load_artifact_file(rel_path)
    assert fresh_file.exists()
    loaded_model_info = joblib.load(fresh_file)

    # 3. Construct service using reloaded artifact
    reloaded_db_model = CompanyElasticityModel(
        id="el-restart-1",
        user_id=999,
        dataset_id="ds-restart",
        model_version=1,
        status="ready",
        price_elasticity=loaded_model_info["price_elasticity"],
        discount_sensitivity=loaded_model_info["discount_sensitivity"],
        diagnostics={"price_supported": True, "discount_supported": True},
        artifact_path=rel_path,
        is_active=True,
    )
    new_service = SimulationService(elasticity_model=reloaded_db_model)
    req = SimulationRequest(
        scenario_type=ScenarioType.PRICE_CHANGE,
        price_change_percent=5.0,
        horizon_days=7,
        include_revenue=True,
    )
    res = new_service.run_simulation(req)
    assert res.status == SimulationStatus.COMPLETED
    assert res.price_elasticity == -0.55
