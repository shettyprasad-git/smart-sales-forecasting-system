from __future__ import annotations

import io
import uuid
from datetime import date, datetime, timedelta, timezone
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
    ModelTrainingJob,
    Product,
    SalesRecord,
    User,
)
from backend.app.schemas.models import ElasticityModelSummary
from backend.app.schemas.simulations import (
    ScenarioType,
    SimulationRequest,
    SimulationResponse,
    SimulationStatus,
)
from backend.app.services.company_model_service import company_model_service
from backend.app.services.simulation_service import (
    SimulationService,
    UnsupportedScenarioError,
)
from backend.app.services.storage_service import storage_service
from ml.training.elasticity_trainer import train_company_elasticity
from tests.conftest import TestingSessionLocal


def register_and_login(client: TestClient, email: str, password: str = "LifecyclePass123!") -> dict[str, str]:
    """Register and log in a user to retrieve auth bearer token."""
    client.post("/api/auth/register", json={"email": email, "password": password})
    res = client.post("/api/auth/login", data={"username": email, "password": password})
    assert res.status_code == 200, f"Login failed for {email}: {res.text}"
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def seed_test_sales_history(
    db_session,
    user_id: int,
    dataset_id: str,
    days: int = 120,
    start_date: date = date(2024, 1, 1),
) -> None:
    """Seed synthetic sales records with realistic price and discount variations."""
    prod_code = f"PROD-LC-{user_id}"
    product = db_session.query(Product).filter(Product.product_id == prod_code).first()
    if not product:
        product = Product(
            product_id=prod_code,
            product_name="Lifecycle Test Product",
            category_id="CAT-ELEC",
            category_name="Electronics",
            unit_price=25.0,
            tenant_id=user_id,
            raw_product_id=prod_code,
        )
        db_session.add(product)
        db_session.flush()

    rng = np.random.default_rng(42)
    for i in range(days):
        d = start_date + timedelta(days=i)
        # Price varies between $20.0 and $35.0
        unit_price = round(float(25.0 + 5.0 * np.sin(2 * np.pi * i / 14) + rng.uniform(-2, 2)), 2)
        # Discount varies between 0% and 20%
        discount_percent = float(10.0 if (i % 7 in (5, 6)) else (15.0 if (i % 14 == 0) else 0.0))
        # Log-log demand with price elasticity ~ -0.6 and discount sensitivity ~ 1.2
        log_q = 6.0 - 0.6 * np.log(unit_price) + 0.012 * discount_percent + rng.normal(0, 0.05)
        quantity = max(10.0, float(np.exp(log_q) * 50.0))
        sales_amount = quantity * unit_price * (1.0 - discount_percent / 100.0)
        profit = sales_amount * 0.25

        sr = SalesRecord(
            user_id=user_id,
            dataset_id=dataset_id,
            product_id=product.id,
            sale_date=d,
            quantity=quantity,
            unit_price=unit_price,
            discount_percent=discount_percent,
            sales_amount=sales_amount,
            profit=profit,
            promotion=(discount_percent > 0),
            holiday_flag=(i % 30 == 0),
        )
        db_session.add(sr)
    db_session.commit()


# ---------------------------------------------------------------------------
# Test 1-5: Full Lifecycle: Training -> DB Record -> Storage Artifact -> Ready State
# ---------------------------------------------------------------------------

def test_company_training_pipeline_triggers_elasticity_and_persists_artifact(db_session):
    """
    Phases 1-5 Verification:
    1. Training pipeline is triggered.
    2. Valid dataset produces 7D, 30D, 90D models AND elasticity model.
    3. CompanyElasticityModel row is created.
    4. Artifact is uploaded to exact expected storage path:
       models/{user_id}/{dataset_id}/v{version}/elasticity/model.joblib
    5. Elasticity model reaches READY and becomes active.
    """
    user = User(email=f"lifecycle_tr_{uuid.uuid4().hex[:6]}@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    ds = DatasetUpload(
        id=f"ds-lc-{uuid.uuid4().hex[:8]}",
        user_id=user.id,
        original_filename="lifecycle_sales.csv",
        dataset_key=f"raw/{user.id}/sales.csv",
        row_count=250,
        status="active",
    )
    db_session.add(ds)
    db_session.flush()

    # Seed sales records with price and discount variation (250 days for 7D, 30D, and 90D horizons)
    seed_test_sales_history(db_session, user.id, ds.id, days=250)

    # Enqueue training job
    job, acquired = company_model_service.enqueue_training_job(db_session, user.id, ds.id)
    assert acquired is True
    assert job.status == "queued"

    # Execute training synchronously using test session
    with patch("backend.app.services.company_model_service.SessionLocal", TestingSessionLocal):
        company_model_service.execute_training_pipeline(user.id, ds.id, job.id)

    db_session.refresh(job)
    assert job.status == "ready", f"Job failed: {job.error_message}"

    # Confirm all 3 forecast models (7D, 30D, 90D) are READY and active
    forecast_models = (
        db_session.query(CompanyModel)
        .filter(CompanyModel.user_id == user.id, CompanyModel.dataset_id == ds.id)
        .all()
    )
    assert len(forecast_models) == 3
    horizons = {m.horizon: m for m in forecast_models}
    for h in [7, 30, 90]:
        assert h in horizons
        assert horizons[h].status == "ready"
        assert horizons[h].is_active is True

    # Confirm CompanyElasticityModel row was created
    elasticity_model = (
        db_session.query(CompanyElasticityModel)
        .filter(CompanyElasticityModel.user_id == user.id, CompanyElasticityModel.dataset_id == ds.id)
        .first()
    )
    assert elasticity_model is not None
    assert elasticity_model.status == "ready"
    assert elasticity_model.is_active is True
    assert elasticity_model.model_version == 1
    assert elasticity_model.model_type == "RidgeLogLog"
    assert elasticity_model.price_elasticity is not None
    assert elasticity_model.price_elasticity < 0
    assert elasticity_model.discount_sensitivity is not None
    assert elasticity_model.discount_sensitivity > 0

    # Confirm exact storage path convention
    expected_path = f"models/{user.id}/{ds.id}/v1/elasticity/model.joblib"
    assert elasticity_model.artifact_path == expected_path

    # Confirm artifact exists in storage and can be unpickled
    local_file = storage_service.load_artifact_file(expected_path)
    assert local_file.exists()
    assert local_file.stat().st_size > 0

    artifact_content = joblib.load(local_file)
    assert "model" in artifact_content
    assert "price_elasticity" in artifact_content
    assert "discount_sensitivity" in artifact_content
    assert artifact_content["price_elasticity"] == elasticity_model.price_elasticity
    assert artifact_content["discount_sensitivity"] == elasticity_model.discount_sensitivity


# ---------------------------------------------------------------------------
# Test 6: API Endpoints (GET /api/models/elasticity & GET /api/models/current)
# ---------------------------------------------------------------------------

def test_api_models_endpoints_return_ready_elasticity(client: TestClient, db_session):
    """
    Phases 6-7 Verification:
    - GET /api/models/elasticity returns active ready model.
    - GET /api/models/current includes elasticity summary without breaking horizon models.
    """
    email = f"api_user_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)

    user = db_session.query(User).filter(User.email == email).first()
    assert user is not None

    ds = DatasetUpload(
        id=f"ds-api-{uuid.uuid4().hex[:8]}",
        user_id=user.id,
        original_filename="api_sales.csv",
        dataset_key="raw/key.csv",
        row_count=100,
        status="active",
    )
    db_session.add(ds)
    db_session.flush()

    # Create active company models (7, 30, 90)
    for h in [7, 30, 90]:
        cm = CompanyModel(
            id=f"cm-h{h}-{uuid.uuid4().hex[:6]}",
            user_id=user.id,
            dataset_id=ds.id,
            horizon=h,
            model_type="HistGradientBoosting",
            artifact_path=f"models/{user.id}/{ds.id}/v1/{h}d/model.joblib",
            model_version=1,
            validation_wape=0.10,
            status="ready",
            is_active=True,
        )
        db_session.add(cm)

    # Create active elasticity model
    em = CompanyElasticityModel(
        id=f"cem-{uuid.uuid4().hex[:8]}",
        user_id=user.id,
        dataset_id=ds.id,
        model_version=1,
        model_type="RidgeLogLog",
        artifact_path=f"models/{user.id}/{ds.id}/v1/elasticity/model.joblib",
        price_elasticity=-0.58,
        discount_sensitivity=1.12,
        r2_score=0.85,
        mae=10.2,
        rmse=13.1,
        training_rows=100,
        status="ready",
        is_active=True,
        diagnostics={"price_supported": True, "discount_supported": True},
    )
    db_session.add(em)
    db_session.commit()

    # 1. GET /api/models/elasticity
    res_el = client.get("/api/models/elasticity", headers=headers)
    assert res_el.status_code == 200
    el_json = res_el.json()
    assert el_json["status"] == "ready"
    assert el_json["price_elasticity"] == pytest.approx(-0.58)
    assert el_json["discount_sensitivity"] == pytest.approx(1.12)
    assert el_json["price_supported"] is True
    assert el_json["discount_supported"] is True
    assert el_json["dataset_id"] == ds.id

    # 2. GET /api/models/current
    res_curr = client.get("/api/models/current", headers=headers)
    assert res_curr.status_code == 200
    curr_json = res_curr.json()
    assert curr_json["active_dataset_id"] == ds.id
    assert len(curr_json["models"]) == 3
    assert curr_json["elasticity_model"] is not None
    assert curr_json["elasticity_model"]["status"] == "ready"
    assert curr_json["elasticity_model"]["price_elasticity"] == pytest.approx(-0.58)
    assert curr_json["elasticity_model"]["discount_sensitivity"] == pytest.approx(1.12)


# ---------------------------------------------------------------------------
# Test 7-8: Simulations resolve ready model & load artifact from storage
# ---------------------------------------------------------------------------

def test_price_change_and_discount_depth_simulations_resolve_and_load_artifact(client: TestClient, db_session):
    """
    Phases 7-8 & 14-15 Verification:
    - Price change simulation resolves ready model.
    - Discount depth simulation resolves ready model.
    - Artifact is loaded from storage backend via storage_service.load_artifact_file.
    - Simulation runs end-to-end via POST /api/simulations.
    """
    email = f"sim_user_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)

    user = db_session.query(User).filter(User.email == email).first()
    assert user is not None

    ds = DatasetUpload(
        id=f"ds-sim-{uuid.uuid4().hex[:8]}",
        user_id=user.id,
        original_filename="sim_sales.csv",
        dataset_key="raw/sim.csv",
        row_count=100,
        status="active",
    )
    db_session.add(ds)
    db_session.flush()

    # Seed sales records so baseline forecast generation succeeds
    seed_test_sales_history(db_session, user.id, ds.id, days=60)

    # Save a real artifact to storage
    rel_path = f"models/{user.id}/{ds.id}/v1/elasticity/model.joblib"
    fake_artifact = {
        "model": "RidgeMock",
        "price_elasticity": -0.62,
        "discount_sensitivity": 1.25,
        "features": ["log_unit_price", "discount_rate"],
    }
    buf = io.BytesIO()
    joblib.dump(fake_artifact, buf)
    storage_service.save_artifact(rel_path, buf.getvalue())

    # Create active CompanyElasticityModel
    em = CompanyElasticityModel(
        id=f"cem-sim-{uuid.uuid4().hex[:8]}",
        user_id=user.id,
        dataset_id=ds.id,
        model_version=1,
        model_type="RidgeLogLog",
        artifact_path=rel_path,
        price_elasticity=-0.62,
        discount_sensitivity=1.25,
        r2_score=0.88,
        mae=8.5,
        rmse=11.2,
        training_rows=100,
        status="ready",
        is_active=True,
        diagnostics={"price_supported": True, "discount_supported": True},
    )
    db_session.add(em)
    db_session.commit()

    # A. Run Price Change simulation (+10% price)
    res_pc = client.post(
        "/api/simulations",
        headers=headers,
        json={
            "scenario_type": "price_change",
            "price_change_percent": 10.0,
            "horizon_days": 7,
            "include_revenue": True,
        },
    )
    assert res_pc.status_code == 200, f"Simulation failed: {res_pc.text}"
    pc_data = res_pc.json()
    assert pc_data["status"] == "completed"
    assert pc_data["elasticity_model_version"] == 1
    assert pc_data["price_elasticity"] == pytest.approx(-0.62)
    # Higher price -> lower quantity
    assert pc_data["scenario"]["total_quantity"] < pc_data["baseline"]["total_quantity"]
    assert len(pc_data["daily_results"]) == 7

    # B. Run Discount Depth simulation (+15% discount)
    res_dc = client.post(
        "/api/simulations",
        headers=headers,
        json={
            "scenario_type": "discount_change",
            "discount_change_percent": 15.0,
            "horizon_days": 7,
            "include_revenue": True,
        },
    )
    assert res_dc.status_code == 200, f"Simulation failed: {res_dc.text}"
    dc_data = res_dc.json()
    assert dc_data["status"] == "completed"
    assert dc_data["elasticity_model_version"] == 1
    assert dc_data["discount_sensitivity"] == pytest.approx(1.25)
    # Greater discount -> expanded volume
    assert dc_data["scenario"]["total_quantity"] > dc_data["baseline"]["total_quantity"]


# ---------------------------------------------------------------------------
# Test 9: Missing model returns 422 with clear diagnostic
# ---------------------------------------------------------------------------

def test_missing_model_returns_422_with_diagnostic(client: TestClient):
    """
    Phase 9 Verification:
    If no elasticity model exists, simulation is rejected with 422 and helpful diagnostic.
    """
    headers = register_and_login(client, f"no_model_{uuid.uuid4().hex[:6]}@example.com")

    res = client.post(
        "/api/simulations",
        headers=headers,
        json={
            "scenario_type": "price_change",
            "price_change_percent": 5.0,
            "horizon_days": 7,
        },
    )
    assert res.status_code == 422
    assert "dedicated elasticity model" in res.text.lower() or "unavailable" in res.text.lower()


# ---------------------------------------------------------------------------
# Test 10: Training in-flight returns 422 with training diagnostic
# ---------------------------------------------------------------------------

def test_training_in_flight_returns_422_with_training_diagnostic(client: TestClient, db_session):
    """
    Verification:
    When a dataset training job is actively in-flight ('processing'),
    simulation returns 422 with 'currently training' message.
    """
    email = f"inflight_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)

    user = db_session.query(User).filter(User.email == email).first()
    assert user is not None

    ds = DatasetUpload(
        id=f"ds-inf-{uuid.uuid4().hex[:8]}",
        user_id=user.id,
        original_filename="inf.csv",
        dataset_key="raw/inf.csv",
        row_count=100,
        status="active",
    )
    db_session.add(ds)
    db_session.flush()

    job = ModelTrainingJob(
        id=f"job-inf-{uuid.uuid4().hex[:8]}",
        user_id=user.id,
        dataset_id=ds.id,
        status="processing",
        progress_stage="Training company models",
    )
    db_session.add(job)
    db_session.commit()

    res = client.post(
        "/api/simulations",
        headers=headers,
        json={
            "scenario_type": "price_change",
            "price_change_percent": 5.0,
            "horizon_days": 7,
        },
    )
    assert res.status_code == 422
    assert "currently training" in res.text.lower()


# ---------------------------------------------------------------------------
# Test 11: Failed model returns 422 with failure diagnostic
# ---------------------------------------------------------------------------

def test_failed_model_returns_422_with_failure_diagnostic(client: TestClient, db_session):
    """
    Phase 10 Verification:
    When elasticity training recorded status='failed', simulation returns 422
    with exact failure diagnostic.
    """
    email = f"failed_user_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)

    user = db_session.query(User).filter(User.email == email).first()
    assert user is not None

    ds = DatasetUpload(
        id=f"ds-fail-{uuid.uuid4().hex[:8]}",
        user_id=user.id,
        original_filename="fail.csv",
        dataset_key="raw/fail.csv",
        row_count=100,
        status="active",
    )
    db_session.add(ds)
    db_session.flush()

    em = CompanyElasticityModel(
        id=f"cem-fail-{uuid.uuid4().hex[:8]}",
        user_id=user.id,
        dataset_id=ds.id,
        model_version=1,
        model_type="None",
        artifact_path="",
        status="failed",
        status_message="Demand sensitivity modeling failed: Matrix singular during fit.",
        is_active=False,
    )
    db_session.add(em)
    db_session.commit()

    # GET /api/models/elasticity should return the failed model diagnostic
    res_get = client.get("/api/models/elasticity", headers=headers)
    assert res_get.status_code == 200
    assert res_get.json()["status"] == "failed"
    assert "Matrix singular" in res_get.json()["status_message"]

    # POST /api/simulations should return 422 with the exact message
    res_post = client.post(
        "/api/simulations",
        headers=headers,
        json={
            "scenario_type": "price_change",
            "price_change_percent": 5.0,
            "horizon_days": 7,
        },
    )
    assert res_post.status_code == 422
    assert "Matrix singular" in res_post.text


# ---------------------------------------------------------------------------
# Test 12: Insufficient data returns 422 with reason diagnostic
# ---------------------------------------------------------------------------

def test_insufficient_data_returns_422_with_specific_reasons(client: TestClient, db_session):
    """
    Phase 11 Verification:
    When price or discount is unsupported due to constant/missing historical variation,
    simulation returns 422 with the exact diagnostic reason.
    """
    email = f"insuff_user_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)

    user = db_session.query(User).filter(User.email == email).first()
    assert user is not None

    ds = DatasetUpload(
        id=f"ds-ins-{uuid.uuid4().hex[:8]}",
        user_id=user.id,
        original_filename="ins.csv",
        dataset_key="raw/ins.csv",
        row_count=100,
        status="active",
    )
    db_session.add(ds)
    db_session.flush()

    # Ready model with price supported but discount UNSUPPORTED
    em = CompanyElasticityModel(
        id=f"cem-ins-{uuid.uuid4().hex[:8]}",
        user_id=user.id,
        dataset_id=ds.id,
        model_version=1,
        model_type="RidgeLogLog",
        artifact_path=f"models/{user.id}/{ds.id}/v1/elasticity/model.joblib",
        price_elasticity=-0.5,
        discount_sensitivity=None,
        status="ready",
        is_active=True,
        diagnostics={
            "price_supported": True,
            "discount_supported": False,
            "discount_reason": "No promotional markdown variation observed in sales history.",
        },
    )
    db_session.add(em)
    db_session.commit()

    # Save mock artifact to allow price change simulation
    buf = io.BytesIO()
    joblib.dump({"model": "RidgeMock", "price_elasticity": -0.5, "discount_sensitivity": None}, buf)
    storage_service.save_artifact(em.artifact_path, buf.getvalue())

    # Price change succeeds
    res_pc = client.post(
        "/api/simulations",
        headers=headers,
        json={"scenario_type": "price_change", "price_change_percent": 5.0, "horizon_days": 7},
    )
    assert res_pc.status_code == 200

    # Discount change fails with specific reason
    res_dc = client.post(
        "/api/simulations",
        headers=headers,
        json={"scenario_type": "discount_change", "discount_change_percent": 10.0, "horizon_days": 7},
    )
    assert res_dc.status_code == 422
    assert "No promotional markdown variation observed" in res_dc.text


# ---------------------------------------------------------------------------
# Test 13: Tenant Isolation
# ---------------------------------------------------------------------------

def test_tenant_isolation_elasticity_model(client: TestClient, db_session):
    """
    Phase 12 Verification:
    User B cannot access or run simulations using User A's elasticity model.
    """
    headers_a = register_and_login(client, f"tenant_a_{uuid.uuid4().hex[:6]}@example.com")
    headers_b = register_and_login(client, f"tenant_b_{uuid.uuid4().hex[:6]}@example.com")

    # Get User A ID
    user_a_info = client.get("/api/auth/me", headers=headers_a).json()
    user_a_id = user_a_info["id"]

    ds_a = DatasetUpload(
        id=f"ds-ten-a-{uuid.uuid4().hex[:8]}",
        user_id=user_a_id,
        original_filename="a.csv",
        dataset_key="raw/a.csv",
        row_count=100,
        status="active",
    )
    db_session.add(ds_a)
    db_session.flush()

    em_a = CompanyElasticityModel(
        id=f"cem-a-{uuid.uuid4().hex[:8]}",
        user_id=user_a_id,
        dataset_id=ds_a.id,
        model_version=1,
        model_type="RidgeLogLog",
        artifact_path=f"models/{user_a_id}/{ds_a.id}/v1/elasticity/model.joblib",
        price_elasticity=-0.75,
        discount_sensitivity=1.35,
        status="ready",
        is_active=True,
        diagnostics={"price_supported": True, "discount_supported": True},
    )
    db_session.add(em_a)
    db_session.commit()

    # User A gets 200
    res_a = client.get("/api/models/elasticity", headers=headers_a)
    assert res_a.status_code == 200
    assert res_a.json()["price_elasticity"] == pytest.approx(-0.75)

    # User B gets 404
    res_b = client.get("/api/models/elasticity", headers=headers_b)
    assert res_b.status_code == 404

    # User B cannot run simulation
    res_b_sim = client.post(
        "/api/simulations",
        headers=headers_b,
        json={"scenario_type": "price_change", "price_change_percent": 10.0, "horizon_days": 7},
    )
    assert res_b_sim.status_code == 422


# ---------------------------------------------------------------------------
# Test 14: Dataset Isolation & Version Superseding
# ---------------------------------------------------------------------------

def test_dataset_isolation_and_version_superseding(db_session):
    """
    Phase 13 Verification:
    When a new dataset is uploaded and trained, it creates a new version
    and deactivates the previous active elasticity model.
    """
    user = User(email=f"iso_user_{uuid.uuid4().hex[:6]}@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    # Dataset 1
    ds1 = DatasetUpload(
        id=f"ds1-{uuid.uuid4().hex[:8]}",
        user_id=user.id,
        original_filename="v1.csv",
        dataset_key="raw/v1.csv",
        row_count=100,
        status="active",
    )
    db_session.add(ds1)
    db_session.flush()

    # Seed sales for ds1
    seed_test_sales_history(db_session, user.id, ds1.id, days=60)

    # Train v1
    job1, _ = company_model_service.enqueue_training_job(db_session, user.id, ds1.id)
    with patch("backend.app.services.company_model_service.SessionLocal", TestingSessionLocal):
        company_model_service.execute_training_pipeline(user.id, ds1.id, job1.id)

    model_v1 = (
        db_session.query(CompanyElasticityModel)
        .filter(CompanyElasticityModel.user_id == user.id, CompanyElasticityModel.dataset_id == ds1.id)
        .first()
    )
    assert model_v1 is not None
    assert model_v1.is_active is True
    assert model_v1.model_version == 1

    # Dataset 2
    ds2 = DatasetUpload(
        id=f"ds2-{uuid.uuid4().hex[:8]}",
        user_id=user.id,
        original_filename="v2.csv",
        dataset_key="raw/v2.csv",
        row_count=100,
        status="active",
    )
    # Activate ds2, deactivate ds1
    ds1.status = "archived"
    db_session.add(ds2)
    db_session.flush()

    # Seed sales for ds2
    seed_test_sales_history(db_session, user.id, ds2.id, days=60, start_date=date(2024, 6, 1))

    # Train v2
    job2, _ = company_model_service.enqueue_training_job(db_session, user.id, ds2.id)
    with patch("backend.app.services.company_model_service.SessionLocal", TestingSessionLocal):
        company_model_service.execute_training_pipeline(user.id, ds2.id, job2.id)

    db_session.refresh(model_v1)
    model_v2 = (
        db_session.query(CompanyElasticityModel)
        .filter(CompanyElasticityModel.user_id == user.id, CompanyElasticityModel.dataset_id == ds2.id)
        .first()
    )
    assert model_v2 is not None

    # Model v1 was deactivated, Model v2 is active with model_version == 2
    assert model_v1.is_active is False
    assert model_v2.is_active is True
    assert model_v2.model_version == 2
    assert model_v2.dataset_id == ds2.id


# ---------------------------------------------------------------------------
# Test 15: Field-Name Consistency
# ---------------------------------------------------------------------------

def test_field_name_consistency():
    """
    Phase 15 Verification:
    Verify field-name consistency across models, schemas, and trainer:
    - price_elasticity
    - discount_sensitivity
    - price_supported
    - discount_supported
    - price_reason
    - discount_reason
    """
    # 1. Trainer output keys
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=40, freq="D"),
        "price": np.random.uniform(20.0, 40.0, 40),
        "discount": np.random.uniform(0.0, 20.0, 40),
        "quantity": np.random.uniform(50.0, 100.0, 40),
    })
    res = train_company_elasticity(df)
    assert "price_elasticity" in res
    assert "discount_sensitivity" in res
    assert "price_supported" in res
    assert "discount_supported" in res
    assert "price_reason" in res
    assert "discount_reason" in res

    # 2. Database column attributes
    col_names = {c.name for c in CompanyElasticityModel.__table__.columns}
    assert "price_elasticity" in col_names
    assert "discount_sensitivity" in col_names
    assert "status_message" in col_names
    assert "diagnostics" in col_names

    # 3. Pydantic ElasticityModelSummary fields
    summary_fields = ElasticityModelSummary.model_fields.keys()
    assert "price_elasticity" in summary_fields
    assert "discount_sensitivity" in summary_fields
    assert "price_supported" in summary_fields
    assert "discount_supported" in summary_fields
    assert "price_reason" in summary_fields
    assert "discount_reason" in summary_fields
    assert "dataset_id" in summary_fields

    # 4. Simulation response fields
    sim_res_fields = SimulationResponse.model_fields.keys()
    assert "price_elasticity" in sim_res_fields
    assert "discount_sensitivity" in sim_res_fields
    assert "elasticity_model_version" in sim_res_fields
