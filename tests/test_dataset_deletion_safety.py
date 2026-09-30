from __future__ import annotations

import io
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.app.database.models import (
    CompanyElasticityModel,
    CompanyModel,
    DatasetUpload,
    ModelTrainingJob,
    Product,
    SalesRecord,
    User,
)
from backend.app.services.company_model_service import training_lock_manager
from backend.app.services.dataset_runtime_service import (
    DatasetConflictError,
    dataset_runtime_service,
)
from backend.app.services.monitoring_service import monitoring_service
from backend.app.services.recommendation_service import recommendation_service
from backend.app.services.simulation_service import _SIMULATION_CACHE
from backend.app.services.storage_service import storage_service


def register_and_login(client: TestClient, email: str, password: str = "SafePass123!") -> dict[str, str]:
    """Register and log in a user to retrieve auth bearer token."""
    client.post("/api/auth/register", json={"email": email, "password": password})
    res = client.post("/api/auth/login", data={"username": email, "password": password})
    assert res.status_code == 200, f"Login failed: {res.text}"
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def make_dataset(user_id: int, status: str = "archived", filename: str = "data.csv", row_count: int = 50) -> DatasetUpload:
    """Helper to create a valid DatasetUpload with required dataset_key."""
    ds_id = f"ds-{uuid.uuid4().hex[:8]}"
    return DatasetUpload(
        id=ds_id,
        user_id=user_id,
        original_filename=filename,
        dataset_key=f"raw/datasets/{user_id}/{ds_id}.csv",
        status=status,
        row_count=row_count,
    )


def make_sales_record(
    user_id: int,
    dataset_id: str,
    product_id: int,
    sale_date: date,
    quantity: int = 1,
    unit_price: float = 10.0,
    profit: float = 2.0,
) -> SalesRecord:
    """Helper to create a valid SalesRecord with all non-null columns."""
    return SalesRecord(
        user_id=user_id,
        dataset_id=dataset_id,
        product_id=product_id,
        sale_date=sale_date,
        quantity=quantity,
        unit_price=unit_price,
        sales_amount=float(quantity * unit_price),
        profit=profit,
    )


# ---------------------------------------------------------------------------
# 1. Successful deletion of archived dataset
# ---------------------------------------------------------------------------

def test_delete_archived_dataset_success(client: TestClient, db_session):
    email = f"user_del_archived_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)
    user = db_session.query(User).filter(User.email == email).first()

    # Create active dataset 1 and archived dataset 2
    ds1 = make_dataset(user.id, status="active", filename="active_data.csv")
    ds2 = make_dataset(user.id, status="archived", filename="archived_data.csv")
    db_session.add_all([ds1, ds2])
    db_session.commit()
    ds1_id = ds1.id
    ds2_id = ds2.id

    # Delete archived dataset 2
    res = client.delete(f"/api/datasets/{ds2_id}", headers=headers)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["status"] == "deleted"
    assert data["dataset_id"] == ds2_id
    assert "successfully" in data["message"]

    # Verify in DB: ds2 gone, ds1 intact
    db_session.expire_all()
    assert db_session.query(DatasetUpload).filter(DatasetUpload.id == ds2_id).first() is None
    assert db_session.query(DatasetUpload).filter(DatasetUpload.id == ds1_id).first() is not None


# ---------------------------------------------------------------------------
# 2. Deleting active dataset rejected
# ---------------------------------------------------------------------------

def test_delete_active_dataset_rejected_when_only_dataset(client: TestClient, db_session):
    email = f"user_del_single_act_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)
    user = db_session.query(User).filter(User.email == email).first()

    # Only one dataset, which is active
    ds1 = make_dataset(user.id, status="active", filename="single_active.csv")
    db_session.add(ds1)
    db_session.commit()
    ds1_id = ds1.id

    res = client.delete(f"/api/datasets/{ds1_id}", headers=headers)
    assert res.status_code == 409
    assert "No active replacement dataset is available" in res.json()["detail"]

    # In DB, ds1 remains active and untouched
    db_session.expire_all()
    check = db_session.query(DatasetUpload).filter(DatasetUpload.id == ds1_id).first()
    assert check is not None
    assert check.status == "active"


def test_delete_active_dataset_rejected_when_other_datasets_exist(client: TestClient, db_session):
    email = f"user_del_act_multi_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)
    user = db_session.query(User).filter(User.email == email).first()

    ds1 = make_dataset(user.id, status="active", filename="active_one.csv")
    ds2 = make_dataset(user.id, status="archived", filename="archived_two.csv")
    db_session.add_all([ds1, ds2])
    db_session.commit()
    ds1_id = ds1.id
    ds2_id = ds2.id

    # Attempt to delete ds1 while it is still the active dataset
    res = client.delete(f"/api/datasets/{ds1_id}", headers=headers)
    assert res.status_code == 409
    assert "Activate another dataset before deleting" in res.json()["detail"]

    # Verify neither dataset was deleted
    db_session.expire_all()
    assert db_session.query(DatasetUpload).filter(DatasetUpload.id == ds1_id).first() is not None
    assert db_session.query(DatasetUpload).filter(DatasetUpload.id == ds2_id).first() is not None


# ---------------------------------------------------------------------------
# 3. Deletion cascades to sales records belonging only to that dataset
# ---------------------------------------------------------------------------

def test_delete_dataset_with_sales_records(client: TestClient, db_session):
    email = f"user_del_sales_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)
    user = db_session.query(User).filter(User.email == email).first()

    ds_act = make_dataset(user.id, status="active", filename="a.csv")
    ds_target = make_dataset(user.id, status="archived", filename="b.csv")
    db_session.add_all([ds_act, ds_target])
    db_session.flush()
    ds_act_id = ds_act.id
    ds_target_id = ds_target.id

    prod = Product(product_id=f"PROD-S-{uuid.uuid4().hex[:6]}", product_name="Test Product", unit_price=20.0, category_name="General")
    db_session.add(prod)
    db_session.flush()

    # Create 3 sales records for target dataset, and 2 for active dataset
    for i in range(3):
        db_session.add(make_sales_record(
            user_id=user.id,
            dataset_id=ds_target_id,
            product_id=prod.id,
            sale_date=date(2024, 1, i + 1),
            quantity=5,
            unit_price=20.0,
            profit=20.0,
        ))
    for i in range(2):
        db_session.add(make_sales_record(
            user_id=user.id,
            dataset_id=ds_act_id,
            product_id=prod.id,
            sale_date=date(2024, 2, i + 1),
            quantity=2,
            unit_price=20.0,
            profit=8.0,
        ))
    db_session.commit()

    # Delete target dataset
    res = client.delete(f"/api/datasets/{ds_target_id}", headers=headers)
    assert res.status_code == 200

    db_session.expire_all()
    target_sales = db_session.query(SalesRecord).filter(SalesRecord.dataset_id == ds_target_id).all()
    act_sales = db_session.query(SalesRecord).filter(SalesRecord.dataset_id == ds_act_id).all()
    assert len(target_sales) == 0
    assert len(act_sales) == 2


# ---------------------------------------------------------------------------
# 4. Deletion cascades to company forecast models
# ---------------------------------------------------------------------------

def test_delete_dataset_with_company_forecast_models(client: TestClient, db_session):
    email = f"user_del_cm_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)
    user = db_session.query(User).filter(User.email == email).first()

    ds_act = make_dataset(user.id, status="active", filename="a.csv")
    ds_target = make_dataset(user.id, status="archived", filename="b.csv")
    db_session.add_all([ds_act, ds_target])
    db_session.flush()
    ds_act_id = ds_act.id
    ds_target_id = ds_target.id

    for h in [7, 30, 90]:
        db_session.add(CompanyModel(
            id=f"cm-tgt-{h}-{uuid.uuid4().hex[:4]}",
            user_id=user.id,
            dataset_id=ds_target_id,
            horizon=h,
            model_type="HistGradientBoosting",
            artifact_path=f"models/{user.id}/{ds_target_id}/v1/{h}d/model.joblib",
            model_version=1,
            validation_wape=0.12,
            status="ready",
            is_active=False,
        ))
        db_session.add(CompanyModel(
            id=f"cm-act-{h}-{uuid.uuid4().hex[:4]}",
            user_id=user.id,
            dataset_id=ds_act_id,
            horizon=h,
            model_type="HistGradientBoosting",
            artifact_path=f"models/{user.id}/{ds_act_id}/v1/{h}d/model.joblib",
            model_version=1,
            validation_wape=0.11,
            status="ready",
            is_active=True,
        ))
    db_session.commit()

    res = client.delete(f"/api/datasets/{ds_target_id}", headers=headers)
    assert res.status_code == 200

    db_session.expire_all()
    tgt_models = db_session.query(CompanyModel).filter(CompanyModel.dataset_id == ds_target_id).all()
    act_models = db_session.query(CompanyModel).filter(CompanyModel.dataset_id == ds_act_id).all()
    assert len(tgt_models) == 0
    assert len(act_models) == 3


# ---------------------------------------------------------------------------
# 5. Deletion cascades to elasticity model
# ---------------------------------------------------------------------------

def test_delete_dataset_with_elasticity_model(client: TestClient, db_session):
    email = f"user_del_em_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)
    user = db_session.query(User).filter(User.email == email).first()

    ds_act = make_dataset(user.id, status="active", filename="a.csv")
    ds_target = make_dataset(user.id, status="archived", filename="b.csv")
    db_session.add_all([ds_act, ds_target])
    db_session.flush()
    ds_act_id = ds_act.id
    ds_target_id = ds_target.id

    em_tgt = CompanyElasticityModel(
        id=f"cem-tgt-{uuid.uuid4().hex[:6]}",
        user_id=user.id,
        dataset_id=ds_target_id,
        model_version=1,
        model_type="RidgeLogLog",
        artifact_path=f"models/{user.id}/{ds_target_id}/v1/elasticity/model.joblib",
        price_elasticity=-0.5,
        discount_sensitivity=1.1,
        status="ready",
        is_active=False,
    )
    em_act = CompanyElasticityModel(
        id=f"cem-act-{uuid.uuid4().hex[:6]}",
        user_id=user.id,
        dataset_id=ds_act_id,
        model_version=1,
        model_type="RidgeLogLog",
        artifact_path=f"models/{user.id}/{ds_act_id}/v1/elasticity/model.joblib",
        price_elasticity=-0.7,
        discount_sensitivity=1.3,
        status="ready",
        is_active=True,
    )
    db_session.add_all([em_tgt, em_act])
    db_session.commit()

    res = client.delete(f"/api/datasets/{ds_target_id}", headers=headers)
    assert res.status_code == 200

    db_session.expire_all()
    assert db_session.query(CompanyElasticityModel).filter(CompanyElasticityModel.dataset_id == ds_target_id).first() is None
    assert db_session.query(CompanyElasticityModel).filter(CompanyElasticityModel.dataset_id == ds_act_id).first() is not None


# ---------------------------------------------------------------------------
# 6. Deletion cascades to completed training job
# ---------------------------------------------------------------------------

def test_delete_dataset_with_completed_training_job(client: TestClient, db_session):
    email = f"user_del_job_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)
    user = db_session.query(User).filter(User.email == email).first()

    ds_act = make_dataset(user.id, status="active", filename="a.csv")
    ds_target = make_dataset(user.id, status="archived", filename="b.csv")
    db_session.add_all([ds_act, ds_target])
    db_session.flush()
    ds_act_id = ds_act.id
    ds_target_id = ds_target.id

    job = ModelTrainingJob(
        id=f"job-cmp-{uuid.uuid4().hex[:6]}",
        user_id=user.id,
        dataset_id=ds_target_id,
        model_version=1,
        status="completed",
        progress_stage="Completed",
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )
    db_session.add(job)
    db_session.commit()
    job_id = job.id

    res = client.delete(f"/api/datasets/{ds_target_id}", headers=headers)
    assert res.status_code == 200

    db_session.expire_all()
    assert db_session.query(ModelTrainingJob).filter(ModelTrainingJob.id == job_id).first() is None


# ---------------------------------------------------------------------------
# 7. Active training job blocks dataset deletion with 409
# ---------------------------------------------------------------------------

def test_delete_dataset_with_active_training_job_rejected(client: TestClient, db_session):
    email = f"user_del_active_train_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)
    user = db_session.query(User).filter(User.email == email).first()

    ds_act = make_dataset(user.id, status="active", filename="a.csv")
    ds_target = make_dataset(user.id, status="archived", filename="b.csv")
    db_session.add_all([ds_act, ds_target])
    db_session.flush()
    ds_act_id = ds_act.id
    ds_target_id = ds_target.id

    job = ModelTrainingJob(
        id=f"job-run-{uuid.uuid4().hex[:6]}",
        user_id=user.id,
        dataset_id=ds_target_id,
        model_version=1,
        status="processing",
        progress_stage="Training 30D horizon",
        started_at=datetime.now(timezone.utc),
    )
    db_session.add(job)
    db_session.commit()
    job_id = job.id

    res = client.delete(f"/api/datasets/{ds_target_id}", headers=headers)
    assert res.status_code == 409
    assert "Dataset cannot be deleted while model training is in progress" in res.json()["detail"]

    # Verify records remain completely untouched
    db_session.expire_all()
    assert db_session.query(DatasetUpload).filter(DatasetUpload.id == ds_target_id).first() is not None
    assert db_session.query(ModelTrainingJob).filter(ModelTrainingJob.id == job_id).first() is not None

    # Test lock manager protection as well
    job.status = "completed"
    db_session.commit()
    training_lock_manager.acquire(user.id, ds_target_id)
    try:
        res2 = client.delete(f"/api/datasets/{ds_target_id}", headers=headers)
        assert res2.status_code == 409
        assert "Dataset cannot be deleted while model training is in progress" in res2.json()["detail"]
    finally:
        training_lock_manager.release(user.id, ds_target_id)


# ---------------------------------------------------------------------------
# 8. Multi-tenant isolation: cannot delete another user's dataset
# ---------------------------------------------------------------------------

def test_delete_dataset_cannot_affect_other_users_dataset(client: TestClient, db_session):
    email_a = f"tenant_a_{uuid.uuid4().hex[:6]}@example.com"
    email_b = f"tenant_b_{uuid.uuid4().hex[:6]}@example.com"
    headers_a = register_and_login(client, email_a)
    headers_b = register_and_login(client, email_b)

    user_a = db_session.query(User).filter(User.email == email_a).first()
    user_b = db_session.query(User).filter(User.email == email_b).first()

    ds_a = make_dataset(user_a.id, status="archived", filename="a.csv")
    ds_b = make_dataset(user_b.id, status="active", filename="b.csv")
    db_session.add_all([ds_a, ds_b])
    db_session.commit()
    ds_a_id = ds_a.id
    ds_b_id = ds_b.id

    # User B attempts to delete User A's dataset
    res = client.delete(f"/api/datasets/{ds_a_id}", headers=headers_b)
    assert res.status_code == 404

    # Verify User A's dataset is completely untouched
    db_session.expire_all()
    assert db_session.query(DatasetUpload).filter(DatasetUpload.id == ds_a_id).first() is not None


# ---------------------------------------------------------------------------
# 9. Preserves shared product records
# ---------------------------------------------------------------------------

def test_delete_dataset_cannot_affect_shared_products(client: TestClient, db_session):
    email = f"user_prod_safety_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)
    user = db_session.query(User).filter(User.email == email).first()

    prod_code = f"PROD-GLOBAL-{uuid.uuid4().hex[:6]}"
    prod = Product(
        product_id=prod_code,
        product_name="Catalog Protected Item",
        category_name="Electronics",
        unit_price=199.99,
        tenant_id=user.id,
    )
    db_session.add(prod)
    db_session.flush()
    prod_id = prod.id

    ds_act = make_dataset(user.id, status="active", filename="a.csv")
    ds_target = make_dataset(user.id, status="archived", filename="b.csv")
    db_session.add_all([ds_act, ds_target])
    db_session.flush()
    ds_act_id = ds_act.id
    ds_target_id = ds_target.id

    db_session.add(make_sales_record(
        user_id=user.id,
        dataset_id=ds_target_id,
        product_id=prod_id,
        sale_date=date(2024, 1, 1),
        quantity=3,
        unit_price=199.99,
        profit=150.0,
    ))
    db_session.commit()

    res = client.delete(f"/api/datasets/{ds_target_id}", headers=headers)
    assert res.status_code == 200

    db_session.expire_all()
    # Product record MUST be intact
    prod_check = db_session.query(Product).filter(Product.id == prod_id).first()
    assert prod_check is not None
    assert prod_check.product_id == prod_code
    assert prod_check.product_name == "Catalog Protected Item"


# ---------------------------------------------------------------------------
# 10. Runtime cache invalidation
# ---------------------------------------------------------------------------

def test_delete_dataset_clears_runtime_caches(client: TestClient, db_session):
    email = f"user_cache_clear_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)
    user = db_session.query(User).filter(User.email == email).first()

    ds_act = make_dataset(user.id, status="active", filename="a.csv")
    ds_target = make_dataset(user.id, status="archived", filename="b.csv")
    db_session.add_all([ds_act, ds_target])
    db_session.commit()
    ds_act_id = ds_act.id
    ds_target_id = ds_target.id

    # Populate caches
    dataset_runtime_service._cache[(user.id, ds_target_id, "daily")] = "cached_df"
    recommendation_service._cache["mock_rec_anomaly_id"] = "cached_rec"
    monitoring_service._cached_category_anomalies[user.id] = []
    _SIMULATION_CACHE[(user.id, ds_target_id)] = "cached_sim"

    res = client.delete(f"/api/datasets/{ds_target_id}", headers=headers)
    assert res.status_code == 200

    # Verify caches invalidated
    assert (user.id, ds_target_id, "daily") not in dataset_runtime_service._cache
    assert "mock_rec_anomaly_id" not in recommendation_service._cache
    assert user.id not in monitoring_service._cached_category_anomalies
    assert (user.id, ds_target_id) not in _SIMULATION_CACHE


# ---------------------------------------------------------------------------
# 11. Storage artifact cleanup
# ---------------------------------------------------------------------------

def test_delete_dataset_storage_cleanup(client: TestClient, db_session):
    email = f"user_storage_clean_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)
    user = db_session.query(User).filter(User.email == email).first()

    ds_act = make_dataset(user.id, status="active", filename="a.csv")
    ds_target = make_dataset(user.id, status="archived", filename="b.csv")
    db_session.add_all([ds_act, ds_target])
    db_session.commit()
    ds_act_id = ds_act.id
    ds_target_id = ds_target.id

    # Create dummy storage artifacts for target dataset and active dataset
    tgt_p1 = f"models/{user.id}/{ds_target_id}/v1/7d/model.joblib"
    tgt_p2 = f"models/{user.id}/{ds_target_id}/v1/elasticity/model.joblib"
    act_p = f"models/{user.id}/{ds_act_id}/v1/7d/model.joblib"

    storage_service.save_artifact(tgt_p1, b"mock_tgt_7d_bytes")
    storage_service.save_artifact(tgt_p2, b"mock_tgt_elasticity_bytes")
    storage_service.save_artifact(act_p, b"mock_act_7d_bytes")

    assert storage_service.backend.exists(tgt_p1) is True
    assert storage_service.backend.exists(tgt_p2) is True
    assert storage_service.backend.exists(act_p) is True

    # Associate them with DB records
    db_session.add(CompanyModel(
        id=f"cm-{uuid.uuid4().hex[:6]}",
        user_id=user.id,
        dataset_id=ds_target_id,
        horizon=7,
        model_type="HistGradientBoosting",
        artifact_path=tgt_p1,
        model_version=1,
        validation_wape=0.1,
        status="ready",
        is_active=False,
    ))
    db_session.add(CompanyElasticityModel(
        id=f"cem-{uuid.uuid4().hex[:6]}",
        user_id=user.id,
        dataset_id=ds_target_id,
        model_version=1,
        model_type="RidgeLogLog",
        artifact_path=tgt_p2,
        price_elasticity=-0.5,
        discount_sensitivity=1.0,
        status="ready",
        is_active=False,
    ))
    db_session.commit()

    # Delete target dataset
    res = client.delete(f"/api/datasets/{ds_target_id}", headers=headers)
    assert res.status_code == 200

    # Target artifacts deleted, active dataset artifact preserved
    assert storage_service.backend.exists(tgt_p1) is False
    assert storage_service.backend.exists(tgt_p2) is False
    assert storage_service.backend.exists(act_p) is True

    # Cleanup
    storage_service.delete_artifact(act_p)


# ---------------------------------------------------------------------------
# 12. Transaction rollback on failure
# ---------------------------------------------------------------------------

def test_delete_dataset_transaction_rollback_on_failure(db_session):
    user = User(email=f"user_rollback_{uuid.uuid4().hex[:6]}@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    ds_act = make_dataset(user.id, status="active", filename="a.csv")
    ds_target = make_dataset(user.id, status="archived", filename="b.csv")
    db_session.add_all([ds_act, ds_target])
    db_session.flush()
    ds_target_id = ds_target.id

    prod = Product(product_id=f"PROD-R-{uuid.uuid4().hex[:6]}", product_name="Test Product", unit_price=10.0, category_name="General")
    db_session.add(prod)
    db_session.flush()

    sale = make_sales_record(
        user_id=user.id,
        dataset_id=ds_target_id,
        product_id=prod.id,
        sale_date=date(2024, 1, 1),
        quantity=1,
        unit_price=10.0,
        profit=2.0,
    )
    db_session.add(sale)
    db_session.commit()

    # Simulate IntegrityError during the deletion transaction
    with patch.object(
        db_session,
        "commit",
        side_effect=IntegrityError("Simulated FK conflict on dataset_uploads", params=None, orig=Exception("table sales_records contains dependent records")),
    ):
        with pytest.raises(DatasetConflictError) as exc_info:
            dataset_runtime_service.delete_dataset(
                db=db_session,
                user_id=user.id,
                dataset_id=ds_target_id,
            )
        assert "dependent records" in str(exc_info.value).lower() or "conflict" in str(exc_info.value).lower()

    # Verify transaction rollback preserved the dataset and sales record
    db_session.rollback()
    db_session.expire_all()
    check_ds = db_session.query(DatasetUpload).filter(DatasetUpload.id == ds_target_id).first()
    check_sale = db_session.query(SalesRecord).filter(SalesRecord.dataset_id == ds_target_id).first()
    assert check_ds is not None
    assert check_sale is not None


# ---------------------------------------------------------------------------
# 13. Nonexistent dataset returns 404
# ---------------------------------------------------------------------------

def test_delete_nonexistent_dataset_404(client: TestClient, db_session):
    email = f"user_404_{uuid.uuid4().hex[:6]}@example.com"
    headers = register_and_login(client, email)

    res = client.delete("/api/datasets/ds-nonexistent-12345", headers=headers)
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 14. Unauthenticated deletion returns 401
# ---------------------------------------------------------------------------

def test_delete_unauthenticated_user_401(client: TestClient):
    res = client.delete("/api/datasets/ds-any")
    assert res.status_code == 401
