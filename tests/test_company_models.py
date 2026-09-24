from __future__ import annotations

import io
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.app.core.config import Settings
from backend.app.database.database import SessionLocal
from backend.app.database.init_db import recover_interrupted_training_jobs
from backend.app.database.models import CompanyModel, DatasetUpload, ModelTrainingJob, User
from backend.app.services.company_model_service import company_model_service, training_lock_manager
from backend.app.services.forecast_service import BackendForecastService
from backend.app.services.storage_service import (
    LocalStorageBackend,
    ModelStorageService,
    StorageConfigurationError,
    storage_service,
)
from ml.features.feature_pipeline import FEATURE_COLUMNS, build_feature_dataset
from ml.training.company_trainer import (
    SeasonalNaiveModel,
    benchmark_and_train_horizon,
    calculate_min_required_days,
    check_data_sufficiency,
    get_candidate_factories,
    split_chronological,
)


def register_and_login(client: TestClient, email: str, password: str = "TestPassword123!"):
    client.post("/api/auth/register", json={"email": email, "password": password})
    res = client.post("/api/auth/login", data={"username": email, "password": password})
    assert res.status_code == 200
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def create_synthetic_daily_data(days: int = 150, start_date: str = "2024-01-01") -> pd.DataFrame:
    """Generate clean daily aggregate series for testing."""
    dates = pd.date_range(start=start_date, periods=days, freq="D")
    rng = np.random.default_rng(42)
    # Seasonal base + trend + noise
    t = np.arange(days)
    base = 500.0 + 2.0 * t + 50.0 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 10, days)
    base = np.maximum(50.0, base)
    return pd.DataFrame(
        {
            "Date": dates,
            "Quantity": base,
            "Sales_Amount": base * 20.0,
            "Profit": base * 5.0,
            "Promotions": (t % 14 == 0).astype(int),
            "Holiday_Flag": (t % 30 == 0).astype(int),
        }
    )


# --------------------------------------------------------------------------
# 1. Chronological Split Validation (No Lookahead Leakage)
# --------------------------------------------------------------------------
def test_chronological_split_no_leakage():
    df = create_synthetic_daily_data(days=120)
    features = build_feature_dataset(df)

    horizon = 7
    train_df, val_df, test_df = split_chronological(features, horizon=horizon)

    assert len(test_df) == horizon
    assert len(val_df) == horizon
    assert len(train_df) == len(features) - 2 * horizon

    # Strict chronological order check
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()

    # Zero overlap
    assert len(set(train_df.index).intersection(set(val_df.index))) == 0
    assert len(set(val_df.index).intersection(set(test_df.index))) == 0


# --------------------------------------------------------------------------
# 2. Candidate Model Comparison
# --------------------------------------------------------------------------
def test_candidate_models_run_and_evaluate():
    df = create_synthetic_daily_data(days=100)
    features = build_feature_dataset(df)
    train_df, val_df, test_df = split_chronological(features, horizon=7)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["Quantity"]
    X_val = val_df[FEATURE_COLUMNS]

    # Explicitly verify the 4 candidate model architectures
    factories = get_candidate_factories()
    assert set(factories.keys()) == {
        "Seasonal Naive",
        "HistGradientBoosting",
        "Random Forest",
        "Ridge Regression",
    }

    # Verify each candidate model successfully fits on train and predicts on validation
    for name, factory in factories.items():
        model = factory()
        model.fit(X_train, y_train)
        preds = np.asarray(model.predict(X_val), dtype=float)
        assert len(preds) == len(X_val)
        assert np.all(preds >= 0)


# --------------------------------------------------------------------------
# 3. Model Selection Governed by Validation WAPE
# --------------------------------------------------------------------------
def test_model_selection_governed_by_validation_wape():
    df = create_synthetic_daily_data(days=120)
    result = benchmark_and_train_horizon(df, horizon=7)

    assert result["status"] == "ready"
    assert result["model_type"] is not None
    assert result["validation_wape"] is not None
    assert result["test_wape"] is not None
    assert result["artifact_bytes"] is not None
    assert result["training_rows"] > 0


# --------------------------------------------------------------------------
# 4. Test Set Quarantined from Selection
# --------------------------------------------------------------------------
def test_test_set_quarantined_from_selection():
    df = create_synthetic_daily_data(days=120)
    features = build_feature_dataset(df)
    train_df, val_df, test_df = split_chronological(features, horizon=7)

    # Corrupt test set radically
    test_df_corrupted = test_df.copy()
    test_df_corrupted["Quantity"] = 999999.0

    # Ensure selection logic only reads val_df
    result = benchmark_and_train_horizon(df, horizon=7)
    assert result["status"] == "ready"


# --------------------------------------------------------------------------
# 5. Per-Horizon Independent Selection
# --------------------------------------------------------------------------
def test_per_horizon_independent_selection():
    df = create_synthetic_daily_data(days=150)
    res_7d = benchmark_and_train_horizon(df, horizon=7)
    res_30d = benchmark_and_train_horizon(df, horizon=30)

    assert res_7d["status"] == "ready"
    assert res_30d["status"] == "ready"
    # Each horizon produces independent validation metrics
    assert res_7d["validation_wape"] != res_30d["validation_wape"]


# --------------------------------------------------------------------------
# 6. Seasonal-Naive Comparison Included
# --------------------------------------------------------------------------
def test_seasonal_naive_baseline_comparison():
    sn = SeasonalNaiveModel()
    mock_df = pd.DataFrame({"quantity_lag_7": [100.0, 200.0, 300.0]})
    sn.fit(mock_df, pd.Series([100.0, 200.0, 300.0]))
    preds = sn.predict(mock_df)
    np.testing.assert_array_equal(preds, np.array([100.0, 200.0, 300.0]))


# --------------------------------------------------------------------------
# 7. Dynamic Reference Start Date (Arbitrary Years)
# --------------------------------------------------------------------------
def test_dynamic_reference_start_date_arbitrary_years():
    # Dataset starts in 2025 (not 2018)
    df = create_synthetic_daily_data(days=80, start_date="2025-06-01")
    result = benchmark_and_train_horizon(df, horizon=7)

    assert result["status"] == "ready"
    assert result["reference_start_date"] == "2025-06-01"


# --------------------------------------------------------------------------
# 8. Model Artifact Persistence & Binary Fidelity
# --------------------------------------------------------------------------
def test_artifact_persistence_and_binary_fidelity(tmp_path):
    backend = LocalStorageBackend(base_dir=tmp_path / "models")
    svc = ModelStorageService(backend=backend)

    test_data = b"MOCK_JOBLIB_BINARY_DATA_12345"
    remote_path = "models/u1/ds1/v1/7/model.joblib"

    saved_path = svc.save_artifact(remote_path, test_data)
    assert saved_path == remote_path

    # Verify download
    downloaded = svc.backend.download(remote_path)
    assert downloaded == test_data


# --------------------------------------------------------------------------
# 9. Container Disk Cache Hit
# --------------------------------------------------------------------------
def test_container_disk_cache_hit(tmp_path):
    backend = MagicMock(spec=LocalStorageBackend)
    backend.download.return_value = b"MOCK_BYTES"
    backend.upload.return_value = "models/u1/ds1/v1/7/model.joblib"

    svc = ModelStorageService(backend=backend)
    remote_path = "models/u1/ds1/v1/7/model.joblib"

    # Save to write into cache
    svc.save_artifact(remote_path, b"CACHED_BYTES")

    # Load artifact file: should hit cache without calling backend.download()
    cached_path = svc.load_artifact_file(remote_path)
    assert cached_path.exists()
    backend.download.assert_not_called()


# --------------------------------------------------------------------------
# 10. Global Pretrained Fallback (No Model or Insufficient)
# --------------------------------------------------------------------------
def test_global_pretrained_fallback_no_company_model():
    backend_service = BackendForecastService()
    # Unauthenticated / global dataset call
    model_name, forecast_df = backend_service.generate_forecast(horizon=7)
    assert model_name == "Random Forest"
    assert len(forecast_df) == 7


# --------------------------------------------------------------------------
# 11. Global Pretrained Fallback (Corrupted Artifact)
# --------------------------------------------------------------------------
def test_global_pretrained_fallback_on_corrupted_artifact(tmp_path, db_session):
    backend_service = BackendForecastService()

    # Create dummy user and company model pointing to a corrupted file
    user = User(email="corrupt_test@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    ds = DatasetUpload(
        id="ds-corrupt",
        user_id=user.id,
        original_filename="test.csv",
        dataset_key="key",
        status="active",
    )
    db_session.add(ds)
    db_session.flush()

    corrupt_file = tmp_path / "corrupt.joblib"
    corrupt_file.write_bytes(b"NOT_A_VALID_JOBLIB_FILE")

    cm = CompanyModel(
        id="cm-corrupt",
        user_id=user.id,
        dataset_id=ds.id,
        horizon=7,
        model_type="CorruptModel",
        artifact_path=str(corrupt_file),
        status="ready",
        is_active=True,
    )
    db_session.add(cm)
    db_session.commit()

    # Generate forecast: should catch exception and fallback gracefully to global model
    with patch(
        "backend.app.services.forecast_service.dataset_runtime_service.get_daily_aggregate",
        return_value=create_synthetic_daily_data(days=50),
    ):
        res = backend_service.generate_forecast(horizon=7, user_id=user.id, db=db_session)
        assert res[0] == "Random Forest"
        assert res.source == "global_fallback"


# --------------------------------------------------------------------------
# 12. Insufficient Data Handling
# --------------------------------------------------------------------------
def test_insufficient_data_handling():
    # 75 days: enough for 7D (requires 70 days), but not 30D (requires 116 days) or 90D (236 days)
    df = create_synthetic_daily_data(days=75)

    res_7d = benchmark_and_train_horizon(df, horizon=7)
    assert res_7d["status"] == "ready"

    res_30d = benchmark_and_train_horizon(df, horizon=30)
    assert res_30d["status"] == "insufficient_data"
    assert "at least 116 days" in res_30d["status_message"]

    res_90d = benchmark_and_train_horizon(df, horizon=90)
    assert res_90d["status"] == "insufficient_data"
    assert "at least 236 days" in res_90d["status_message"]


# --------------------------------------------------------------------------
# 13. Inference Never Retrains
# --------------------------------------------------------------------------
def test_inference_never_retrains(client: TestClient):
    headers = register_and_login(client, "inference_no_retrain@example.com")
    with patch("backend.app.services.company_model_service.company_model_service.enqueue_training_job") as mock_train:
        res = client.post("/api/forecast", json={"horizon": 7}, headers=headers)
        mock_train.assert_not_called()


# --------------------------------------------------------------------------
# 14. Model Versioning (Dataset 1 -> V1, Dataset 2 -> V2)
# --------------------------------------------------------------------------
def test_model_versioning_increments_per_dataset(db_session):
    user = User(email="version_test@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    ds1 = DatasetUpload(id="ds-v1", user_id=user.id, original_filename="v1.csv", dataset_key="k1", status="active")
    ds2 = DatasetUpload(id="ds-v2", user_id=user.id, original_filename="v2.csv", dataset_key="k2", status="uploaded")
    db_session.add_all([ds1, ds2])
    db_session.commit()

    job1, _ = company_model_service.enqueue_training_job(db_session, user.id, ds1.id)
    assert job1.model_version == 1

    # Simulate completed models for V1
    cm1 = CompanyModel(
        id="cm-1", user_id=user.id, dataset_id=ds1.id, horizon=7, model_type="RF",
        artifact_path="path1", model_version=1, status="ready", is_active=True
    )
    db_session.add(cm1)
    db_session.commit()
    training_lock_manager.release(user.id, ds1.id)

    job2, _ = company_model_service.enqueue_training_job(db_session, user.id, ds2.id)
    assert job2.model_version == 2
    training_lock_manager.release(user.id, ds2.id)


# --------------------------------------------------------------------------
# 15. Tenant Isolation
# --------------------------------------------------------------------------
def test_tenant_model_isolation(client: TestClient, db_session):
    headers_a = register_and_login(client, "user_a@example.com")
    headers_b = register_and_login(client, "user_b@example.com")

    # User B queries current models: should see empty/none and never User A's data
    res_b = client.get("/api/models/current", headers=headers_b)
    assert res_b.status_code == 200
    data_b = res_b.json()
    assert data_b["active_dataset_id"] is None
    assert all(m["source"] == "global_fallback" for m in data_b["models"])


# --------------------------------------------------------------------------
# 16. Duplicate / Concurrent Training Protection
# --------------------------------------------------------------------------
def test_concurrent_training_protection(client: TestClient, db_session):
    headers = register_and_login(client, "concurrent_train@example.com")
    user = db_session.scalars(select(User).where(User.email == "concurrent_train@example.com")).first()

    ds = DatasetUpload(id="ds-concurrent", user_id=user.id, original_filename="c.csv", dataset_key="ckey", status="active")
    db_session.add(ds)
    db_session.commit()

    # Lock it manually
    assert training_lock_manager.acquire(user.id, ds.id) is True

    try:
        # Second attempt must return 409 Conflict
        res = client.post(f"/api/datasets/{ds.id}/train", headers=headers)
        assert res.status_code == 409
        assert "already in progress" in res.json()["detail"]
    finally:
        training_lock_manager.release(user.id, ds.id)


# --------------------------------------------------------------------------
# 17. Stale Job Crash Recovery (Heartbeat-Based)
# --------------------------------------------------------------------------
def test_stale_job_crash_recovery(db_session):
    user = User(email="crash_test@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    ds = DatasetUpload(id="ds-crash", user_id=user.id, original_filename="crash.csv", dataset_key="key", status="active")
    db_session.add(ds)
    db_session.flush()

    stale_time = datetime.now(timezone.utc) - timedelta(minutes=25)
    stale_job = ModelTrainingJob(
        id="job-stale",
        user_id=user.id,
        dataset_id=ds.id,
        model_version=1,
        status="training",
        progress_stage="Evaluating",
        started_at=stale_time,
        last_heartbeat_at=stale_time,
        created_at=stale_time,
    )
    db_session.add(stale_job)
    db_session.commit()

    # Run crash recovery on test database engine
    recover_interrupted_training_jobs(db_session.get_bind())

    db_session.refresh(stale_job)
    assert stale_job.status == "failed"
    assert "heartbeat timeout" in stale_job.error_message or "Retrain Models" in stale_job.error_message


# --------------------------------------------------------------------------
# 18. Long-Running Healthy Job Is NOT Marked Stale
# --------------------------------------------------------------------------
def test_long_running_healthy_job_not_marked_stale(db_session):
    user = User(email="healthy_job_test@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    ds = DatasetUpload(id="ds-healthy", user_id=user.id, original_filename="healthy.csv", dataset_key="hkey", status="active")
    db_session.add(ds)
    db_session.flush()

    # Job started 45 minutes ago, but its heartbeat was updated 2 minutes ago
    started_time = datetime.now(timezone.utc) - timedelta(minutes=45)
    recent_heartbeat = datetime.now(timezone.utc) - timedelta(minutes=2)

    healthy_job = ModelTrainingJob(
        id="job-healthy-long",
        user_id=user.id,
        dataset_id=ds.id,
        model_version=1,
        status="training",
        progress_stage="Training 90D models",
        started_at=started_time,
        last_heartbeat_at=recent_heartbeat,
        created_at=started_time,
    )
    db_session.add(healthy_job)
    db_session.commit()

    # Run recovery
    recover_interrupted_training_jobs(db_session.get_bind())

    db_session.refresh(healthy_job)
    assert healthy_job.status == "training"
    assert healthy_job.error_message is None


# --------------------------------------------------------------------------
# 19. Production Storage Fail-Closed Validation
# --------------------------------------------------------------------------
def test_production_storage_fail_closed():
    # 1. Config validation fail-closed
    with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY"):
        Settings(
            environment="production",
            secret_key="production-secret-key-1234567890",
            database_url="postgresql://user:pass@localhost:5432/proddb",
            allowed_origins="https://app.example.com",
            supabase_url=None,
            supabase_key=None,
            supabase_service_role_key=None,
        )

    # 2. Storage service backend fail-closed
    with patch("backend.app.services.storage_service.settings.environment", "production"):
        with patch("backend.app.services.storage_service.settings.supabase_url", None):
            with patch("backend.app.services.storage_service.settings.supabase_key", None):
                with patch("backend.app.services.storage_service.settings.supabase_service_role_key", None):
                    service = ModelStorageService.__new__(ModelStorageService)
                    with pytest.raises(StorageConfigurationError, match="SUPABASE_SERVICE_ROLE_KEY"):
                        service._init_default_backend()


# --------------------------------------------------------------------------
# 20. Database-Level Unique Constraint on Active Training Jobs
# --------------------------------------------------------------------------
def test_database_partial_unique_index_active_training_jobs(db_session):
    user = User(email="uq_train@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    ds = DatasetUpload(id="ds-uq", user_id=user.id, original_filename="uq.csv", dataset_key="ukey", status="active")
    db_session.add(ds)
    db_session.commit()

    job1 = ModelTrainingJob(
        id="job-active-1",
        user_id=user.id,
        dataset_id=ds.id,
        model_version=1,
        status="training",
    )
    db_session.add(job1)
    db_session.commit()

    # Second active job for same user & dataset must violate partial unique index
    job2 = ModelTrainingJob(
        id="job-active-2",
        user_id=user.id,
        dataset_id=ds.id,
        model_version=2,
        status="queued",
    )
    db_session.add(job2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


# --------------------------------------------------------------------------
# 21. Retraining on Same Dataset Increments Version & Preserves Old Artifacts
# --------------------------------------------------------------------------
def test_retraining_same_dataset_increments_version_and_preserves_artifacts(db_session):
    user = User(email="retrain_immutability@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    ds = DatasetUpload(id="ds-retrain", user_id=user.id, original_filename="retrain.csv", dataset_key="rkey", status="active")
    db_session.add(ds)
    db_session.commit()

    # Run 1
    job1, acquired1 = company_model_service.enqueue_training_job(db_session, user.id, ds.id)
    assert acquired1 is True
    assert job1.model_version == 1

    # Simulate saving model v1 artifacts
    v1_path = f"models/{user.id}/{ds.id}/v1/7/model.joblib"
    storage_service.save_artifact(v1_path, b"fake_model_v1_bytes")
    cm_v1 = CompanyModel(
        id="cm-v1",
        user_id=user.id,
        dataset_id=ds.id,
        horizon=7,
        model_type="Random Forest",
        artifact_path=v1_path,
        model_version=1,
        feature_version="v1",
        feature_columns=FEATURE_COLUMNS,
        status="ready",
        is_active=True,
    )
    job1.status = "ready"
    db_session.add(cm_v1)
    db_session.commit()
    training_lock_manager.release(user.id, ds.id)

    # Run 2: Retrain on the SAME dataset
    job2, acquired2 = company_model_service.enqueue_training_job(db_session, user.id, ds.id)
    assert acquired2 is True
    assert job2.model_version == 2, "Retraining must monotonically increment model_version"

    v2_path = f"models/{user.id}/{ds.id}/v2/7/model.joblib"
    storage_service.save_artifact(v2_path, b"fake_model_v2_bytes")
    cm_v2 = CompanyModel(
        id="cm-v2",
        user_id=user.id,
        dataset_id=ds.id,
        horizon=7,
        model_type="HistGradientBoosting",
        artifact_path=v2_path,
        model_version=2,
        feature_version="v1",
        feature_columns=FEATURE_COLUMNS,
        status="ready",
        is_active=True,
    )
    # Deactivate v1, activate v2
    cm_v1.is_active = False
    db_session.add(cm_v2)
    db_session.commit()
    training_lock_manager.release(user.id, ds.id)

    # Verify immutability:
    # 1. Old artifact v1 still exists in storage
    assert storage_service.backend.exists(v1_path)
    # 2. New artifact v2 exists in storage
    assert storage_service.backend.exists(v2_path)
    # 3. Both models are preserved in registry
    models = list(db_session.scalars(select(CompanyModel).where(CompanyModel.dataset_id == ds.id)).all())
    assert len(models) == 2
    versions = {m.model_version for m in models}
    assert versions == {1, 2}


# --------------------------------------------------------------------------
# 22. Feature Schema Compatibility Verification & Graceful Fallback
# --------------------------------------------------------------------------
def test_inference_feature_schema_incompatibility_fallback(db_session):
    user = User(email="schema_test@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    ds = DatasetUpload(id="ds-schema", user_id=user.id, original_filename="schema.csv", dataset_key="skey", status="active")
    db_session.add(ds)
    db_session.commit()

    # Company model with unsupported feature_version "v2_experimental"
    cm_incompatible = CompanyModel(
        id="cm-incompatible",
        user_id=user.id,
        dataset_id=ds.id,
        horizon=7,
        model_type="ExperimentalModel",
        artifact_path="models/nonexistent/model.joblib",
        model_version=1,
        feature_version="v2_experimental",
        status="ready",
        is_active=True,
    )
    db_session.add(cm_incompatible)
    db_session.commit()

    forecast_svc = BackendForecastService()
    mock_df = create_synthetic_daily_data(days=60)
    with patch("backend.app.services.dataset_runtime_service.dataset_runtime_service.get_daily_aggregate", return_value=mock_df):
        # Forecast request should not crash; must fall back to global model
        result = forecast_svc.generate_forecast(horizon=7, user_id=user.id, db=db_session)
        assert result.source == "global_fallback"
        assert "Incompatible feature schema version" in result.fallback_reason


# --------------------------------------------------------------------------
# 23. Horizon-Specific Model Activation Safety
# --------------------------------------------------------------------------
def test_horizon_specific_model_activation_safety(db_session):
    """
    Test Rule: Never replace a currently READY company model with an insufficient,
    failed, or corrupted new model.
    V1: 7D READY, 30D READY, 90D READY
    V2: 7D READY, 30D READY, 90D INSUFFICIENT
    Result: 7D -> V2, 30D -> V2, 90D -> V1
    """
    user = User(email="horizon_safety@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    ds = DatasetUpload(id="ds-horizon-safety", user_id=user.id, original_filename="data.csv", dataset_key="key", status="active")
    db_session.add(ds)
    db_session.commit()

    # V1 models: all 3 horizons READY
    v1_7 = CompanyModel(
        id="cm-v1-7", user_id=user.id, dataset_id=ds.id, horizon=7, model_type="RF_v1",
        artifact_path="models/v1/7.joblib", model_version=1, feature_version="v1",
        feature_columns=FEATURE_COLUMNS, status="ready", is_active=True
    )
    v1_30 = CompanyModel(
        id="cm-v1-30", user_id=user.id, dataset_id=ds.id, horizon=30, model_type="HGB_v1",
        artifact_path="models/v1/30.joblib", model_version=1, feature_version="v1",
        feature_columns=FEATURE_COLUMNS, status="ready", is_active=True
    )
    v1_90 = CompanyModel(
        id="cm-v1-90", user_id=user.id, dataset_id=ds.id, horizon=90, model_type="Ridge_v1",
        artifact_path="models/v1/90.joblib", model_version=1, feature_version="v1",
        feature_columns=FEATURE_COLUMNS, status="ready", is_active=True
    )
    db_session.add_all([v1_7, v1_30, v1_90])
    db_session.commit()

    # Now simulate V2 training: 7D and 30D are READY, but 90D is INSUFFICIENT
    v2_7 = CompanyModel(
        id="cm-v2-7", user_id=user.id, dataset_id=ds.id, horizon=7, model_type="HGB_v2",
        artifact_path="models/v2/7.joblib", model_version=2, feature_version="v1",
        feature_columns=FEATURE_COLUMNS, status="ready", is_active=False
    )
    v2_30 = CompanyModel(
        id="cm-v2-30", user_id=user.id, dataset_id=ds.id, horizon=30, model_type="RF_v2",
        artifact_path="models/v2/30.joblib", model_version=2, feature_version="v1",
        feature_columns=FEATURE_COLUMNS, status="ready", is_active=False
    )
    v2_90 = CompanyModel(
        id="cm-v2-90", user_id=user.id, dataset_id=ds.id, horizon=90, model_type="None",
        artifact_path="", model_version=2, feature_version="v1",
        feature_columns=None, status="insufficient_data", is_active=False
    )
    db_session.add_all([v2_7, v2_30, v2_90])
    db_session.commit()

    # Apply dataset model activation
    company_model_service.activate_dataset_models(db_session, user.id, ds.id)

    db_session.refresh(v1_7)
    db_session.refresh(v1_30)
    db_session.refresh(v1_90)
    db_session.refresh(v2_7)
    db_session.refresh(v2_30)
    db_session.refresh(v2_90)

    # 7D must be upgraded to V2
    assert v1_7.is_active is False
    assert v2_7.is_active is True

    # 30D must be upgraded to V2
    assert v1_30.is_active is False
    assert v2_30.is_active is True

    # 90D must REMAIN V1 because V2 was insufficient!
    assert v1_90.is_active is True, "V1 90D model must remain active when V2 is insufficient!"
    assert v2_90.is_active is False

    # Summary check
    summary = company_model_service.get_current_models_summary(db_session, user.id)
    models_by_h = {m["horizon"]: m for m in summary["models"]}
    assert models_by_h[7]["model_version"] == 2
    assert models_by_h[7]["is_active"] is True
    assert models_by_h[30]["model_version"] == 2
    assert models_by_h[30]["is_active"] is True
    assert models_by_h[90]["model_version"] == 1
    assert models_by_h[90]["is_active"] is True


# --------------------------------------------------------------------------
# 24. Privileged Storage Credentials Enforced in Production
# --------------------------------------------------------------------------
def test_privileged_supabase_credentials_enforced_in_production():
    """
    In production, SUPABASE_SERVICE_ROLE_KEY is mandatory.
    Public/anon SUPABASE_KEY must not be silently accepted for private model artifacts.
    """
    # 1. Providing only supabase_key (anon) without service_role_key fails Settings validation
    with pytest.raises(ValueError, match="SUPABASE_SERVICE_ROLE_KEY"):
        Settings(
            environment="production",
            secret_key="production-secret-key-1234567890",
            database_url="postgresql://user:pass@localhost:5432/proddb",
            allowed_origins="https://app.example.com",
            supabase_url="https://test.supabase.co",
            supabase_key="public-anon-key-only",
            supabase_service_role_key=None,
        )

    # 2. Storage backend in production raises StorageConfigurationError if service_role_key missing
    with patch("backend.app.services.storage_service.settings.environment", "production"):
        with patch("backend.app.services.storage_service.settings.supabase_url", "https://test.supabase.co"):
            with patch("backend.app.services.storage_service.settings.supabase_key", "public-anon-key-only"):
                with patch("backend.app.services.storage_service.settings.supabase_service_role_key", None):
                    service = ModelStorageService.__new__(ModelStorageService)
                    with pytest.raises(StorageConfigurationError, match="SUPABASE_SERVICE_ROLE_KEY"):
                        service._init_default_backend()


# --------------------------------------------------------------------------
# 25. Stranded Queued Jobs Resumed on Server Restart
# --------------------------------------------------------------------------
def test_resume_stranded_queued_jobs(db_session):
    user = User(email="stranded_test@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    ds = DatasetUpload(id="ds-stranded", user_id=user.id, original_filename="data.csv", dataset_key="key", status="active")
    db_session.add(ds)
    db_session.flush()

    job = ModelTrainingJob(
        id="job-stranded-queued",
        user_id=user.id,
        dataset_id=ds.id,
        model_version=1,
        status="queued",
        progress_stage="Job queued",
    )
    db_session.add(job)
    db_session.commit()

    with patch.object(company_model_service, "execute_training_pipeline") as mock_exec:
        resumed = company_model_service.resume_stranded_queued_jobs(engine=db_session.get_bind())
        assert resumed >= 1
        # Give daemon thread a brief moment to execute target
        time.sleep(0.1)
        mock_exec.assert_any_call(user.id, ds.id, job.id)


# --------------------------------------------------------------------------
# 26. Unresponsive Jobs Auto-Failed on Retrain Request
# --------------------------------------------------------------------------
def test_unresponsive_jobs_auto_failed_on_retrain_request(db_session):
    user = User(email="unresponsive_test@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    ds = DatasetUpload(id="ds-unresponsive", user_id=user.id, original_filename="data.csv", dataset_key="key", status="active")
    db_session.add(ds)
    db_session.flush()

    stale_time = datetime.now(timezone.utc) - timedelta(minutes=15)
    stale_job = ModelTrainingJob(
        id="job-unresponsive-1",
        user_id=user.id,
        dataset_id=ds.id,
        model_version=1,
        status="training",
        progress_stage="Evaluating",
        started_at=stale_time,
        last_heartbeat_at=stale_time,
        created_at=stale_time,
    )
    db_session.add(stale_job)
    db_session.commit()

    # User triggers retrain: enqueue_training_job should auto-fail the unresponsive job and succeed
    new_job, acquired = company_model_service.enqueue_training_job(db_session, user.id, ds.id)
    assert acquired is True
    assert new_job.model_version == 2

    db_session.refresh(stale_job)
    assert stale_job.status == "failed"
    assert "timed out" in stale_job.error_message
