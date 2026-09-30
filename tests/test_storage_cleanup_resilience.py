from __future__ import annotations

import io
import uuid
import urllib.error
from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.database.models import (
    CompanyElasticityModel,
    CompanyModel,
    DatasetUpload,
    ModelTrainingJob,
    Product,
    SalesRecord,
    User,
)
from backend.app.services.dataset_runtime_service import dataset_runtime_service
from backend.app.services.storage_service import (
    DeleteResult,
    LocalStorageBackend,
    ModelStorageService,
    StorageCleanupResult,
    SupabaseStorageBackend,
    build_model_artifact_path,
    classify_supabase_storage_error,
    storage_service,
    validate_artifact_path,
)


def make_dataset(user_id: int, status: str = "archived", filename: str = "data.csv") -> DatasetUpload:
    ds_id = f"ds-{uuid.uuid4().hex[:8]}"
    return DatasetUpload(
        id=ds_id,
        user_id=user_id,
        original_filename=filename,
        dataset_key=f"raw/datasets/{user_id}/{ds_id}.csv",
        status=status,
        row_count=50,
    )


# ---------------------------------------------------------------------------
# 1. Path validation & Canonical Generation
# ---------------------------------------------------------------------------

def test_validate_artifact_path():
    # Valid model paths
    assert validate_artifact_path("models/1/ds-123/v1/7/model.joblib") is True
    assert validate_artifact_path("models/1/ds-123/v1/7d/model.joblib") is True
    assert validate_artifact_path("models/u1/ds-abc/v2/30/model.joblib") is True
    assert validate_artifact_path("models/1/ds-123/v1/elasticity/model.joblib") is True
    assert validate_artifact_path("models/user_99/ds-test/v10/elasticity/model.json") is True

    # Invalid / dangerous paths
    assert validate_artifact_path("") is False
    assert validate_artifact_path(None) is False
    assert validate_artifact_path("..") is False
    assert validate_artifact_path("models/../etc/passwd") is False
    assert validate_artifact_path("models/1/ds/v1/../../secret") is False
    assert validate_artifact_path("not_models/1/ds/v1/7/model.joblib") is False
    assert validate_artifact_path("models/1/ds-123") is False
    assert validate_artifact_path("models/1/ds-123/v1/7/model.exe") is False


def test_build_model_artifact_path():
    p1 = build_model_artifact_path(user_id=1, dataset_id="ds-100", model_version=1, horizon_or_type=7)
    assert p1 == "models/1/ds-100/v1/7/model.joblib"

    p2 = build_model_artifact_path(user_id="user_2", dataset_id="ds-200", model_version="v2", horizon_or_type="30d")
    assert p2 == "models/user_2/ds-200/v2/30d/model.joblib"

    p3 = build_model_artifact_path(user_id=5, dataset_id="ds-300", model_version=3, horizon_or_type="elasticity")
    assert p3 == "models/5/ds-300/v3/elasticity/model.joblib"


# ---------------------------------------------------------------------------
# 2. Supabase Storage: Successful Deletion (HTTP 200/204)
# ---------------------------------------------------------------------------

def test_supabase_storage_successful_object_deletion():
    backend = SupabaseStorageBackend(
        supabase_url="https://mock.supabase.co",
        supabase_key="mock-key",
        bucket_name="company-models",
    )
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_cm):
        res = backend.delete("models/1/ds-123/v1/7/model.joblib")
        assert res.result == StorageCleanupResult.DELETED
        assert res.status_code == 200
        assert res.is_success_or_benign is True


# ---------------------------------------------------------------------------
# 3. Supabase Storage: Object Already Absent (HTTP 404 & HTTP 400 not found)
# ---------------------------------------------------------------------------

def test_supabase_storage_object_already_absent_404():
    backend = SupabaseStorageBackend(
        supabase_url="https://mock.supabase.co",
        supabase_key="mock-key",
        bucket_name="company-models",
    )
    err = urllib.error.HTTPError(
        url="https://mock.supabase.co/storage/v1/object/company-models/path",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=io.BytesIO(b'{"statusCode":"404","message":"The resource was not found"}'),
    )

    with patch("urllib.request.urlopen", side_effect=err):
        res = backend.delete("models/1/ds-123/v1/7/model.joblib")
        assert res.result == StorageCleanupResult.ALREADY_ABSENT
        assert res.status_code == 404
        assert res.is_success_or_benign is True


def test_supabase_storage_object_already_absent_400_not_found():
    backend = SupabaseStorageBackend(
        supabase_url="https://mock.supabase.co",
        supabase_key="mock-key",
        bucket_name="company-models",
    )
    err = urllib.error.HTTPError(
        url="https://mock.supabase.co/storage/v1/object/company-models/path",
        code=400,
        msg="Bad Request",
        hdrs={},
        fp=io.BytesIO(b'{"statusCode":"400","error":"Bad Request","message":"The resource was not found"}'),
    )

    with patch("urllib.request.urlopen", side_effect=err):
        res = backend.delete("models/1/ds-123/v1/7/model.joblib")
        assert res.result == StorageCleanupResult.ALREADY_ABSENT
        assert res.status_code == 400
        assert res.is_success_or_benign is True


def test_supabase_storage_actionable_400_not_classified_as_absent():
    backend = SupabaseStorageBackend(
        supabase_url="https://mock.supabase.co",
        supabase_key="mock-key",
        bucket_name="company-models",
    )
    # HTTP 400 with a message that does NOT indicate resource was absent
    err = urllib.error.HTTPError(
        url="https://mock.supabase.co/storage/v1/object/company-models/path",
        code=400,
        msg="Bad Request",
        hdrs={},
        fp=io.BytesIO(b'{"statusCode":"400","error":"Bad Request","message":"Invalid payload or parameters"}'),
    )

    with patch("urllib.request.urlopen", side_effect=err):
        res = backend.delete("models/1/ds-123/v1/7/model.joblib")
        # Must be treated as actionable failure, NOT ALREADY_ABSENT
        assert res.result == StorageCleanupResult.INVALID_PATH
        assert res.status_code == 400
        assert res.is_success_or_benign is False
        assert "Invalid payload" in (res.message or "")


# ---------------------------------------------------------------------------
# 4. Supabase Storage: Invalid Object Path
# ---------------------------------------------------------------------------

def test_supabase_storage_invalid_object_path():
    backend = SupabaseStorageBackend(
        supabase_url="https://mock.supabase.co",
        supabase_key="mock-key",
        bucket_name="company-models",
    )

    with patch("urllib.request.urlopen") as mock_open:
        res = backend.delete("../invalid/traversal/path")
        assert res.result == StorageCleanupResult.INVALID_PATH
        assert res.is_success_or_benign is False
        mock_open.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Supabase Storage: Permission / RLS Error (HTTP 401 & 403)
# ---------------------------------------------------------------------------

def test_supabase_storage_permission_error():
    backend = SupabaseStorageBackend(
        supabase_url="https://mock.supabase.co",
        supabase_key="mock-key",
        bucket_name="company-models",
    )
    err_403 = urllib.error.HTTPError(
        url="https://mock.supabase.co/storage/v1/object/company-models/path",
        code=403,
        msg="Forbidden",
        hdrs={},
        fp=io.BytesIO(b'{"statusCode":"403","message":"Access denied by RLS policy"}'),
    )

    with patch("urllib.request.urlopen", side_effect=err_403):
        res = backend.delete("models/1/ds-123/v1/7/model.joblib")
        assert res.result == StorageCleanupResult.PERMISSION_DENIED
        assert res.status_code == 403
        assert res.is_success_or_benign is False


# ---------------------------------------------------------------------------
# 6. Supabase Storage: Transient Error (HTTP 500/503/Timeout)
# ---------------------------------------------------------------------------

def test_supabase_storage_transient_error():
    backend = SupabaseStorageBackend(
        supabase_url="https://mock.supabase.co",
        supabase_key="mock-key",
        bucket_name="company-models",
    )
    err_503 = urllib.error.HTTPError(
        url="https://mock.supabase.co/storage/v1/object/company-models/path",
        code=503,
        msg="Service Unavailable",
        hdrs={},
        fp=io.BytesIO(b'Service Unavailable'),
    )

    with patch("urllib.request.urlopen", side_effect=err_503):
        res = backend.delete("models/1/ds-123/v1/7/model.joblib")
        assert res.result == StorageCleanupResult.TRANSIENT_ERROR
        assert res.status_code == 503
        assert res.is_success_or_benign is False

    with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out")):
        res = backend.delete("models/1/ds-123/v1/7/model.joblib")
        assert res.result == StorageCleanupResult.TRANSIENT_ERROR
        assert res.is_success_or_benign is False


# ---------------------------------------------------------------------------
# 7. Repeated / Idempotent Cleanup
# ---------------------------------------------------------------------------

def test_repeated_idempotent_cleanup(tmp_path):
    backend = LocalStorageBackend(base_dir=tmp_path / "models")
    svc = ModelStorageService(backend=backend)
    path = "models/1/ds-123/v1/7/model.joblib"

    svc.save_artifact(path, b"test_artifact_bytes")
    assert svc.backend.exists(path) is True

    # First delete -> DELETED
    res1 = svc.delete_artifact(path)
    assert res1.result == StorageCleanupResult.DELETED
    assert svc.backend.exists(path) is False

    # Second delete on same path -> ALREADY_ABSENT (benign, no exception)
    res2 = svc.delete_artifact(path)
    assert res2.result == StorageCleanupResult.ALREADY_ABSENT
    assert res2.is_success_or_benign is True


# ---------------------------------------------------------------------------
# 8. Post-Commit Safety: Commit succeeds even if storage cleanup fails
# ---------------------------------------------------------------------------

def test_database_commit_succeeds_even_when_post_commit_cleanup_fails(db_session):
    user = User(email=f"user_storage_fail_{uuid.uuid4().hex[:6]}@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    ds_act = make_dataset(user.id, status="active", filename="act.csv")
    ds_tgt = make_dataset(user.id, status="archived", filename="tgt.csv")
    db_session.add_all([ds_act, ds_tgt])
    db_session.flush()
    ds_tgt_id = ds_tgt.id

    cm = CompanyModel(
        id=f"cm-{uuid.uuid4().hex[:6]}",
        user_id=user.id,
        dataset_id=ds_tgt_id,
        horizon=7,
        model_type="HistGradientBoosting",
        artifact_path=f"models/{user.id}/{ds_tgt_id}/v1/7/model.joblib",
        model_version=1,
        status="ready",
        is_active=False,
    )
    db_session.add(cm)
    db_session.commit()

    # Simulate storage backend raising an unexpected exception during post-commit cleanup
    with patch.object(storage_service, "delete_artifact", side_effect=Exception("Critical Supabase down")):
        # Must NOT raise exception, and must NOT rollback the database deletion
        dataset_runtime_service.delete_dataset(
            db=db_session,
            user_id=user.id,
            dataset_id=ds_tgt_id,
        )

    # Verify that database records remain committed as deleted
    check_ds = db_session.query(DatasetUpload).filter(DatasetUpload.id == ds_tgt_id).first()
    check_cm = db_session.query(CompanyModel).filter(CompanyModel.dataset_id == ds_tgt_id).first()
    assert check_ds is None
    assert check_cm is None


# ---------------------------------------------------------------------------
# 9. Forecast & Elasticity Model Cleanup on Dataset Deletion
# ---------------------------------------------------------------------------

def test_forecast_and_elasticity_model_cleanup(db_session):
    user = User(email=f"user_fe_clean_{uuid.uuid4().hex[:6]}@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    ds_act = make_dataset(user.id, status="active", filename="act.csv")
    ds_tgt = make_dataset(user.id, status="archived", filename="tgt.csv")
    db_session.add_all([ds_act, ds_tgt])
    db_session.flush()
    ds_tgt_id = ds_tgt.id

    p_7 = f"models/{user.id}/{ds_tgt_id}/v1/7/model.joblib"
    p_30 = f"models/{user.id}/{ds_tgt_id}/v1/30/model.joblib"
    p_90 = f"models/{user.id}/{ds_tgt_id}/v1/90/model.joblib"
    p_el = f"models/{user.id}/{ds_tgt_id}/v1/elasticity/model.joblib"

    storage_service.save_artifact(p_7, b"bytes_7")
    storage_service.save_artifact(p_30, b"bytes_30")
    storage_service.save_artifact(p_90, b"bytes_90")
    storage_service.save_artifact(p_el, b"bytes_el")

    for h, p in [(7, p_7), (30, p_30), (90, p_90)]:
        db_session.add(CompanyModel(
            id=f"cm-{uuid.uuid4().hex[:6]}",
            user_id=user.id,
            dataset_id=ds_tgt_id,
            horizon=h,
            model_type="HistGradientBoosting",
            artifact_path=p,
            model_version=1,
            status="ready",
            is_active=False,
        ))

    db_session.add(CompanyElasticityModel(
        id=f"cem-{uuid.uuid4().hex[:6]}",
        user_id=user.id,
        dataset_id=ds_tgt_id,
        model_version=1,
        model_type="RidgeLogLog",
        artifact_path=p_el,
        price_elasticity=-0.5,
        discount_sensitivity=1.0,
        status="ready",
        is_active=False,
    ))
    db_session.commit()

    assert storage_service.backend.exists(p_7) is True
    assert storage_service.backend.exists(p_30) is True
    assert storage_service.backend.exists(p_90) is True
    assert storage_service.backend.exists(p_el) is True

    # Delete dataset
    dataset_runtime_service.delete_dataset(
        db=db_session,
        user_id=user.id,
        dataset_id=ds_tgt_id,
    )

    # Verify all artifacts removed
    assert storage_service.backend.exists(p_7) is False
    assert storage_service.backend.exists(p_30) is False
    assert storage_service.backend.exists(p_90) is False
    assert storage_service.backend.exists(p_el) is False


# ---------------------------------------------------------------------------
# 10. Multi-Version Dataset Deletion Cleanup
# ---------------------------------------------------------------------------

def test_dataset_deletion_with_multiple_model_versions(db_session):
    user = User(email=f"user_mv_clean_{uuid.uuid4().hex[:6]}@example.com", password_hash="hash")
    db_session.add(user)
    db_session.flush()

    ds_act = make_dataset(user.id, status="active", filename="act.csv")
    ds_tgt = make_dataset(user.id, status="archived", filename="tgt.csv")
    db_session.add_all([ds_act, ds_tgt])
    db_session.flush()
    ds_tgt_id = ds_tgt.id

    # Create v1 and v2 artifacts
    p_v1 = f"models/{user.id}/{ds_tgt_id}/v1/7/model.joblib"
    p_v2 = f"models/{user.id}/{ds_tgt_id}/v2/7/model.joblib"
    storage_service.save_artifact(p_v1, b"v1_bytes")
    storage_service.save_artifact(p_v2, b"v2_bytes")

    db_session.add(CompanyModel(
        id=f"cm-v1-{uuid.uuid4().hex[:6]}",
        user_id=user.id,
        dataset_id=ds_tgt_id,
        horizon=7,
        model_type="HistGradientBoosting",
        artifact_path=p_v1,
        model_version=1,
        status="ready",
        is_active=False,
    ))
    db_session.add(CompanyModel(
        id=f"cm-v2-{uuid.uuid4().hex[:6]}",
        user_id=user.id,
        dataset_id=ds_tgt_id,
        horizon=7,
        model_type="HistGradientBoosting",
        artifact_path=p_v2,
        model_version=2,
        status="ready",
        is_active=False,
    ))
    db_session.add(ModelTrainingJob(
        id=f"job-v2-{uuid.uuid4().hex[:6]}",
        user_id=user.id,
        dataset_id=ds_tgt_id,
        model_version=2,
        status="completed",
    ))
    db_session.commit()

    assert storage_service.backend.exists(p_v1) is True
    assert storage_service.backend.exists(p_v2) is True

    # Delete dataset
    dataset_runtime_service.delete_dataset(
        db=db_session,
        user_id=user.id,
        dataset_id=ds_tgt_id,
    )

    # Both versions cleaned up
    assert storage_service.backend.exists(p_v1) is False
    assert storage_service.backend.exists(p_v2) is False
