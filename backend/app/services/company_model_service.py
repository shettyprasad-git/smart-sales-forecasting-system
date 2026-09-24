from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.database.database import SessionLocal
from backend.app.database.models import CompanyModel, DatasetUpload, ModelTrainingJob
from backend.app.services.dataset_runtime_service import dataset_runtime_service
from backend.app.services.storage_service import storage_service
from ml.training.company_trainer import FEATURE_COLUMNS, benchmark_and_train_horizon

logger = logging.getLogger(__name__)


class TrainingConcurrencyError(Exception):
    """Raised when a concurrent training job is requested for an active dataset."""
    pass


class TrainingLockManager:
    """Thread-safe concurrency manager preventing duplicate training runs per (user_id, dataset_id)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_jobs: set[tuple[int, str]] = set()

    def acquire(self, user_id: int, dataset_id: str) -> bool:
        with self._lock:
            key = (user_id, dataset_id)
            if key in self._active_jobs:
                return False
            self._active_jobs.add(key)
            return True

    def release(self, user_id: int, dataset_id: str) -> None:
        with self._lock:
            self._active_jobs.discard((user_id, dataset_id))

    def is_locked(self, user_id: int, dataset_id: str) -> bool:
        with self._lock:
            return (user_id, dataset_id) in self._active_jobs


training_lock_manager = TrainingLockManager()


class CompanyModelService:
    """
    Orchestrates the asynchronous company model training lifecycle,
    model versioning, persistent storage uploads, and database registry updates.
    """

    SUPPORTED_HORIZONS = (7, 30, 90)

    def enqueue_training_job(
        self,
        db: Session,
        user_id: int,
        dataset_id: str,
    ) -> tuple[ModelTrainingJob, bool]:
        """
        Register a queued training job. Returns (job, acquired_lock).
        If already locked, returns existing active job and False.
        """
        # Check for existing job in progress in DB
        existing_job = db.scalars(
            select(ModelTrainingJob)
            .where(
                ModelTrainingJob.user_id == user_id,
                ModelTrainingJob.dataset_id == dataset_id,
                ModelTrainingJob.status.in_(("queued", "processing", "training", "evaluating")),
            )
            .order_by(ModelTrainingJob.created_at.desc())
        ).first()

        if existing_job:
            # Check if existing job is unresponsive (> 5 minutes without heartbeat)
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
            last_hb = existing_job.last_heartbeat_at or existing_job.started_at or existing_job.created_at
            is_stale = False
            if last_hb:
                if last_hb.tzinfo is None:
                    is_stale = (last_hb < cutoff.replace(tzinfo=None))
                else:
                    is_stale = (last_hb < cutoff)
            if is_stale:
                logger.warning(
                    "Found unresponsive active training job %s (last heartbeat: %s). Auto-failing to unblock new training.",
                    existing_job.id,
                    last_hb,
                )
                existing_job.status = "failed"
                existing_job.error_message = "Training job timed out (no heartbeat detected). Replaced by new run."
                existing_job.completed_at = datetime.now(timezone.utc)
                db.commit()
                training_lock_manager.release(user_id, dataset_id)
            else:
                return existing_job, False

        if not training_lock_manager.acquire(user_id, dataset_id):
            return existing_job, False

        # Monotonically increment model_version across all runs for this user.
        # This guarantees that retraining the same dataset creates an immutable new version
        # (e.g. v1 -> v2) without overwriting earlier artifacts or registry entries.
        max_job_ver = db.scalar(
            select(func.max(ModelTrainingJob.model_version)).where(
                ModelTrainingJob.user_id == user_id,
            )
        ) or 0
        max_model_ver = db.scalar(
            select(func.max(CompanyModel.model_version)).where(
                CompanyModel.user_id == user_id,
            )
        ) or 0
        model_version = max(max_job_ver, max_model_ver) + 1

        job_id = f"job-{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        job = ModelTrainingJob(
            id=job_id,
            user_id=user_id,
            dataset_id=dataset_id,
            model_version=model_version,
            status="queued",
            progress_stage="Job queued",
            started_at=None,
            last_heartbeat_at=now,
            created_at=now,
        )
        try:
            db.add(job)
            db.commit()
            db.refresh(job)
            return job, True
        except IntegrityError:
            db.rollback()
            training_lock_manager.release(user_id, dataset_id)
            existing_job = db.scalars(
                select(ModelTrainingJob)
                .where(
                    ModelTrainingJob.user_id == user_id,
                    ModelTrainingJob.dataset_id == dataset_id,
                    ModelTrainingJob.status.in_(("queued", "processing", "training", "evaluating")),
                )
                .order_by(ModelTrainingJob.created_at.desc())
            ).first()
            if existing_job:
                cutoff = datetime.now(timezone.utc) - timedelta(minutes=5)
                last_hb = existing_job.last_heartbeat_at or existing_job.started_at or existing_job.created_at
                is_stale = False
                if last_hb:
                    if last_hb.tzinfo is None:
                        is_stale = (last_hb < cutoff.replace(tzinfo=None))
                    else:
                        is_stale = (last_hb < cutoff)
                if is_stale:
                    existing_job.status = "failed"
                    existing_job.error_message = "Training job timed out (no heartbeat detected)."
                    existing_job.completed_at = datetime.now(timezone.utc)
                    db.commit()
                else:
                    return existing_job, False
            raise TrainingConcurrencyError("A training job is already in progress for this dataset.")

    def execute_training_pipeline(
        self,
        user_id: int,
        dataset_id: str,
        job_id: str,
    ) -> None:
        """
        Synchronous worker entrypoint for company model training.

        Worker Durability Semantics:
        - In current deployment, this is dispatched in-process via FastAPI BackgroundTasks.
        - BackgroundTasks is restart-aware, but not truly durable across process termination.
          If the web process restarts or is killed, in-flight training does not automatically
          resume mid-flight; interrupted jobs are marked failed at startup and require retry.
        - This function is decoupled from the HTTP request context (takes primitive IDs:
          user_id, dataset_id, job_id) so it can be migrated to an external durable worker
          (e.g., Celery, RQ, AWS SQS) without modifying the API contract or database schema.
        """
        logger.info("Starting background company model training for user=%s, dataset=%s, job=%s", user_id, dataset_id, job_id)
        db: Session = SessionLocal()

        try:
            job = db.scalars(select(ModelTrainingJob).where(ModelTrainingJob.id == job_id)).first()
            if not job:
                logger.error("Training job %s not found in database.", job_id)
                return

            now = datetime.now(timezone.utc)
            job.status = "processing"
            job.started_at = now
            job.last_heartbeat_at = now
            job.progress_stage = "Aggregating daily historical sales"
            db.commit()

            # 1. Fetch daily aggregate dataset
            daily_df = dataset_runtime_service.get_daily_aggregate(
                user_id=user_id,
                db=db,
                dataset_id=dataset_id,
            )

            if daily_df.empty:
                job.status = "failed"
                job.error_message = "Historical forecasting dataset is empty."
                job.completed_at = datetime.now(timezone.utc)
                job.last_heartbeat_at = datetime.now(timezone.utc)
                db.commit()
                return

            model_version = job.model_version

            # Check if this dataset is currently active for the user
            active_ds = dataset_runtime_service.get_active_dataset(db, user_id)
            dataset_is_active = (active_ds is not None and active_ds.id == dataset_id)

            # 2. Iterate through each horizon
            for horizon in self.SUPPORTED_HORIZONS:
                job.status = "training"
                job.last_heartbeat_at = datetime.now(timezone.utc)
                job.progress_stage = f"Benchmarking {horizon}-day forecasting models"
                db.commit()

                # Train and benchmark candidates
                result = benchmark_and_train_horizon(daily_df, horizon)

                cm_id = f"cm-{uuid.uuid4().hex[:12]}"
                trained_now = datetime.now(timezone.utc)

                if result["status"] == "ready":
                    remote_path = f"models/{user_id}/{dataset_id}/v{model_version}/{horizon}/model.joblib"
                    # Upload to persistent storage (and cache locally)
                    storage_service.save_artifact(remote_path, result["artifact_bytes"])

                    # If dataset is currently active, deactivate older models for this horizon
                    if dataset_is_active:
                        db.execute(
                            update(CompanyModel)
                            .where(
                                CompanyModel.user_id == user_id,
                                CompanyModel.horizon == horizon,
                                CompanyModel.is_active == True,
                            )
                            .values(is_active=False)
                        )

                    company_model = CompanyModel(
                        id=cm_id,
                        user_id=user_id,
                        dataset_id=dataset_id,
                        horizon=horizon,
                        model_type=result["model_type"],
                        artifact_path=remote_path,
                        model_version=model_version,
                        validation_wape=result["validation_wape"],
                        validation_mae=result["validation_mae"],
                        validation_rmse=result["validation_rmse"],
                        test_wape=result["test_wape"],
                        test_mae=result["test_mae"],
                        test_rmse=result["test_rmse"],
                        training_rows=result["training_rows"],
                        feature_version="v1",
                        feature_columns=FEATURE_COLUMNS,
                        target_column="Quantity",
                        reference_start_date=result.get("reference_start_date"),
                        status="ready",
                        status_message=None,
                        is_active=dataset_is_active,
                        trained_at=trained_now,
                        created_at=trained_now,
                    )
                    db.add(company_model)
                    db.commit()

                elif result["status"] == "insufficient_data":
                    company_model = CompanyModel(
                        id=cm_id,
                        user_id=user_id,
                        dataset_id=dataset_id,
                        horizon=horizon,
                        model_type="None",
                        artifact_path="",
                        model_version=model_version,
                        validation_wape=None,
                        validation_mae=None,
                        validation_rmse=None,
                        test_wape=None,
                        test_mae=None,
                        test_rmse=None,
                        training_rows=None,
                        feature_version="v1",
                        feature_columns=None,
                        target_column="Quantity",
                        reference_start_date=result.get("reference_start_date"),
                        status="insufficient_data",
                        status_message=result["status_message"],
                        is_active=False,
                        trained_at=trained_now,
                        created_at=trained_now,
                    )
                    db.add(company_model)
                    db.commit()

                # Heartbeat after saving horizon
                job.last_heartbeat_at = datetime.now(timezone.utc)
                job.progress_stage = f"Completed {horizon}-day forecasting model"
                db.commit()

            # Completed successfully
            job.status = "ready"
            job.last_heartbeat_at = datetime.now(timezone.utc)
            job.progress_stage = "All forecasting horizons successfully benchmarked and saved"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            logger.info("Training pipeline completed successfully for job=%s", job_id)

        except Exception as exc:
            db.rollback()
            logger.exception("Unexpected error during training pipeline for job=%s: %s", job_id, exc)
            try:
                job = db.scalars(select(ModelTrainingJob).where(ModelTrainingJob.id == job_id)).first()
                if job:
                    job.status = "failed"
                    job.error_message = str(exc)
                    job.completed_at = datetime.now(timezone.utc)
                    job.last_heartbeat_at = datetime.now(timezone.utc)
                    db.commit()
            except Exception:
                pass
        finally:
            training_lock_manager.release(user_id, dataset_id)
            db.close()

    def activate_dataset_models(self, db: Session, user_id: int, dataset_id: str) -> None:
        """
        Synchronize active model flags when a dataset is activated.
        Applies strict horizon-level safety:
        For each horizon (7D, 30D, 90D):
        Activates the latest model belonging to this dataset that has status == 'ready'.
        Never replaces a READY model with an insufficient, failed, or corrupted model.
        """
        for horizon in self.SUPPORTED_HORIZONS:
            # Find latest ready model for this dataset and this specific horizon
            latest_ready = db.scalars(
                select(CompanyModel)
                .where(
                    CompanyModel.user_id == user_id,
                    CompanyModel.dataset_id == dataset_id,
                    CompanyModel.horizon == horizon,
                    CompanyModel.status == "ready",
                )
                .order_by(CompanyModel.model_version.desc(), CompanyModel.created_at.desc())
            ).first()

            if latest_ready:
                # Deactivate any other active model for this user and this horizon
                db.execute(
                    update(CompanyModel)
                    .where(
                        CompanyModel.user_id == user_id,
                        CompanyModel.horizon == horizon,
                        CompanyModel.id != latest_ready.id,
                        CompanyModel.is_active == True,
                    )
                    .values(is_active=False)
                )
                latest_ready.is_active = True
            else:
                # If this dataset has NO ready model for this horizon:
                # Deactivate any active model for this horizon that belonged to another dataset
                db.execute(
                    update(CompanyModel)
                    .where(
                        CompanyModel.user_id == user_id,
                        CompanyModel.horizon == horizon,
                        CompanyModel.dataset_id != dataset_id,
                        CompanyModel.is_active == True,
                    )
                    .values(is_active=False)
                )

        db.commit()
        logger.info("Updated active company model pointers for user_id=%s, dataset_id=%s with horizon-specific safety", user_id, dataset_id)

    def get_current_models_summary(self, db: Session, user_id: int) -> dict[str, Any]:
        """
        Retrieve current active models across 7D, 30D, and 90D for the active dataset.
        """
        active_ds = dataset_runtime_service.get_active_dataset(db, user_id)
        if not active_ds:
            return {
                "active_dataset_id": None,
                "active_model_version": None,
                "training_job": None,
                "models": [],
            }

        # Latest training job for this dataset
        latest_job = db.scalars(
            select(ModelTrainingJob)
            .where(
                ModelTrainingJob.user_id == user_id,
                ModelTrainingJob.dataset_id == active_ds.id,
            )
            .order_by(ModelTrainingJob.created_at.desc())
        ).first()

        job_dict = None
        if latest_job:
            job_dict = {
                "job_id": latest_job.id,
                "status": latest_job.status,
                "progress_stage": latest_job.progress_stage,
                "error_message": latest_job.error_message,
                "started_at": latest_job.started_at,
                "completed_at": latest_job.completed_at,
            }

        # Query models for this dataset
        models_query = (
            select(CompanyModel)
            .where(
                CompanyModel.user_id == user_id,
                CompanyModel.dataset_id == active_ds.id,
            )
            .order_by(CompanyModel.horizon.asc(), CompanyModel.created_at.desc())
        )
        all_models = list(db.scalars(models_query).all())

        # For each horizon, prioritize the currently active model (e.g. V1 ready while V2 is insufficient),
        # otherwise pick the latest model record for that horizon.
        horizon_models: dict[int, CompanyModel] = {}
        for m in all_models:
            if m.is_active:
                horizon_models[m.horizon] = m

        for m in all_models:
            if m.horizon not in horizon_models:
                horizon_models[m.horizon] = m

        active_version = None
        models_list = []
        for h in self.SUPPORTED_HORIZONS:
            m = horizon_models.get(h)
            if m:
                if active_version is None and m.model_version:
                    active_version = m.model_version
                models_list.append(
                    {
                        "horizon": m.horizon,
                        "status": m.status,
                        "model_type": m.model_type if m.status == "ready" else None,
                        "model_version": m.model_version,
                        "validation_wape": m.validation_wape,
                        "test_wape": m.test_wape,
                        "validation_mae": m.validation_mae,
                        "test_mae": m.test_mae,
                        "validation_rmse": m.validation_rmse,
                        "test_rmse": m.test_rmse,
                        "training_rows": m.training_rows,
                        "trained_at": m.trained_at,
                        "is_active": m.is_active,
                        "source": "company" if m.status == "ready" and m.is_active else "global_fallback",
                        "status_message": m.status_message,
                    }
                )
            else:
                models_list.append(
                    {
                        "horizon": h,
                        "status": "queued" if latest_job and latest_job.status in ("queued", "processing", "training", "evaluating") else "none",
                        "model_type": None,
                        "model_version": active_version or 1,
                        "validation_wape": None,
                        "test_wape": None,
                        "validation_mae": None,
                        "test_mae": None,
                        "validation_rmse": None,
                        "test_rmse": None,
                        "training_rows": None,
                        "trained_at": None,
                        "is_active": False,
                        "source": "global_fallback",
                        "status_message": "Company model not yet trained. Using global pre-trained model.",
                    }
                )

        return {
            "active_dataset_id": active_ds.id,
            "active_model_version": active_version,
            "training_job": job_dict,
            "models": models_list,
        }

    def resume_stranded_queued_jobs(self, engine: Any = None) -> int:
        """
        Scans for queued training jobs that were stranded by a server restart
        and resumes their execution in background daemon threads.
        Returns the count of resumed jobs.
        """
        db = SessionLocal() if engine is None else Session(bind=engine)
        resumed_count = 0
        try:
            queued_jobs = list(
                db.scalars(
                    select(ModelTrainingJob)
                    .where(ModelTrainingJob.status == "queued")
                    .order_by(ModelTrainingJob.created_at.asc())
                ).all()
            )
            for job in queued_jobs:
                logger.info(
                    "Resuming execution of stranded queued training job %s for user=%s, dataset=%s",
                    job.id,
                    job.user_id,
                    job.dataset_id,
                )
                training_lock_manager.acquire(job.user_id, job.dataset_id)
                t = threading.Thread(
                    target=self.execute_training_pipeline,
                    args=(job.user_id, job.dataset_id, job.id),
                    daemon=True,
                    name=f"training-worker-{job.id}",
                )
                t.start()
                resumed_count += 1
        except Exception as exc:
            logger.warning("Could not resume stranded queued training jobs: %s", exc)
        finally:
            db.close()
        return resumed_count


company_model_service = CompanyModelService()
