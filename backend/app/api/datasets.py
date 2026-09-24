from __future__ import annotations

import logging
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Path as FastPath,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.database import get_db
from backend.app.database.models import DatasetUpload, User
from backend.app.dependencies import get_current_user
from backend.app.schemas.datasets import (
    DatasetActionResponse,
    DatasetHistoryItem,
    DatasetHistoryResponse,
    DatasetSummaryResponse,
)
from backend.app.schemas.models import TrainTriggerResponse
from backend.app.services.company_model_service import (
    TrainingConcurrencyError,
    company_model_service,
    training_lock_manager,
)
from backend.app.services.dataset_runtime_service import (
    DatasetConflictError,
    DatasetValidationError,
    dataset_runtime_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/datasets",
    tags=["Datasets"],
)


@router.post(
    "/upload",
    response_model=DatasetSummaryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and transactionally import a sales dataset for runtime model inference",
)
async def upload_dataset(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DatasetSummaryResponse:
    """
    Accepts an uploaded CSV file, validates columns and data rows deterministically,
    stores normalized sales records in PostgreSQL, activates the dataset for the current user,
    and enqueues asynchronous company model benchmarking and training.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename.",
        )

    ext = Path(file.filename).suffix.lower()
    if ext != ".csv":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Only standard CSV files (.csv) are accepted.",
        )

    # Streamed read with max size check
    max_bytes = settings.max_dataset_upload_bytes
    chunks: list[bytes] = []
    total_bytes = 0

    chunk_size = 1024 * 1024  # 1MB
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded file exceeds maximum allowed size of {max_bytes // (1024 * 1024)} MB.",
            )
        chunks.append(chunk)

    file_content = b"".join(chunks)

    if not file_content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    try:
        summary = dataset_runtime_service.process_and_import_csv(
            db=db,
            user_id=current_user.id,
            original_filename=file.filename,
            file_content=file_content,
        )

        # Enqueue background company model training
        try:
            job, _ = company_model_service.enqueue_training_job(
                db=db,
                user_id=current_user.id,
                dataset_id=summary.dataset_id,
            )
            background_tasks.add_task(
                company_model_service.execute_training_pipeline,
                user_id=current_user.id,
                dataset_id=summary.dataset_id,
                job_id=job.id,
            )
        except Exception as exc:
            logger.warning("Could not auto-enqueue model training on upload: %s", exc)

        return summary
    except DatasetValidationError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": exc.message,
                "errors": exc.errors[:20],
            },
        ) from exc
    except DatasetConflictError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        db.rollback()
        logger.exception("Unexpected error during dataset upload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dataset processing failed: {exc}",
        ) from exc


@router.get(
    "/current",
    response_model=DatasetSummaryResponse,
    summary="Get metadata for the current user's active dataset",
)
def get_current_dataset(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DatasetSummaryResponse:
    """Return summary metadata for the currently active dataset belonging to the authenticated user."""
    active_ds = dataset_runtime_service.get_active_dataset(db, current_user.id)
    if not active_ds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active dataset found for this account.",
        )

    return DatasetSummaryResponse(
        dataset_id=active_ds.id,
        filename=active_ds.original_filename,
        rows_imported=active_ds.row_count,
        products=active_ds.product_count,
        categories=active_ds.category_count,
        start_date=active_ds.min_date.isoformat() if active_ds.min_date else None,
        end_date=active_ds.max_date.isoformat() if active_ds.max_date else None,
        status=active_ds.status,
        validation_warnings=[],
        validation_errors=[],
    )


@router.get(
    "/history",
    response_model=DatasetHistoryResponse,
    summary="List all datasets uploaded by the current user",
)
def get_dataset_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DatasetHistoryResponse:
    """Return historical dataset uploads for the current user."""
    stmt = (
        select(DatasetUpload)
        .where(DatasetUpload.user_id == current_user.id)
        .order_by(DatasetUpload.created_at.desc())
    )
    records = list(db.scalars(stmt).all())

    items = [
        DatasetHistoryItem(
            id=r.id,
            original_filename=r.original_filename,
            row_count=r.row_count,
            product_count=r.product_count,
            category_count=r.category_count,
            min_date=r.min_date.isoformat() if r.min_date else None,
            max_date=r.max_date.isoformat() if r.max_date else None,
            status=r.status,
            created_at=r.created_at,
            activated_at=r.activated_at,
        )
        for r in records
    ]

    return DatasetHistoryResponse(
        datasets=items,
        total=len(items),
    )


@router.post(
    "/{dataset_id}/activate",
    response_model=DatasetSummaryResponse,
    summary="Activate an uploaded dataset for the current user",
)
def activate_dataset_endpoint(
    dataset_id: str = FastPath(..., description="Unique dataset identifier"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DatasetSummaryResponse:
    """
    Transactionally activate a dataset belonging to the current user.
    Archives previously active datasets.
    """
    try:
        summary = dataset_runtime_service.activate_dataset(
            db=db,
            user_id=current_user.id,
            dataset_id=dataset_id,
        )
        # Synchronize active company model pointers for this user
        try:
            company_model_service.activate_dataset_models(
                db=db,
                user_id=current_user.id,
                dataset_id=dataset_id,
            )
        except Exception as exc:
            logger.warning("Could not sync active model pointers on dataset activation: %s", exc)

        return summary
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except DatasetConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.post(
    "/{dataset_id}/train",
    response_model=TrainTriggerResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger asynchronous company model training and benchmarking for a dataset",
)
def train_dataset_models_endpoint(
    dataset_id: str = FastPath(..., description="Unique dataset identifier"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> TrainTriggerResponse:
    """
    Manually trigger company forecasting model benchmarking across 7D, 30D, and 90D horizons.
    Quarantines test data from model selection and persists winning architectures to persistent storage.
    """
    # Verify dataset belongs to current user
    ds = db.scalars(
        select(DatasetUpload).where(
            DatasetUpload.id == dataset_id,
            DatasetUpload.user_id == current_user.id,
        )
    ).first()
    if not ds:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_id}' not found for current user.",
        )

    if training_lock_manager.is_locked(current_user.id, dataset_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Model training is already in progress for this dataset.",
        )

    try:
        job, acquired = company_model_service.enqueue_training_job(
            db=db,
            user_id=current_user.id,
            dataset_id=dataset_id,
        )
        if not acquired:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Model training is already in progress for this dataset.",
            )
    except TrainingConcurrencyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    background_tasks.add_task(
        company_model_service.execute_training_pipeline,
        user_id=current_user.id,
        dataset_id=dataset_id,
        job_id=job.id,
    )

    return TrainTriggerResponse(
        dataset_id=dataset_id,
        job_id=job.id,
        status=job.status,
        message="Company model benchmarking and training queued for horizons [7, 30, 90].",
    )


@router.delete(
    "/{dataset_id}",
    response_model=DatasetActionResponse,
    summary="Delete a dataset and its sales records",
)
def delete_dataset_endpoint(
    dataset_id: str = FastPath(..., description="Unique dataset identifier"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DatasetActionResponse:
    """
    Delete an uploaded dataset belonging to the current user.
    Cascades delete to associated sales records while preserving shared products.
    """
    try:
        dataset_runtime_service.delete_dataset(
            db=db,
            user_id=current_user.id,
            dataset_id=dataset_id,
        )
        return DatasetActionResponse(
            message="Dataset deleted successfully.",
            dataset_id=dataset_id,
            status="deleted",
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
