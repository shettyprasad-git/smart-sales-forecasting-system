from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.database.models import User
from backend.app.dependencies import get_current_user
from backend.app.schemas.models import CurrentModelsResponse, ElasticityModelSummary
from backend.app.services.company_model_service import company_model_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/models",
    tags=["Models"],
)


@router.get(
    "/current",
    response_model=CurrentModelsResponse,
    summary="Get company forecasting model registry status and metrics for current user",
)
def get_current_models(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CurrentModelsResponse:
    """
    Returns the company-specific forecasting models (7D, 30D, 90D) for the currently active dataset,
    including validation WAPE, unbiased test WAPE, selected algorithm family, model version, and training status.
    """
    try:
        summary = company_model_service.get_current_models_summary(
            db=db,
            user_id=current_user.id,
        )
        return CurrentModelsResponse(**summary)
    except Exception as exc:
        logger.exception("Failed to retrieve current models for user %s: %s", current_user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve current models: {exc}",
        ) from exc


@router.get(
    "/elasticity",
    response_model=ElasticityModelSummary,
    summary="Get company demand sensitivity and elasticity model status for current user",
)
def get_elasticity_model(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ElasticityModelSummary:
    try:
        from sqlalchemy import select
        from backend.app.database.models import CompanyElasticityModel

        # Priority 1: Query the authenticated user's active READY elasticity model
        active_model = db.scalars(
            select(CompanyElasticityModel)
            .where(
                CompanyElasticityModel.user_id == current_user.id,
                CompanyElasticityModel.is_active == True,
                CompanyElasticityModel.status == "ready",
            )
            .order_by(CompanyElasticityModel.created_at.desc())
        ).first()

        if active_model:
            diag = active_model.diagnostics or {}
            logger.info(
                "elasticity_model_resolved: user_id=%s, model_id=%s, model_version=%s, status=ready",
                current_user.id,
                active_model.id,
                active_model.model_version,
            )
            return ElasticityModelSummary(
                status=active_model.status,
                model_type=active_model.model_type,
                model_version=active_model.model_version,
                dataset_id=active_model.dataset_id,
                price_elasticity=active_model.price_elasticity,
                discount_sensitivity=active_model.discount_sensitivity,
                r2_score=active_model.r2_score,
                mae=active_model.mae,
                rmse=active_model.rmse,
                training_rows=active_model.training_rows,
                price_supported=bool(diag.get("price_supported", active_model.price_elasticity is not None)),
                discount_supported=bool(diag.get("discount_supported", active_model.discount_sensitivity is not None)),
                price_reason=diag.get("price_reason"),
                discount_reason=diag.get("discount_reason"),
                status_message=active_model.status_message,
                is_active=active_model.is_active,
                trained_at=active_model.trained_at,
            )

        # Priority 2: Check current_models_summary (which catches in-flight training or inactive model on active dataset)
        summary = company_model_service.get_current_models_summary(
            db=db,
            user_id=current_user.id,
        )
        em = summary.get("elasticity_model")
        if em:
            return ElasticityModelSummary(**em)

        # Priority 3: Check if any elasticity model exists for user (e.g. failed/unavailable)
        latest_any = db.scalars(
            select(CompanyElasticityModel)
            .where(CompanyElasticityModel.user_id == current_user.id)
            .order_by(CompanyElasticityModel.created_at.desc())
        ).first()
        if latest_any:
            diag = latest_any.diagnostics or {}
            return ElasticityModelSummary(
                status=latest_any.status,
                model_type=latest_any.model_type,
                model_version=latest_any.model_version,
                dataset_id=latest_any.dataset_id,
                price_elasticity=latest_any.price_elasticity,
                discount_sensitivity=latest_any.discount_sensitivity,
                r2_score=latest_any.r2_score,
                mae=latest_any.mae,
                rmse=latest_any.rmse,
                training_rows=latest_any.training_rows,
                price_supported=bool(diag.get("price_supported", latest_any.price_elasticity is not None)),
                discount_supported=bool(diag.get("discount_supported", latest_any.discount_sensitivity is not None)),
                price_reason=diag.get("price_reason"),
                discount_reason=diag.get("discount_reason"),
                status_message=latest_any.status_message,
                is_active=latest_any.is_active,
                trained_at=latest_any.trained_at,
            )

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active elasticity model found for this tenant.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to retrieve elasticity model for user %s: %s", current_user.id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve elasticity model: {exc}",
        ) from exc
