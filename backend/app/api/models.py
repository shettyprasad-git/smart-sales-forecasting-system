from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.database.models import User
from backend.app.dependencies import get_current_user
from backend.app.schemas.models import CurrentModelsResponse
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
