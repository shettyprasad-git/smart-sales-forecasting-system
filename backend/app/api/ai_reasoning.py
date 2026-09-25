from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.database import get_db
from backend.app.database.models import User
from backend.app.dependencies import get_current_user_optional
from backend.app.llm.provider import (
    LLMConfigurationError,
    LLMMalformedResponseError,
    LLMProviderError,
    LLMProviderUnavailableError,
    LLMResponseValidationError,
)
from backend.app.schemas.ai_reasoning import AIReasoningResponse
from backend.app.services.ai_reasoning_service import AIReasoningService
from backend.app.services.dataset_runtime_service import NoActiveDatasetError

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/ai-reasoning",
    tags=["AI Reasoning"],
)

ai_reasoning_service = AIReasoningService()


@router.get(
    "/{anomaly_id}",
    response_model=AIReasoningResponse,
    summary="Generate AI-grounded commercial reasoning for a sales anomaly",
    description=(
        "Synthesize empirical statistical evidence (from Phase 6.1-6.3) into structured "
        "executive reasoning using Gemini, covering competing hypotheses, uncertainties, "
        "and human validation questions without altering authoritative statistical metrics."
    ),
)
def get_anomaly_ai_reasoning(
    anomaly_id: str = Path(
        ...,
        description="Unique anomaly identifier (e.g. anom-20251228-sal-agg)",
    ),
    refresh: bool = Query(
        default=False,
        description="If True, bypasses the in-memory cache and generates fresh reasoning",
    ),
    current_user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    if settings.is_production and current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        user_id = current_user.id if current_user else None
        return ai_reasoning_service.reason_about_anomaly(
            anomaly_id=anomaly_id,
            refresh=refresh,
            user_id=user_id,
            db=db,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except NoActiveDatasetError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except LLMConfigurationError as exc:
        logger.warning("AI Reasoning unavailable due to configuration: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI reasoning provider is currently unavailable. Provider is not configured or credentials are invalid.",
        ) from exc
    except LLMProviderUnavailableError as exc:
        logger.warning("AI Reasoning provider unavailable / capacity: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI reasoning provider is currently unavailable or experiencing capacity limits. Please try again later.",
        ) from exc
    except (LLMMalformedResponseError, LLMResponseValidationError) as exc:
        logger.error("AI Reasoning malformed response: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI reasoning provider returned a malformed response.",
        ) from exc
    except LLMProviderError as exc:
        logger.error("AI Reasoning provider call failed for %s: %s", anomaly_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI reasoning provider encountered an error. Please try again later.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected failure during AI reasoning for %s: %s", anomaly_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI reasoning encountered an unexpected error. Internal server error.",
        ) from exc
