from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status
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
from backend.app.schemas.recommendations import RecommendationResponse
from backend.app.services.dataset_runtime_service import NoActiveDatasetError
from backend.app.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recommendations", tags=["Prescriptive Recommendations"])

# Module-level singleton
recommendation_service = RecommendationService()


@router.get(
    "/{anomaly_id}",
    response_model=RecommendationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get prescriptive action recommendations for an anomaly",
    description=(
        "Synthesizes structured, non-autonomous prescriptive recommendations for a sales anomaly. "
        "Evaluates deterministic eligibility rules across 8 categories, leverages Gemini for executive phrasing "
        "and trade-offs, and gracefully degrades to deterministic policy fallback if AI is unavailable. "
        "Human approval is mandatory for all recommendations."
    ),
)
def get_recommendations(
    anomaly_id: str,
    refresh: bool = Query(
        default=False,
        description="Bypass in-memory cache and re-evaluate recommendation synthesis",
    ),
    fallback: bool = Query(
        default=True,
        description="Fallback to deterministic rule-based recommendations if AI inference fails",
    ),
    current_user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
) -> RecommendationResponse:
    if settings.is_production and current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        user_id = current_user.id if current_user else None
        return recommendation_service.recommend_for_anomaly(
            anomaly_id=anomaly_id,
            refresh=refresh,
            fallback=fallback,
            user_id=user_id,
            db=db,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Anomaly with ID '{anomaly_id}' not found.",
        ) from exc
    except NoActiveDatasetError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation AI provider is currently unavailable. Provider is unconfigured or credentials are invalid.",
        ) from exc
    except LLMProviderUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation AI provider is currently unavailable or experiencing capacity limits. Please try again later.",
        ) from exc
    except (LLMMalformedResponseError, LLMResponseValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Recommendation AI provider returned a malformed response.",
        ) from exc
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation AI provider encountered an error. Please try again later.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error generating recommendations for %s", anomaly_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations. Internal server error.",
        ) from exc
