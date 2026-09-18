from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, Query, status

from backend.app.llm.provider import LLMConfigurationError, LLMProviderError
from backend.app.schemas.recommendations import RecommendationResponse
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
) -> RecommendationResponse:
    try:
        return recommendation_service.recommend_for_anomaly(
            anomaly_id=anomaly_id,
            refresh=refresh,
            fallback=fallback,
        )
    except LLMConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Recommendation AI provider is unconfigured: {exc}",
        ) from exc
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Recommendation AI provider error: {exc}",
        ) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Anomaly with ID '{anomaly_id}' not found.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error generating recommendations for %s", anomaly_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations. Internal server error.",
        ) from exc
