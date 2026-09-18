from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Path, Query, status

from backend.app.llm.provider import (
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseValidationError,
)
from backend.app.schemas.ai_reasoning import AIReasoningResponse
from backend.app.services.ai_reasoning_service import AIReasoningService

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
):
    try:
        return ai_reasoning_service.reason_about_anomaly(
            anomaly_id=anomaly_id,
            refresh=refresh,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except LLMConfigurationError as exc:
        logger.warning("AI Reasoning unavailable due to configuration: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "AI reasoning is currently unavailable. Statistical investigation "
                "and executive reports remain fully functional. Reason: " + str(exc)
            ),
        ) from exc
    except (LLMProviderError, LLMResponseValidationError) as exc:
        logger.error("AI Reasoning provider call failed for %s: %s", anomaly_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI reasoning provider encountered an error: {exc}",
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
