from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Path, Query, status

from backend.app.schemas.explanations import ExecutiveExplanation
from backend.app.services.explanation_service import ExplanationService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/explanations",
    tags=["Explanations"],
)

explanation_service = ExplanationService()


@router.get(
    "/{anomaly_id}",
    response_model=ExecutiveExplanation,
    summary="Generate executive narrative explanation for a sales anomaly",
    description=(
        "Synthesize structured root-cause attribution evidence into deterministic, "
        "executive-ready narrative reporting without using non-deterministic models or LLMs."
    ),
)
def get_anomaly_explanation(
    anomaly_id: str = Path(
        ...,
        description="Unique anomaly identifier (e.g. anom-20251228-sal-agg)",
    ),
    top_n: int = Query(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of primary driver factors to highlight in executive narrative",
    ),
    format: Literal["json"] = Query(
        default="json",
        description="Response format (currently supports 'json')",
    ),
):
    try:
        return explanation_service.explain_anomaly(
            anomaly_id=anomaly_id,
            top_n=top_n,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Explanation generation failed for anomaly %s: %s", anomaly_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation generation failed: {exc}",
        ) from exc
