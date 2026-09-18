from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.dependencies import get_current_user
from backend.app.schemas.decisions import (
    DecisionActionRequest,
    DecisionCreateRequest,
    DecisionResubmitRequest,
    DecisionResponse,
    DecisionStatus,
)
from backend.app.services.decision_service import (
    DecisionConflictError,
    DecisionNotFoundError,
    DecisionPermissionError,
    DecisionService,
    DecisionValidationError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/decisions", tags=["Human Decision Governance"])

decision_service = DecisionService()


@router.post(
    "",
    response_model=DecisionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a decision review record",
    description="Submits a recommendation for human governance review. Captures empirical evidence and simulation snapshots.",
)
def create_decision(
    request: DecisionCreateRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> DecisionResponse:
    try:
        record = decision_service.create_decision(db, current_user.id, request)
        return DecisionResponse.model_validate(record)
    except Exception as exc:
        logger.exception("Unexpected error creating decision record: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create decision record: {exc}",
        ) from exc


@router.get(
    "",
    response_model=list[DecisionResponse],
    status_code=status.HTTP_200_OK,
    summary="List decision review records",
    description="Retrieves decisions owned by the authenticated user with optional status, anomaly, or recommendation type filters.",
)
def list_decisions(
    status_filter: DecisionStatus | None = Query(default=None, alias="status"),
    anomaly_id: str | None = Query(default=None),
    recommendation_type: str | None = Query(default=None),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> list[DecisionResponse]:
    records = decision_service.list_decisions(
        db=db,
        user_id=current_user.id,
        status=status_filter,
        anomaly_id=anomaly_id,
        recommendation_type=recommendation_type,
        skip=skip,
        limit=limit,
    )
    return [DecisionResponse.model_validate(r) for r in records]


@router.get(
    "/{decision_id}",
    response_model=DecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get a decision review record",
    description="Retrieves an existing decision record with immutable snapshots and audit trail.",
)
def get_decision(
    decision_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> DecisionResponse:
    try:
        record = decision_service.get_decision(db, decision_id, current_user.id)
        return DecisionResponse.model_validate(record)
    except DecisionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DecisionPermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


@router.post(
    "/{decision_id}/approve",
    response_model=DecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Approve recommendation",
    description="Approves the recommendation for manual human consideration. Does not automatically execute any business action.",
)
def approve_decision(
    decision_id: str,
    request: DecisionActionRequest = DecisionActionRequest(),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> DecisionResponse:
    try:
        record = decision_service.approve_decision(db, decision_id, current_user.id, request)
        return DecisionResponse.model_validate(record)
    except DecisionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DecisionPermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except DecisionConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post(
    "/{decision_id}/reject",
    response_model=DecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Reject recommendation",
    description="Rejects the recommendation. Requires a human decision rationale note.",
)
def reject_decision(
    decision_id: str,
    request: DecisionActionRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> DecisionResponse:
    try:
        record = decision_service.reject_decision(db, decision_id, current_user.id, request)
        return DecisionResponse.model_validate(record)
    except DecisionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DecisionPermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except DecisionConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except DecisionValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/{decision_id}/request-changes",
    response_model=DecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Request changes on recommendation",
    description="Requests revision or clarification on the recommendation. Requires an explanatory note.",
)
def request_changes(
    decision_id: str,
    request: DecisionActionRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> DecisionResponse:
    try:
        record = decision_service.request_changes(db, decision_id, current_user.id, request)
        return DecisionResponse.model_validate(record)
    except DecisionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DecisionPermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except DecisionConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except DecisionValidationError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post(
    "/{decision_id}/resubmit",
    response_model=DecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Resubmit recommendation for review",
    description="Resubmits a decision in 'changes_requested' status back to 'pending_review' with revised action text.",
)
def resubmit_decision(
    decision_id: str,
    request: DecisionResubmitRequest = DecisionResubmitRequest(),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> DecisionResponse:
    try:
        record = decision_service.resubmit_decision(db, decision_id, current_user.id, request)
        return DecisionResponse.model_validate(record)
    except DecisionNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except DecisionPermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except DecisionConflictError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
