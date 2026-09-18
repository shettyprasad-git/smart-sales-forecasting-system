from __future__ import annotations

import datetime as dt
from typing import Literal

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.database.models import User
from backend.app.dependencies import get_current_user
from backend.app.schemas.monitoring import (
    AlertDetailResponse,
    AlertListResponse,
    AlertResponse,
    MonitoringRunItemResponse,
    MonitoringRunRequest,
    MonitoringRunResponse,
    MonitoringSummary,
)
from backend.app.services.monitoring_service import MonitoringService

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])
monitoring_service = MonitoringService()


@router.post(
    "/run",
    response_model=MonitoringRunResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute sales data monitoring scan",
)
def run_monitoring(
    request: MonitoringRunRequest | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MonitoringRunResponse:
    """
    Run proactive sales monitoring scan across available data.
    Evaluates newly significant anomalies, detects repeated patterns,
    suppresses duplicates, and creates prioritized alerts.
    Does NOT mutate sales records.
    """
    req = request or MonitoringRunRequest()
    return monitoring_service.run_monitoring(
        db=db,
        user_id=current_user.id,
        request=req,
    )


@router.get(
    "/alerts",
    response_model=AlertListResponse,
    status_code=status.HTTP_200_OK,
    summary="List monitoring alerts",
)
def list_alerts(
    status_filter: str | None = Query(None, alias="status", description="Filter by alert status ('new', 'acknowledged', 'resolved', 'dismissed')"),
    severity: str | None = Query(None, description="Filter by severity ('low', 'medium', 'high', 'critical')"),
    alert_type: str | None = Query(None, description="Filter by alert type"),
    start_date: dt.date | None = Query(None, description="Filter events on or after this date"),
    end_date: dt.date | None = Query(None, description="Filter events on or before this date"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AlertListResponse:
    """Retrieve filtered, paginated monitoring alerts scoped to the current user."""
    return monitoring_service.list_alerts(
        db=db,
        user_id=current_user.id,
        status_filter=status_filter,
        severity=severity,
        alert_type=alert_type,
        start_date=start_date,
        end_date=end_date,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/alerts/{alert_id}",
    response_model=AlertDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get single alert detail",
)
def get_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AlertDetailResponse:
    """Retrieve detailed alert context with investigation, simulation, and governance linkages."""
    return monitoring_service.get_alert(
        db=db,
        alert_id=alert_id,
        user_id=current_user.id,
    )


@router.post(
    "/alerts/{alert_id}/acknowledge",
    response_model=AlertResponse,
    status_code=status.HTTP_200_OK,
    summary="Acknowledge alert",
)
def acknowledge_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AlertResponse:
    """Mark an alert as acknowledged by human reviewer."""
    return monitoring_service.acknowledge_alert(
        db=db,
        alert_id=alert_id,
        user_id=current_user.id,
    )


@router.post(
    "/alerts/{alert_id}/resolve",
    response_model=AlertResponse,
    status_code=status.HTTP_200_OK,
    summary="Resolve alert",
)
def resolve_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AlertResponse:
    """Mark an alert as resolved by human reviewer."""
    return monitoring_service.resolve_alert(
        db=db,
        alert_id=alert_id,
        user_id=current_user.id,
    )


@router.post(
    "/alerts/{alert_id}/dismiss",
    response_model=AlertResponse,
    status_code=status.HTTP_200_OK,
    summary="Dismiss alert",
)
def dismiss_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AlertResponse:
    """Mark an alert as dismissed without action."""
    return monitoring_service.dismiss_alert(
        db=db,
        alert_id=alert_id,
        user_id=current_user.id,
    )


@router.get(
    "/summary",
    response_model=MonitoringSummary,
    status_code=status.HTTP_200_OK,
    summary="Get monitoring summary KPIs",
)
def get_monitoring_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MonitoringSummary:
    """Retrieve executive telemetry KPIs for proactive alerts."""
    return monitoring_service.get_monitoring_summary(
        db=db,
        user_id=current_user.id,
    )


@router.get(
    "/runs",
    response_model=list[MonitoringRunItemResponse],
    status_code=status.HTTP_200_OK,
    summary="List recent monitoring runs",
)
def list_monitoring_runs(
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[MonitoringRunItemResponse]:
    """Retrieve recent monitoring execution runs for observability."""
    return monitoring_service.list_runs(
        db=db,
        user_id=current_user.id,
        limit=limit,
    )
