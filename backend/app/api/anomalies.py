import datetime as dt
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, status

from backend.app.schemas.anomalies import (
    AnomalyItem,
    AnomalyListResponse,
    AnomalySummary,
)
from backend.app.services.anomaly_service import AnomalyDetectionService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/anomalies",
    tags=["Anomalies"],
)

anomaly_service = AnomalyDetectionService()


@router.get(
    "",
    response_model=AnomalyListResponse,
    summary="List sales anomalies",
    description="Retrieve sales and demand anomalies with multi-dimensional filtering, pagination, and summary statistics.",
)
def list_anomalies(
    metric: Literal["quantity", "sales_amount"] | None = Query(
        default=None,
        description="Filter by metric: quantity or sales_amount",
    ),
    severity: Literal["low", "medium", "high", "critical"] | None = Query(
        default=None,
        description="Filter by exact severity level",
    ),
    min_severity: Literal["low", "medium", "high", "critical"] | None = Query(
        default=None,
        description="Filter by minimum severity level (e.g. 'high' returns high and critical)",
    ),
    direction: Literal["spike", "drop"] | None = Query(
        default=None,
        description="Filter by direction: spike (above baseline) or drop (below baseline)",
    ),
    entity_type: Literal["aggregate", "product", "category"] | None = Query(
        default="aggregate",
        description="Granularity level: aggregate (total sales), category, or product",
    ),
    entity_id: str | None = Query(
        default=None,
        description="Filter by entity ID (e.g. Category ID or Product ID)",
    ),
    start_date: dt.date | None = Query(
        default=None,
        description="Filter observations on or after this date (YYYY-MM-DD)",
    ),
    end_date: dt.date | None = Query(
        default=None,
        description="Filter observations on or before this date (YYYY-MM-DD)",
    ),
    skip: int = Query(default=0, ge=0, description="Pagination skip offset"),
    limit: int = Query(default=50, ge=1, le=500, description="Pagination limit"),
    window: int = Query(
        default=28,
        ge=7,
        le=90,
        description="Rolling window size in days for historical baseline calculation",
    ),
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date cannot be later than end_date.",
        )

    try:
        return anomaly_service.get_anomalies(
            metric=metric,
            severity=severity,
            min_severity=min_severity,
            direction=direction,
            entity_type=entity_type,
            entity_id=entity_id,
            start_date=start_date,
            end_date=end_date,
            skip=skip,
            limit=limit,
            window=window,
        )
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
        logger.exception("Failed to retrieve anomalies: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {exc}",
        ) from exc


@router.get(
    "/summary",
    response_model=AnomalySummary,
    summary="Get anomaly summary statistics",
    description="Retrieve aggregated anomaly KPIs, severity counts, and direction breakdown.",
)
def get_anomaly_summary(
    metric: Literal["quantity", "sales_amount"] | None = Query(default=None),
    severity: Literal["low", "medium", "high", "critical"] | None = Query(default=None),
    direction: Literal["spike", "drop"] | None = Query(default=None),
    entity_type: Literal["aggregate", "product", "category"] | None = Query(default="aggregate"),
    start_date: dt.date | None = Query(default=None),
    end_date: dt.date | None = Query(default=None),
    window: int = Query(default=28, ge=7, le=90),
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_date cannot be later than end_date.",
        )

    try:
        response = anomaly_service.get_anomalies(
            metric=metric,
            severity=severity,
            direction=direction,
            entity_type=entity_type,
            start_date=start_date,
            end_date=end_date,
            skip=0,
            limit=1,
            window=window,
        )
        return response.summary
    except Exception as exc:
        logger.exception("Failed to compute anomaly summary: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate anomaly summary: {exc}",
        ) from exc


@router.get(
    "/recent",
    response_model=list[AnomalyItem],
    summary="Get recent anomalies",
    description="Quickly fetch the most recent N detected anomalies.",
)
def get_recent_anomalies(
    limit: int = Query(default=10, ge=1, le=100),
    metric: Literal["quantity", "sales_amount"] | None = Query(default=None),
    severity: Literal["low", "medium", "high", "critical"] | None = Query(default=None),
):
    try:
        response = anomaly_service.get_anomalies(
            metric=metric,
            severity=severity,
            entity_type="aggregate",
            skip=0,
            limit=limit,
        )
        return response.items
    except Exception as exc:
        logger.exception("Failed to retrieve recent anomalies: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch recent anomalies: {exc}",
        ) from exc


@router.get(
    "/products/{product_id}",
    response_model=AnomalyListResponse,
    summary="Get product-level anomalies",
    description="Retrieve detected anomalies for a specific product ID.",
)
def get_product_anomalies(
    product_id: int,
    metric: Literal["quantity", "sales_amount"] | None = Query(default=None),
    severity: Literal["low", "medium", "high", "critical"] | None = Query(default=None),
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
):
    try:
        return anomaly_service.get_anomalies(
            metric=metric,
            severity=severity,
            entity_type="product",
            entity_id=str(product_id),
            skip=skip,
            limit=limit,
        )
    except Exception as exc:
        logger.exception("Failed to retrieve product anomalies: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch product anomalies: {exc}",
        ) from exc
