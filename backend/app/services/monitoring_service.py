from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import Any

from fastapi import HTTPException, status
import pandas as pd
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from backend.app.database.models import DecisionRecord, MonitoringAlert, MonitoringRun
from backend.app.schemas.anomalies import AnomalyItem
from backend.app.schemas.monitoring import (
    AlertDetailResponse,
    AlertListResponse,
    AlertPriority,
    AlertResponse,
    AlertSeverity,
    AlertStatus,
    AlertType,
    MonitoringRunItemResponse,
    MonitoringRunRequest,
    MonitoringRunResponse,
    MonitoringSummary,
)
from backend.app.services.anomaly_service import (
    SEVERITY_RANKS,
    AnomalyDetectionService,
)
from backend.app.services.investigation_service import InvestigationService
from backend.app.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)


class MonitoringService:
    """
    Proactive Intelligence & Monitoring Service for evaluating sales data,
    detecting newly significant conditions, suppressing duplicates, and managing
    the human-governed alert lifecycle.
    """

    def __init__(
        self,
        anomaly_service: AnomalyDetectionService | None = None,
        investigation_service: InvestigationService | None = None,
        recommendation_service: RecommendationService | None = None,
    ) -> None:
        self.anomaly_service = anomaly_service or AnomalyDetectionService()
        self.investigation_service = (
            investigation_service
            or InvestigationService(anomaly_service=self.anomaly_service)
        )
        self.recommendation_service = (
            recommendation_service
            or RecommendationService(
                investigation_service=self.investigation_service
            )
        )
        self._cached_category_anomalies: list[AnomalyItem] | None = None

    def _determine_priority(self, severity: str) -> str:
        """Map statistical severity to operational priority deterministically."""
        if severity == "critical":
            return AlertPriority.URGENT.value
        elif severity == "high":
            return AlertPriority.HIGH.value
        elif severity == "medium":
            return AlertPriority.MEDIUM.value
        else:
            return AlertPriority.LOW.value

    def _generate_fingerprint(
        self,
        event_date: dt.date,
        metric: str,
        entity_type: str,
        entity_id: str | None,
        alert_type: str,
    ) -> str:
        """Compute deterministic alert fingerprint to prevent duplicates."""
        ent = entity_id or "all"
        return f"{event_date.strftime('%Y%m%d')}_{metric}_{entity_type}_{ent}_{alert_type}"

    def run_monitoring(
        self,
        db: Session,
        user_id: int,
        request: MonitoringRunRequest,
    ) -> MonitoringRunResponse:
        """
        Execute deterministic sales monitoring scan across available data.
        Does NOT mutate sales or product data.
        """
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        now = dt.datetime.now(dt.timezone.utc)
        run_record = MonitoringRun(
            id=run_id,
            user_id=user_id,
            started_at=now,
            status="running",
            total_records_evaluated=0,
            anomalies_found=0,
            alerts_created=0,
            duplicates_suppressed=0,
        )
        db.add(run_record)
        db.commit()

        try:
            # 1. Determine date range for scan
            daily_df = self.anomaly_service._load_daily_data()
            if daily_df.empty:
                max_date = dt.date.today()
            else:
                max_date = daily_df["Date"].max().date()

            start_date = max_date - dt.timedelta(days=request.lookback_days)

            # 2. Collect anomalies across requested entities
            anomalies: list[AnomalyItem] = []

            # Aggregate anomalies
            agg_resp = self.anomaly_service.get_anomalies(
                start_date=start_date,
                end_date=max_date,
                entity_type="aggregate",
                limit=500,
            )
            anomalies.extend(agg_resp.items)
            eval_records = len(daily_df[daily_df["Date"].dt.date >= start_date]) * 2

            # Category anomalies
            if request.include_categories:
                try:
                    if self._cached_category_anomalies is None:
                        self._cached_category_anomalies = self.anomaly_service.detect_category_anomalies()
                    cat_filtered = [
                        a
                        for a in self._cached_category_anomalies
                        if start_date <= a.date <= max_date
                    ]
                    anomalies.extend(cat_filtered)
                    eval_records += len(cat_filtered) * 5
                except Exception as e:
                    logger.warning("Failed to scan category anomalies: %s", e)

            # Product anomalies (optional)
            if request.include_products:
                pass  # product scanning is on-demand to keep monitoring runs lightweight

            # 3. Frequency analysis for repeated anomaly detection
            # Key: (entity_type, entity_id, metric)
            frequency_map: dict[tuple[str, str | None, str], list[AnomalyItem]] = {}
            for anom in anomalies:
                key = (anom.entity_type, anom.entity_id, anom.metric)
                frequency_map.setdefault(key, []).append(anom)

            alerts_created = 0
            duplicates_suppressed = 0
            new_alerts_list: list[MonitoringAlert] = []

            min_rank = SEVERITY_RANKS.get(request.min_severity, 2)

            # 4. Evaluate each anomaly against materiality & suppression
            for anom in anomalies:
                anom_rank = SEVERITY_RANKS.get(anom.severity, 1)
                group_events = frequency_map.get(
                    (anom.entity_type, anom.entity_id, anom.metric), []
                )
                is_repeated = len(group_events) >= 2

                # Materiality rule:
                # low: skipped unless repeated
                # medium: eligible if requested min_severity allows or dev >= 15%
                # high & critical: always material
                if anom.severity == "low" and not is_repeated and min_rank > 1:
                    continue

                if anom_rank < min_rank and not is_repeated:
                    continue

                # Determine Alert Type
                if is_repeated and anom.severity in ("low", "medium"):
                    alert_type = AlertType.REPEATED_ANOMALY.value
                    priority = AlertPriority.HIGH.value
                    title = f"Repeated Anomaly ({len(group_events)} events): {anom.entity_name or anom.entity_type} {anom.metric.replace('_', ' ')}"
                elif anom.entity_type == "category":
                    alert_type = AlertType.CATEGORY_DEVIATION.value
                    priority = self._determine_priority(anom.severity)
                    title = f"Category Deviation: {anom.entity_name} ({anom.deviation_percent:+.1f}%)"
                elif anom.entity_type == "product":
                    alert_type = AlertType.PRODUCT_DEVIATION.value
                    priority = self._determine_priority(anom.severity)
                    title = f"Product Deviation: {anom.entity_name} ({anom.deviation_percent:+.1f}%)"
                elif anom.direction == "spike":
                    alert_type = AlertType.SALES_SPIKE.value
                    priority = self._determine_priority(anom.severity)
                    title = f"{anom.severity.capitalize()} Sales Spike: {anom.deviation_percent:+.1f}% vs baseline on {anom.entity_name or 'System Sales'}"
                else:
                    alert_type = AlertType.SALES_DROP.value
                    priority = self._determine_priority(anom.severity)
                    title = f"{anom.severity.capitalize()} Sales Drop: {anom.deviation_percent:+.1f}% vs baseline on {anom.entity_name or 'System Sales'}"

                # Deterministic fingerprint
                fingerprint = self._generate_fingerprint(
                    event_date=anom.date,
                    metric=anom.metric,
                    entity_type=anom.entity_type,
                    entity_id=anom.entity_id,
                    alert_type=alert_type,
                )

                # Check duplicate suppression in DB
                existing_unresolved = (
                    db.query(MonitoringAlert)
                    .filter(
                        MonitoringAlert.user_id == user_id,
                        (
                            (MonitoringAlert.fingerprint == fingerprint)
                            | (MonitoringAlert.anomaly_id == anom.id)
                        ),
                        MonitoringAlert.status.in_(["new", "acknowledged"]),
                    )
                    .first()
                )

                if existing_unresolved is not None:
                    duplicates_suppressed += 1
                    continue

                # Build compact evidence snapshot
                evidence_snapshot: dict[str, Any] = {
                    "anomaly_id": anom.id,
                    "date": anom.date.isoformat(),
                    "metric": anom.metric,
                    "entity_type": anom.entity_type,
                    "entity_id": anom.entity_id,
                    "entity_name": anom.entity_name,
                    "actual_value": anom.actual_value,
                    "expected_value": anom.expected_value,
                    "deviation": anom.deviation,
                    "deviation_percent": anom.deviation_percent,
                    "anomaly_score": anom.anomaly_score,
                    "severity": anom.severity,
                    "explanation": anom.explanation,
                }

                # Optionally attach lightweight investigation summary if available
                try:
                    inv = self.investigation_service.investigate_anomaly(anom.id)
                    evidence_snapshot["top_drivers"] = [
                        {
                            "driver_type": d.driver_type,
                            "driver_name": d.driver_name,
                            "difference_percent": d.difference_percent,
                            "contribution_score": d.contribution_score,
                            "confidence": d.confidence,
                        }
                        for d in inv.drivers[:3]
                    ]
                    evidence_snapshot["estimated_impact"] = {
                        "impact_value": inv.estimated_impact.impact_value,
                        "estimated_revenue_impact": inv.estimated_impact.estimated_revenue_impact,
                        "interpretation": inv.estimated_impact.interpretation,
                    }
                except Exception:
                    evidence_snapshot["top_drivers"] = []

                alert_id = f"alert-{uuid.uuid4().hex[:12]}"
                new_alert = MonitoringAlert(
                    id=alert_id,
                    user_id=user_id,
                    anomaly_id=anom.id,
                    alert_type=alert_type,
                    title=title,
                    metric=anom.metric,
                    severity=anom.severity,
                    priority=priority,
                    status=AlertStatus.NEW.value,
                    event_date=anom.date,
                    actual_value=anom.actual_value,
                    expected_value=anom.expected_value,
                    deviation=anom.deviation,
                    deviation_percent=anom.deviation_percent,
                    fingerprint=fingerprint,
                    evidence_snapshot=evidence_snapshot,
                    created_at=dt.datetime.now(dt.timezone.utc),
                )
                db.add(new_alert)
                new_alerts_list.append(new_alert)
                alerts_created += 1

            # Update run record
            run_record.completed_at = dt.datetime.now(dt.timezone.utc)
            run_record.total_records_evaluated = eval_records
            run_record.anomalies_found = len(anomalies)
            run_record.alerts_created = alerts_created
            run_record.duplicates_suppressed = duplicates_suppressed
            run_record.status = "completed"

            db.commit()

            summary = self.get_monitoring_summary(db, user_id)

            return MonitoringRunResponse(
                run_id=run_record.id,
                status=run_record.status,
                started_at=run_record.started_at,
                completed_at=run_record.completed_at,
                total_records_evaluated=run_record.total_records_evaluated,
                anomalies_found=run_record.anomalies_found,
                alerts_created=run_record.alerts_created,
                duplicates_suppressed=run_record.duplicates_suppressed,
                summary=summary,
                human_review_required=True,
                automatic_execution=False,
            )

        except Exception as e:
            logger.exception("Monitoring run failed: %s", e)
            db.rollback()
            # Update run record to failed
            failed_run = db.query(MonitoringRun).filter(MonitoringRun.id == run_id).first()
            if failed_run:
                failed_run.status = "failed"
                failed_run.error_message = str(e)[:1000]
                failed_run.completed_at = dt.datetime.now(dt.timezone.utc)
                db.commit()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Monitoring scan failed. Internal server error.",
            )

    def list_alerts(
        self,
        db: Session,
        user_id: int,
        status_filter: str | None = None,
        severity: str | None = None,
        alert_type: str | None = None,
        start_date: dt.date | None = None,
        end_date: dt.date | None = None,
        skip: int = 0,
        limit: int = 50,
    ) -> AlertListResponse:
        """Query alerts scoped to the authenticated user with filters and pagination."""
        query = db.query(MonitoringAlert).filter(MonitoringAlert.user_id == user_id)

        if status_filter:
            query = query.filter(MonitoringAlert.status == status_filter)
        if severity:
            query = query.filter(MonitoringAlert.severity == severity)
        if alert_type:
            query = query.filter(MonitoringAlert.alert_type == alert_type)
        if start_date:
            query = query.filter(MonitoringAlert.event_date >= start_date)
        if end_date:
            query = query.filter(MonitoringAlert.event_date <= end_date)

        total = query.count()
        items = (
            query.order_by(
                desc(MonitoringAlert.event_date),
                desc(MonitoringAlert.created_at),
            )
            .offset(skip)
            .limit(limit)
            .all()
        )

        return AlertListResponse(
            items=[AlertResponse.model_validate(it) for it in items],
            total=total,
            skip=skip,
            limit=limit,
        )

    def get_alert(
        self,
        db: Session,
        alert_id: str,
        user_id: int,
    ) -> AlertDetailResponse:
        """Retrieve detailed alert context with investigation, simulation, and decision linkages."""
        alert = db.query(MonitoringAlert).filter(MonitoringAlert.id == alert_id).first()
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {alert_id}",
            )

        if alert.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this alert",
            )

        resp = AlertDetailResponse.model_validate(alert)

        # Populate explanation & drivers from snapshot
        snap = alert.evidence_snapshot or {}
        resp.explanation = snap.get("explanation")
        resp.top_drivers = snap.get("top_drivers", [])

        # Check recommendation availability
        try:
            recs_resp = self.recommendation_service.get_recommendations(alert.anomaly_id)
            if recs_resp and recs_resp.recommendations:
                resp.has_recommendation = True
                resp.recommendation_type = recs_resp.recommendations[0].recommendation_type
        except Exception:
            resp.has_recommendation = False

        # Simulation availability
        resp.simulation_available = True
        resp.recommended_scenario = (
            "demand_multiplier" if alert.deviation > 0 else "temporary_shock"
        )

        # Check existing decision record
        decision = (
            db.query(DecisionRecord)
            .filter(
                DecisionRecord.user_id == user_id,
                DecisionRecord.anomaly_id == alert.anomaly_id,
            )
            .first()
        )
        if decision:
            resp.decision_id = decision.id
            resp.decision_status = decision.status

        return resp

    def acknowledge_alert(
        self,
        db: Session,
        alert_id: str,
        user_id: int,
    ) -> AlertResponse:
        """Transition alert to acknowledged status."""
        alert = db.query(MonitoringAlert).filter(MonitoringAlert.id == alert_id).first()
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {alert_id}",
            )
        if alert.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to modify this alert",
            )

        if alert.status in (AlertStatus.RESOLVED.value, AlertStatus.DISMISSED.value):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot acknowledge alert in '{alert.status}' state",
            )

        alert.status = AlertStatus.ACKNOWLEDGED.value
        alert.acknowledged_at = dt.datetime.now(dt.timezone.utc)
        db.commit()
        db.refresh(alert)
        return AlertResponse.model_validate(alert)

    def resolve_alert(
        self,
        db: Session,
        alert_id: str,
        user_id: int,
    ) -> AlertResponse:
        """Transition alert to resolved status."""
        alert = db.query(MonitoringAlert).filter(MonitoringAlert.id == alert_id).first()
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {alert_id}",
            )
        if alert.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to modify this alert",
            )

        if alert.status in (AlertStatus.RESOLVED.value, AlertStatus.DISMISSED.value):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot resolve alert in '{alert.status}' state",
            )

        alert.status = AlertStatus.RESOLVED.value
        alert.resolved_at = dt.datetime.now(dt.timezone.utc)
        db.commit()
        db.refresh(alert)
        return AlertResponse.model_validate(alert)

    def dismiss_alert(
        self,
        db: Session,
        alert_id: str,
        user_id: int,
    ) -> AlertResponse:
        """Transition alert to dismissed status."""
        alert = db.query(MonitoringAlert).filter(MonitoringAlert.id == alert_id).first()
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert not found: {alert_id}",
            )
        if alert.user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to modify this alert",
            )

        if alert.status in (AlertStatus.RESOLVED.value, AlertStatus.DISMISSED.value):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot dismiss alert in '{alert.status}' state",
            )

        alert.status = AlertStatus.DISMISSED.value
        db.commit()
        db.refresh(alert)
        return AlertResponse.model_validate(alert)

    def get_monitoring_summary(
        self,
        db: Session,
        user_id: int,
    ) -> MonitoringSummary:
        """Aggregate executive monitoring KPIs."""
        now = dt.datetime.now(dt.timezone.utc)
        total_alerts = db.query(MonitoringAlert).filter(MonitoringAlert.user_id == user_id).count()
        new_count = db.query(MonitoringAlert).filter(
            MonitoringAlert.user_id == user_id,
            MonitoringAlert.status == AlertStatus.NEW.value,
        ).count()
        crit_count = db.query(MonitoringAlert).filter(
            MonitoringAlert.user_id == user_id,
            MonitoringAlert.severity == "critical",
        ).count()
        high_count = db.query(MonitoringAlert).filter(
            MonitoringAlert.user_id == user_id,
            MonitoringAlert.severity == "high",
        ).count()
        med_count = db.query(MonitoringAlert).filter(
            MonitoringAlert.user_id == user_id,
            MonitoringAlert.severity == "medium",
        ).count()
        unresolved = db.query(MonitoringAlert).filter(
            MonitoringAlert.user_id == user_id,
            MonitoringAlert.status.in_([AlertStatus.NEW.value, AlertStatus.ACKNOWLEDGED.value]),
        ).count()

        latest_run = (
            db.query(MonitoringRun)
            .filter(MonitoringRun.user_id == user_id)
            .order_by(desc(MonitoringRun.started_at))
            .first()
        )

        suppressed = latest_run.duplicates_suppressed if latest_run else 0
        scan_time = latest_run.completed_at or latest_run.started_at if latest_run else now

        return MonitoringSummary(
            scan_timestamp=scan_time,
            total_anomalies=total_alerts,
            new_alerts=new_count,
            critical_alerts=crit_count,
            high_alerts=high_count,
            medium_alerts=med_count,
            suppressed_duplicates=suppressed,
            unresolved_alerts=unresolved,
        )

    def list_runs(
        self,
        db: Session,
        user_id: int,
        limit: int = 10,
    ) -> list[MonitoringRunItemResponse]:
        """List recent monitoring runs."""
        runs = (
            db.query(MonitoringRun)
            .filter(MonitoringRun.user_id == user_id)
            .order_by(desc(MonitoringRun.started_at))
            .limit(limit)
            .all()
        )
        return [MonitoringRunItemResponse.model_validate(r) for r in runs]
