from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import Any

from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.app.database.models import DecisionAuditEvent, DecisionRecord
from backend.app.schemas.decisions import (
    AnomalySnapshot,
    DecisionActionRequest,
    DecisionCreateRequest,
    DecisionPackage,
    DecisionResubmitRequest,
    DecisionStatus,
    RecommendationSnapshot,
    SimulationSnapshot,
)
from backend.app.services.investigation_service import InvestigationService
from backend.app.services.recommendation_service import RecommendationService
from backend.app.services.simulation_service import SimulationService

logger = logging.getLogger(__name__)


class DecisionServiceError(Exception):
    """Base exception for decision governance operations."""


class DecisionNotFoundError(DecisionServiceError):
    """Raised when a decision record cannot be found."""


class DecisionPermissionError(DecisionServiceError):
    """Raised when a user attempts to access or mutate a decision they do not own."""


class DecisionConflictError(DecisionServiceError):
    """Raised when attempting an invalid state transition or mutating a finalized decision."""


class DecisionValidationError(DecisionServiceError):
    """Raised when validation rules or rationale requirements are violated."""


class DecisionService:
    """
    Service managing the human approval lifecycle and immutable audit trail.
    Ensures zero autonomous execution and strict non-repudiation.
    """

    def __init__(
        self,
        investigation_service: InvestigationService | None = None,
        recommendation_service: RecommendationService | None = None,
        simulation_service: SimulationService | None = None,
    ) -> None:
        self.investigation_service = investigation_service or InvestigationService()
        self.recommendation_service = recommendation_service or RecommendationService()
        self.simulation_service = simulation_service or SimulationService()

    def create_decision(
        self,
        db: Session,
        user_id: int,
        request: DecisionCreateRequest,
    ) -> DecisionRecord:
        """
        Creates a new decision review record and captures empirical evidence snapshots.
        Initial status: pending_review.
        """
        sim_snapshot: dict[str, Any] | None = None
        anomaly_snapshot: AnomalySnapshot | None = None

        # 1. Capture Anomaly Snapshot if anchored
        if request.anomaly_id:
            try:
                inv = self.investigation_service.investigate_anomaly(request.anomaly_id)
                anomaly_snapshot = AnomalySnapshot(
                    anomaly_id=inv.anomaly_id,
                    date=str(inv.anomaly_date),
                    metric=inv.metric,
                    actual=inv.actual_value,
                    baseline=inv.expected_baseline,
                    deviation=inv.deviation,
                    severity=inv.severity,
                    associated_contributors=[
                        f"{d.driver_name} ({d.driver_type})" for d in inv.drivers[:3]
                    ],
                )
            except Exception as exc:
                logger.warning("Could not build anomaly snapshot for %s: %s", request.anomaly_id, exc)

        # 2. Capture Simulation Snapshot if referenced
        if request.simulation_id:
            try:
                sim = self.simulation_service.get_simulation(request.simulation_id)
                if sim and sim.baseline and sim.scenario and sim.delta:
                    sim_obj = SimulationSnapshot(
                        simulation_id=sim.simulation_id,
                        scenario_type=sim.scenario_type.value if hasattr(sim.scenario_type, "value") else str(sim.scenario_type),
                        horizon_days=sim.horizon_days,
                        baseline_quantity=sim.baseline.total_quantity,
                        scenario_quantity=sim.scenario.total_quantity,
                        quantity_delta=sim.delta.quantity_delta,
                        baseline_revenue=sim.baseline.total_revenue,
                        scenario_revenue=sim.scenario.total_revenue,
                        revenue_delta=sim.delta.revenue_delta,
                        assumptions=sim.assumptions,
                        limitations=sim.limitations,
                        confidence=str(sim.confidence),
                    )
                    sim_snapshot = sim_obj.model_dump()
            except Exception as exc:
                logger.warning("Could not build simulation snapshot for %s: %s", request.simulation_id, exc)

        # 3. Capture Recommendation Snapshot
        rec_snapshot: RecommendationSnapshot
        if request.recommendation_payload:
            payload = request.recommendation_payload
            rec_snapshot = RecommendationSnapshot(
                recommendation_id=payload.get("recommendation_id") or request.recommendation_id or f"rec-{uuid.uuid4().hex[:8]}",
                recommendation_type=payload.get("recommendation_type") or request.recommendation_type,
                title=payload.get("title"),
                action=payload.get("action") or request.proposed_action,
                priority=payload.get("priority"),
                risk=payload.get("risk"),
                confidence=payload.get("confidence"),
                supporting_evidence=payload.get("supporting_evidence") or [],
                assumptions=payload.get("assumptions") or [],
                tradeoffs=payload.get("tradeoffs") or [],
                limitations=payload.get("limitations") or [],
            )
        else:
            rec_snapshot = RecommendationSnapshot(
                recommendation_id=request.recommendation_id or f"rec-{uuid.uuid4().hex[:8]}",
                recommendation_type=request.recommendation_type,
                title=f"{request.recommendation_type.replace('_', ' ').title()}",
                action=request.proposed_action,
                priority="medium",
                risk="medium",
                confidence="high",
                supporting_evidence=[f"Submitted for governance review under category {request.recommendation_type}."],
                assumptions=["Operational review required before execution."],
                tradeoffs=["Human validation required."],
                limitations=["No autonomous execution."],
            )

        # 4. Construct DecisionPackage
        package = DecisionPackage(
            recommendation=rec_snapshot,
            anomaly_context=anomaly_snapshot,
            simulation=SimulationSnapshot.model_validate(sim_snapshot) if sim_snapshot else None,
            assumptions=rec_snapshot.assumptions,
            limitations=rec_snapshot.limitations,
            evidence_quality="high",
            approval_requirement="Human review and approval required before consideration/execution",
        )

        decision_id = f"dec-{uuid.uuid4().hex[:12]}"
        now = dt.datetime.now(dt.timezone.utc)

        record = DecisionRecord(
            id=decision_id,
            user_id=user_id,
            recommendation_id=request.recommendation_id or rec_snapshot.recommendation_id,
            anomaly_id=request.anomaly_id,
            simulation_id=request.simulation_id,
            recommendation_type=request.recommendation_type,
            proposed_action=request.proposed_action,
            modified_action=request.modified_action,
            status=DecisionStatus.PENDING_REVIEW.value,
            decision_note=request.decision_note,
            rationale=request.decision_note,
            evidence_snapshot=package.model_dump(),
            simulation_snapshot=sim_snapshot,
            created_at=now,
            updated_at=now,
        )

        db.add(record)
        db.flush()

        # 5. Append initial creation audit event
        audit_event = DecisionAuditEvent(
            decision_id=decision_id,
            actor_user_id=user_id,
            event_type="created",
            previous_status=None,
            new_status=DecisionStatus.PENDING_REVIEW.value,
            note=request.decision_note or "Submitted for human approval review",
            event_metadata={"recommendation_type": request.recommendation_type},
            created_at=now,
        )
        db.add(audit_event)
        db.commit()
        db.refresh(record)

        return record

    def get_decision(
        self,
        db: Session,
        decision_id: str,
        user_id: int,
    ) -> DecisionRecord:
        """
        Retrieves a decision record, enforcing ownership authorization.
        """
        record = db.query(DecisionRecord).filter(DecisionRecord.id == decision_id).first()
        if not record:
            raise DecisionNotFoundError(f"Decision record '{decision_id}' not found.")
        if record.user_id != user_id:
            raise DecisionPermissionError("You do not have authorization to access this decision record.")
        return record

    def list_decisions(
        self,
        db: Session,
        user_id: int,
        status: DecisionStatus | None = None,
        anomaly_id: str | None = None,
        recommendation_type: str | None = None,
        skip: int = 0,
        limit: int = 100,
    ) -> list[DecisionRecord]:
        """
        Lists decision records owned by the authenticated user with optional filters.
        """
        query = db.query(DecisionRecord).filter(DecisionRecord.user_id == user_id)
        if status:
            query = query.filter(DecisionRecord.status == status.value)
        if anomaly_id:
            query = query.filter(DecisionRecord.anomaly_id == anomaly_id)
        if recommendation_type:
            query = query.filter(DecisionRecord.recommendation_type == recommendation_type)

        return query.order_by(desc(DecisionRecord.created_at)).offset(skip).limit(limit).all()

    def approve_decision(
        self,
        db: Session,
        decision_id: str,
        user_id: int,
        request: DecisionActionRequest,
    ) -> DecisionRecord:
        """
        Approves a recommendation for human execution/consideration.
        Finalizes the decision into 'approved'.
        """
        record = self.get_decision(db, decision_id, user_id)

        # Check finalization and state machine rules
        if record.status in (DecisionStatus.APPROVED.value, DecisionStatus.REJECTED.value):
            raise DecisionConflictError(
                f"Decision '{decision_id}' is already finalized in status '{record.status}' and cannot be modified."
            )
        if record.status not in (DecisionStatus.PENDING_REVIEW.value, DecisionStatus.CHANGES_REQUESTED.value):
            raise DecisionConflictError(f"Cannot approve decision with current status '{record.status}'.")

        prev_status = record.status
        now = dt.datetime.now(dt.timezone.utc)

        record.status = DecisionStatus.APPROVED.value
        record.reviewed_at = now
        record.updated_at = now
        if request.decision_note:
            record.decision_note = request.decision_note
            record.rationale = request.decision_note
        if request.modified_action:
            record.modified_action = request.modified_action

        audit_event = DecisionAuditEvent(
            decision_id=decision_id,
            actor_user_id=user_id,
            event_type="approved",
            previous_status=prev_status,
            new_status=DecisionStatus.APPROVED.value,
            note=request.decision_note or "Approved by human reviewer for manual execution consideration",
            event_metadata={
                "modified_action": request.modified_action,
                "reviewed_at": now.isoformat(),
            },
            created_at=now,
        )
        db.add(audit_event)
        db.commit()
        db.refresh(record)

        return record

    def reject_decision(
        self,
        db: Session,
        decision_id: str,
        user_id: int,
        request: DecisionActionRequest,
    ) -> DecisionRecord:
        """
        Rejects a recommendation. Mandatory decision rationale required.
        Finalizes the decision into 'rejected'.
        """
        if not request.decision_note or not request.decision_note.strip():
            raise DecisionValidationError("A decision rationale note is required when rejecting a recommendation.")

        record = self.get_decision(db, decision_id, user_id)

        # Check finalization and state machine rules
        if record.status in (DecisionStatus.APPROVED.value, DecisionStatus.REJECTED.value):
            raise DecisionConflictError(
                f"Decision '{decision_id}' is already finalized in status '{record.status}' and cannot be modified."
            )
        if record.status not in (DecisionStatus.PENDING_REVIEW.value, DecisionStatus.CHANGES_REQUESTED.value):
            raise DecisionConflictError(f"Cannot reject decision with current status '{record.status}'.")

        prev_status = record.status
        now = dt.datetime.now(dt.timezone.utc)

        record.status = DecisionStatus.REJECTED.value
        record.reviewed_at = now
        record.updated_at = now
        record.decision_note = request.decision_note.strip()
        record.rationale = request.decision_note.strip()

        audit_event = DecisionAuditEvent(
            decision_id=decision_id,
            actor_user_id=user_id,
            event_type="rejected",
            previous_status=prev_status,
            new_status=DecisionStatus.REJECTED.value,
            note=request.decision_note.strip(),
            event_metadata={"rejected_at": now.isoformat()},
            created_at=now,
        )
        db.add(audit_event)
        db.commit()
        db.refresh(record)

        return record

    def request_changes(
        self,
        db: Session,
        decision_id: str,
        user_id: int,
        request: DecisionActionRequest,
    ) -> DecisionRecord:
        """
        Requests modifications/clarification. Mandatory decision note required.
        Transitions status to 'changes_requested'.
        """
        if not request.decision_note or not request.decision_note.strip():
            raise DecisionValidationError("A decision note specifying the requested changes is required.")

        record = self.get_decision(db, decision_id, user_id)

        # Check finalization and state machine rules
        if record.status in (DecisionStatus.APPROVED.value, DecisionStatus.REJECTED.value):
            raise DecisionConflictError(
                f"Decision '{decision_id}' is already finalized in status '{record.status}' and cannot be modified."
            )
        if record.status != DecisionStatus.PENDING_REVIEW.value:
            raise DecisionConflictError(f"Cannot request changes on decision with status '{record.status}'.")

        prev_status = record.status
        now = dt.datetime.now(dt.timezone.utc)

        record.status = DecisionStatus.CHANGES_REQUESTED.value
        record.updated_at = now
        record.decision_note = request.decision_note.strip()
        record.rationale = request.decision_note.strip()
        if request.modified_action:
            record.modified_action = request.modified_action

        audit_event = DecisionAuditEvent(
            decision_id=decision_id,
            actor_user_id=user_id,
            event_type="changes_requested",
            previous_status=prev_status,
            new_status=DecisionStatus.CHANGES_REQUESTED.value,
            note=request.decision_note.strip(),
            event_metadata={"changes_requested_at": now.isoformat()},
            created_at=now,
        )
        db.add(audit_event)
        db.commit()
        db.refresh(record)

        return record

    def resubmit_decision(
        self,
        db: Session,
        decision_id: str,
        user_id: int,
        request: DecisionResubmitRequest,
    ) -> DecisionRecord:
        """
        Resubmits a decision back to 'pending_review' after revising action wording.
        """
        record = self.get_decision(db, decision_id, user_id)

        if record.status in (DecisionStatus.APPROVED.value, DecisionStatus.REJECTED.value):
            raise DecisionConflictError(
                f"Decision '{decision_id}' is already finalized in status '{record.status}' and cannot be modified."
            )
        if record.status != DecisionStatus.CHANGES_REQUESTED.value:
            raise DecisionConflictError(
                f"Only decisions with status 'changes_requested' can be resubmitted (current status: '{record.status}')."
            )

        prev_status = record.status
        now = dt.datetime.now(dt.timezone.utc)

        record.status = DecisionStatus.PENDING_REVIEW.value
        record.updated_at = now
        if request.modified_action:
            record.modified_action = request.modified_action
        if request.decision_note:
            record.decision_note = request.decision_note

        audit_event = DecisionAuditEvent(
            decision_id=decision_id,
            actor_user_id=user_id,
            event_type="reopened",
            previous_status=prev_status,
            new_status=DecisionStatus.PENDING_REVIEW.value,
            note=request.decision_note or "Resubmitted for review with revised action wording",
            event_metadata={"resubmitted_at": now.isoformat()},
            created_at=now,
        )
        db.add(audit_event)
        db.commit()
        db.refresh(record)

        return record
