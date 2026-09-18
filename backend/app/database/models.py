from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.database.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    sales_records: Mapped[list["SalesRecord"]] = relationship(
        "SalesRecord",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    decisions: Mapped[list["DecisionRecord"]] = relationship(
        "DecisionRecord",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    alerts: Mapped[list["MonitoringAlert"]] = relationship(
        "MonitoringAlert",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    monitoring_runs: Mapped[list["MonitoringRun"]] = relationship(
        "MonitoringRun",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class Product(Base):
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    product_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        index=True,
        nullable=False,
    )

    product_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    category_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    category_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    unit_price: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    sales_records: Mapped[list["SalesRecord"]] = relationship(
        "SalesRecord",
        back_populates="product",
    )


class SalesRecord(Base):
    __tablename__ = "sales_records"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    product_id: Mapped[int] = mapped_column(
        ForeignKey("products.id"),
        nullable=False,
        index=True,
    )

    sale_date: Mapped[datetime] = mapped_column(
        Date,
        nullable=False,
        index=True,
    )

    quantity: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    unit_price: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    discount_percent: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )

    promotion: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    holiday_flag: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )

    sales_amount: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    profit: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="sales_records",
    )

    product: Mapped["Product"] = relationship(
        "Product",
        back_populates="sales_records",
    )


class DecisionRecord(Base):
    __tablename__ = "decisions"

    id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
        index=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    recommendation_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    anomaly_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    simulation_id: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )

    recommendation_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    proposed_action: Mapped[str] = mapped_column(
        String(2000),
        nullable=False,
    )

    modified_action: Mapped[str | None] = mapped_column(
        String(2000),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        default="pending_review",
        nullable=False,
        index=True,
    )

    rationale: Mapped[str | None] = mapped_column(
        String(2000),
        nullable=True,
    )

    decision_note: Mapped[str | None] = mapped_column(
        String(2000),
        nullable=True,
    )

    evidence_snapshot: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    simulation_snapshot: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="decisions",
    )

    audit_events: Mapped[list["DecisionAuditEvent"]] = relationship(
        "DecisionAuditEvent",
        back_populates="decision",
        cascade="all, delete-orphan",
        order_by="DecisionAuditEvent.created_at.asc()",
    )


class DecisionAuditEvent(Base):
    __tablename__ = "decision_audit_events"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    decision_id: Mapped[str] = mapped_column(
        ForeignKey("decisions.id"),
        nullable=False,
        index=True,
    )

    actor_user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    previous_status: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    new_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    note: Mapped[str | None] = mapped_column(
        String(2000),
        nullable=True,
    )

    event_metadata: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    decision: Mapped["DecisionRecord"] = relationship(
        "DecisionRecord",
        back_populates="audit_events",
    )

    actor: Mapped["User"] = relationship(
        "User",
    )


class MonitoringAlert(Base):
    __tablename__ = "monitoring_alerts"

    id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
        index=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    anomaly_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    alert_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    metric: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    severity: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
    )

    priority: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        default="new",
        nullable=False,
        index=True,
    )

    event_date: Mapped[datetime] = mapped_column(
        Date,
        nullable=False,
        index=True,
    )

    actual_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    expected_value: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    deviation: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    deviation_percent: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    fingerprint: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    evidence_snapshot: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    acknowledged_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="alerts",
    )


class MonitoringRun(Base):
    __tablename__ = "monitoring_runs"

    id: Mapped[str] = mapped_column(
        String(50),
        primary_key=True,
        index=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    total_records_evaluated: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    anomalies_found: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    alerts_created: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    duplicates_suppressed: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        default="running",
        nullable=False,
        index=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="monitoring_runs",
    )


