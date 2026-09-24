from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    text,
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

    datasets: Mapped[list["DatasetUpload"]] = relationship(
        "DatasetUpload",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    company_models: Mapped[list["CompanyModel"]] = relationship(
        "CompanyModel",
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

    tenant_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    raw_product_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
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


class DatasetUpload(Base):
    __tablename__ = "dataset_uploads"
    __table_args__ = (
        Index(
            "uq_dataset_uploads_user_active",
            "user_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
            sqlite_where=text("status = 'active'"),
        ),
    )

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

    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    dataset_key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    row_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    product_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    category_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    min_date: Mapped[datetime | None] = mapped_column(
        Date,
        nullable=True,
    )

    max_date: Mapped[datetime | None] = mapped_column(
        Date,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        default="active",
        nullable=False,
        index=True,
    )

    validation_summary: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )

    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="datasets",
    )

    sales_records: Mapped[list["SalesRecord"]] = relationship(
        "SalesRecord",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )

    company_models: Mapped[list["CompanyModel"]] = relationship(
        "CompanyModel",
        back_populates="dataset",
        cascade="all, delete-orphan",
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

    dataset_id: Mapped[str | None] = mapped_column(
        String(50),
        ForeignKey("dataset_uploads.id", ondelete="CASCADE"),
        nullable=True,
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

    dataset: Mapped["DatasetUpload | None"] = relationship(
        "DatasetUpload",
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
        index=True,
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
        index=True,
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
        index=True,
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


class CompanyModel(Base):
    __tablename__ = "company_models"
    __table_args__ = (
        Index(
            "uq_company_models_user_horizon_active",
            "user_id",
            "horizon",
            unique=True,
            postgresql_where=text("is_active = true"),
            sqlite_where=text("is_active = 1"),
        ),
        Index("ix_company_models_user_dataset", "user_id", "dataset_id"),
        Index("ix_company_models_user_version", "user_id", "model_version"),
    )

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

    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_uploads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    horizon: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        index=True,
    )

    model_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    artifact_path: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    model_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    validation_wape: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    validation_mae: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    validation_rmse: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    test_wape: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    test_mae: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    test_rmse: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    training_rows: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    feature_version: Mapped[str] = mapped_column(
        String(50),
        default="v1",
        nullable=False,
    )

    feature_columns: Mapped[list | None] = mapped_column(
        JSON,
        nullable=True,
    )

    target_column: Mapped[str] = mapped_column(
        String(50),
        default="Quantity",
        nullable=False,
    )

    reference_start_date: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="queued",
        nullable=False,
        index=True,
    )

    status_message: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
    )

    trained_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user: Mapped["User"] = relationship(
        "User",
        back_populates="company_models",
    )

    dataset: Mapped["DatasetUpload"] = relationship(
        "DatasetUpload",
        back_populates="company_models",
    )


class ModelTrainingJob(Base):
    __tablename__ = "model_training_jobs"
    __table_args__ = (
        Index("ix_model_training_jobs_user_dataset", "user_id", "dataset_id"),
        Index(
            "uq_active_training_job_per_dataset",
            "user_id",
            "dataset_id",
            unique=True,
            postgresql_where=text("status IN ('queued', 'processing', 'training', 'evaluating')"),
            sqlite_where=text("status IN ('queued', 'processing', 'training', 'evaluating')"),
        ),
    )

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

    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("dataset_uploads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    model_version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        default="queued",
        nullable=False,
        index=True,
    )

    progress_stage: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    error_message: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    started_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    last_heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user: Mapped["User"] = relationship("User")
    dataset: Mapped["DatasetUpload"] = relationship("DatasetUpload")


