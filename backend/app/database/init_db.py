"""
Database initialization and bootstrap management for Smart Sales Forecasting System.
Provides idempotent schema creation and administrator account bootstrapping
compatible with Supabase PostgreSQL and SQLite.
"""

from __future__ import annotations

import logging
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import settings
from backend.app.core.security import hash_password
from backend.app.database.database import (
    Base,
    engine as default_engine,
    SessionLocal as default_session_factory,
)
# Ensure all SQLAlchemy models are imported and registered on Base.metadata before create_all
import backend.app.database.models  # noqa: F401
from backend.app.database.models import User
from backend.app.database.user_crud import create_user, get_user_by_email

logger = logging.getLogger(__name__)


def ensure_schema_migrations(engine: Engine) -> None:
    """
    Safely and idempotently verifies that newly added columns exist in existing tables.
    Never drops or alters existing columns or tables.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    if "sales_records" in table_names:
        columns = [col["name"] for col in inspector.get_columns("sales_records")]
        if "dataset_id" not in columns:
            logger.info("Migrating schema: adding dataset_id column to sales_records...")
            with engine.begin() as conn:
                if engine.dialect.name == "postgresql":
                    conn.execute(
                        text(
                            "ALTER TABLE sales_records ADD COLUMN IF NOT EXISTS dataset_id VARCHAR(50) REFERENCES dataset_uploads(id) ON DELETE CASCADE;"
                        )
                    )
                else:
                    # SQLite or other dialects
                    conn.execute(
                        text(
                            "ALTER TABLE sales_records ADD COLUMN dataset_id VARCHAR(50);"
                        )
                    )
            logger.info("Successfully added dataset_id column to sales_records.")

    if "products" in table_names:
        columns = [col["name"] for col in inspector.get_columns("products")]
        with engine.begin() as conn:
            if "tenant_id" not in columns:
                logger.info("Migrating schema: adding tenant_id column to products...")
                if engine.dialect.name == "postgresql":
                    conn.execute(
                        text(
                            "ALTER TABLE products ADD COLUMN IF NOT EXISTS tenant_id INTEGER REFERENCES users(id) ON DELETE SET NULL;"
                        )
                    )
                else:
                    conn.execute(
                        text(
                            "ALTER TABLE products ADD COLUMN tenant_id INTEGER;"
                        )
                    )
                logger.info("Successfully added tenant_id column to products.")

            if "raw_product_id" not in columns:
                logger.info("Migrating schema: adding raw_product_id column to products...")
                if engine.dialect.name == "postgresql":
                    conn.execute(
                        text(
                            "ALTER TABLE products ADD COLUMN IF NOT EXISTS raw_product_id VARCHAR(255);"
                        )
                    )
                else:
                    conn.execute(
                        text(
                            "ALTER TABLE products ADD COLUMN raw_product_id VARCHAR(255);"
                        )
                    )
                logger.info("Successfully added raw_product_id column to products.")

    if "model_training_jobs" in table_names:
        columns = {col["name"] for col in inspector.get_columns("model_training_jobs")}
        with engine.begin() as conn:
            if "last_heartbeat_at" not in columns:
                logger.info("Migrating schema: adding last_heartbeat_at column to model_training_jobs...")
                if engine.dialect.name == "postgresql":
                    conn.execute(text("ALTER TABLE model_training_jobs ADD COLUMN IF NOT EXISTS last_heartbeat_at TIMESTAMP;"))
                else:
                    conn.execute(text("ALTER TABLE model_training_jobs ADD COLUMN last_heartbeat_at TIMESTAMP;"))
                logger.info("Successfully added last_heartbeat_at column to model_training_jobs.")

        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "CREATE UNIQUE INDEX IF NOT EXISTS uq_active_training_job_per_dataset "
                        "ON model_training_jobs (user_id, dataset_id) "
                        "WHERE status IN ('queued', 'processing', 'training', 'evaluating');"
                    )
                )
            logger.info("Verified/created partial unique index uq_active_training_job_per_dataset on model_training_jobs.")
        except Exception as exc:
            logger.warning("Could not create partial unique index uq_active_training_job_per_dataset: %s", exc)

    if "dataset_uploads" in table_names:
        try:
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "CREATE UNIQUE INDEX IF NOT EXISTS uq_dataset_uploads_user_active "
                        "ON dataset_uploads (user_id) WHERE status = 'active';"
                    )
                )
            logger.info("Verified/created partial unique index uq_dataset_uploads_user_active on dataset_uploads.")
        except Exception as exc:
            logger.warning("Could not create partial unique index uq_dataset_uploads_user_active: %s", exc)

    if "company_models" in table_names:
        cm_columns = {col["name"] for col in inspector.get_columns("company_models")}
        with engine.begin() as conn:
            if "feature_columns" not in cm_columns:
                logger.info("Migrating schema: adding feature_columns column to company_models...")
                if engine.dialect.name == "postgresql":
                    conn.execute(text("ALTER TABLE company_models ADD COLUMN IF NOT EXISTS feature_columns JSON;"))
                else:
                    conn.execute(text("ALTER TABLE company_models ADD COLUMN feature_columns JSON;"))
                logger.info("Successfully added feature_columns column to company_models.")
            if "target_column" not in cm_columns:
                logger.info("Migrating schema: adding target_column column to company_models...")
                if engine.dialect.name == "postgresql":
                    conn.execute(text("ALTER TABLE company_models ADD COLUMN IF NOT EXISTS target_column VARCHAR(50) DEFAULT 'Quantity';"))
                else:
                    conn.execute(text("ALTER TABLE company_models ADD COLUMN target_column VARCHAR(50) DEFAULT 'Quantity';"))
                logger.info("Successfully added target_column column to company_models.")
            if "reference_start_date" not in cm_columns:
                logger.info("Migrating schema: adding reference_start_date column to company_models...")
                if engine.dialect.name == "postgresql":
                    conn.execute(text("ALTER TABLE company_models ADD COLUMN IF NOT EXISTS reference_start_date VARCHAR(20);"))
                else:
                    conn.execute(text("ALTER TABLE company_models ADD COLUMN reference_start_date VARCHAR(20);"))
                logger.info("Successfully added reference_start_date column to company_models.")

        try:
            with engine.begin() as conn:
                if engine.dialect.name == "sqlite":
                    conn.execute(
                        text(
                            "CREATE UNIQUE INDEX IF NOT EXISTS uq_company_models_user_horizon_active "
                            "ON company_models (user_id, horizon) WHERE is_active = 1;"
                        )
                    )
                else:
                    conn.execute(
                        text(
                            "CREATE UNIQUE INDEX IF NOT EXISTS uq_company_models_user_horizon_active "
                            "ON company_models (user_id, horizon) WHERE is_active = true;"
                        )
                    )
            logger.info("Verified/created partial unique index uq_company_models_user_horizon_active on company_models.")
        except Exception as exc:
            logger.warning("Could not create partial unique index uq_company_models_user_horizon_active: %s", exc)


def recover_interrupted_training_jobs(engine: Engine, stale_minutes: int = 10) -> None:
    """
    Recovers from Render container restarts by transitioning stale in-flight training jobs to 'failed'.
    Uses heartbeat-based stale detection: a job running for > 10m is NOT failed if its
    heartbeat was recently updated. Only jobs whose last_heartbeat_at is older than stale_minutes
    (or missing) are considered stale.
    """
    from datetime import datetime, timezone, timedelta
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "model_training_jobs" not in inspector.get_table_names():
        return

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")

    try:
        with engine.begin() as conn:
            if engine.dialect.name == "sqlite":
                conn.execute(
                    text(
                        "UPDATE model_training_jobs "
                        "SET status = 'failed', "
                        "error_message = 'Training job became unresponsive (heartbeat timeout). Please click Retrain Models.' "
                        "WHERE status IN ('queued', 'processing', 'training', 'evaluating') "
                        "AND ("
                        "  (last_heartbeat_at IS NOT NULL AND datetime(last_heartbeat_at) < datetime(:cutoff)) "
                        "  OR (last_heartbeat_at IS NULL AND started_at IS NOT NULL AND datetime(started_at) < datetime(:cutoff)) "
                        "  OR (last_heartbeat_at IS NULL AND started_at IS NULL AND datetime(created_at) < datetime(:cutoff))"
                        ");"
                    ),
                    {"cutoff": cutoff_str},
                )
                conn.execute(
                    text(
                        "UPDATE company_models "
                        "SET status = 'failed', "
                        "status_message = 'Training job became unresponsive (heartbeat timeout). Please click Retrain Models.' "
                        "WHERE status IN ('queued', 'processing', 'training', 'evaluating') "
                        "AND datetime(created_at) < datetime(:cutoff);"
                    ),
                    {"cutoff": cutoff_str},
                )
            else:
                conn.execute(
                    text(
                        "UPDATE model_training_jobs "
                        "SET status = 'failed', "
                        "error_message = 'Training job became unresponsive (heartbeat timeout). Please click Retrain Models.' "
                        "WHERE status IN ('queued', 'processing', 'training', 'evaluating') "
                        "AND ("
                        "  (last_heartbeat_at IS NOT NULL AND last_heartbeat_at < :cutoff) "
                        "  OR (last_heartbeat_at IS NULL AND started_at IS NOT NULL AND started_at < :cutoff) "
                        "  OR (last_heartbeat_at IS NULL AND started_at IS NULL AND created_at < :cutoff)"
                        ");"
                    ),
                    {"cutoff": cutoff},
                )
                conn.execute(
                    text(
                        "UPDATE company_models "
                        "SET status = 'failed', "
                        "status_message = 'Training job became unresponsive (heartbeat timeout). Please click Retrain Models.' "
                        "WHERE status IN ('queued', 'processing', 'training', 'evaluating') "
                        "AND created_at < :cutoff;"
                    ),
                    {"cutoff": cutoff},
                )
    except Exception as exc:
        logger.warning("Could not check/recover interrupted training jobs: %s", exc)


def init_db(engine: Engine | None = None) -> None:
    """
    Idempotently creates all tables defined in Base.metadata if they do not exist.
    Never drops, truncates, or modifies existing tables or data.
    """
    target_engine = engine or default_engine
    logger.info("Verifying/initializing database schema for dialect: %s", target_engine.dialect.name)
    Base.metadata.create_all(bind=target_engine)
    ensure_schema_migrations(target_engine)
    recover_interrupted_training_jobs(target_engine)
    try:
        from backend.app.services.company_model_service import company_model_service
        resumed = company_model_service.resume_stranded_queued_jobs(engine=target_engine)
        if resumed > 0:
            logger.info("Resumed %d stranded queued training job(s) on startup.", resumed)
    except Exception as exc:
        logger.warning("Could not check/resume stranded training jobs on startup: %s", exc)
    logger.info("Database schema verification/initialization complete.")


def seed_admin_user(
    db: Session,
    admin_username: str | None = None,
    admin_email: str | None = None,
    admin_password: str | None = None,
) -> User | None:
    """
    Idempotently seeds the system bootstrap administrator account.

    Bootstrap Password Policy Notice:
    The bootstrap administrator password is read directly from the ADMIN_PASSWORD
    environment variable (or parameter) and hashed using the system's Argon2id password
    hasher. This allows system-level provisioning of the bootstrap account without
    weakening or altering the public registration schema (UserCreate min_length=8)
    enforced for ordinary users.

    Idempotency Guarantees:
    - Creates the admin account only if it does not already exist.
    - Never creates duplicates.
    - Never deletes or overwrites existing administrator data.
    - Never logs plaintext passwords.
    """
    target_email = (admin_email or settings.admin_email or "admin@smart-sales.local").strip().lower()
    target_password = admin_password if admin_password is not None else settings.admin_password

    if not target_password or not target_password.strip():
        logger.info("ADMIN_PASSWORD not configured; skipping bootstrap administrator seeding.")
        return None

    existing_user = get_user_by_email(db, target_email)
    if existing_user:
        logger.info("Administrator account already exists (%s); skipping creation.", target_email)
        return existing_user

    logger.info("Seeding bootstrap administrator account (%s)...", target_email)
    hashed = hash_password(target_password)
    admin_user = create_user(
        db=db,
        email=target_email,
        password_hash=hashed,
    )
    logger.info("Bootstrap administrator account created successfully (%s).", target_email)
    return admin_user


def init_database_and_seed(
    engine: Engine | None = None,
    session_factory: sessionmaker | None = None,
) -> None:
    """
    Orchestrates idempotent database schema initialization followed by
    administrator account seeding.
    """
    target_engine = engine or default_engine
    target_session_factory = session_factory or default_session_factory

    init_db(engine=target_engine)

    db = target_session_factory()
    try:
        seed_admin_user(db)
    finally:
        db.close()


if __name__ == "__main__":
    init_database_and_seed()
