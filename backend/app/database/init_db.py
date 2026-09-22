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


def init_db(engine: Engine | None = None) -> None:
    """
    Idempotently creates all tables defined in Base.metadata if they do not exist.
    Never drops, truncates, or modifies existing tables or data.
    """
    target_engine = engine or default_engine
    logger.info("Verifying/initializing database schema for dialect: %s", target_engine.dialect.name)
    Base.metadata.create_all(bind=target_engine)
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
