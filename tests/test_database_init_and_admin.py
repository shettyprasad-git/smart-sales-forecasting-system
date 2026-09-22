"""
Tests for idempotent database initialization and administrator account bootstrapping.
Validates all requirements:
- Safe, non-destructive schema creation (PostgreSQL & SQLite compatible)
- Idempotent table creation and row preservation
- Bootstrap administrator seeding without hardcoded passwords
- Username 'admin' resolution to configured admin email
- Unaltered, strictly enforced general user registration password validation (min_length=8)
- Valid JWT token issuance and /api/auth/me integration
"""

from __future__ import annotations

from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.config import settings
from backend.app.core.security import decode_access_token, verify_password
from backend.app.database.database import Base
from backend.app.database.init_db import init_db, seed_admin_user
from backend.app.database.models import Product, User
from backend.app.database.user_crud import get_user_by_email
from backend.app.schemas.auth import UserCreate
import pytest


@pytest.fixture(autouse=True)
def clean_admin_user(db_session: Session):
    """Ensure clean slate for admin user in test database."""
    db_session.query(User).filter(User.email == "admin@smart-sales.local").delete()
    db_session.commit()
    yield
    db_session.query(User).filter(User.email == "admin@smart-sales.local").delete()
    db_session.commit()


def test_database_initialization_idempotent(tmp_path):
    """
    1. Database initialization is idempotent.
    Repeated runs on the same database must succeed without error and leave tables intact.
    """
    test_db_path = tmp_path / "idempotent_test.db"
    test_engine = create_engine(f"sqlite:///{test_db_path}")

    try:
        # First initialization run
        init_db(engine=test_engine)
        inspector1 = inspect(test_engine)
        tables1 = sorted(inspector1.get_table_names())

        # Second initialization run (must not error, drop, or recreate)
        init_db(engine=test_engine)
        inspector2 = inspect(test_engine)
        tables2 = sorted(inspector2.get_table_names())

        assert tables1 == tables2
        assert len(tables1) >= 7
    finally:
        test_engine.dispose()


def test_all_expected_sqlalchemy_tables_created(tmp_path):
    """
    2. All expected SQLAlchemy tables are created.
    3. The 'users' table exists after initialization.
    """
    test_db_path = tmp_path / "tables_test.db"
    test_engine = create_engine(f"sqlite:///{test_db_path}")

    try:
        init_db(engine=test_engine)
        inspector = inspect(test_engine)
        tables = set(inspector.get_table_names())

        expected_tables = {
            "users",
            "products",
            "sales_records",
            "decisions",
            "decision_audit_events",
            "monitoring_alerts",
            "monitoring_runs",
        }
        assert expected_tables.issubset(tables)
        assert "users" in tables

        # Direct SQL probe to verify table accessibility
        with test_engine.connect() as conn:
            result = conn.execute(text("SELECT count(*) FROM users")).scalar()
            assert result == 0
    finally:
        test_engine.dispose()


def test_existing_database_rows_preserved(tmp_path):
    """
    4. Existing database rows are preserved across repeated initialization runs.
    """
    test_db_path = tmp_path / "preserve_test.db"
    test_engine = create_engine(f"sqlite:///{test_db_path}")
    SessionFactory = sessionmaker(bind=test_engine)

    try:
        init_db(engine=test_engine)

        # Insert test records
        with SessionFactory() as db:
            product = Product(
                product_id="PRD-PRESERVE-001",
                product_name="Preserved Product",
                unit_price=49.99,
            )
            db.add(product)
            db.commit()

        # Run init_db again
        init_db(engine=test_engine)

        # Verify record is preserved
        with SessionFactory() as db:
            preserved = db.query(Product).filter(Product.product_id == "PRD-PRESERVE-001").first()
            assert preserved is not None
            assert preserved.product_name == "Preserved Product"
            assert preserved.unit_price == 49.99
    finally:
        test_engine.dispose()


def test_admin_seeding_idempotent_and_not_duplicated(db_session: Session):
    """
    5. Admin seeding is idempotent.
    6. Admin is not duplicated.
    """
    admin_pw = "test-bootstrap-pw-123"

    # First seeding run
    admin1 = seed_admin_user(
        db=db_session,
        admin_username="admin",
        admin_email="admin@smart-sales.local",
        admin_password=admin_pw,
    )
    assert admin1 is not None
    admin1_id = admin1.id

    # Second seeding run (idempotent; must return existing admin without duplicating)
    admin2 = seed_admin_user(
        db=db_session,
        admin_username="admin",
        admin_email="admin@smart-sales.local",
        admin_password=admin_pw,
    )
    assert admin2 is not None
    assert admin2.id == admin1_id

    # Count admin records in database
    admin_users = db_session.query(User).filter(User.email == "admin@smart-sales.local").all()
    assert len(admin_users) == 1


def test_admin_password_stored_as_hash_never_plaintext(db_session: Session):
    """
    7. Admin password is stored only as a password hash.
    8. Plaintext password is never stored.
    """
    admin_pw = "test-confidential-secret-999"

    admin = seed_admin_user(
        db=db_session,
        admin_username="admin",
        admin_email="admin@smart-sales.local",
        admin_password=admin_pw,
    )
    assert admin is not None

    # Verify stored value is a cryptographic hash, not plaintext
    assert admin.password_hash != admin_pw
    assert admin_pw not in admin.password_hash
    assert admin.password_hash.startswith("$argon2")

    # Verify password hash validates with verify_password
    assert verify_password(admin_pw, admin.password_hash) is True
    assert verify_password("incorrect-password", admin.password_hash) is False


def test_admin_login_username_resolution(client: TestClient, db_session: Session):
    """
    9. Username 'admin' resolves to the configured admin email.
    11. Admin login returns a valid JWT.
    12. /api/auth/me works with the JWT.
    """
    admin_pw = "test-admin-secret-pass"
    seed_admin_user(
        db=db_session,
        admin_username="admin",
        admin_email="admin@smart-sales.local",
        admin_password=admin_pw,
    )

    # 1. Login with username = 'admin' (OAuth2 password flow)
    login_response = client.post(
        "/api/auth/login",
        data={
            "username": "admin",
            "password": admin_pw,
        },
    )
    assert login_response.status_code == status.HTTP_200_OK
    payload = login_response.json()

    assert payload["token_type"] == "bearer"
    assert "access_token" in payload
    assert payload["user"]["email"] == "admin@smart-sales.local"

    # 2. Validate token claims
    token = payload["access_token"]
    decoded = decode_access_token(token)
    assert decoded["sub"] == "admin@smart-sales.local"

    # 3. Access /api/auth/me with the issued JWT
    me_response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_response.status_code == status.HTTP_200_OK
    me_data = me_response.json()
    assert me_data["email"] == "admin@smart-sales.local"


def test_admin_direct_email_login_works(client: TestClient, db_session: Session):
    """
    The administrator account can also log in directly using its full email address.
    """
    admin_pw = "test-direct-email-pass"
    seed_admin_user(
        db=db_session,
        admin_username="admin",
        admin_email="admin@smart-sales.local",
        admin_password=admin_pw,
    )

    response = client.post(
        "/api/auth/login",
        data={
            "username": "admin@smart-sales.local",
            "password": admin_pw,
        },
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["user"]["email"] == "admin@smart-sales.local"


def test_normal_email_login_still_works(client: TestClient):
    """
    10. Normal email login still works alongside admin login.
    """
    normal_email = "regular_executive@example.com"
    normal_password = "ExecutivePassword123!"

    # Register normal user
    reg_response = client.post(
        "/api/auth/register",
        json={"email": normal_email, "password": normal_password},
    )
    assert reg_response.status_code == status.HTTP_201_CREATED

    # Login as normal user
    login_response = client.post(
        "/api/auth/login",
        data={"username": normal_email, "password": normal_password},
    )
    assert login_response.status_code == status.HTTP_200_OK
    token = login_response.json()["access_token"]

    # Verify /api/auth/me
    me_response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me_response.status_code == status.HTTP_200_OK
    assert me_response.json()["email"] == normal_email


def test_invalid_admin_password_returns_401(client: TestClient, db_session: Session):
    """
    13. Invalid admin password returns 401 Unauthorized.
    """
    admin_pw = "test-correct-admin-pw"
    seed_admin_user(
        db=db_session,
        admin_username="admin",
        admin_email="admin@smart-sales.local",
        admin_password=admin_pw,
    )

    response = client.post(
        "/api/auth/login",
        data={
            "username": "admin",
            "password": "wrong-password-attempt",
        },
    )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Incorrect email or password" in response.json()["detail"]


def test_registration_validation_remains_unchanged(client: TestClient):
    """
    14. Existing registration validation remains unchanged.
    - Passwords shorter than 8 characters are strictly rejected (HTTP 422).
    - Public registration cannot hijack the reserved admin email.
    """
    # 1. Short password (< 8 chars) rejected by Pydantic validation
    short_res = client.post(
        "/api/auth/register",
        json={"email": "short_pw_user@example.com", "password": "admin"},
    )
    assert short_res.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # 2. Short password of 5 characters rejected
    short_res2 = client.post(
        "/api/auth/register",
        json={"email": "another_short@example.com", "password": "12345"},
    )
    assert short_res2.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    # 3. Attempt to publicly register reserved admin email is rejected
    admin_email_res = client.post(
        "/api/auth/register",
        json={"email": "admin@smart-sales.local", "password": "ValidPassword888!"},
    )
    assert admin_email_res.status_code == status.HTTP_409_CONFLICT

    # 4. Valid normal registration succeeds
    valid_res = client.post(
        "/api/auth/register",
        json={"email": "valid_user_policy@example.com", "password": "ValidPassword888!"},
    )
    assert valid_res.status_code == status.HTTP_201_CREATED


def test_admin_bootstrap_with_short_configured_credential(client: TestClient, db_session: Session):
    """
    Verifies that when ADMIN_PASSWORD is set to a bootstrap string (such as the requested 'admin'),
    the system bootstrap seeder hashes and stores it without error, and login with
    username='admin' and password='admin' succeeds, while public registration still rejects
    short passwords.
    """
    # Seed bootstrap credential directly
    bootstrap_pw = "admin"
    admin = seed_admin_user(
        db=db_session,
        admin_username="admin",
        admin_email="admin@smart-sales.local",
        admin_password=bootstrap_pw,
    )
    assert admin is not None

    # Login with username='admin' and password='admin' succeeds
    login_res = client.post(
        "/api/auth/login",
        data={"username": "admin", "password": bootstrap_pw},
    )
    assert login_res.status_code == status.HTTP_200_OK
    assert login_res.json()["user"]["email"] == "admin@smart-sales.local"
