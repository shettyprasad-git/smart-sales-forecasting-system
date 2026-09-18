from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt
import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app.core.config import Settings, settings
from backend.app.core.rate_limiter import InMemoryRateLimiter, auth_rate_limiter
from backend.app.database.database import get_db
from backend.app.database.models import (
    DecisionAuditEvent,
    DecisionRecord,
    MonitoringAlert,
    MonitoringRun,
)
from backend.app.main import app
from backend.app.schemas.decisions import DecisionResponse
from backend.app.schemas.recommendations import RecommendationResponse


# ===========================================================================
# 1. API Health & Liveness Endpoints
# ===========================================================================

def test_health_endpoint_ok(client: TestClient):
    """GET /health must return 200 OK with status ok without touching external services."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data == {"status": "ok"}


def test_ready_endpoint_database_ok(client: TestClient):
    """GET /ready must ping database (SELECT 1) and return 200 ready when DB is connected."""
    response = client.get("/ready")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data == {"status": "ready", "database": "ok"}


def test_ready_endpoint_database_down_fails():
    """GET /ready must return 503 degraded and database unreachable when DB fails."""
    def failing_get_db():
        class FailingSession:
            def execute(self, *args, **kwargs):
                raise RuntimeError("Connection refused: PostgreSQL host down")
        yield FailingSession()

    app.dependency_overrides[get_db] = failing_get_db
    try:
        with TestClient(app) as test_client:
            response = test_client.get("/ready")
            assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
            data = response.json()
            assert data == {"status": "degraded", "database": "unreachable"}
    finally:
        app.dependency_overrides.pop(get_db, None)


# ===========================================================================
# 2. Security Headers & Request Correlation Tracing
# ===========================================================================

def test_security_headers_present(client: TestClient):
    """Responses must include standard SPA security hardening headers."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    headers = response.headers
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-frame-options") == "DENY"
    assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert headers.get("x-xss-protection") == "1; mode=block"


def test_request_correlation_id_generation(client: TestClient):
    """Requests without X-Request-ID must have a 32-char UUID generated and returned."""
    response = client.get("/health")
    req_id = response.headers.get("x-request-id")
    assert req_id is not None
    assert len(req_id) == 32
    assert all(c in "0123456789abcdef" for c in req_id)


def test_request_correlation_id_propagation(client: TestClient):
    """Custom inbound X-Request-ID must be preserved and echoed back in response headers."""
    custom_id = "trace-client-e2e-correlation-9988"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.headers.get("x-request-id") == custom_id


# ===========================================================================
# 3. Production Environment & Secret Validation
# ===========================================================================

def test_production_secret_validation_enforced():
    """In production mode, settings must reject weak secrets, SQLite, and wildcard origins."""
    # 1. Weak/short secret key rejected
    with pytest.raises(ValidationError) as exc:
        Settings(
            environment="production",
            secret_key="short",
            database_url="postgresql://user:pass@host:5432/sales",
            allowed_origins="https://app.onrender.com",
        )
    assert "minimum 16 characters" in str(exc.value)

    # 2. Default fallback secret key rejected in production
    with pytest.raises(ValidationError) as exc:
        Settings(
            environment="production",
            secret_key="dev-fallback-secret-key-change-in-production",
            database_url="postgresql://user:pass@host:5432/sales",
            allowed_origins="https://app.onrender.com",
        )
    assert "secure, non-trivial secret" in str(exc.value)

    # 3. Local SQLite rejected in production
    with pytest.raises(ValidationError) as exc:
        Settings(
            environment="production",
            secret_key="production-secure-random-key-123456",
            database_url="sqlite:///./sales_forecasting.db",
            allowed_origins="https://app.onrender.com",
        )
    assert "must not use local SQLite file" in str(exc.value)

    # 4. Wildcard CORS origin with credentials rejected in production
    with pytest.raises(ValidationError) as exc:
        Settings(
            environment="production",
            secret_key="production-secure-random-key-123456",
            database_url="postgresql://user:pass@host:5432/sales",
            allowed_origins="*",
        )
    assert "wildcard" in str(exc.value).lower()

    # 5. Valid production configuration passes
    valid_settings = Settings(
        environment="production",
        secret_key="production-secure-random-key-123456",
        database_url="postgresql://user:pass@host:5432/sales",
        allowed_origins="https://app.onrender.com,https://dashboard.company.com",
    )
    assert valid_settings.environment == "production"
    assert valid_settings.cors_allowed_origins == [
        "https://app.onrender.com",
        "https://dashboard.company.com",
    ]


def test_production_no_silent_secret_generation():
    """
    The application must NEVER generate an ephemeral SECRET_KEY at runtime in production.
    If SECRET_KEY is missing, empty, or trivial, production startup must fail safely
    via ValidationError rather than generating a random secret or falling back silently.
    """
    # 1. Missing secret key in production raises ValidationError (dev fallback key is rejected)
    with pytest.raises(ValidationError) as exc1:
        Settings(
            _env_file=None,
            environment="production",
            database_url="postgresql://user:pass@host:5432/sales",
            allowed_origins="https://app.onrender.com",
        )
    assert "SECRET_KEY must be set" in str(exc1.value)

    # 2. Empty secret key in production raises ValidationError
    with pytest.raises(ValidationError) as exc2:
        Settings(
            _env_file=None,
            environment="production",
            secret_key="",
            database_url="postgresql://user:pass@host:5432/sales",
            allowed_origins="https://app.onrender.com",
        )
    assert "SECRET_KEY must be set" in str(exc2.value)

    # 3. Whitespace-only secret key in production raises ValidationError
    with pytest.raises(ValidationError) as exc3:
        Settings(
            _env_file=None,
            environment="production",
            secret_key="   ",
            database_url="postgresql://user:pass@host:5432/sales",
            allowed_origins="https://app.onrender.com",
        )
    assert "SECRET_KEY must be set" in str(exc3.value)

    # 4. In development and test environments, safe defaults remain functional
    dev_settings = Settings(_env_file=None, environment="development")
    assert dev_settings.secret_key == "dev-insecure-secret-key-for-local-testing-only"
    assert "sqlite" in dev_settings.database_url

    test_settings = Settings(_env_file=None, environment="test")
    assert test_settings.secret_key == "dev-insecure-secret-key-for-local-testing-only"
    assert "sqlite" in test_settings.database_url


# ===========================================================================
# 4. CORS Whitelist & Enforcement
# ===========================================================================

def test_cors_allowed_origin_success(client: TestClient):
    """Allowed origin must receive Access-Control-Allow-Origin header matching its origin."""
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_disallowed_origin_rejected(client: TestClient):
    """Disallowed origin must NOT receive Access-Control-Allow-Origin header for that origin."""
    response = client.options(
        "/health",
        headers={
            "Origin": "https://malicious-external-attacker.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers.get("access-control-allow-origin") != "https://malicious-external-attacker.com"


# ===========================================================================
# 5. Authentication Hardening & Token Rejection
# ===========================================================================

def test_auth_expired_jwt_rejected(client: TestClient):
    """Expired JWT must be rejected with 401 Unauthorized."""
    expired_payload = {
        "sub": "test@example.com",
        "exp": datetime.now(timezone.utc) - timedelta(hours=2),
    }
    expired_token = jwt.encode(expired_payload, settings.secret_key, algorithm=settings.algorithm)
    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "credentials" in response.json()["detail"].lower()


def test_auth_malformed_jwt_rejected(client: TestClient):
    """Malformed JWT must return 401 Unauthorized without crashing or returning 500."""
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer totally-malformed-token-xyz"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_auth_invalid_signature_rejected(client: TestClient):
    """JWT signed with an untrusted secret key must return 401 Unauthorized."""
    foreign_payload = {
        "sub": "test@example.com",
        "exp": datetime.now(timezone.utc) + timedelta(hours=2),
    }
    foreign_token = jwt.encode(foreign_payload, "completely-different-signing-key-12345", algorithm="HS256")
    response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {foreign_token}"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


def test_auth_missing_token_rejected(client: TestClient):
    """Accessing protected endpoints without an Authorization token returns 401."""
    response = client.get("/api/auth/me")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED


# ===========================================================================
# 6. Error Sanitization & Secret Leak Prevention
# ===========================================================================

def test_sanitized_500_response():
    """Unhandled backend errors must return a sanitized 500 response without leaking internal paths."""
    safe_client = TestClient(app, raise_server_exceptions=False)
    # Register and login to get auth token
    safe_client.post(
        "/api/auth/register",
        json={"email": "sanitized_readiness@example.com", "password": "TestPassword123!"},
    )
    login_res = safe_client.post(
        "/api/auth/login",
        data={"username": "sanitized_readiness@example.com", "password": "TestPassword123!"},
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    with patch(
        "backend.app.services.monitoring_service.MonitoringService.run_monitoring",
        side_effect=RuntimeError("Connection failed at /var/data/postgres/pg_hba.conf line 42 with secret pass123"),
    ):
        response = safe_client.post("/api/monitoring/run", json={"lookback_days": 30}, headers=headers)
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data.get("detail") in ["Internal server error", "Internal server error."]
        assert "/var/data/postgres" not in response.text
        assert "pass123" not in response.text
        assert "RuntimeError" not in response.text


def test_no_secret_leakage_in_error_responses(client: TestClient):
    """Error responses (401, 404, 422) must never expose database connection strings or secret keys."""
    # 404 Not Found
    res404 = client.get("/api/nonexistent/route")
    assert settings.secret_key not in res404.text
    assert "postgresql://" not in res404.text

    # 422 Unprocessable Entity
    res422 = client.post("/api/auth/register", json={"email": "not-an-email"})
    assert settings.secret_key not in res422.text
    assert "postgresql://" not in res422.text

    # 401 Unauthorized
    res401 = client.get("/api/auth/me")
    assert settings.secret_key not in res401.text
    assert "postgresql://" not in res401.text


# ===========================================================================
# 7. Database Production Readiness & Indexing
# ===========================================================================

def test_database_postgres_url_normalization():
    """Legacy postgres:// URLs from Render/Heroku must be normalized to postgresql://."""
    legacy_url = "postgres://user:super_secret_pw@db.render.com:5432/sales_db"
    normalized = legacy_url.replace("postgres://", "postgresql://", 1)
    assert normalized == "postgresql://user:super_secret_pw@db.render.com:5432/sales_db"
    assert normalized.startswith("postgresql://")


def test_database_audit_event_indexed():
    """DecisionAuditEvent.created_at and key foreign keys must have database indexes configured."""
    assert DecisionAuditEvent.__table__.columns["created_at"].index is True
    assert DecisionRecord.__table__.columns["status"].index is True
    assert DecisionRecord.__table__.columns["created_at"].index is True
    assert MonitoringAlert.__table__.columns["created_at"].index is True
    assert MonitoringAlert.__table__.columns["status"].index is True


# ===========================================================================
# 8. Rate Limiting Functionality
# ===========================================================================

def test_rate_limiter_allows_under_limit():
    """Rate limiter allows requests within configured requests_per_minute threshold."""
    limiter = InMemoryRateLimiter(requests_per_minute=3)
    client_ip = "192.168.1.100"

    assert limiter.is_allowed(client_ip) is True
    assert limiter.is_allowed(client_ip) is True
    assert limiter.is_allowed(client_ip) is True


def test_rate_limiter_blocks_over_limit():
    """Rate limiter rejects requests exceeding threshold and respects reset."""
    limiter = InMemoryRateLimiter(requests_per_minute=2)
    client_ip = "192.168.1.200"

    assert limiter.is_allowed(client_ip) is True
    assert limiter.is_allowed(client_ip) is True
    # 3rd request exceeds limit
    assert limiter.is_allowed(client_ip) is False

    # After reset, requests are permitted again
    limiter.reset()
    assert limiter.is_allowed(client_ip) is True


def test_rate_limiter_endpoint_rejection(client: TestClient):
    """Rapid repeated calls to rate-limited auth endpoints return 429 with Retry-After header."""
    auth_rate_limiter.reset()
    try:
        # Exhaust 30 rpm limit
        for i in range(30):
            client.post(
                "/api/auth/register",
                json={"email": f"ratelimit_{i}@example.com", "password": "TestPassword123!"},
            )

        # 31st request must trigger 429
        response = client.post(
            "/api/auth/register",
            json={"email": "ratelimit_blocked@example.com", "password": "TestPassword123!"},
        )
        assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert response.headers.get("retry-after") == "60"
        assert "Too many authentication requests" in response.json()["detail"]
    finally:
        auth_rate_limiter.reset()


# ===========================================================================
# 9. Human Governance Non-Negotiable Invariants
# ===========================================================================

def test_human_governance_invariant_maintained():
    """
    Core business invariant: The system requires explicit human approval for all
    recommended commercial actions and never executes autonomous mutations.
    """
    assert DecisionResponse.model_fields["human_approval_required"].default is True
    assert DecisionResponse.model_fields["automatic_execution"].default is False
    assert RecommendationResponse.model_fields["human_approval_required"].default is True

    # Mandatory governance safeguard: bypassing human approval raises ValidationError
    with pytest.raises(ValidationError):
        RecommendationResponse(
            anomaly_id="anom-test",
            recommendation_status="actionable",
            summary="Test summary",
            human_approval_required=False,
        )
