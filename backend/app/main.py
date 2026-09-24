from contextlib import asynccontextmanager
import logging
import re
import time
import uuid

from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.api.ai_reasoning import router as ai_reasoning_router
from backend.app.api.anomalies import router as anomalies_router
from backend.app.api.auth import router as auth_router
from backend.app.api.datasets import router as datasets_router
from backend.app.api.decisions import router as decisions_router
from backend.app.api.explanations import router as explanations_router
from backend.app.api.forecast import router as forecast_router
from backend.app.api.investigations import router as investigations_router
from backend.app.api.monitoring import router as monitoring_router
from backend.app.api.products import router as products_router
from backend.app.api.recommendations import router as recommendations_router
from backend.app.api.sales import router as sales_router
from backend.app.api.simulations import router as simulations_router
from backend.app.core.config import settings
from backend.app.core.exceptions import global_exception_handler
from backend.app.core.logging_config import configure_logging
from backend.app.database.database import get_db
from backend.app.database.init_db import init_database_and_seed

configure_logging()

logger = logging.getLogger(__name__)

CORRELATION_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes database tables idempotently and seeds the bootstrap administrator
    account before processing incoming requests.
    """
    try:
        init_database_and_seed()
    except Exception as exc:
        logger.warning(
            "Database initialization encountered an error during startup: %s. "
            "Continuing application startup; database connectivity will be verified by /ready.",
            exc,
        )
    yield


app = FastAPI(
    title="Smart Sales Forecasting API",
    description=(
        "Production backend API for the "
        "Smart Sales Forecasting System."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_exception_handler(
    Exception,
    global_exception_handler,
)

# Enterprise CORS: read from settings, rejecting wildcard '*' when credentials are active
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_lifecycle_middleware(
    request: Request,
    call_next,
):
    """
    HTTP middleware enforcing request correlation tracing, structured logging,
    and foundational security headers.
    """
    incoming_req_id = request.headers.get("X-Request-ID")
    if incoming_req_id and CORRELATION_ID_REGEX.match(incoming_req_id):
        request_id = incoming_req_id
    else:
        request_id = uuid.uuid4().hex

    request.state.request_id = request_id
    start_time = time.perf_counter()

    logger.info(
        "[%s] Request started: %s %s",
        request_id,
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "[%s] Request completed: %s %s | status=%s | time=%.2fms",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
    except Exception:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "[%s] Request failed: %s %s | time=%.2fms",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise

    # Security headers (compatible with browser Single Page Applications)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    return response


app.include_router(anomalies_router)
app.include_router(investigations_router)
app.include_router(explanations_router)
app.include_router(ai_reasoning_router)
app.include_router(recommendations_router)
app.include_router(auth_router)
app.include_router(forecast_router)
app.include_router(products_router)
app.include_router(sales_router)
app.include_router(datasets_router)
app.include_router(simulations_router)
app.include_router(decisions_router)
app.include_router(monitoring_router)


@app.get(
    "/",
    summary="Root API Info",
    tags=["System"],
)
def root():
    return {
        "message": "Smart Sales Forecasting API is running",
        "version": "1.0.0",
        "environment": settings.environment,
    }


@app.get(
    "/health",
    summary="Liveness Health Probe",
    tags=["System"],
)
def health_check():
    """
    Lightweight liveness probe indicating backend HTTP process is active.
    Returns 200 OK without touching external services.
    """
    return {
        "status": "ok",
    }


@app.get(
    "/ready",
    summary="Readiness Health Probe",
    tags=["System"],
)
def readiness_check(db: Session = Depends(get_db)):
    """
    Readiness probe verifying operational relational database connectivity.
    Executes a fast ping query. Does NOT call external Gemini LLM APIs.
    """
    try:
        db.execute(text("SELECT 1"))
        return {
            "status": "ready",
            "database": "ok",
        }
    except Exception as exc:
        logger.error("Readiness check database probe failed: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "degraded",
                "database": "unreachable",
            },
        )
