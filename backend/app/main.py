import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.anomalies import router as anomalies_router
from backend.app.api.auth import router as auth_router
from backend.app.api.forecast import router as forecast_router
from backend.app.api.investigations import router as investigations_router
from backend.app.api.products import router as products_router
from backend.app.api.sales import router as sales_router
from backend.app.core.exceptions import global_exception_handler
from backend.app.core.logging_config import configure_logging


configure_logging()

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Smart Sales Forecasting API",
    description=(
        "Production backend API for the "
        "Smart Sales Forecasting System."
    ),
    version="1.0.0",
)


app.add_exception_handler(
    Exception,
    global_exception_handler,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next,
):
    start_time = time.perf_counter()

    logger.info(
        "Request started: %s %s",
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "Request completed: %s %s | status=%s | time=%.2fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

        return response

    except Exception:
        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.exception(
            "Request failed: %s %s | time=%.2fms",
            request.method,
            request.url.path,
            elapsed_ms,
        )

        raise


app.include_router(anomalies_router)
app.include_router(investigations_router)
app.include_router(auth_router)
app.include_router(forecast_router)
app.include_router(products_router)
app.include_router(sales_router)


@app.get("/")
def root():
    return {
        "message": "Smart Sales Forecasting API is running",
        "version": "1.0.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
