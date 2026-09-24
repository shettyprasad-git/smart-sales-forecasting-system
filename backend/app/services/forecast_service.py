from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import CompanyModel
from backend.app.services.dataset_runtime_service import (
    dataset_runtime_service,
)
from backend.app.services.storage_service import storage_service
from ml.features.feature_pipeline import get_feature_columns
from ml.inference.forecast_service import (
    ForecastService as MLForecastService,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_CONFIG = {
    7: {
        "model_name": "Random Forest",
        "model_path": (
            PROJECT_ROOT
            / "ml"
            / "artifacts"
            / "demand_7d_random_forest.joblib"
        ),
    },
    30: {
        "model_name": "HistGradientBoosting",
        "model_path": (
            PROJECT_ROOT
            / "ml"
            / "artifacts"
            / "demand_30d_gradient_boosting.joblib"
        ),
    },
    90: {
        "model_name": "Linear Regression",
        "model_path": (
            PROJECT_ROOT
            / "ml"
            / "artifacts"
            / "demand_90d_linear_regression.joblib"
        ),
    },
}

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_daily_forecasting_features.csv"
)


class ForecastResult(tuple):
    """
    Two-element tuple (model_name, forecast_df) subclass that attaches rich
    provenance metadata while allowing seamless 2-element tuple unpacking:
    model_name, forecast_df = service.generate_forecast(...)
    """

    def __new__(
        cls,
        model_name: str,
        forecast_df: pd.DataFrame,
        source: str = "global",
        model_type: str | None = None,
        model_version: int | None = None,
        validation_wape: float | None = None,
        test_wape: float | None = None,
        fallback_reason: str | None = None,
    ):
        return super().__new__(cls, (model_name, forecast_df))

    def __init__(
        self,
        model_name: str,
        forecast_df: pd.DataFrame,
        source: str = "global",
        model_type: str | None = None,
        model_version: int | None = None,
        validation_wape: float | None = None,
        test_wape: float | None = None,
        fallback_reason: str | None = None,
    ):
        self.source = source
        self.model_type = model_type
        self.model_version = model_version
        self.validation_wape = validation_wape
        self.test_wape = test_wape
        self.fallback_reason = fallback_reason

    @property
    def model_name(self) -> str:
        return self[0]

    @property
    def forecast_df(self) -> pd.DataFrame:
        return self[1]


class BackendForecastService:
    """
    Backend wrapper around production forecasting services.
    Dynamically loads tenant company-specific models when available and in 'ready' status,
    with seamless fallback to production global pre-trained models.
    Inference requests NEVER trigger retraining.
    """

    def __init__(self) -> None:
        self._services: dict[int, MLForecastService] = {}

    def _get_service(
        self,
        horizon: int,
    ) -> MLForecastService:

        if horizon not in MODEL_CONFIG:
            raise ValueError(
                "Supported forecast horizons are 7, 30, and 90 days."
            )

        if horizon not in self._services:

            model_path = MODEL_CONFIG[
                horizon
            ]["model_path"]

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Production model artifact not found: "
                    f"{model_path}"
                )

            self._services[horizon] = MLForecastService(
                model_path=model_path
            )

        return self._services[horizon]

    def generate_forecast(
        self,
        horizon: int,
        events: pd.DataFrame | None = None,
        user_id: int | None = None,
        db: Any = None,
    ) -> ForecastResult:

        # 1. Fetch user daily aggregate
        df = dataset_runtime_service.get_daily_aggregate(
            user_id=user_id,
            db=db,
        )

        if df.empty:
            raise ValueError(
                "Historical forecasting dataset is empty."
            )

        df = df.sort_values("Date").reset_index(drop=True)

        history = pd.Series(
            df["Quantity"].astype(float).values
        )

        if len(history) < 28:
            raise ValueError(
                f"Insufficient historical data for forecasting: "
                f"dataset contains {len(history)} daily observations, "
                f"but at least 28 are required."
            )

        last_date = pd.Timestamp(
            df["Date"].max()
        ).normalize()

        start_date = (
            last_date
            + pd.Timedelta(days=1)
        )

        if events is None:
            events = pd.DataFrame(
                columns=[
                    "Promotions",
                    "Holiday_Flag",
                ]
            )
            events.index = pd.DatetimeIndex([])

        # 2. Resolve Model: Company Model -> Global Fallback
        service: MLForecastService
        display_name: str
        source: str
        model_type: str | None = None
        model_version: int | None = None
        validation_wape: float | None = None
        test_wape: float | None = None
        fallback_reason: str | None = None

        company_model: CompanyModel | None = None
        if user_id is not None and db is not None:
            try:
                stmt = (
                    select(CompanyModel)
                    .where(
                        CompanyModel.user_id == user_id,
                        CompanyModel.horizon == horizon,
                        CompanyModel.status == "ready",
                        CompanyModel.is_active == True,
                    )
                    .order_by(CompanyModel.created_at.desc())
                )
                company_model = db.scalars(stmt).first()
            except Exception as exc:
                logger.warning("Error querying company model for user_id=%s, horizon=%s: %s", user_id, horizon, exc)

        if company_model and company_model.artifact_path:
            try:
                # 1. Feature schema compatibility verification
                expected_feature_columns = get_feature_columns()
                if company_model.feature_version != "v1":
                    raise ValueError(
                        f"Incompatible feature schema version: '{company_model.feature_version}' (expected 'v1')"
                    )
                if company_model.horizon != horizon:
                    raise ValueError(
                        f"Horizon mismatch: model horizon={company_model.horizon}, requested={horizon}"
                    )
                if company_model.feature_columns and set(company_model.feature_columns) != set(expected_feature_columns):
                    raise ValueError(
                        f"Feature columns schema mismatch: expected {expected_feature_columns}, found {company_model.feature_columns}"
                    )

                cached_file = storage_service.load_artifact_file(company_model.artifact_path)
                service = MLForecastService(model_path=cached_file)

                # Verify loaded artifact internal schema if present
                if hasattr(service, "features") and service.features:
                    if set(service.features) != set(expected_feature_columns):
                        raise ValueError(
                            f"Artifact feature columns mismatch: expected {expected_feature_columns}, found {service.features}"
                        )

                display_name = f"Company Model ({company_model.model_type})"
                source = "company"
                model_type = company_model.model_type
                model_version = company_model.model_version
                validation_wape = company_model.validation_wape
                test_wape = company_model.test_wape
            except Exception as exc:
                logger.warning(
                    "Failed to load or validate company model artifact '%s' for user_id=%s, horizon=%s: %s. Falling back to global model.",
                    company_model.artifact_path,
                    user_id,
                    horizon,
                    exc,
                )
                service = self._get_service(horizon)
                display_name = MODEL_CONFIG[horizon]["model_name"]
                source = "global_fallback"
                fallback_reason = f"Company model schema/loading error: {exc}"
        else:
            service = self._get_service(horizon)
            display_name = MODEL_CONFIG[horizon]["model_name"]
            source = "global" if user_id is None else "global_fallback"
            if user_id is not None:
                if company_model and company_model.status == "insufficient_data":
                    fallback_reason = company_model.status_message or "Insufficient historical data for this horizon."
                else:
                    fallback_reason = "No ready company model found for this horizon."

        # 3. Perform recursive forecast (Inference NEVER retrains)
        forecast_df = service.forecast(
            history=history,
            start_date=start_date,
            horizon=horizon,
            events=events,
        )

        return ForecastResult(
            display_name,
            forecast_df,
            source=source,
            model_type=model_type,
            model_version=model_version,
            validation_wape=validation_wape,
            test_wape=test_wape,
            fallback_reason=fallback_reason,
        )
