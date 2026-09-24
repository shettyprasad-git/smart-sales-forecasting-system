from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml.inference.forecast_service import (
    ForecastService as MLForecastService,
)


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


from backend.app.services.dataset_runtime_service import (
    dataset_runtime_service,
)


class BackendForecastService:
    """
    Backend wrapper around the production ML ForecastService.
    Loads active user dataset via DatasetRuntimeService with user-isolation.
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
    ) -> tuple[str, pd.DataFrame]:

        service = self._get_service(horizon)

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

        forecast_df = service.forecast(
            history=history,
            start_date=start_date,
            horizon=horizon,
            events=events,
        )

        return (
            MODEL_CONFIG[horizon]["model_name"],
            forecast_df,
        )
