from __future__ import annotations

from pathlib import Path

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


class BackendForecastService:
    """
    Backend wrapper around the production ML ForecastService.
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
    ) -> tuple[str, pd.DataFrame]:

        service = self._get_service(horizon)

        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: "
                f"{DATASET_PATH}"
            )

        df = pd.read_csv(
            DATASET_PATH,
            parse_dates=["Date"],
        )

        if df.empty:
            raise ValueError(
                "Processed forecasting dataset is empty."
            )

        df = df.sort_values("Date")

        history = pd.Series(
            df["Quantity"].astype(float).values
        )

        if len(history) < 28:
            raise ValueError(
                "At least 28 historical observations "
                "are required for forecasting."
            )

        last_date = pd.Timestamp(
            df["Date"].max()
        ).normalize()

        start_date = (
            last_date
            + pd.Timedelta(days=1)
        )

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
