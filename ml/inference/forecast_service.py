from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from ml.features.feature_pipeline import (
    build_recursive_feature_row,
    get_feature_columns,
)


class ForecastService:
    """
    Production forecasting service.

    Loads a trained model artifact and generates
    recursive multi-day demand forecasts.
    """

    def __init__(
        self,
        model_path: str | Path,
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {self.model_path}"
            )

        artifact = joblib.load(self.model_path)

        # -----------------------------------------------------
        # Load production artifact
        # -----------------------------------------------------

        if isinstance(artifact, dict):

            if "model" not in artifact:
                raise ValueError(
                    "Model artifact does not contain "
                    "a 'model' entry."
                )

            self.model = artifact["model"]

            self.features = artifact.get(
                "features",
                get_feature_columns(),
            )

            self.target = artifact.get(
                "target",
                "Quantity",
            )

            self.horizon = artifact.get(
                "horizon",
                None,
            )

            self.model_name = artifact.get(
                "model_name",
                self.model.__class__.__name__,
            )

            # Critical for days_since_start.
            self.reference_start_date = artifact.get(
                "reference_start_date",
                None,
            )

            if self.reference_start_date is None:
                raise ValueError(
                    "Model artifact does not contain "
                    "'reference_start_date'. "
                    "Retrain the production models."
                )

            self.reference_start_date = pd.Timestamp(
                self.reference_start_date
            )

        else:

            # Backward compatibility for raw model artifacts.
            self.model = artifact
            self.features = get_feature_columns()
            self.target = "Quantity"
            self.horizon = None
            self.model_name = (
                self.model.__class__.__name__
            )

            raise ValueError(
                "Raw model artifact detected. "
                "Production artifacts must contain "
                "'reference_start_date'. "
                "Retrain the production models."
            )

    def forecast(
        self,
        history: pd.Series | list[float],
        start_date: str | pd.Timestamp,
        horizon: int,
        events: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate a recursive demand forecast.

        Parameters
        ----------
        history:
            Historical Quantity observations.

        start_date:
            First forecast date.

        horizon:
            Number of future days.

        events:
            Optional DataFrame indexed by Date containing:

                Promotions
                Holiday_Flag
        """

        if horizon <= 0:
            raise ValueError(
                "Horizon must be greater than zero."
            )

        history = pd.Series(
            history,
            dtype=float,
        ).reset_index(drop=True)

        if len(history) < 28:
            raise ValueError(
                "At least 28 historical observations "
                "are required."
            )

        start_date = pd.Timestamp(
            start_date
        ).normalize()

        # -----------------------------------------------------
        # Validate forecast date against training reference
        # -----------------------------------------------------

        if start_date < self.reference_start_date:
            raise ValueError(
                "Forecast start date cannot be earlier "
                "than the model reference start date."
            )

        # -----------------------------------------------------
        # Prepare future event data
        # -----------------------------------------------------

        if events is None:

            events = pd.DataFrame(
                columns=[
                    "Promotions",
                    "Holiday_Flag",
                ]
            )

            events.index = pd.DatetimeIndex([])

        else:

            events = events.copy()

            if not isinstance(
                events.index,
                pd.DatetimeIndex,
            ):
                events.index = pd.to_datetime(
                    events.index
                )

            events.index = events.index.normalize()

            if "Promotions" not in events.columns:
                events["Promotions"] = 0

            if "Holiday_Flag" not in events.columns:
                events["Holiday_Flag"] = 0

        # -----------------------------------------------------
        # Recursive forecasting
        # -----------------------------------------------------

        predictions = []

        for step in range(horizon):

            forecast_date = (
                start_date
                + pd.Timedelta(days=step)
            )

            # -------------------------------------------------
            # Event values
            # -------------------------------------------------

            if forecast_date in events.index:

                promotion = events.loc[
                    forecast_date,
                    "Promotions",
                ]

                holiday = events.loc[
                    forecast_date,
                    "Holiday_Flag",
                ]

                # Handle duplicate event dates safely.
                if isinstance(
                    promotion,
                    pd.Series,
                ):
                    promotion = promotion.iloc[0]

                if isinstance(
                    holiday,
                    pd.Series,
                ):
                    holiday = holiday.iloc[0]

            else:

                promotion = 0
                holiday = 0

            # -------------------------------------------------
            # Build features
            # -------------------------------------------------

            feature_row = build_recursive_feature_row(
                history=history,
                future_date=forecast_date,
                promotion=promotion,
                holiday=holiday,
                reference_start_date=(
                    self.reference_start_date
                ),
            )

            # Use the exact feature list stored
            # inside the model artifact.
            X = feature_row[self.features]

            # -------------------------------------------------
            # Prediction
            # -------------------------------------------------

            prediction = float(
                self.model.predict(X)[0]
            )

            # Demand cannot be negative.
            prediction = max(
                0.0,
                prediction,
            )

            predictions.append(
                {
                    "Date": forecast_date,
                    "Predicted_Quantity": prediction,
                }
            )

            # -------------------------------------------------
            # Feed prediction back into history
            # -------------------------------------------------

            history = pd.concat(
                [
                    history,
                    pd.Series([prediction]),
                ],
                ignore_index=True,
            )

        return pd.DataFrame(
            predictions
        )


def forecast_from_model(
    model_path: str | Path,
    history: pd.Series | list[float],
    start_date: str | pd.Timestamp,
    horizon: int,
    events: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Convenience function for generating forecasts.
    """

    service = ForecastService(
        model_path
    )

    return service.forecast(
        history=history,
        start_date=start_date,
        horizon=horizon,
        events=events,
    )
