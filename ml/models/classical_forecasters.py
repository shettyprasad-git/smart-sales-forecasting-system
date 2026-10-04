from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt


def _validate_history(
    history: Sequence[float] | np.ndarray | pd.Series,
    min_observations: int,
    model_name: str,
) -> np.ndarray:
    """
    Validate input historical observations, ensuring finite numbers and sufficient length.
    Guarantees no in-place mutation of the caller's data structures.
    """
    if history is None:
        raise ValueError(f"{model_name} requires a non-None historical series.")

    if isinstance(history, pd.Series):
        arr = history.to_numpy(dtype=float, copy=True)
    elif isinstance(history, np.ndarray):
        arr = history.astype(float, copy=True).flatten()
    else:
        try:
            arr = np.array(list(history), dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{model_name} history contains invalid non-numeric elements: {exc}") from exc

    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError(f"{model_name} history cannot be empty and must be 1-dimensional.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{model_name} history contains NaN or infinite values.")

    if len(arr) < min_observations:
        raise ValueError(
            f"{model_name} requires at least {min_observations} historical observations, "
            f"but received only {len(arr)}."
        )

    return arr


def _validate_steps(steps: int, model_name: str) -> int:
    """Validate forecast horizon length."""
    if not isinstance(steps, (int, np.integer)):
        raise ValueError(f"{model_name} forecast steps must be an integer, got {type(steps).__name__}.")
    if steps < 0:
        raise ValueError(f"{model_name} forecast steps must be non-negative, got {steps}.")
    return int(steps)


class HoltAdditiveModel:
    """
    Holt linear exponential smoothing with additive trend.

    Model:
      level:  l_t = alpha * y_t + (1 - alpha) * (l_{t-1} + b_{t-1})
      trend:  b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
      forecast: y_hat_{t+h} = max(0.0, l_t + h * b_t)

    Fits strictly on historical observations provided to each forecast() call.
    Applies non-negative clipping on final demand forecasts: max(0.0, prediction).
    """

    name: str = "Holt Additive"

    def __init__(self, initialization_method: str = "estimated"):
        self.name = "Holt Additive"
        self.initialization_method = initialization_method

    def forecast(
        self,
        history: Sequence[float] | np.ndarray | pd.Series,
        steps: int,
    ) -> list[float]:
        steps = _validate_steps(steps, self.name)
        if steps == 0:
            return []

        # Minimum 3 observations required for level + trend slope initialization
        y = _validate_history(history, min_observations=3, model_name=self.name)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = Holt(
                    y,
                    exponential=False,
                    damped_trend=False,
                    initialization_method=self.initialization_method,
                )
                fitted = model.fit(optimized=True)
                raw_preds = fitted.forecast(steps)
        except Exception as exc:
            raise ValueError(f"Failed to fit {self.name} on history (N={len(y)}): {exc}") from exc

        preds = np.atleast_1d(raw_preds)
        if not np.all(np.isfinite(preds)):
            raise ValueError(f"{self.name} generated non-finite predictions: {preds}")

        return [max(0.0, float(val)) for val in preds]

    def __repr__(self) -> str:
        return f"HoltAdditiveModel(name='{self.name}', initialization_method='{self.initialization_method}')"


class HoltWintersAdditiveModel:
    """
    Holt-Winters exponential smoothing with additive trend and additive weekly seasonality.

    Model:
      level:    l_t = alpha * (y_t - s_{t-m}) + (1 - alpha) * (l_{t-1} + b_{t-1})
      trend:    b_t = beta * (l_t - l_{t-1}) + (1 - beta) * b_{t-1}
      seasonal: s_t = gamma * (y_t - l_{t-1} - b_{t-1}) + (1 - gamma) * s_{t-m}
      forecast: y_hat_{t+h} = max(0.0, l_t + h * b_t + s_{t+h-m*(k+1)})

    Fits strictly on historical observations provided to each forecast() call.
    Applies non-negative clipping on final demand forecasts: max(0.0, prediction).
    """

    name: str = "Holt-Winters Additive Weekly"

    def __init__(
        self,
        seasonal_periods: int = 7,
        initialization_method: str = "estimated",
    ):
        if seasonal_periods <= 1:
            raise ValueError(f"seasonal_periods must be > 1, got {seasonal_periods}.")
        self.name = "Holt-Winters Additive Weekly"
        self.seasonal_periods = seasonal_periods
        self.initialization_method = initialization_method

    def forecast(
        self,
        history: Sequence[float] | np.ndarray | pd.Series,
        steps: int,
    ) -> list[float]:
        steps = _validate_steps(steps, self.name)
        if steps == 0:
            return []

        # Minimum 2 full seasonal cycles required for seasonal pattern estimation
        min_obs = 2 * self.seasonal_periods
        y = _validate_history(history, min_observations=min_obs, model_name=self.name)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    y,
                    trend="add",
                    damped_trend=False,
                    seasonal="add",
                    seasonal_periods=self.seasonal_periods,
                    initialization_method=self.initialization_method,
                )
                fitted = model.fit(optimized=True)
                raw_preds = fitted.forecast(steps)
        except Exception as exc:
            raise ValueError(f"Failed to fit {self.name} on history (N={len(y)}): {exc}") from exc

        preds = np.atleast_1d(raw_preds)
        if not np.all(np.isfinite(preds)):
            raise ValueError(f"{self.name} generated non-finite predictions: {preds}")

        return [max(0.0, float(val)) for val in preds]

    def __repr__(self) -> str:
        return (
            f"HoltWintersAdditiveModel(name='{self.name}', "
            f"seasonal_periods={self.seasonal_periods}, "
            f"initialization_method='{self.initialization_method}')"
        )


class DampedHoltWintersModel:
    """
    Damped Holt-Winters exponential smoothing with additive trend and additive weekly seasonality.

    Damps trend growth by factor phi in (0, 1):
      level:    l_t = alpha * (y_t - s_{t-m}) + (1 - alpha) * (l_{t-1} + phi * b_{t-1})
      trend:    b_t = beta * (l_t - l_{t-1}) + (1 - beta) * phi * b_{t-1}
      seasonal: s_t = gamma * (y_t - l_{t-1} - phi * b_{t-1}) + (1 - gamma) * s_{t-m}
      forecast: y_hat_{t+h} = max(0.0, l_t + sum_{i=1}^h phi^i * b_t + s_{t+h-m*(k+1)})

    Fits strictly on historical observations provided to each forecast() call.
    Applies non-negative clipping on final demand forecasts: max(0.0, prediction).
    """

    name: str = "Damped Holt-Winters Additive Weekly"

    def __init__(
        self,
        seasonal_periods: int = 7,
        initialization_method: str = "estimated",
    ):
        if seasonal_periods <= 1:
            raise ValueError(f"seasonal_periods must be > 1, got {seasonal_periods}.")
        self.name = "Damped Holt-Winters Additive Weekly"
        self.seasonal_periods = seasonal_periods
        self.initialization_method = initialization_method

    def forecast(
        self,
        history: Sequence[float] | np.ndarray | pd.Series,
        steps: int,
    ) -> list[float]:
        steps = _validate_steps(steps, self.name)
        if steps == 0:
            return []

        min_obs = 2 * self.seasonal_periods
        y = _validate_history(history, min_observations=min_obs, model_name=self.name)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ExponentialSmoothing(
                    y,
                    trend="add",
                    damped_trend=True,
                    seasonal="add",
                    seasonal_periods=self.seasonal_periods,
                    initialization_method=self.initialization_method,
                )
                fitted = model.fit(optimized=True)
                raw_preds = fitted.forecast(steps)
        except Exception as exc:
            raise ValueError(f"Failed to fit {self.name} on history (N={len(y)}): {exc}") from exc

        preds = np.atleast_1d(raw_preds)
        if not np.all(np.isfinite(preds)):
            raise ValueError(f"{self.name} generated non-finite predictions: {preds}")

        return [max(0.0, float(val)) for val in preds]

    def __repr__(self) -> str:
        return (
            f"DampedHoltWintersModel(name='{self.name}', "
            f"seasonal_periods={self.seasonal_periods}, "
            f"initialization_method='{self.initialization_method}')"
        )


def forecast_holt_additive(
    history: Sequence[float] | np.ndarray | pd.Series,
    steps: int,
) -> list[float]:
    """Convenience function for Holt Additive forecasting."""
    return HoltAdditiveModel().forecast(history=history, steps=steps)


def forecast_holt_winters_additive(
    history: Sequence[float] | np.ndarray | pd.Series,
    steps: int,
    seasonal_periods: int = 7,
) -> list[float]:
    """Convenience function for Holt-Winters Additive Weekly forecasting."""
    return HoltWintersAdditiveModel(seasonal_periods=seasonal_periods).forecast(history=history, steps=steps)


def forecast_damped_holt_winters(
    history: Sequence[float] | np.ndarray | pd.Series,
    steps: int,
    seasonal_periods: int = 7,
) -> list[float]:
    """Convenience function for Damped Holt-Winters Additive Weekly forecasting."""
    return DampedHoltWintersModel(seasonal_periods=seasonal_periods).forecast(history=history, steps=steps)


__all__ = [
    "HoltAdditiveModel",
    "HoltWintersAdditiveModel",
    "DampedHoltWintersModel",
    "forecast_holt_additive",
    "forecast_holt_winters_additive",
    "forecast_damped_holt_winters",
]
