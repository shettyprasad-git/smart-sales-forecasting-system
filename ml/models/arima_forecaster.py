from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def minimum_history_for_order(order: tuple[int, int, int]) -> int:
    """
    Derive the minimum required historical observations for an ARIMA(p, d, q) model.

    Formula:
        min_history = max(d + 2, d + p + q + 2)

    Theoretical Justification:
      1. Differencing (d): Differencing d times consumes d observations, leaving N - d
         observations in the stationary differenced series. To have at least one usable
         differenced value and one lagged state, N - d >= 2, so N >= d + 2.
      2. Estimation Degrees of Freedom: Estimating p autoregressive (AR) parameters and
         q moving average (MA) parameters plus the innovation variance requires at least
         (p + q + 1) points in the differenced series, with at least 1 degree of freedom
         for residual variance estimation. Thus:
             (N - d) >= p + q + 2  ==>  N >= d + p + q + 2.
      3. Numerical Stability: For all candidate orders in the grid (up to p=2, d=1, q=2),
         this provides between 3 and 7 observations, guaranteeing sufficient data for
         Kalman filter state-space initialization and numerical convergence.
    """
    p, d, q = order
    return max(d + 2, d + p + q + 2)


def _validate_order(order: Any) -> tuple[int, int, int]:
    """Validate ARIMA (p, d, q) order tuple."""
    if not isinstance(order, (tuple, list)) or len(order) != 3:
        raise ValueError(f"ARIMA order must be a 3-element tuple (p, d, q), got {order}.")

    p, d, q = order
    for name, val in [("p", p), ("d", d), ("q", q)]:
        if not isinstance(val, (int, np.integer)) or isinstance(val, bool):
            raise ValueError(f"ARIMA order parameter '{name}' must be an integer, got {type(val).__name__} ({val}).")
        if val < 0:
            raise ValueError(f"ARIMA order parameter '{name}' must be non-negative, got {val}.")

    return int(p), int(d), int(q)


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
    if not isinstance(steps, (int, np.integer)) or isinstance(steps, bool):
        raise ValueError(f"{model_name} forecast steps must be an integer, got {type(steps).__name__}.")
    if steps < 0:
        raise ValueError(f"{model_name} forecast steps must be non-negative, got {steps}.")
    return int(steps)


class ARIMAForecaster:
    """
    Classical Autoregressive Integrated Moving Average (ARIMA) forecasting model.

    Model Formulation:
      ARIMA(p, d, q) models the d-th difference of the series y_t:
        (1 - sum_{i=1}^p phi_i B^i) (1 - B)^d y_t = c + (1 + sum_{j=1}^q theta_j B^j) epsilon_t

      Where:
        - p: order of the autoregressive (AR) polynomial
        - d: degree of differencing (enforces stationarity)
        - q: order of the moving average (MA) polynomial
        - B: backshift lag operator, B^k y_t = y_{t-k}
        - epsilon_t: white-noise innovation process ~ N(0, sigma^2)
        - c: optional deterministic trend term

    Invariants Enforced:
      - Fits strictly on historical observations provided to each forecast() call.
      - Never consumes or requires future calendar or target features.
      - Dispatches recursively across multi-step horizons via state-space representation.
      - Demand physical constraint: clips final forecasts to non-negative values: max(0.0, prediction).
      - Never mutates input history objects in-place.
    """

    name: str

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        trend: str | None = None,
        name: Optional[str] = None,
    ):
        self.order: tuple[int, int, int] = _validate_order(order)
        self.trend: str | None = trend
        self.name: str = name or f"ARIMA({self.order[0]},{self.order[1]},{self.order[2]})"
        self.min_history: int = minimum_history_for_order(self.order)

    def forecast(
        self,
        history: Sequence[float] | np.ndarray | pd.Series,
        steps: int,
    ) -> list[float]:
        """
        Generate out-of-sample demand forecasts for exactly `steps` ahead.

        Args:
            history: Sequence of historical target demand values strictly prior to origin.
            steps: Integer count of horizon steps to predict forward.

        Returns:
            List of non-negative float predictions of length `steps`.
        """
        steps = _validate_steps(steps, self.name)
        if steps == 0:
            return []

        y = _validate_history(history, min_observations=self.min_history, model_name=self.name)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(
                    y,
                    order=self.order,
                    trend=self.trend,
                )
                fitted = model.fit()
                raw_preds = fitted.forecast(steps=steps)
        except Exception as exc:
            raise ValueError(f"Failed to fit {self.name} on history (N={len(y)}): {exc}") from exc

        preds = np.atleast_1d(raw_preds)
        if not np.all(np.isfinite(preds)):
            raise ValueError(f"{self.name} generated non-finite predictions: {preds}")

        return [max(0.0, float(val)) for val in preds]

    def __repr__(self) -> str:
        return f"ARIMAForecaster(order={self.order}, trend={self.trend!r}, name={self.name!r})"


def forecast_arima(
    history: Sequence[float] | np.ndarray | pd.Series,
    steps: int,
    order: tuple[int, int, int] = (1, 1, 1),
    trend: str | None = None,
) -> list[float]:
    """
    Convenience function for ARIMA multi-step demand forecasting.
    """
    model = ARIMAForecaster(order=order, trend=trend)
    return model.forecast(history=history, steps=steps)
