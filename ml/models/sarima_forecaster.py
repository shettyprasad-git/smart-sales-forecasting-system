from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def minimum_history_for_sarima(
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> int:
    """
    Derive the minimum required historical observations for SARIMA(p, d, q)(P, D, Q, s).

    Formula:
        min_history = max(2 * s + d + D * s, d + D * s + p + q + (P + Q) * s + 2)

    Theoretical Justification:
      1. Differencing Loss: Non-seasonal differencing d consumes d initial observations;
         seasonal differencing D consumes D * s initial observations. The combined
         differencing operator (1 - B)^d (1 - B^s)^D requires at least d + D * s observations
         to produce a single stationary differenced value.
      2. Seasonal Cycle Visibility: A seasonal process cannot reliably estimate seasonal
         autoregressive or moving average components without observing at least two complete
         seasonal cycles in addition to differencing: 2 * s + d + D * s.
      3. Lag Reach & Estimation Degrees of Freedom: To accommodate non-seasonal lag order (p + q)
         and seasonal lag span ((P + Q) * s) in the state-space representation, plus at least
         two degrees of freedom for residual error estimation:
             d + D * s + p + q + (P + Q) * s + 2.
      4. Numerical Stability: For weekly seasonal retail sales (s=7), this rule enforces
         between 15 and 26 observations depending on orders, preventing singular Kalman
         covariance matrix initializations.
    """
    p, d, q = order
    P, D, Q, s = seasonal_order
    return max(2 * s + d + D * s, d + D * s + p + q + (P + Q) * s + 2)


def _validate_order(order: Any) -> tuple[int, int, int]:
    """Validate non-seasonal ARIMA (p, d, q) order tuple."""
    if not isinstance(order, (tuple, list)) or len(order) != 3:
        raise ValueError(f"SARIMA non-seasonal order must be a 3-element tuple (p, d, q), got {order}.")

    p, d, q = order
    for name, val in [("p", p), ("d", d), ("q", q)]:
        if not isinstance(val, (int, np.integer)) or isinstance(val, bool):
            raise ValueError(f"SARIMA order parameter '{name}' must be an integer, got {type(val).__name__} ({val}).")
        if val < 0:
            raise ValueError(f"SARIMA order parameter '{name}' must be non-negative, got {val}.")

    return int(p), int(d), int(q)


def _validate_seasonal_order(
    seasonal_order: Any,
    allowed_seasonal_period: Optional[int] = 7,
) -> tuple[int, int, int, int]:
    """Validate seasonal SARIMA (P, D, Q, s) order tuple."""
    if not isinstance(seasonal_order, (tuple, list)) or len(seasonal_order) != 4:
        raise ValueError(
            f"SARIMA seasonal_order must be a 4-element tuple (P, D, Q, s), got {seasonal_order}."
        )

    P, D, Q, s = seasonal_order
    for name, val in [("P", P), ("D", D), ("Q", Q), ("s", s)]:
        if not isinstance(val, (int, np.integer)) or isinstance(val, bool):
            raise ValueError(
                f"SARIMA seasonal parameter '{name}' must be an integer, got {type(val).__name__} ({val})."
            )
        if val < 0:
            raise ValueError(f"SARIMA seasonal parameter '{name}' must be non-negative, got {val}.")

    P, D, Q, s = int(P), int(D), int(Q), int(s)
    if s <= 1:
        raise ValueError(f"SARIMA seasonal period s must be greater than 1, got {s}.")

    if allowed_seasonal_period is not None and s != allowed_seasonal_period:
        raise ValueError(
            f"SARIMA seasonal period s must be {allowed_seasonal_period} for weekly retail forecasting, got {s}."
        )

    return P, D, Q, s


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


class SARIMAForecaster:
    """
    Classical Seasonal Autoregressive Integrated Moving Average (SARIMA) forecasting model.

    Model Formulation:
      SARIMA(p, d, q)(P, D, Q, s) models non-seasonal and seasonal dynamics:
        Phi_P(B^s) phi_p(B) (1 - B)^d (1 - B^s)^D y_t = c + Theta_Q(B^s) theta_q(B) epsilon_t

      Where:
        - p, d, q: Non-seasonal AR order, differencing degree, and MA order
        - P, D, Q: Seasonal AR order, seasonal differencing degree, and seasonal MA order
        - s: Seasonal period length (default: 7 for weekly retail sales)
        - B: Backshift lag operator
        - epsilon_t: White noise innovation process ~ N(0, sigma^2)
        - c: Optional trend / constant term

    Invariants Enforced:
      - Fits strictly on historical observations provided to each forecast() call.
      - Dispatches natively to RollingEvaluator via forecast(history, steps).
      - Demands physical realism: clips final forecasts to non-negative values: max(0.0, prediction).
      - Never consumes future target or calendar data.
      - Never mutates input history objects in-place.
    """

    name: str

    def __init__(
        self,
        order: tuple[int, int, int] = (0, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 7),
        trend: str | None = None,
        name: Optional[str] = None,
        allowed_seasonal_period: Optional[int] = 7,
    ):
        self.order: tuple[int, int, int] = _validate_order(order)
        self.seasonal_order: tuple[int, int, int, int] = _validate_seasonal_order(
            seasonal_order, allowed_seasonal_period=allowed_seasonal_period
        )
        self.trend: str | None = trend
        p, d, q = self.order
        P, D, Q, s = self.seasonal_order
        self.name: str = name or f"SARIMA({p},{d},{q})({P},{D},{Q},{s})"
        self.min_history: int = minimum_history_for_sarima(self.order, self.seasonal_order)

    def forecast(
        self,
        history: Sequence[float] | np.ndarray | pd.Series,
        steps: int,
    ) -> list[float]:
        """
        Generate out-of-sample seasonal demand forecasts for exactly `steps` ahead.

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
                model = SARIMAX(
                    y,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    trend=self.trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False)
                raw_preds = fitted.forecast(steps=steps)
        except Exception as exc:
            raise ValueError(f"Failed to fit {self.name} on history (N={len(y)}): {exc}") from exc

        preds = np.atleast_1d(raw_preds)
        if not np.all(np.isfinite(preds)):
            raise ValueError(f"{self.name} generated non-finite predictions: {preds}")

        return [max(0.0, float(val)) for val in preds]

    def __repr__(self) -> str:
        return (
            f"SARIMAForecaster(order={self.order}, seasonal_order={self.seasonal_order}, "
            f"trend={self.trend!r}, name={self.name!r})"
        )


def forecast_sarima(
    history: Sequence[float] | np.ndarray | pd.Series,
    steps: int,
    order: tuple[int, int, int] = (0, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 7),
    trend: str | None = None,
) -> list[float]:
    """
    Convenience function for SARIMA multi-step demand forecasting.
    """
    model = SARIMAForecaster(order=order, seasonal_order=seasonal_order, trend=trend)
    return model.forecast(history=history, steps=steps)
