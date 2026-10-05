from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from ml.models.sarima_forecaster import (
    SARIMAForecaster,
    forecast_sarima,
    minimum_history_for_sarima,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def weekly_seasonal_history() -> list[float]:
    """28-day synthetic history with strong weekly cyclical pattern and positive trend."""
    days = np.arange(28)
    seasonal = 15.0 * np.sin(2 * np.pi * days / 7.0)
    baseline = 100.0 + 0.5 * days
    return (baseline + seasonal).tolist()


@pytest.fixture
def trending_history() -> list[float]:
    """28-day synthetic history with pure linear trend."""
    return [50.0 + 2.0 * i for i in range(28)]


@pytest.fixture
def constant_history() -> list[float]:
    """28-day constant history."""
    return [75.0] * 28


@pytest.fixture
def zero_history() -> list[float]:
    """28-day all-zero history."""
    return [0.0] * 28


@pytest.fixture
def candidate_sarima_models() -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    return [
        ((0, 1, 0), (1, 0, 0, 7)),
        ((0, 1, 1), (0, 1, 1, 7)),
        ((1, 1, 0), (1, 0, 1, 7)),
        ((1, 1, 1), (0, 1, 1, 7)),
    ]


# ==============================================================================
# 1. Forecast Length and Steps Validation Tests
# ==============================================================================

def test_valid_forecast_lengths(weekly_seasonal_history):
    """Verify SARIMA generates exact requested forecast horizon steps."""
    model = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))
    for steps in [1, 7, 14, 30]:
        preds = model.forecast(weekly_seasonal_history, steps)
        assert isinstance(preds, list)
        assert len(preds) == steps
        assert all(isinstance(p, float) for p in preds)
        assert all(np.isfinite(p) for p in preds)


def test_zero_steps_returns_empty_list(weekly_seasonal_history):
    """Verify requesting 0 forecast steps returns an empty list."""
    model = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))
    assert model.forecast(weekly_seasonal_history, steps=0) == []


def test_negative_steps_rejected(weekly_seasonal_history):
    """Verify requesting negative forecast steps raises ValueError."""
    model = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))
    with pytest.raises(ValueError, match="forecast steps must be non-negative"):
        model.forecast(weekly_seasonal_history, steps=-4)


@pytest.mark.parametrize("invalid_step", [3.5, "7", True, None, [7]])
def test_non_integer_steps_rejected(weekly_seasonal_history, invalid_step):
    """Verify non-integer step types raise ValueError."""
    model = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))
    with pytest.raises(ValueError, match="forecast steps must be an integer"):
        model.forecast(weekly_seasonal_history, steps=invalid_step)


# ==============================================================================
# 2. Non-Negative Demand Clipping
# ==============================================================================

def test_non_negative_clipping_on_downward_trend():
    """Verify downward trending series clips to 0.0 rather than producing negative demand."""
    downward = [100.0 - 4.0 * i for i in range(25)]
    model = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))
    preds = model.forecast(downward, steps=25)

    assert len(preds) == 25
    assert all(p >= 0.0 for p in preds)
    assert any(p == 0.0 for p in preds)


# ==============================================================================
# 3. Determinism and Idempotence
# ==============================================================================

def test_deterministic_forecast_reproducibility(weekly_seasonal_history):
    """Verify identical inputs yield bit-for-bit identical forecasts."""
    model = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))
    run_1 = model.forecast(weekly_seasonal_history, steps=14)
    run_2 = model.forecast(weekly_seasonal_history, steps=14)

    assert run_1 == run_2


# ==============================================================================
# 4. History Immutability
# ==============================================================================

def test_history_list_not_mutated_in_place(weekly_seasonal_history):
    """Verify forecasting does not modify caller's input list."""
    original_copy = copy.deepcopy(weekly_seasonal_history)
    model = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))
    _ = model.forecast(weekly_seasonal_history, steps=10)

    assert weekly_seasonal_history == original_copy


def test_history_numpy_array_not_mutated_in_place(weekly_seasonal_history):
    """Verify forecasting does not modify caller's numpy array."""
    arr = np.array(weekly_seasonal_history, dtype=float)
    arr_copy = arr.copy()
    model = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))
    _ = model.forecast(arr, steps=10)

    np.testing.assert_array_equal(arr, arr_copy)


# ==============================================================================
# 5. History Input Validation: NaN, Infinite, Empty, None, Non-numeric
# ==============================================================================

def test_none_history_rejected():
    """Verify None input raises ValueError."""
    model = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))
    with pytest.raises(ValueError, match="requires a non-None historical series"):
        model.forecast(None, steps=7)


def test_empty_history_rejected():
    """Verify empty history raises ValueError."""
    model = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))
    with pytest.raises(ValueError, match="history cannot be empty"):
        model.forecast([], steps=7)


def test_nan_history_rejected(weekly_seasonal_history):
    """Verify history containing NaN raises ValueError."""
    history = weekly_seasonal_history.copy()
    history[10] = np.nan
    model = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))
    with pytest.raises(ValueError, match="contains NaN or infinite values"):
        model.forecast(history, steps=7)


def test_inf_history_rejected(weekly_seasonal_history):
    """Verify history containing inf raises ValueError."""
    history = weekly_seasonal_history.copy()
    history[12] = np.inf
    model = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))
    with pytest.raises(ValueError, match="contains NaN or infinite values"):
        model.forecast(history, steps=7)


def test_string_elements_in_history_rejected():
    """Verify history containing non-numeric strings raises ValueError."""
    model = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))
    with pytest.raises(ValueError, match="invalid non-numeric elements"):
        model.forecast([10.0, 20.0, "invalid", 40.0, 50.0], steps=3)


# ==============================================================================
# 6. Minimum History Enforcement and Exact Boundaries
# ==============================================================================

@pytest.mark.parametrize(
    "order, seasonal_order, expected_min",
    [
        ((0, 1, 0), (1, 0, 0, 7), 15),
        ((0, 1, 1), (0, 0, 1, 7), 15),
        ((0, 1, 1), (0, 1, 1, 7), 22),
        ((1, 1, 0), (1, 0, 1, 7), 18),
        ((1, 1, 1), (0, 1, 1, 7), 22),
        ((1, 1, 1), (1, 1, 0, 7), 22),
    ],
)
def test_minimum_history_formula(order, seasonal_order, expected_min):
    """Verify exact formula max(2*s + d + D*s, d + D*s + p + q + (P + Q)*s + 2)."""
    assert minimum_history_for_sarima(order, seasonal_order) == expected_min


@pytest.mark.parametrize(
    "order, seasonal_order",
    [
        ((0, 1, 0), (1, 0, 0, 7)),
        ((0, 1, 1), (0, 1, 1, 7)),
        ((1, 1, 0), (1, 0, 1, 7)),
        ((1, 1, 1), (0, 1, 1, 7)),
    ],
)
def test_exact_minimum_history_boundary(order, seasonal_order):
    """
    Test smallest valid history (length == min_history) and smallest invalid history
    (length == min_history - 1).
    """
    min_obs = minimum_history_for_sarima(order, seasonal_order)
    model = SARIMAForecaster(order=order, seasonal_order=seasonal_order)

    # 1. Invalid: min_obs - 1 observations must raise ValueError
    invalid_history = [20.0 + float(i) for i in range(min_obs - 1)]
    with pytest.raises(ValueError, match="requires at least"):
        model.forecast(invalid_history, steps=3)

    # 2. Valid: exactly min_obs observations must succeed
    valid_history = [20.0 + float(i) + 5.0 * np.sin(2 * np.pi * i / 7.0) for i in range(min_obs)]
    preds = model.forecast(valid_history, steps=3)
    assert len(preds) == 3
    assert all(isinstance(p, float) for p in preds)
    assert all(np.isfinite(p) for p in preds)


# ==============================================================================
# 7. Order & Seasonal Order Validation Tests
# ==============================================================================

@pytest.mark.parametrize(
    "invalid_order",
    [
        (1, 1),
        (1, 1, 1, 1),
        (-1, 1, 0),
        (1, -1, 0),
        (1, 1, -1),
        (1.5, 1, 0),
        (True, 1, 0),
        "011",
    ],
)
def test_invalid_order_rejection(invalid_order):
    """Verify invalid non-seasonal order specifications raise ValueError."""
    with pytest.raises(ValueError):
        SARIMAForecaster(order=invalid_order, seasonal_order=(0, 1, 1, 7))


@pytest.mark.parametrize(
    "invalid_s_order",
    [
        (0, 1, 1),          # only 3 elements
        (0, 1, 1, 7, 1),    # 5 elements
        (-1, 1, 1, 7),      # negative P
        (0, -1, 1, 7),      # negative D
        (0, 1, -1, 7),      # negative Q
        (0, 1, 1, 0),       # s = 0
        (0, 1, 1, 1),       # s = 1
        (0, 1, 1, 12),      # s != 7 (weekly phase requirement)
        (0, 1, 1, "7"),     # string s
        (0, 1, True, 7),    # boolean Q
    ],
)
def test_invalid_seasonal_order_rejection(invalid_s_order):
    """Verify invalid seasonal order specifications raise ValueError."""
    with pytest.raises(ValueError):
        SARIMAForecaster(order=(0, 1, 1), seasonal_order=invalid_s_order)


# ==============================================================================
# 8. Synthetic Pattern Tests
# ==============================================================================

def test_constant_history(constant_history, candidate_sarima_models):
    """Verify SARIMA fits and forecasts on constant demand without crash."""
    for order, s_order in candidate_sarima_models:
        model = SARIMAForecaster(order=order, seasonal_order=s_order)
        preds = model.forecast(constant_history, steps=7)
        assert len(preds) == 7
        assert all(np.isfinite(p) for p in preds)


def test_zero_history(zero_history, candidate_sarima_models):
    """Verify SARIMA fits and forecasts on all-zero demand history."""
    for order, s_order in candidate_sarima_models:
        model = SARIMAForecaster(order=order, seasonal_order=s_order)
        preds = model.forecast(zero_history, steps=7)
        assert len(preds) == 7
        assert all(p == 0.0 for p in preds)


def test_trending_history(trending_history, candidate_sarima_models):
    """Verify SARIMA fits and forecasts on linearly trending demand."""
    for order, s_order in candidate_sarima_models:
        model = SARIMAForecaster(order=order, seasonal_order=s_order)
        preds = model.forecast(trending_history, steps=7)
        assert len(preds) == 7
        assert all(p >= 0.0 for p in preds)
        assert all(np.isfinite(p) for p in preds)


def test_weekly_seasonal_series(weekly_seasonal_history, candidate_sarima_models):
    """Verify SARIMA fits on seasonal series and reproduces finite positive forecasts."""
    for order, s_order in candidate_sarima_models:
        model = SARIMAForecaster(order=order, seasonal_order=s_order)
        preds = model.forecast(weekly_seasonal_history, steps=14)
        assert len(preds) == 14
        assert all(p >= 0.0 for p in preds)
        assert all(np.isfinite(p) for p in preds)


# ==============================================================================
# 9. Model Representation & Helpers
# ==============================================================================

def test_model_repr_and_name():
    """Verify custom name, default name, and repr."""
    m1 = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))
    assert m1.name == "SARIMA(0,1,1)(0,1,1,7)"
    assert "order=(0, 1, 1)" in repr(m1)
    assert "seasonal_order=(0, 1, 1, 7)" in repr(m1)

    m2 = SARIMAForecaster(
        order=(1, 1, 0),
        seasonal_order=(1, 0, 1, 7),
        name="CustomSARIMA",
        trend="c",
    )
    assert m2.name == "CustomSARIMA"
    assert "trend='c'" in repr(m2)


def test_convenience_function(weekly_seasonal_history):
    """Verify forecast_sarima convenience function."""
    preds = forecast_sarima(
        weekly_seasonal_history,
        steps=7,
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 7),
    )
    assert len(preds) == 7
    assert all(isinstance(p, float) for p in preds)
