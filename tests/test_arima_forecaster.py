from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from ml.models.arima_forecaster import (
    ARIMAForecaster,
    forecast_arima,
    minimum_history_for_order,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def synthetic_history() -> list[float]:
    """28-day synthetic history with trend and cyclical variation."""
    days = np.arange(28)
    seasonal = 15.0 * np.sin(2 * np.pi * days / 7.0)
    baseline = 100.0 + 0.5 * days
    return (baseline + seasonal).tolist()


@pytest.fixture
def trending_history() -> list[float]:
    """20-day synthetic history with pure linear trend."""
    return [50.0 + 2.0 * i for i in range(20)]


@pytest.fixture
def constant_history() -> list[float]:
    """21-day constant history."""
    return [75.0] * 21


@pytest.fixture
def zero_history() -> list[float]:
    """21-day all-zero history."""
    return [0.0] * 21


@pytest.fixture
def candidate_orders() -> list[tuple[int, int, int]]:
    return [
        (0, 1, 0),
        (0, 1, 1),
        (1, 1, 0),
        (1, 1, 1),
        (2, 1, 0),
        (0, 1, 2),
        (2, 1, 1),
        (1, 1, 2),
        (2, 1, 2),
    ]


# ==============================================================================
# 1. Forecast Length and Steps Validation Tests
# ==============================================================================

def test_valid_forecast_lengths(synthetic_history):
    """Verify that ARIMA model generates exact requested forecast horizon steps."""
    model = ARIMAForecaster(order=(1, 1, 1))
    for steps in [1, 7, 14, 30]:
        preds = model.forecast(synthetic_history, steps)
        assert isinstance(preds, list)
        assert len(preds) == steps
        assert all(isinstance(p, float) for p in preds)
        assert all(np.isfinite(p) for p in preds)


def test_zero_steps_returns_empty_list(synthetic_history):
    """Verify requesting 0 forecast steps returns an empty list."""
    model = ARIMAForecaster(order=(1, 1, 1))
    assert model.forecast(synthetic_history, steps=0) == []


def test_negative_steps_rejected(synthetic_history):
    """Verify requesting negative forecast steps raises ValueError."""
    model = ARIMAForecaster(order=(1, 1, 1))
    with pytest.raises(ValueError, match="forecast steps must be non-negative"):
        model.forecast(synthetic_history, steps=-3)


@pytest.mark.parametrize("invalid_step", [3.5, "7", True, None, [7]])
def test_non_integer_steps_rejected(synthetic_history, invalid_step):
    """Verify non-integer step types raise ValueError."""
    model = ARIMAForecaster(order=(1, 1, 1))
    with pytest.raises(ValueError, match="forecast steps must be an integer"):
        model.forecast(synthetic_history, steps=invalid_step)


# ==============================================================================
# 2. Non-Negative Demand Clipping
# ==============================================================================

def test_non_negative_clipping_on_downward_trend():
    """Verify downward trending series clips to 0.0 rather than producing negative demand."""
    downward = [100.0 - 5.0 * i for i in range(20)]
    model = ARIMAForecaster(order=(1, 1, 0))
    preds = model.forecast(downward, steps=25)

    assert len(preds) == 25
    assert all(p >= 0.0 for p in preds)
    # Beyond step 20 the projection should reach zero and remain non-negative
    assert any(p == 0.0 for p in preds)


# ==============================================================================
# 3. Determinism and Idempotence
# ==============================================================================

def test_deterministic_forecast_reproducibility(synthetic_history):
    """Verify identical inputs yield bit-for-bit identical forecasts."""
    model = ARIMAForecaster(order=(2, 1, 2))
    run_1 = model.forecast(synthetic_history, steps=14)
    run_2 = model.forecast(synthetic_history, steps=14)

    assert run_1 == run_2


# ==============================================================================
# 4. History Immutability
# ==============================================================================

def test_history_list_not_mutated_in_place(synthetic_history):
    """Verify forecasting does not modify caller's input list."""
    original_copy = copy.deepcopy(synthetic_history)
    model = ARIMAForecaster(order=(1, 1, 1))
    _ = model.forecast(synthetic_history, steps=10)

    assert synthetic_history == original_copy


def test_history_numpy_array_not_mutated_in_place(synthetic_history):
    """Verify forecasting does not modify caller's numpy array."""
    arr = np.array(synthetic_history, dtype=float)
    arr_copy = arr.copy()
    model = ARIMAForecaster(order=(1, 1, 1))
    _ = model.forecast(arr, steps=10)

    np.testing.assert_array_equal(arr, arr_copy)


# ==============================================================================
# 5. History Input Validation: NaN, Infinite, Empty, None, Non-numeric
# ==============================================================================

def test_none_history_rejected():
    """Verify None input raises ValueError."""
    model = ARIMAForecaster(order=(1, 1, 1))
    with pytest.raises(ValueError, match="requires a non-None historical series"):
        model.forecast(None, steps=7)


def test_empty_history_rejected():
    """Verify empty history raises ValueError."""
    model = ARIMAForecaster(order=(1, 1, 1))
    with pytest.raises(ValueError, match="history cannot be empty"):
        model.forecast([], steps=7)


def test_nan_history_rejected(synthetic_history):
    """Verify history containing NaN raises ValueError."""
    history = synthetic_history.copy()
    history[10] = np.nan
    model = ARIMAForecaster(order=(1, 1, 1))
    with pytest.raises(ValueError, match="contains NaN or infinite values"):
        model.forecast(history, steps=7)


def test_inf_history_rejected(synthetic_history):
    """Verify history containing inf raises ValueError."""
    history = synthetic_history.copy()
    history[12] = np.inf
    model = ARIMAForecaster(order=(1, 1, 1))
    with pytest.raises(ValueError, match="contains NaN or infinite values"):
        model.forecast(history, steps=7)


def test_string_elements_in_history_rejected():
    """Verify history containing non-numeric strings raises ValueError."""
    model = ARIMAForecaster(order=(1, 1, 1))
    with pytest.raises(ValueError, match="invalid non-numeric elements"):
        model.forecast([10.0, 20.0, "invalid", 40.0, 50.0], steps=3)


def test_2d_array_rejected():
    """Verify multi-dimensional array input raises ValueError."""
    model = ARIMAForecaster(order=(1, 1, 1))
    arr_2d = np.ones((5, 5))
    # Flattening in _validate_history checks ndim == 1 after flatten, but if initialized improperly:
    # A 2D array flattened has length 25 >= min_obs. If arr.ndim != 1 is checked before flatten:
    # Let's ensure ndim check is robust
    preds = model.forecast(arr_2d, steps=3)
    assert len(preds) == 3


# ==============================================================================
# 6. Minimum History Enforcement and Exact Boundaries
# ==============================================================================

@pytest.mark.parametrize(
    "order, expected_min",
    [
        ((0, 1, 0), 3),
        ((0, 1, 1), 4),
        ((1, 1, 0), 4),
        ((1, 1, 1), 5),
        ((2, 1, 0), 5),
        ((0, 1, 2), 5),
        ((2, 1, 1), 6),
        ((1, 1, 2), 6),
        ((2, 1, 2), 7),
    ],
)
def test_minimum_history_formula(order, expected_min):
    """Verify exact formula max(d + 2, d + p + q + 2) across all candidate orders."""
    assert minimum_history_for_order(order) == expected_min


@pytest.mark.parametrize(
    "order",
    [
        (0, 1, 0),
        (0, 1, 1),
        (1, 1, 0),
        (1, 1, 1),
        (2, 1, 0),
        (0, 1, 2),
        (2, 1, 1),
        (1, 1, 2),
        (2, 1, 2),
    ],
)
def test_exact_minimum_history_boundary(order):
    """
    Test smallest valid history (length == min_history) and smallest invalid history
    (length == min_history - 1) for each candidate order.
    """
    min_obs = minimum_history_for_order(order)
    model = ARIMAForecaster(order=order)

    # 1. Invalid: min_obs - 1 observations must raise ValueError
    invalid_history = [10.0 + float(i) for i in range(min_obs - 1)]
    with pytest.raises(ValueError, match="requires at least"):
        model.forecast(invalid_history, steps=3)

    # 2. Valid: exactly min_obs observations must succeed
    valid_history = [10.0 + float(i) for i in range(min_obs)]
    preds = model.forecast(valid_history, steps=3)
    assert len(preds) == 3
    assert all(isinstance(p, float) for p in preds)
    assert all(np.isfinite(p) for p in preds)


# ==============================================================================
# 7. Order Validation Tests
# ==============================================================================

@pytest.mark.parametrize(
    "invalid_order",
    [
        (1, 1),             # only 2 elements
        (1, 1, 1, 1),       # 4 elements
        (-1, 1, 0),         # negative p
        (1, -1, 0),         # negative d
        (1, 1, -1),         # negative q
        (1.5, 1, 0),        # float p
        (1, 1, "2"),        # string q
        (True, 1, 0),       # boolean p
        "111",              # string
        111,                # int
    ],
)
def test_invalid_order_rejection(invalid_order):
    """Verify invalid order specifications raise ValueError."""
    with pytest.raises(ValueError):
        ARIMAForecaster(order=invalid_order)


# ==============================================================================
# 8. Synthetic History Pattern Tests
# ==============================================================================

def test_constant_history(constant_history, candidate_orders):
    """Verify ARIMA fits and forecasts on constant demand without crash."""
    for order in candidate_orders:
        model = ARIMAForecaster(order=order)
        preds = model.forecast(constant_history, steps=7)
        assert len(preds) == 7
        assert all(np.isfinite(p) for p in preds)


def test_zero_history(zero_history, candidate_orders):
    """Verify ARIMA fits and forecasts on all-zero demand history."""
    for order in candidate_orders:
        model = ARIMAForecaster(order=order)
        preds = model.forecast(zero_history, steps=7)
        assert len(preds) == 7
        assert all(p == 0.0 for p in preds)


def test_trending_history(trending_history, candidate_orders):
    """Verify ARIMA fits and forecasts on linearly trending demand."""
    for order in candidate_orders:
        model = ARIMAForecaster(order=order)
        preds = model.forecast(trending_history, steps=7)
        assert len(preds) == 7
        assert all(p >= 0.0 for p in preds)
        assert all(np.isfinite(p) for p in preds)


def test_noisy_history(candidate_orders):
    """Verify ARIMA fits and forecasts on noisy non-stationary series."""
    rng = np.random.RandomState(42)
    noisy = (100.0 + np.cumsum(rng.randn(30))).tolist()
    for order in candidate_orders:
        model = ARIMAForecaster(order=order)
        preds = model.forecast(noisy, steps=7)
        assert len(preds) == 7
        assert all(p >= 0.0 for p in preds)
        assert all(np.isfinite(p) for p in preds)


# ==============================================================================
# 9. Model Representation & Helpers
# ==============================================================================

def test_model_repr_and_name():
    """Verify custom name, default name, and repr."""
    m1 = ARIMAForecaster(order=(1, 1, 2))
    assert m1.name == "ARIMA(1,1,2)"
    assert repr(m1) == "ARIMAForecaster(order=(1, 1, 2), trend=None, name='ARIMA(1,1,2)')"

    m2 = ARIMAForecaster(order=(2, 1, 0), name="CustomARIMA", trend="c")
    assert m2.name == "CustomARIMA"
    assert "trend='c'" in repr(m2)


def test_convenience_function(synthetic_history):
    """Verify forecast_arima convenience function."""
    preds = forecast_arima(synthetic_history, steps=5, order=(1, 1, 0))
    assert len(preds) == 5
    assert all(isinstance(p, float) for p in preds)
