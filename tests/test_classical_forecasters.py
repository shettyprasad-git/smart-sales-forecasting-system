from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from ml.models.classical_forecasters import (
    DampedHoltWintersModel,
    HoltAdditiveModel,
    HoltWintersAdditiveModel,
    forecast_damped_holt_winters,
    forecast_holt_additive,
    forecast_holt_winters_additive,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def weekly_seasonal_history() -> list[float]:
    """28-day synthetic history with strong weekly pattern and positive trend."""
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


# ==============================================================================
# 1. Forecast Length and Steps Validation Tests
# ==============================================================================

@pytest.mark.parametrize(
    "model_cls",
    [HoltAdditiveModel, HoltWintersAdditiveModel, DampedHoltWintersModel],
)
def test_valid_forecast_lengths(model_cls, weekly_seasonal_history):
    """Verify that models generate exact requested forecast horizon steps."""
    model = model_cls()
    for steps in [1, 7, 14, 30]:
        preds = model.forecast(weekly_seasonal_history, steps)
        assert isinstance(preds, list)
        assert len(preds) == steps
        assert all(isinstance(p, float) for p in preds)
        assert all(np.isfinite(p) for p in preds)


@pytest.mark.parametrize(
    "model_cls",
    [HoltAdditiveModel, HoltWintersAdditiveModel, DampedHoltWintersModel],
)
def test_zero_steps_returns_empty_list(model_cls, weekly_seasonal_history):
    """Verify requesting 0 forecast steps returns empty list."""
    model = model_cls()
    assert model.forecast(weekly_seasonal_history, steps=0) == []


@pytest.mark.parametrize(
    "model_cls",
    [HoltAdditiveModel, HoltWintersAdditiveModel, DampedHoltWintersModel],
)
def test_negative_steps_rejected(model_cls, weekly_seasonal_history):
    """Verify requesting negative forecast steps raises ValueError."""
    model = model_cls()
    with pytest.raises(ValueError, match="forecast steps must be non-negative"):
        model.forecast(weekly_seasonal_history, steps=-5)


# ==============================================================================
# 2. Non-Negative Demand Clipping
# ==============================================================================

@pytest.mark.parametrize(
    "model_cls",
    [HoltAdditiveModel, HoltWintersAdditiveModel, DampedHoltWintersModel],
)
def test_non_negative_clipping_on_downward_trend(model_cls):
    """Verify downward trending or negative-inclined projections clip to 0.0."""
    # Steep downward trend that would project negative without clipping
    downward = [100.0 - 5.0 * i for i in range(20)]
    model = model_cls()
    preds = model.forecast(downward, steps=15)
    assert all(p >= 0.0 for p in preds)
    assert all(np.isfinite(p) for p in preds)


# ==============================================================================
# 3. Determinism
# ==============================================================================

@pytest.mark.parametrize(
    "model_cls",
    [HoltAdditiveModel, HoltWintersAdditiveModel, DampedHoltWintersModel],
)
def test_deterministic_repeated_forecasts(model_cls, weekly_seasonal_history):
    """Verify multiple executions on the same history yield bit-for-bit identical forecasts."""
    model = model_cls()
    preds_1 = model.forecast(weekly_seasonal_history, steps=14)
    preds_2 = model.forecast(weekly_seasonal_history, steps=14)
    assert preds_1 == preds_2


# ==============================================================================
# 4. History Length and Quality Validation
# ==============================================================================

def test_holt_insufficient_history():
    """Verify Holt requires at least 3 historical observations."""
    model = HoltAdditiveModel()
    with pytest.raises(ValueError, match="requires at least 3 historical observations"):
        model.forecast([10.0, 12.0], steps=3)


@pytest.mark.parametrize("model_cls", [HoltWintersAdditiveModel, DampedHoltWintersModel])
def test_holt_winters_insufficient_history(model_cls):
    """Verify Holt-Winters models require at least 2 full seasonal cycles (14 days for weekly)."""
    model = model_cls(seasonal_periods=7)
    with pytest.raises(ValueError, match="requires at least 14 historical observations"):
        model.forecast([10.0] * 13, steps=7)


@pytest.mark.parametrize(
    "model_cls",
    [HoltAdditiveModel, HoltWintersAdditiveModel, DampedHoltWintersModel],
)
def test_empty_or_none_history_rejected(model_cls):
    """Verify None or empty history raises ValueError."""
    model = model_cls()
    with pytest.raises(ValueError, match="requires a non-None historical series"):
        model.forecast(None, steps=5)
    with pytest.raises(ValueError, match="history cannot be empty"):
        model.forecast([], steps=5)


@pytest.mark.parametrize(
    "model_cls",
    [HoltAdditiveModel, HoltWintersAdditiveModel, DampedHoltWintersModel],
)
def test_nan_or_infinite_history_rejected(model_cls, weekly_seasonal_history):
    """Verify history containing NaN or Inf raises ValueError."""
    model = model_cls()
    bad_nan = list(weekly_seasonal_history)
    bad_nan[5] = float("nan")
    with pytest.raises(ValueError, match="history contains NaN or infinite values"):
        model.forecast(bad_nan, steps=5)

    bad_inf = list(weekly_seasonal_history)
    bad_inf[5] = float("inf")
    with pytest.raises(ValueError, match="history contains NaN or infinite values"):
        model.forecast(bad_inf, steps=5)


# ==============================================================================
# 5. Distinct Demand Patterns (Zero, Constant, Trending, Seasonal)
# ==============================================================================

@pytest.mark.parametrize(
    "model_cls",
    [HoltAdditiveModel, HoltWintersAdditiveModel, DampedHoltWintersModel],
)
def test_zero_demand_history(model_cls, zero_history):
    """Verify handling of all-zero historical demand series without crash or NaN."""
    model = model_cls()
    preds = model.forecast(zero_history, steps=7)
    assert len(preds) == 7
    assert all(p == 0.0 for p in preds)
    assert all(np.isfinite(p) for p in preds)


@pytest.mark.parametrize(
    "model_cls",
    [HoltAdditiveModel, HoltWintersAdditiveModel, DampedHoltWintersModel],
)
def test_constant_demand_history(model_cls, constant_history):
    """Verify constant demand series produces constant non-negative forecasts."""
    model = model_cls()
    preds = model.forecast(constant_history, steps=7)
    assert len(preds) == 7
    # Should predict approximately the constant level (75.0)
    for p in preds:
        assert pytest.approx(p, abs=1e-3) == 75.0


def test_trending_demand_history(trending_history):
    """Verify Holt model captures upward linear trend."""
    model = HoltAdditiveModel()
    preds = model.forecast(trending_history, steps=5)
    # History ends at 50 + 2*19 = 88.0. Next steps should be ~ 90, 92, 94, 96, 98
    assert preds[0] > trending_history[-1]
    for i in range(1, len(preds)):
        assert preds[i] > preds[i - 1]


def test_damped_trend_growth_is_bounded_relative_to_linear(trending_history):
    """Verify damped Holt-Winters dampens slope compared to pure linear Holt."""
    h_model = HoltAdditiveModel()
    dhw_model = DampedHoltWintersModel(seasonal_periods=7)

    # Need at least 14 obs for DHW
    h_preds = h_model.forecast(trending_history, steps=30)
    dhw_preds = dhw_model.forecast(trending_history, steps=30)

    # The 30th step of damped trend should be less than or equal to unconstrained linear extrapolation
    assert dhw_preds[-1] <= h_preds[-1]


def test_weekly_seasonality_reproduced(weekly_seasonal_history):
    """Verify Holt-Winters captures weekly cyclical peaks and troughs."""
    model = HoltWintersAdditiveModel(seasonal_periods=7)
    preds = model.forecast(weekly_seasonal_history, steps=14)

    # Pattern over 7 steps should closely mirror pattern over next 7 steps (modulo trend)
    diff_week1 = preds[6] - preds[0]
    diff_week2 = preds[13] - preds[7]
    assert pytest.approx(diff_week1, abs=2.0) == diff_week2


# ==============================================================================
# 6. Input Immutability Tests
# ==============================================================================

@pytest.mark.parametrize(
    "model_cls",
    [HoltAdditiveModel, HoltWintersAdditiveModel, DampedHoltWintersModel],
)
def test_input_history_not_mutated_list(model_cls, weekly_seasonal_history):
    """Verify that passing a Python list does not mutate the list in place."""
    original = list(weekly_seasonal_history)
    model = model_cls()
    _ = model.forecast(weekly_seasonal_history, steps=7)
    assert weekly_seasonal_history == original


@pytest.mark.parametrize(
    "model_cls",
    [HoltAdditiveModel, HoltWintersAdditiveModel, DampedHoltWintersModel],
)
def test_input_history_not_mutated_series(model_cls, weekly_seasonal_history):
    """Verify that passing a pandas Series does not mutate the Series in place."""
    s = pd.Series(weekly_seasonal_history, name="Quantity")
    s_copy = s.copy(deep=True)
    model = model_cls()
    _ = model.forecast(s, steps=7)
    pd.testing.assert_series_equal(s, s_copy)


# ==============================================================================
# 7. Convenience Functions Tests
# ==============================================================================

def test_convenience_functions(weekly_seasonal_history):
    """Verify convenience function wrappers yield matching forecasts."""
    p_h = forecast_holt_additive(weekly_seasonal_history, steps=7)
    assert len(p_h) == 7
    assert p_h == HoltAdditiveModel().forecast(weekly_seasonal_history, steps=7)

    p_hw = forecast_holt_winters_additive(weekly_seasonal_history, steps=7)
    assert len(p_hw) == 7
    assert p_hw == HoltWintersAdditiveModel().forecast(weekly_seasonal_history, steps=7)

    p_dhw = forecast_damped_holt_winters(weekly_seasonal_history, steps=7)
    assert len(p_dhw) == 7
    assert p_dhw == DampedHoltWintersModel().forecast(weekly_seasonal_history, steps=7)
