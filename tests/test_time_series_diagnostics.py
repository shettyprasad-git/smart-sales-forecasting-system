from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.analysis.time_series_diagnostics import (
    TimeSeriesDiagnosticsConfig,
    TimeSeriesDiagnosticsResult,
    diagnose_time_series,
    validate_input_data,
)
import ml.analysis.time_series_diagnostics as tsd_module
from ml.data.canonical_series import assess_time_series_quality


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def synthetic_stationary_series() -> pd.DataFrame:
    """Deterministic stationary AR(1) time series (N = 150)."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.4 * y[t - 1] + np.random.normal(0, 1)
    # Shift to positive demand values
    y += 100.0

    return pd.DataFrame({"Date": dates, "Quantity": y})


@pytest.fixture
def synthetic_random_walk() -> pd.DataFrame:
    """Deterministic non-stationary random walk with drift (N = 150)."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    innovations = np.random.normal(0.5, 1.0, size=n)
    y = np.cumsum(innovations) + 100.0

    return pd.DataFrame({"Date": dates, "Quantity": y})


@pytest.fixture
def synthetic_weekly_seasonal() -> pd.DataFrame:
    """Deterministic 7-day cyclical time series (N = 150)."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    t = np.arange(n)
    seasonal = 25.0 * np.sin(2 * np.pi * t / 7.0)
    noise = np.random.normal(0, 1.0, size=n)
    y = 100.0 + seasonal + noise

    return pd.DataFrame({"Date": dates, "Quantity": y})


# ==============================================================================
# Required Tests 1 through 16
# ==============================================================================

def test_1_continuous_synthetic_series_accepted(synthetic_stationary_series: pd.DataFrame):
    """1. Continuous synthetic series is accepted by diagnostics."""
    res = diagnose_time_series(synthetic_stationary_series)
    assert isinstance(res, TimeSeriesDiagnosticsResult)
    assert res.profile.observations == 150
    assert res.profile.start_date == "2024-01-01"
    assert res.profile.end_date == "2024-05-29"
    assert res.sufficiency.sufficient_for_diagnostics is True


def test_2_missing_calendar_date_rejected():
    """2. Missing calendar dates are rejected."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Date": dates, "Quantity": 100.0})
    # Remove day 15 to introduce a calendar discontinuity
    df = df[df["Date"] != "2024-01-15"].reset_index(drop=True)

    with pytest.raises(ValueError, match="fails canonical daily continuity|missing calendar days"):
        diagnose_time_series(df)


def test_3_is_missing_true_rejected():
    """3. Explicit is_missing=True indicators are rejected."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Date": dates, "Quantity": 100.0})
    df["is_missing"] = False
    df.loc[14, "is_missing"] = True

    with pytest.raises(ValueError, match="missing calendar day indicators"):
        diagnose_time_series(df)


def test_4_explicit_zero_demand_days_accepted():
    """4. Explicit zero-demand days (Quantity=0.0) remain valid demand."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    y = np.ones(30) * 50.0
    y[5] = 0.0
    y[12] = 0.0
    df = pd.DataFrame({"Date": dates, "Quantity": y})

    res = diagnose_time_series(df)
    assert res.profile.zero_demand_days == 2
    assert res.variance_transformation.has_zero_demand is True
    assert res.profile.min == 0.0
    assert res.profile.observations == 30


def test_5_empty_series_rejected():
    """5. Empty series is rejected."""
    empty_df = pd.DataFrame(columns=["Date", "Quantity"])
    with pytest.raises(ValueError, match="empty|None"):
        diagnose_time_series(empty_df)


def test_6_one_observation_series_handled_safely():
    """6. One-observation series is handled safely without crashing."""
    single_df = pd.DataFrame({"Date": ["2024-01-01"], "Quantity": [105.5]})
    res = diagnose_time_series(single_df)

    assert res.profile.observations == 1
    assert res.profile.mean == 105.5
    assert res.profile.std == 0.0
    assert res.profile.min == 105.5
    assert res.profile.max == 105.5
    assert res.sufficiency.sufficient_for_diagnostics is False
    assert res.stationarity.stationarity_assessment == "inconclusive"
    assert res.trend.trend_slope is None
    assert res.autocorrelation.acf_values == []
    assert res.partial_autocorrelation.pacf_values == []


def test_7_short_series_does_not_request_invalid_pacf_lags():
    """7. Short series does not request invalid PACF lags beyond sample capacity."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    df = pd.DataFrame({"Date": dates, "Quantity": [10.0, 12.0, 15.0, 14.0, 16.0, 18.0]})

    config = TimeSeriesDiagnosticsConfig(max_pacf_lags=35)
    res = diagnose_time_series(df, config=config)

    # For N=6, safe max lag is min(35, (6 // 2) - 1) = 2
    assert res.partial_autocorrelation.max_evaluated_lag <= 2
    assert len(res.partial_autocorrelation.pacf_values) <= 2


def test_8_stationary_synthetic_series_adf(synthetic_stationary_series: pd.DataFrame):
    """8. Stationary synthetic series produces valid stationary ADF result."""
    res = diagnose_time_series(synthetic_stationary_series)
    adf = res.stationarity

    assert adf.test_statistic is not None
    assert adf.critical_values is not None
    assert adf.p_value is not None
    assert adf.test_statistic < adf.critical_values["5%"]
    assert adf.stationarity_assessment == "stationary_at_5_percent"


def test_9_non_stationary_trend_series_adf(synthetic_random_walk: pd.DataFrame):
    """9. Non-stationary random walk series produces different ADF behavior (non-stationary)."""
    res = diagnose_time_series(synthetic_random_walk)
    adf = res.stationarity

    assert adf.test_statistic is not None
    # For random walk with drift, unit root is not rejected in levels
    assert adf.stationarity_assessment == "non_stationary_at_5_percent"
    assert res.conclusion.stationarity == "non_stationary_at_5_percent"


def test_10_first_differencing_diagnostic(synthetic_random_walk: pd.DataFrame):
    """10. First differencing diagnostic works and detects change in stationarity."""
    res = diagnose_time_series(synthetic_random_walk)
    diff = res.differencing

    assert diff.original_series.stationarity_assessment == "non_stationary_at_5_percent"
    assert diff.first_difference.stationarity_assessment == "stationary_at_5_percent"
    assert diff.first_difference_changed_stationarity is True
    assert res.conclusion.differencing_signal == "first_differencing_recommended"


def test_11_seasonal_lag_7_autocorrelation(synthetic_weekly_seasonal: pd.DataFrame):
    """11. Seasonal lag-7 autocorrelation is measurable and signals weekly periodicity."""
    res = diagnose_time_series(synthetic_weekly_seasonal)
    seas = res.seasonality

    assert seas.lag_7_acf is not None
    assert seas.lag_7_acf > 0.70
    assert seas.lag_14_acf is not None
    assert seas.lag_14_acf > 0.60
    assert seas.seasonal_strength_7d is not None
    assert seas.seasonal_strength_7d > 0.70
    assert res.conclusion.weekly_autocorrelation == "strong_weekly_autocorrelation"


def test_12_deterministic_repeated_runs(synthetic_weekly_seasonal: pd.DataFrame):
    """12. Deterministic repeated runs return identical diagnostic values."""
    res1 = diagnose_time_series(synthetic_weekly_seasonal)
    res2 = diagnose_time_series(synthetic_weekly_seasonal)

    assert res1.to_dict() == res2.to_dict()


def test_13_input_dataframe_not_mutated(synthetic_stationary_series: pd.DataFrame):
    """13. Input DataFrame is completely unmutated."""
    df_original = synthetic_stationary_series.copy(deep=True)
    _ = diagnose_time_series(synthetic_stationary_series)

    pd.testing.assert_frame_equal(synthetic_stationary_series, df_original)


def test_14_json_serialization_succeeds_without_nan():
    """14. JSON serialization succeeds without NaN/Infinity errors even on sparse data."""
    # Test on minimal 2-day series where some rolling/seasonal metrics are None/NaN
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Quantity": [10.0, 20.0]})
    res = diagnose_time_series(df)

    as_dict = res.to_dict()
    # json.dumps must not raise ValueError
    json_str = json.dumps(as_dict)
    assert "NaN" not in json_str
    assert "Infinity" not in json_str

    loaded = json.loads(json_str)
    assert loaded["profile"]["observations"] == 2
    assert loaded["stationarity"]["test_statistic"] is None
    assert loaded["seasonality"]["seasonal_strength_7d"] is None


def test_15_plot_generation_is_optional_and_does_not_affect_diagnostics(
    synthetic_weekly_seasonal: pd.DataFrame,
    tmp_path: Path,
):
    """15. Plot generation is optional and does not affect numerical diagnostics."""
    # Run 1: without plots
    cfg1 = TimeSeriesDiagnosticsConfig(generate_plots=False)
    res1 = diagnose_time_series(synthetic_weekly_seasonal, config=cfg1)
    assert res1.plot_artifacts == []

    # Run 2: with plots
    cfg2 = TimeSeriesDiagnosticsConfig(generate_plots=True, plots_output_dir=tmp_path)
    res2 = diagnose_time_series(synthetic_weekly_seasonal, config=cfg2)
    assert len(res2.plot_artifacts) == 3
    for p in res2.plot_artifacts:
        assert Path(p).exists()

    # Numerical profile, stationarity, and seasonality must match bit-for-bit
    assert res1.profile.mean == res2.profile.mean
    assert res1.stationarity.test_statistic == res2.stationarity.test_statistic
    assert res1.seasonality.lag_7_acf == res2.seasonality.lag_7_acf


def test_16_p02_continuity_layer_is_reused():
    """16. Verifies P0.2 canonical continuity layer (assess_time_series_quality) is reused."""
    assert tsd_module.assess_time_series_quality is assess_time_series_quality
