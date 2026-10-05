from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.analysis.residual_diagnostics import (
    DEFAULT_ARTIFACTS_DIR,
    LjungBoxRecord,
    OutlierRecord,
    SeriesResidualDiagnostics,
    analyze_residual_series,
    compute_residuals,
    export_diagnostics_artifacts,
    generate_horizon_diagnostic_plots,
    validate_residual_input,
)
from ml.analysis.run_residual_diagnostics import (
    execute_residual_diagnostics,
    load_all_prediction_artifacts,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def synthetic_white_noise() -> np.ndarray:
    """Deterministic Gaussian white noise with zero mean and unit variance (N=250)."""
    np.random.seed(42)
    return np.random.normal(0.0, 1.0, size=250)


@pytest.fixture
def synthetic_ar1_series() -> np.ndarray:
    """Deterministic strongly autocorrelated AR(1) series (phi=0.75, N=250)."""
    np.random.seed(42)
    n = 250
    arr = np.zeros(n)
    innovations = np.random.normal(0.0, 1.0, size=n)
    for t in range(1, n):
        arr[t] = 0.75 * arr[t - 1] + innovations[t]
    return arr


@pytest.fixture
def synthetic_seasonal_series() -> np.ndarray:
    """Deterministic 7-day cyclical series with additive noise (N=250)."""
    np.random.seed(42)
    n = 250
    t = np.arange(n)
    seasonal = 10.0 * np.sin(2 * np.pi * t / 7.0)
    noise = np.random.normal(0.0, 1.0, size=n)
    return seasonal + noise


# ==============================================================================
# 1. Input Validation Tests
# ==============================================================================

def test_validate_residual_input_accepts_valid_types():
    """validate_residual_input must accept list, ndarray, and pd.Series."""
    data_list = [1.0, 2.0, 3.0] * 10
    arr1 = validate_residual_input(data_list, min_observations=10)
    assert isinstance(arr1, np.ndarray)
    assert len(arr1) == 30

    data_np = np.array(data_list)
    arr2 = validate_residual_input(data_np, min_observations=10)
    assert isinstance(arr2, np.ndarray)
    assert len(arr2) == 30

    data_series = pd.Series(data_list)
    arr3 = validate_residual_input(data_series, min_observations=10)
    assert isinstance(arr3, np.ndarray)
    assert len(arr3) == 30


def test_validate_residual_input_immutability():
    """validate_residual_input must copy and not mutate the original data."""
    orig_np = np.array([10.0, 20.0, 30.0] * 10)
    orig_copy = orig_np.copy()
    validated = validate_residual_input(orig_np, min_observations=10)
    validated[0] = 999.0
    assert orig_np[0] == orig_copy[0]
    np.testing.assert_array_equal(orig_np, orig_copy)


def test_validate_residual_input_rejects_invalid_inputs():
    """validate_residual_input must strictly reject None, empty, NaN, Inf, non-numeric, short series."""
    with pytest.raises(ValueError, match="requires non-None"):
        validate_residual_input(None)

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_residual_input([])

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_residual_input(np.array([]))

    with pytest.raises(ValueError, match="contains NaN or infinite"):
        validate_residual_input([1.0, np.nan, 3.0] * 10)

    with pytest.raises(ValueError, match="contains NaN or infinite"):
        validate_residual_input([1.0, np.inf, 3.0] * 10)

    with pytest.raises(ValueError, match="non-numeric elements"):
        validate_residual_input(["a", "b", "c"] * 10)

    with pytest.raises(ValueError, match="requires at least 29 observations"):
        validate_residual_input([1.0, 2.0, 3.0], min_observations=29)


# ==============================================================================
# 2. Residual Calculation & Sign Convention Tests
# ==============================================================================

def test_compute_residuals_sign_convention():
    """Residual must follow convention: residual = actual - predicted."""
    actuals = np.array([100.0, 150.0, 80.0])
    predictions = np.array([90.0, 160.0, 80.0])
    res = compute_residuals(actuals, predictions)

    # 100 - 90 = +10 (under-forecast)
    # 150 - 160 = -10 (over-forecast)
    # 80 - 80 = 0 (exact)
    expected = np.array([10.0, -10.0, 0.0])
    np.testing.assert_array_almost_equal(res, expected)


def test_compute_residuals_length_mismatch():
    """compute_residuals must raise ValueError when actual and predicted lengths differ."""
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_residuals([1.0, 2.0, 3.0], [1.0, 2.0])


def test_compute_residuals_immutability():
    """compute_residuals must not mutate the caller's input arrays."""
    act = np.array([10.0, 20.0, 30.0])
    pred = np.array([5.0, 15.0, 25.0])
    act_copy = act.copy()
    pred_copy = pred.copy()

    res = compute_residuals(act, pred)
    res[0] = 999.0
    np.testing.assert_array_equal(act, act_copy)
    np.testing.assert_array_equal(pred, pred_copy)


# ==============================================================================
# 3. Statistical Analysis on Known Signals
# ==============================================================================

def test_analyze_residual_zero_residuals():
    """Zero residuals (perfect predictions) must handle variance=0 safely."""
    n = 50
    zeros = np.zeros(n)
    diag = analyze_residual_series(zeros, model_name="PerfectModel", horizon_days=7)

    assert diag.count == n
    assert diag.mean_residual == 0.0
    assert diag.std_residual == 0.0
    assert diag.mae == 0.0
    assert diag.rmse == 0.0
    assert diag.bias == 0.0
    assert diag.skewness == 0.0
    assert diag.kurtosis == 0.0
    assert diag.outlier_count_3sigma == 0
    assert np.isnan(diag.variance_ratio)
    for lb in diag.ljung_box:
        assert lb.statistic == 0.0
        assert lb.p_value == 1.0
        assert lb.significant_at_0_05 is False


def test_analyze_residual_constant_residuals():
    """Constant non-zero residuals must have zero variance and no false outliers."""
    n = 60
    constants = np.full(n, 5.0)
    diag = analyze_residual_series(constants, model_name="ConstantBiasModel", horizon_days=7)

    assert diag.count == n
    assert diag.mean_residual == 5.0
    assert diag.std_residual == 0.0
    assert diag.mae == 5.0
    assert diag.rmse == 5.0
    assert diag.bias == 5.0
    assert diag.outlier_count_3sigma == 0
    assert diag.pct_positive_residuals == 100.0
    assert diag.pct_negative_residuals == 0.0


def test_analyze_residual_white_noise(synthetic_white_noise: np.ndarray):
    """Gaussian white noise must exhibit near-zero autocorrelations and pass Ljung-Box."""
    diag = analyze_residual_series(
        synthetic_white_noise,
        model_name="WhiteNoiseModel",
        horizon_days=7,
    )

    assert diag.count == 250
    assert abs(diag.mean_residual) < 0.15
    assert abs(diag.bias) < 0.15
    assert abs(diag.acf_lag_1) < 0.15
    assert abs(diag.acf_lag_7) < 0.15
    assert abs(diag.acf_lag_14) < 0.15

    # Ljung-Box test for white noise: p-values should NOT be significant at 0.05
    for lb in diag.ljung_box:
        assert lb.significant_at_0_05 is False
        assert lb.p_value > 0.05


def test_analyze_residual_autocorrelated_ar1(synthetic_ar1_series: np.ndarray):
    """Strongly autocorrelated AR(1) series must exhibit high lag-1 ACF and reject Ljung-Box."""
    diag = analyze_residual_series(
        synthetic_ar1_series,
        model_name="AR1Model",
        horizon_days=7,
    )

    assert diag.count == 250
    # True AR(1) phi is 0.75, sample ACF at lag 1 should be > 0.6
    assert diag.acf_lag_1 > 0.6
    # Ljung-Box portmanteau tests should strongly reject white noise (p < 0.001)
    for lb in diag.ljung_box:
        assert lb.significant_at_0_05 is True
        assert lb.p_value < 0.001


def test_analyze_residual_seasonal_pattern(synthetic_seasonal_series: np.ndarray):
    """Weekly cyclical residuals must exhibit strong lag-7 autocorrelation and fail Ljung-Box."""
    diag = analyze_residual_series(
        synthetic_seasonal_series,
        model_name="SeasonalModel",
        horizon_days=7,
    )

    assert diag.count == 250
    # Period is 7, so lag 7 should have high positive ACF (> 0.7)
    assert diag.acf_lag_7 > 0.7
    lb_7 = next(lb for lb in diag.ljung_box if lb.lag == 7)
    assert lb_7.significant_at_0_05 is True
    assert lb_7.p_value < 0.001


# ==============================================================================
# 4. Variance Stability Tests
# ==============================================================================

def test_analyze_residual_variance_stability_constant_variance():
    """Homoskedastic series should yield variance ratio close to 1.0."""
    np.random.seed(123)
    n = 200
    e = np.random.normal(0.0, 2.0, size=n)
    diag = analyze_residual_series(e, model_name="HomoskedasticModel", horizon_days=7)

    # Variance ratio should be reasonably close to 1.0 (between 0.7 and 1.45)
    assert 0.7 < diag.variance_ratio < 1.45


def test_analyze_residual_variance_stability_regime_shift():
    """Heteroskedastic series with doubled standard deviation in second half yields variance ratio ~ 4.0."""
    np.random.seed(123)
    n = 200
    first_half = np.random.normal(0.0, 1.0, size=100)
    second_half = np.random.normal(0.0, 2.0, size=100)
    e = np.concatenate([first_half, second_half])
    diag = analyze_residual_series(e, model_name="HeteroskedasticModel", horizon_days=7)

    # Variance ratio = (std2^2) / (std1^2) ~ 4.0 (between 2.8 and 5.5)
    assert 2.8 < diag.variance_ratio < 5.5
    assert diag.second_half_std > diag.first_half_std * 1.5


# ==============================================================================
# 5. Outlier Detection Tests
# ==============================================================================

def test_analyze_residual_outlier_detection_with_spikes():
    """Explicitly injected extreme spikes (|z| > 3.0) must be detected and ordered by |z|."""
    np.random.seed(42)
    n = 100
    e = np.random.normal(0.0, 1.0, size=n)
    dates = [f"2025-01-{i+1:02d}" for i in range(n)]
    actuals = [100.0] * n
    preds = [100.0 - val for val in e]

    # Inject extreme outliers
    e[10] = 12.0  # z ~ +10.5
    e[50] = -10.0  # z ~ -8.8

    diag = analyze_residual_series(
        e,
        model_name="SpikeModel",
        horizon_days=7,
        dates=dates,
        actuals=actuals,
        predictions=preds,
    )

    assert diag.outlier_count_3sigma >= 2
    assert diag.outlier_pct_3sigma >= 2.0
    assert len(diag.top_outliers) == diag.outlier_count_3sigma

    # Top outlier should be the +12.0 spike (highest |z|)
    top = diag.top_outliers[0]
    assert top.date == dates[10]
    assert top.residual == 12.0
    assert top.z_score > 3.0

    second = diag.top_outliers[1]
    assert second.date == dates[50]
    assert second.residual == -10.0
    assert second.z_score < -3.0


# ==============================================================================
# 6. Artifact Export Tests
# ==============================================================================

def test_export_diagnostics_artifacts(tmp_path: Path, synthetic_white_noise: np.ndarray):
    """export_diagnostics_artifacts must persist all 5 required data artifacts cleanly."""
    diag1 = analyze_residual_series(
        synthetic_white_noise,
        model_name="ModelAlpha",
        horizon_days=7,
    )
    diag2 = analyze_residual_series(
        synthetic_white_noise * 1.5,
        model_name="ModelBeta",
        horizon_days=30,
    )

    paths = export_diagnostics_artifacts([diag1, diag2], output_dir=tmp_path)

    # 1. Verify all 5 keys returned
    assert "summary" in paths
    assert "ljung_box" in paths
    assert "autocorrelation" in paths
    assert "outliers" in paths
    assert "json" in paths

    # 2. Check files exist on disk
    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0

    # 3. Verify summary CSV structure
    summary_df = pd.read_csv(paths["summary"])
    assert len(summary_df) == 2
    assert "model" in summary_df.columns
    assert "horizon_days" in summary_df.columns
    assert "MAE" in summary_df.columns
    assert "RMSE" in summary_df.columns
    assert "variance_ratio" in summary_df.columns

    # 4. Verify Ljung-Box CSV structure
    lb_df = pd.read_csv(paths["ljung_box"])
    assert len(lb_df) == 6  # 2 models * 3 lags
    assert set(lb_df["lag"]) == {7, 14, 28}

    # 5. Verify JSON structure
    with open(paths["json"], "r", encoding="utf-8") as f:
        json_content = json.load(f)
    assert json_content["analysis_version"] == "P1.0.2d"
    assert json_content["series_count"] == 2
    assert "ModelAlpha" in json_content["models"]
    assert "ModelBeta" in json_content["models"]


# ==============================================================================
# 7. Diagnostic Plot Generation Tests
# ==============================================================================

def test_generate_horizon_diagnostic_plots(tmp_path: Path):
    """generate_horizon_diagnostic_plots must render 3 valid PNG figure files."""
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    df_m1 = pd.DataFrame(
        {
            "forecast_date": dates,
            "actual_quantity": 100.0 + np.random.normal(0, 5, 60),
            "predicted_quantity": 100.0,
        }
    )
    df_m2 = pd.DataFrame(
        {
            "forecast_date": dates,
            "actual_quantity": df_m1["actual_quantity"],
            "predicted_quantity": 95.0,
        }
    )

    horizon_preds = {
        "SARIMA(1,1,1)(0,1,1,7)": df_m1,
        "ARIMA(2,1,2)": df_m2,
    }

    plot_paths = generate_horizon_diagnostic_plots(
        horizon_predictions=horizon_preds,
        horizon=7,
        output_dir=tmp_path,
        primary_models=["SARIMA(1,1,1)(0,1,1,7)", "ARIMA(2,1,2)"],
    )

    assert "timeseries_7d" in plot_paths
    assert "acf_7d" in plot_paths
    assert "pacf_7d" in plot_paths

    for p in plot_paths.values():
        assert p.exists()
        assert p.stat().st_size > 1000  # Non-trivial image file


# ==============================================================================
# 8. Runner Integration Tests
# ==============================================================================

def test_load_all_prediction_artifacts_succeeds():
    """load_all_prediction_artifacts loads all 4 holdout prediction files without error."""
    df = load_all_prediction_artifacts()

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    expected_cols = {
        "model",
        "horizon_days",
        "forecast_date",
        "actual_quantity",
        "predicted_quantity",
        "source_artifact",
    }
    assert expected_cols.issubset(set(df.columns))

    # Horizons must be 7, 30, 90
    assert set(df["horizon_days"].unique()) == {7, 30, 90}

    # Ensure no duplicate predictions for the same model, horizon, and forecast_date
    dups = df.duplicated(subset=["model", "horizon_days", "forecast_date"]).sum()
    assert dups == 0


def test_execute_residual_diagnostics_e2e(tmp_path: Path):
    """execute_residual_diagnostics runs end-to-end and writes all artifacts to tmp_path."""
    res = execute_residual_diagnostics(output_dir=tmp_path)

    assert "diagnostics" in res
    assert "artifact_paths" in res
    assert "plot_paths" in res
    assert "summary_df" in res

    # 21 distinct (model, horizon) combinations analyzed
    assert len(res["diagnostics"]) == 21

    # Verify all 5 data artifacts exist
    for p in res["artifact_paths"].values():
        assert Path(p).exists()
        assert Path(p).stat().st_size > 0

    # Verify all 9 plot artifacts exist
    assert len(res["plot_paths"]) == 9
    for p in res["plot_paths"].values():
        assert Path(p).exists()
        assert Path(p).stat().st_size > 1000
