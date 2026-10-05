from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from ml.evaluation.rolling_evaluator import (
    EvaluationConfig,
    EvaluationResult,
    RollingEvaluator,
    SeasonalNaiveModel,
)
from ml.evaluation.select_sarima_order import (
    DEFAULT_SARIMA_CANDIDATE_GRID,
    SELECTION_CUTOFF_DATE,
    select_sarima_order,
)
from ml.features.feature_pipeline import FEATURE_COLUMNS, build_feature_dataset
from ml.models.arima_forecaster import ARIMAForecaster
from ml.models.classical_forecasters import HoltAdditiveModel, HoltWintersAdditiveModel
from ml.models.sarima_forecaster import SARIMAForecaster


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def synthetic_eval_data() -> pd.DataFrame:
    """
    Generate a 140-day continuous daily dataset (2024-01-01 to 2024-05-19).
    Sufficient for initial history (28+ days) and rolling origins across horizons 7, 30.
    """
    dates = pd.date_range(start="2024-01-01", end="2024-05-19", freq="D")
    n = len(dates)
    days = np.arange(n)
    seasonal_7d = 12.0 * np.sin(2 * np.pi * days / 7.0)
    baseline = 100.0 + 0.1 * days
    quantity = np.maximum(5.0, baseline + seasonal_7d)

    return pd.DataFrame(
        {
            "Date": dates,
            "Quantity": quantity,
            "Promotions": (dates.dayofweek == 4).astype(int),
            "Holiday_Flag": (dates.day == 1).astype(int),
            "Sales_Amount": quantity * 20.0,
            "Profit": quantity * 4.0,
        }
    )


@pytest.fixture
def pre_and_post_2025_data() -> pd.DataFrame:
    """
    Generate continuous dataset spanning 2024 through 2025 (2024-06-01 to 2025-06-30).
    Pre-2025 data (2024-06-01 to 2024-12-31) has 214 days.
    2025 holdout data (2025-01-01 to 2025-06-30) has 181 days.
    """
    dates = pd.date_range(start="2024-06-01", end="2025-06-30", freq="D")
    n = len(dates)
    days = np.arange(n)
    seasonal_7d = 10.0 * np.sin(2 * np.pi * days / 7.0)
    baseline = 150.0 + 0.2 * days
    quantity = np.maximum(5.0, baseline + seasonal_7d)

    return pd.DataFrame(
        {
            "Date": dates,
            "Quantity": quantity,
        }
    )


@pytest.fixture
def trained_ridge_model(synthetic_eval_data: pd.DataFrame) -> tuple[Ridge, pd.Timestamp]:
    """Train a simple Ridge model to verify existing ML model path remains functional."""
    train_slice = synthetic_eval_data[synthetic_eval_data["Date"] < "2024-02-15"].copy()
    train_slice = train_slice.set_index("Date")
    feature_df = build_feature_dataset(train_slice)

    valid_mask = feature_df[FEATURE_COLUMNS].notna().all(axis=1) & feature_df["Quantity"].notna()
    X = feature_df.loc[valid_mask, FEATURE_COLUMNS]
    y = feature_df.loc[valid_mask, "Quantity"]

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X, y)
    reference_start = train_slice.index.min()
    return model, reference_start


# ==============================================================================
# 1. SARIMAForecaster through RollingEvaluator
# ==============================================================================

def test_1_sarima_forecaster_rolling_evaluator(synthetic_eval_data):
    """Verify SARIMAForecaster passes through RollingEvaluator alongside Seasonal Naive baseline."""
    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2024-02-15",
        evaluation_end="2024-03-15",
        min_history_days=28,
        baseline_model_name="Seasonal Naive",
    )
    evaluator = RollingEvaluator(config=config)
    sarima = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))

    result = evaluator.evaluate_series(
        models=[sarima],
        daily=synthetic_eval_data,
        include_baseline=True,
    )

    assert isinstance(result, EvaluationResult)
    models_present = result.summary_df["model"].tolist()
    assert sarima.name in models_present
    assert "Seasonal Naive" in models_present
    assert len(result.predictions_df) > 0
    assert len(result.origin_metrics_df) > 0

    sarima_row = result.summary_df[result.summary_df["model"] == sarima.name].iloc[0]
    assert not np.isnan(sarima_row["WAPE_delta_vs_baseline"])
    assert not np.isnan(sarima_row["MAE_delta_vs_baseline"])
    assert not np.isnan(sarima_row["RMSE_delta_vs_baseline"])


def test_2_multiple_horizons_evaluation(synthetic_eval_data):
    """Verify evaluation across 7-day and 30-day horizons generates complete metric rows."""
    config = EvaluationConfig(
        horizons=(7, 30),
        evaluation_start="2024-02-15",
        evaluation_end="2024-04-15",
        min_history_days=28,
        baseline_model_name="Seasonal Naive",
    )
    evaluator = RollingEvaluator(config=config)
    sarima = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))

    result = evaluator.evaluate_series(
        models=[sarima],
        daily=synthetic_eval_data,
        include_baseline=True,
    )

    summary_df = result.summary_df
    assert set(summary_df["horizon_days"]) == {7, 30}
    for h in [7, 30]:
        h_slice = summary_df[summary_df["horizon_days"] == h]
        assert len(h_slice) == 2  # SARIMA + Baseline
        for _, row in h_slice.iterrows():
            assert row["WAPE"] >= 0.0
            assert row["MAE"] >= 0.0
            assert row["RMSE"] >= 0.0


# ==============================================================================
# 2. Interoperability with ARIMA, ETS, and Feature-Based ML
# ==============================================================================

def test_3_sarima_arima_ets_and_ml_paths_coexist(synthetic_eval_data, trained_ridge_model):
    """Verify SARIMA, ARIMA, ETS (Holt-Winters), and Ridge evaluate simultaneously without conflict."""
    ridge_model, ref_start = trained_ridge_model
    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2024-02-15",
        evaluation_end="2024-03-01",
        min_history_days=28,
        baseline_model_name="Seasonal Naive",
    )
    evaluator = RollingEvaluator(config=config)

    ml_spec = {
        "model": ridge_model,
        "horizon": 7,
        "model_name": "Ridge Regression",
        "reference_start_date": ref_start,
        "feature_columns": FEATURE_COLUMNS,
    }
    ets_model = HoltWintersAdditiveModel(seasonal_periods=7)
    arima_model = ARIMAForecaster(order=(1, 1, 0))
    sarima_model = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))

    result = evaluator.evaluate_series(
        models=[ml_spec, ets_model, arima_model, sarima_model],
        daily=synthetic_eval_data,
        include_baseline=True,
    )

    models_evaluated = set(result.summary_df["model"])
    assert "Ridge Regression" in models_evaluated
    assert "Holt-Winters Additive Weekly" in models_evaluated
    assert "ARIMA(1,1,0)" in models_evaluated
    assert "SARIMA(0,1,1)(0,1,1,7)" in models_evaluated
    assert "Seasonal Naive" in models_evaluated


# ==============================================================================
# 3. Rolling Origin Forecast Zero-Leakage Invariant
# ==============================================================================

def test_4_future_target_mutation_zero_backward_leakage(synthetic_eval_data):
    """
    Adversarial verification: mutating future demand strictly AFTER the first origin's
    forecast block causes ZERO change to forecasts generated at that first origin.
    """
    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2024-02-15",
        evaluation_end="2024-03-15",
        min_history_days=28,
    )
    evaluator = RollingEvaluator(config=config)
    sarima = SARIMAForecaster(order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))

    # Run 1: clean series
    result_clean = evaluator.evaluate_series(
        models=[sarima],
        daily=synthetic_eval_data,
        include_baseline=True,
    )

    first_origin = "2024-02-15"
    clean_origin_0_preds = result_clean.predictions_df[
        result_clean.predictions_df["origin_date"] == first_origin
    ].copy().sort_values(["model", "forecast_date"]).reset_index(drop=True)

    # Run 2: tamper drastically with all observations strictly AFTER the first 7-day block
    tampered_data = synthetic_eval_data.copy(deep=True)
    future_mask = pd.to_datetime(tampered_data["Date"]) >= "2024-02-23"
    tampered_data.loc[future_mask, "Quantity"] *= 500.0

    result_tampered = evaluator.evaluate_series(
        models=[sarima],
        daily=tampered_data,
        include_baseline=True,
    )

    tampered_origin_0_preds = result_tampered.predictions_df[
        result_tampered.predictions_df["origin_date"] == first_origin
    ].copy().sort_values(["model", "forecast_date"]).reset_index(drop=True)

    assert len(clean_origin_0_preds) == len(tampered_origin_0_preds)
    np.testing.assert_array_equal(
        clean_origin_0_preds["predicted_quantity"].values,
        tampered_origin_0_preds["predicted_quantity"].values,
    )


# ==============================================================================
# 4. Holdout Protection & Mandatory Adversarial Order-Selection Leakage Test
# ==============================================================================

def test_5_order_selection_rejects_holdout_dates(pre_and_post_2025_data):
    """Verify select_sarima_order raises ValueError if cutoff_date enters the 2025 holdout."""
    with pytest.raises(ValueError, match="violates the holdout rule"):
        select_sarima_order(
            daily=pre_and_post_2025_data,
            cutoff_date="2025-01-01",
        )


def test_6_order_is_locked_during_evaluation(synthetic_eval_data):
    """Verify that a SARIMAForecaster maintains its locked specification across all rolling origins."""
    locked_order = (0, 1, 1)
    locked_seasonal = (0, 1, 1, 7)
    sarima = SARIMAForecaster(order=locked_order, seasonal_order=locked_seasonal)

    assert sarima.order == locked_order
    assert sarima.seasonal_order == locked_seasonal

    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2024-02-15",
        evaluation_end="2024-03-01",
        min_history_days=28,
    )
    evaluator = RollingEvaluator(config=config)
    evaluator.evaluate_series(
        models=[sarima],
        daily=synthetic_eval_data,
        include_baseline=False,
    )

    assert sarima.order == locked_order
    assert sarima.seasonal_order == locked_seasonal


def test_7_mandatory_adversarial_order_selection_leakage(pre_and_post_2025_data):
    """
    MANDATORY ADVERSARIAL LEAKAGE TEST:
    1. Run SARIMA order selection on dataset spanning pre-2025 and 2025 holdout.
    2. Modify ONLY 2025 observations (multiply by 500x and add large constant).
    3. Re-run order selection.
    4. Assert selected order, selected seasonal order, and AIC/BIC results are bit-for-bit identical.
    """
    small_candidate_grid = [
        ((0, 1, 0), (1, 0, 0, 7)),
        ((0, 1, 1), (0, 1, 1, 7)),
        ((1, 1, 0), (1, 0, 1, 7)),
    ]

    # Run 1: Clean data
    sel_order_clean, sel_s_clean, df_clean, meta_clean = select_sarima_order(
        daily=pre_and_post_2025_data,
        candidate_grid=small_candidate_grid,
        cutoff_date="2024-12-31",
    )

    # Run 2: Tamper with 2025 observations only
    tampered_data = pre_and_post_2025_data.copy(deep=True)
    mask_2025 = pd.to_datetime(tampered_data["Date"]) >= "2025-01-01"
    tampered_data.loc[mask_2025, "Quantity"] = (
        tampered_data.loc[mask_2025, "Quantity"] * 500.0 + 88888.0
    )

    sel_order_tamp, sel_s_tamp, df_tamp, meta_tamp = select_sarima_order(
        daily=tampered_data,
        candidate_grid=small_candidate_grid,
        cutoff_date="2024-12-31",
    )

    # 1. Selected orders must match exactly
    assert sel_order_clean == sel_order_tamp
    assert sel_s_clean == sel_s_tamp

    # 2. Information criteria and likelihood must match bit-for-bit
    assert meta_clean["selected_aic"] == meta_tamp["selected_aic"]
    assert meta_clean["selected_bic"] == meta_tamp["selected_bic"]

    # 3. Complete candidate evaluation tables must be identical
    pd.testing.assert_frame_equal(df_clean, df_tamp)
