from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from ml.evaluation.metrics import evaluate, wape
from ml.evaluation.recursive_forecast import forecast_block
from ml.evaluation.rolling_evaluator import (
    EvaluationConfig,
    EvaluationResult,
    RollingEvaluator,
    SeasonalNaiveModel,
    forecast_seasonal_naive,
)
from ml.features.feature_pipeline import FEATURE_COLUMNS, build_feature_dataset


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def synthetic_daily_data() -> pd.DataFrame:
    """
    Generate a deterministic continuous daily dataset from 2024-01-01 to 2025-03-31 (456 days).
    """
    dates = pd.date_range(start="2024-01-01", end="2025-03-31", freq="D")
    n = len(dates)

    # 7-day cyclical component + linear trend + positive baseline
    days = np.arange(n)
    seasonal_7d = 15.0 * np.sin(2 * np.pi * days / 7.0)
    seasonal_30d = 5.0 * np.cos(2 * np.pi * days / 30.0)
    baseline = 100.0 + 0.05 * days
    quantity = np.maximum(5.0, baseline + seasonal_7d + seasonal_30d)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Quantity": quantity,
            "Promotions": (dates.dayofweek == 4).astype(int),  # Promotions on Fridays
            "Holiday_Flag": (dates.day == 1).astype(int),      # Holidays on 1st of month
            "Sales_Amount": quantity * 25.0,
            "Profit": quantity * 5.0,
        }
    )
    return df


@pytest.fixture
def trained_ridge_model(synthetic_daily_data: pd.DataFrame) -> tuple[Ridge, pd.Timestamp]:
    """
    Train a simple Ridge model on 2024 data to test ML model evaluation.
    """
    train_slice = synthetic_daily_data[synthetic_daily_data["Date"] < "2025-01-01"].copy()
    train_slice = train_slice.set_index("Date")
    feature_df = build_feature_dataset(train_slice)

    valid_mask = feature_df[FEATURE_COLUMNS].notna().all(axis=1) & feature_df["Quantity"].notna()
    X = feature_df.loc[valid_mask, FEATURE_COLUMNS]
    y = feature_df.loc[valid_mask, "Quantity"]

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X, y)
    reference_start = train_slice.index.min()
    return model, reference_start


# =====================================================================
# 1. Seasonal Naive Baseline Tests
# =====================================================================

def test_seasonal_naive_7d_repetition():
    """Verify SeasonalNaiveModel cyclically repeats the last 7 observed days."""
    model = SeasonalNaiveModel(seasonal_period=7)
    history = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

    preds = model.forecast(history=history, steps=14)
    assert len(preds) == 14
    # First 7 steps match history
    assert preds[:7] == history
    # Next 7 steps recursively repeat the first 7 steps
    assert preds[7:14] == history


def test_seasonal_naive_non_negative_clipping():
    """Verify that SeasonalNaiveModel clips negative observations to 0.0."""
    model = SeasonalNaiveModel(seasonal_period=3)
    history = [-10.0, 0.0, 50.0]

    preds = model.forecast(history=history, steps=4)
    assert preds[0] == 0.0
    assert preds[1] == 0.0
    assert preds[2] == 50.0
    assert preds[3] == 0.0  # Recursive step repeats clipped 0.0


def test_seasonal_naive_insufficient_history():
    """Verify ValueError is raised when history length is less than seasonal_period."""
    model = SeasonalNaiveModel(seasonal_period=7)
    with pytest.raises(ValueError, match="at least 7"):
        model.forecast(history=[1.0, 2.0, 3.0], steps=5)


def test_seasonal_naive_predict_interface():
    """Verify SeasonalNaiveModel.predict works with feature DataFrames."""
    model = SeasonalNaiveModel(seasonal_period=7)
    X = pd.DataFrame(
        {
            "quantity_lag_7": [15.5, -5.0, 22.0],
            "day_of_week": [0, 1, 2],
        }
    )
    preds = model.predict(X)
    np.testing.assert_allclose(preds, [15.5, 0.0, 22.0])


# =====================================================================
# 2. Origin Generation & Partitioning Tests
# =====================================================================

def test_origin_generation_non_overlapping_blocks(synthetic_daily_data: pd.DataFrame):
    """
    Verify origin generation partitions test holdout into non-overlapping blocks
    with a short final block.
    """
    config = EvaluationConfig(
        horizons=(7, 30, 90),
        evaluation_start="2025-01-01",
        evaluation_end="2025-01-31",  # 31 days total
    )
    evaluator = RollingEvaluator(config=config)
    series_index = pd.DatetimeIndex(synthetic_daily_data["Date"])

    # Horizon 7: 31 days = 4 blocks of 7 + 1 block of 3 = 5 origins
    origins_7 = evaluator.generate_origins(series_index, horizon=7)
    assert len(origins_7) == 5
    assert len(origins_7[0][1]) == 7
    assert len(origins_7[1][1]) == 7
    assert len(origins_7[2][1]) == 7
    assert len(origins_7[3][1]) == 7
    assert len(origins_7[4][1]) == 3  # short final block
    total_dates_7 = sum(len(b) for _, b in origins_7)
    assert total_dates_7 == 31

    # Horizon 30: 31 days = 1 block of 30 + 1 block of 1 = 2 origins
    origins_30 = evaluator.generate_origins(series_index, horizon=30)
    assert len(origins_30) == 2
    assert len(origins_30[0][1]) == 30
    assert len(origins_30[1][1]) == 1  # short final block
    total_dates_30 = sum(len(b) for _, b in origins_30)
    assert total_dates_30 == 31


def test_expected_block_lengths_across_365_days():
    """
    Test G: Verify exact origin count and total days for a 365-day year.
    """
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    config = EvaluationConfig(
        evaluation_start="2025-01-01",
        evaluation_end="2025-12-31",
    )
    evaluator = RollingEvaluator(config=config)

    for h, expected_origins, expected_final_block in [(7, 53, 1), (30, 13, 5), (90, 5, 5)]:
        origins = evaluator.generate_origins(dates, horizon=h)
        assert len(origins) == expected_origins
        assert len(origins[-1][1]) == expected_final_block
        total_days = sum(len(b) for _, b in origins)
        assert total_days == 365


# =====================================================================
# 3. Canonical Continuity Guard (Test E)
# =====================================================================

def test_continuity_guard_missing_dates_raises_error(synthetic_daily_data: pd.DataFrame):
    """
    Test E: A time series with missing calendar days must be rejected by the evaluator.
    """
    discontinuous_df = synthetic_daily_data.copy()
    # Drop one day: 2025-01-15
    discontinuous_df = discontinuous_df[discontinuous_df["Date"] != "2025-01-15"].reset_index(drop=True)

    config = EvaluationConfig(
        evaluation_start="2025-01-01",
        evaluation_end="2025-01-31",
    )
    evaluator = RollingEvaluator(config=config)

    with pytest.raises(ValueError, match="continuity defects|missing calendar days"):
        evaluator.validate_time_series(discontinuous_df)


def test_continuity_guard_missing_indicator_raises_error(synthetic_daily_data: pd.DataFrame):
    """
    Test E2: An explicit is_missing=True indicator must trigger ValueError.
    """
    flagged_df = synthetic_daily_data.copy()
    flagged_df["is_missing"] = False
    flagged_df.loc[flagged_df["Date"] == "2025-01-15", "is_missing"] = True

    config = EvaluationConfig(
        evaluation_start="2025-01-01",
        evaluation_end="2025-01-31",
    )
    evaluator = RollingEvaluator(config=config)

    with pytest.raises(ValueError, match="missing calendar day indicators"):
        evaluator.validate_time_series(flagged_df)


# =====================================================================
# 4. Zero Demand Handling (Test F)
# =====================================================================

def test_zero_demand_evaluation(synthetic_daily_data: pd.DataFrame):
    """
    Test F: Verify zero-demand days (Quantity=0.0) do not crash metrics,
    do not cause division by zero, and set is_zero_actual=True.
    """
    zero_df = synthetic_daily_data.copy()
    # Set intermittent zeros in the test period
    zero_df.loc[zero_df["Date"] == "2025-01-10", "Quantity"] = 0.0
    zero_df.loc[zero_df["Date"] == "2025-01-11", "Quantity"] = 0.0

    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2025-01-01",
        evaluation_end="2025-01-21",
    )
    evaluator = RollingEvaluator(config=config)
    result = evaluator.evaluate_series(
        models=[SeasonalNaiveModel(seasonal_period=7)],
        daily=zero_df,
        include_baseline=False,
    )

    preds = result.predictions_df
    zero_rows = preds[preds["is_zero_actual"]]
    assert len(zero_rows) == 2
    assert (zero_rows["actual_quantity"] == 0.0).all()
    # APE should be NaN for zero actuals to avoid division by zero
    assert zero_rows["absolute_percentage_error"].isna().all()

    # Overall WAPE should still be valid and finite
    summary = result.summary_df.iloc[0]
    assert np.isfinite(summary["WAPE"])
    assert summary["WAPE"] > 0


# =====================================================================
# 5. Adversarial Future Target Leakage Tests
# =====================================================================

def test_adversarial_actuals_changed_in_block(
    synthetic_daily_data: pd.DataFrame,
    trained_ridge_model: tuple[Ridge, pd.Timestamp],
):
    """
    Test A: Radically modifying the actual targets inside the forecast block
    must NOT alter the recursive predictions produced for that block.
    """
    model, ref_start = trained_ridge_model

    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2025-01-01",
        evaluation_end="2025-01-07",
    )
    evaluator = RollingEvaluator(config=config)

    # 1. Base run
    res_base = evaluator.evaluate_series(
        models=[
            {
                "name": "RidgeModel",
                "model": model,
                "horizon": 7,
                "reference_start_date": ref_start,
            }
        ],
        daily=synthetic_daily_data,
        include_baseline=False,
    )
    preds_base = res_base.predictions_df["predicted_quantity"].values

    # 2. Corrupt actuals inside the block by multiplying by 100
    corrupted_df = synthetic_daily_data.copy()
    corrupt_mask = (corrupted_df["Date"] >= "2025-01-01") & (corrupted_df["Date"] <= "2025-01-07")
    corrupted_df.loc[corrupt_mask, "Quantity"] *= 100.0

    res_corrupt = evaluator.evaluate_series(
        models=[
            {
                "name": "RidgeModel",
                "model": model,
                "horizon": 7,
                "reference_start_date": ref_start,
            }
        ],
        daily=corrupted_df,
        include_baseline=False,
    )
    preds_corrupt = res_corrupt.predictions_df["predicted_quantity"].values

    # The predictions must be strictly identical - zero target leakage
    np.testing.assert_allclose(preds_base, preds_corrupt, rtol=1e-10)


def test_adversarial_actuals_changed_after_block(
    synthetic_daily_data: pd.DataFrame,
    trained_ridge_model: tuple[Ridge, pd.Timestamp],
):
    """
    Test B: Modifying future actuals after a block must NOT alter
    the predictions of that block.
    """
    model, ref_start = trained_ridge_model

    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2025-01-01",
        evaluation_end="2025-01-14",
    )
    evaluator = RollingEvaluator(config=config)

    # 1. Base run
    res_base = evaluator.evaluate_series(
        models=[
            {
                "name": "RidgeModel",
                "model": model,
                "horizon": 7,
                "reference_start_date": ref_start,
            }
        ],
        daily=synthetic_daily_data,
        include_baseline=False,
    )
    preds_block0_base = res_base.predictions_df[
        res_base.predictions_df["origin_date"] == "2025-01-01"
    ]["predicted_quantity"].values

    # 2. Corrupt dates in block 1 (2025-01-08 to 2025-01-14)
    corrupted_df = synthetic_daily_data.copy()
    corrupt_mask = corrupted_df["Date"] >= "2025-01-08"
    corrupted_df.loc[corrupt_mask, "Quantity"] = 9999.0

    res_corrupt = evaluator.evaluate_series(
        models=[
            {
                "name": "RidgeModel",
                "model": model,
                "horizon": 7,
                "reference_start_date": ref_start,
            }
        ],
        daily=corrupted_df,
        include_baseline=False,
    )
    preds_block0_corrupt = res_corrupt.predictions_df[
        res_corrupt.predictions_df["origin_date"] == "2025-01-01"
    ]["predicted_quantity"].values

    # Block 0 predictions must be completely unaffected
    np.testing.assert_allclose(preds_block0_base, preds_block0_corrupt, rtol=1e-10)


def test_history_reveals_actuals_only_after_block(
    synthetic_daily_data: pd.DataFrame,
    trained_ridge_model: tuple[Ridge, pd.Timestamp],
):
    """
    Test C: Verify that at origin 2, the ground truth from block 1 is present
    in the history used for forecasting block 2.
    """
    model, ref_start = trained_ridge_model

    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2025-01-01",
        evaluation_end="2025-01-14",
    )
    evaluator = RollingEvaluator(config=config)

    # If we replace block 1 actuals with zeros, block 2's step 1 (which uses lag 1)
    # should reflect that changed history
    modified_df = synthetic_daily_data.copy()
    # Change the last day of block 0 (2025-01-07) to 0.0
    modified_df.loc[modified_df["Date"] == "2025-01-07", "Quantity"] = 0.0

    res_mod = evaluator.evaluate_series(
        models=[
            {
                "name": "RidgeModel",
                "model": model,
                "horizon": 7,
                "reference_start_date": ref_start,
            }
        ],
        daily=modified_df,
        include_baseline=False,
    )

    res_orig = evaluator.evaluate_series(
        models=[
            {
                "name": "RidgeModel",
                "model": model,
                "horizon": 7,
                "reference_start_date": ref_start,
            }
        ],
        daily=synthetic_daily_data,
        include_baseline=False,
    )

    preds_mod_b1 = res_mod.predictions_df[
        res_mod.predictions_df["origin_date"] == "2025-01-08"
    ]["predicted_quantity"].values
    preds_orig_b1 = res_orig.predictions_df[
        res_orig.predictions_df["origin_date"] == "2025-01-08"
    ]["predicted_quantity"].values

    # Predictions for block 1 must differ because the revealed history changed
    assert not np.allclose(preds_mod_b1, preds_orig_b1)


def test_event_flags_from_forecast_date(synthetic_daily_data: pd.DataFrame):
    """
    Test D: Verify that known future event flags (promotion, holiday) are passed
    to the forecast step without leaking target quantity.
    """
    events_df = synthetic_daily_data[["Date", "Promotions", "Holiday_Flag"]].copy()
    # Set a special promotion on 2025-01-03
    events_df.loc[events_df["Date"] == "2025-01-03", "Promotions"] = 1

    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2025-01-01",
        evaluation_end="2025-01-07",
    )
    evaluator = RollingEvaluator(config=config)

    # Use a dummy model that checks promotion feature
    class PromotionAwareModel:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return 100.0 + 50.0 * X["Promotions"].to_numpy(dtype=float)

    result = evaluator.evaluate_series(
        models=[{"name": "PromoModel", "model": PromotionAwareModel(), "horizon": 7}],
        daily=synthetic_daily_data,
        events=events_df,
        include_baseline=False,
    )

    preds = result.predictions_df
    # On 2025-01-03, prediction should be 150.0; on other days 100.0
    day3_pred = preds.loc[preds["forecast_date"] == "2025-01-03", "predicted_quantity"].values[0]
    day1_pred = preds.loc[preds["forecast_date"] == "2025-01-01", "predicted_quantity"].values[0]
    assert day3_pred == 150.0
    assert day1_pred == 100.0


# =====================================================================
# 6. Metrics & Baseline Delta Tests
# =====================================================================

def test_pooled_vs_origin_level_metrics(synthetic_daily_data: pd.DataFrame):
    """
    Verify summary contains both pooled metrics and origin distribution statistics.
    """
    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2025-01-01",
        evaluation_end="2025-01-21",  # 3 origins
    )
    evaluator = RollingEvaluator(config=config)
    result = evaluator.evaluate_series(
        models=[SeasonalNaiveModel(seasonal_period=7)],
        daily=synthetic_daily_data,
        include_baseline=False,
    )

    summary = result.summary_df.iloc[0]
    origins = result.origin_metrics_df

    assert summary["number_of_origins"] == 3
    assert summary["total_forecast_days"] == 21

    # Verify origin stats match calculations from origin_metrics_df
    np.testing.assert_allclose(summary["mean_origin_WAPE"], origins["WAPE"].mean())
    np.testing.assert_allclose(summary["min_origin_WAPE"], origins["WAPE"].min())
    np.testing.assert_allclose(summary["max_origin_WAPE"], origins["WAPE"].max())
    np.testing.assert_allclose(summary["mean_origin_Bias"], origins["Bias"].mean())


def test_baseline_delta_calculation(
    synthetic_daily_data: pd.DataFrame,
    trained_ridge_model: tuple[Ridge, pd.Timestamp],
):
    """
    Verify WAPE_delta_vs_baseline, MAE_delta_vs_baseline, RMSE_delta_vs_baseline.
    """
    model, ref_start = trained_ridge_model

    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2025-01-01",
        evaluation_end="2025-01-21",
    )
    evaluator = RollingEvaluator(config=config)
    result = evaluator.evaluate_series(
        models=[
            {
                "name": "RidgeModel",
                "model": model,
                "horizon": 7,
                "reference_start_date": ref_start,
            }
        ],
        daily=synthetic_daily_data,
        include_baseline=True,
    )

    summary = result.summary_df
    assert len(summary) == 2

    baseline_row = summary[summary["model"] == "Seasonal Naive"].iloc[0]
    model_row = summary[summary["model"] == "RidgeModel"].iloc[0]

    # Baseline deltas must be 0.0
    assert baseline_row["WAPE_delta_vs_baseline"] == 0.0
    assert baseline_row["MAE_delta_vs_baseline"] == 0.0
    assert baseline_row["RMSE_delta_vs_baseline"] == 0.0

    # Model delta = model - baseline
    expected_wape_delta = round(model_row["WAPE"] - baseline_row["WAPE"], 6)
    expected_mae_delta = round(model_row["MAE"] - baseline_row["MAE"], 6)
    expected_rmse_delta = round(model_row["RMSE"] - baseline_row["RMSE"], 6)

    assert model_row["WAPE_delta_vs_baseline"] == expected_wape_delta
    assert model_row["MAE_delta_vs_baseline"] == expected_mae_delta
    assert model_row["RMSE_delta_vs_baseline"] == expected_rmse_delta


def test_bias_convention():
    """
    Verify Bias definition: Bias = actual - prediction.
    Positive bias = model under-predicts (actual > prediction).
    Negative bias = model over-predicts (actual < prediction).
    """
    actuals = np.array([100.0, 100.0])
    under_preds = np.array([80.0, 80.0])
    over_preds = np.array([120.0, 120.0])

    metrics_under = evaluate(actuals, under_preds)
    assert metrics_under["Bias"] == 20.0  # Positive -> under-prediction

    metrics_over = evaluate(actuals, over_preds)
    assert metrics_over["Bias"] == -20.0  # Negative -> over-prediction


# =====================================================================
# 7. Deterministic Repeatability (Test H)
# =====================================================================

def test_deterministic_repeatability(
    synthetic_daily_data: pd.DataFrame,
    trained_ridge_model: tuple[Ridge, pd.Timestamp],
):
    """
    Test H: Evaluating the same data and models twice must yield identical results.
    """
    model, ref_start = trained_ridge_model

    config = EvaluationConfig(
        horizons=(7, 30),
        evaluation_start="2025-01-01",
        evaluation_end="2025-01-31",
        run_id="repeatable_test",
    )
    evaluator = RollingEvaluator(config=config)

    models = [
        {
            "name": "RidgeModel",
            "model": model,
            "reference_start_date": ref_start,
        }
    ]

    res1 = evaluator.evaluate_series(models=models, daily=synthetic_daily_data, include_baseline=True)
    res2 = evaluator.evaluate_series(models=models, daily=synthetic_daily_data, include_baseline=True)

    pd.testing.assert_frame_equal(res1.summary_df, res2.summary_df)
    pd.testing.assert_frame_equal(res1.predictions_df, res2.predictions_df)
    pd.testing.assert_frame_equal(res1.origin_metrics_df, res2.origin_metrics_df)
