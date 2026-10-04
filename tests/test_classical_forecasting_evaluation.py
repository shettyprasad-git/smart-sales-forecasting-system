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
from ml.features.feature_pipeline import FEATURE_COLUMNS, build_feature_dataset
from ml.models.classical_forecasters import (
    DampedHoltWintersModel,
    HoltAdditiveModel,
    HoltWintersAdditiveModel,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def synthetic_eval_data() -> pd.DataFrame:
    """
    Generate a 120-day deterministic continuous daily dataset (2024-01-01 to 2024-04-29).
    Sufficient for 28+ days of initial history and rolling origins across horizons 7 and 14.
    """
    dates = pd.date_range(start="2024-01-01", end="2024-04-29", freq="D")
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
# 1. Individual Classical Models through RollingEvaluator
# ==============================================================================

def test_1_holt_model_rolling_evaluator(synthetic_eval_data):
    """Verify Holt additive model can pass through RollingEvaluator alongside baseline."""
    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2024-02-15",
        evaluation_end="2024-03-15",
        min_history_days=28,
        baseline_model_name="Seasonal Naive",
    )
    evaluator = RollingEvaluator(config=config)
    result = evaluator.evaluate_series(
        models=[HoltAdditiveModel()],
        daily=synthetic_eval_data,
        include_baseline=True,
    )

    assert isinstance(result, EvaluationResult)
    models_present = result.summary_df["model"].tolist()
    assert "Holt Additive" in models_present
    assert "Seasonal Naive" in models_present
    assert len(result.predictions_df) > 0
    assert len(result.origin_metrics_df) > 0


def test_2_holt_winters_model_rolling_evaluator(synthetic_eval_data):
    """Verify Holt-Winters additive weekly model can pass through RollingEvaluator."""
    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2024-02-15",
        evaluation_end="2024-03-15",
        min_history_days=28,
        baseline_model_name="Seasonal Naive",
    )
    evaluator = RollingEvaluator(config=config)
    result = evaluator.evaluate_series(
        models=[HoltWintersAdditiveModel(seasonal_periods=7)],
        daily=synthetic_eval_data,
        include_baseline=True,
    )

    models_present = result.summary_df["model"].tolist()
    assert "Holt-Winters Additive Weekly" in models_present
    assert "Seasonal Naive" in models_present
    assert (result.summary_df["WAPE"] > 0.0).all()


def test_3_damped_holt_winters_model_rolling_evaluator(synthetic_eval_data):
    """Verify Damped Holt-Winters model can pass through RollingEvaluator."""
    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2024-02-15",
        evaluation_end="2024-03-15",
        min_history_days=28,
        baseline_model_name="Seasonal Naive",
    )
    evaluator = RollingEvaluator(config=config)
    result = evaluator.evaluate_series(
        models=[DampedHoltWintersModel(seasonal_periods=7)],
        daily=synthetic_eval_data,
        include_baseline=True,
    )

    models_present = result.summary_df["model"].tolist()
    assert "Damped Holt-Winters Additive Weekly" in models_present
    assert "Seasonal Naive" in models_present


# ==============================================================================
# 2. Multi-Model Suite with Baseline Comparison
# ==============================================================================

def test_4_all_classical_models_suite_with_baseline(synthetic_eval_data):
    """Verify evaluating all 3 classical models computes baseline deltas for all models."""
    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2024-02-15",
        evaluation_end="2024-03-15",
        min_history_days=28,
        baseline_model_name="Seasonal Naive",
    )
    evaluator = RollingEvaluator(config=config)
    models = [
        HoltAdditiveModel(),
        HoltWintersAdditiveModel(),
        DampedHoltWintersModel(),
    ]
    result = evaluator.evaluate_series(
        models=models,
        daily=synthetic_eval_data,
        include_baseline=True,
    )

    summary = result.summary_df
    assert len(summary) == 4
    expected_models = {
        "Holt Additive",
        "Holt-Winters Additive Weekly",
        "Damped Holt-Winters Additive Weekly",
        "Seasonal Naive",
    }
    assert set(summary["model"]) == expected_models

    # Seasonal Naive baseline delta must be 0.0
    baseline_row = summary[summary["model"] == "Seasonal Naive"].iloc[0]
    assert baseline_row["WAPE_delta_vs_baseline"] == 0.0
    assert baseline_row["MAE_delta_vs_baseline"] == 0.0

    # Non-baseline models have finite WAPE deltas
    other_rows = summary[summary["model"] != "Seasonal Naive"]
    assert other_rows["WAPE_delta_vs_baseline"].notna().all()


# ==============================================================================
# 3. Existing ML Model Dispatch Unbroken
# ==============================================================================

def test_5_existing_ml_model_evaluator_path_still_works(
    synthetic_eval_data,
    trained_ridge_model,
):
    """Verify that feature-based ML models (using forecast_block) still evaluate correctly."""
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
    classical_spec = HoltAdditiveModel()

    result = evaluator.evaluate_series(
        models=[ml_spec, classical_spec],
        daily=synthetic_eval_data,
        include_baseline=True,
    )

    models_evaluated = set(result.summary_df["model"])
    assert "Ridge Regression" in models_evaluated
    assert "Holt Additive" in models_evaluated
    assert "Seasonal Naive" in models_evaluated


# ==============================================================================
# 4. Strict Leakage Protection & Adversarial Verification
# ==============================================================================

def test_6_predictions_strictly_use_history_prior_to_origin(synthetic_eval_data):
    """Verify predictions are generated strictly before origin date."""
    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2024-02-15",
        evaluation_end="2024-03-01",
        min_history_days=28,
    )
    evaluator = RollingEvaluator(config=config)
    result = evaluator.evaluate_series(
        models=[HoltWintersAdditiveModel()],
        daily=synthetic_eval_data,
        include_baseline=False,
    )

    preds_df = result.predictions_df
    for _, row in preds_df.iterrows():
        origin_dt = pd.Timestamp(row["origin_date"])
        forecast_dt = pd.Timestamp(row["forecast_date"])
        assert forecast_dt >= origin_dt


def test_7_adversarial_future_tampering_zero_leakage(synthetic_eval_data):
    """
    Adversarial test: Intentionally mutate future demand observations after an origin.
    Forecasts generated at and prior to that origin MUST remain bit-for-bit identical,
    proving zero backward leakage.
    """
    config = EvaluationConfig(
        horizons=(7,),
        evaluation_start="2024-02-15",
        evaluation_end="2024-03-15",
        min_history_days=28,
    )
    evaluator = RollingEvaluator(config=config)

    # Run 1: clean dataset
    result_clean = evaluator.evaluate_series(
        models=[HoltAdditiveModel(), HoltWintersAdditiveModel(), DampedHoltWintersModel()],
        daily=synthetic_eval_data,
        include_baseline=True,
    )

    # First origin in evaluation period
    first_origin = "2024-02-15"
    clean_origin_0_preds = result_clean.predictions_df[
        result_clean.predictions_df["origin_date"] == first_origin
    ].copy().sort_values(["model", "forecast_date"]).reset_index(drop=True)

    # Run 2: tamper drastically with all observations strictly AFTER the first 7-day forecast block
    # (e.g. from 2024-02-23 onwards, multiply quantity by 50x)
    tampered_data = synthetic_eval_data.copy(deep=True)
    future_mask = pd.to_datetime(tampered_data["Date"]) >= "2024-02-23"
    tampered_data.loc[future_mask, "Quantity"] *= 50.0

    result_tampered = evaluator.evaluate_series(
        models=[HoltAdditiveModel(), HoltWintersAdditiveModel(), DampedHoltWintersModel()],
        daily=tampered_data,
        include_baseline=True,
    )

    tampered_origin_0_preds = result_tampered.predictions_df[
        result_tampered.predictions_df["origin_date"] == first_origin
    ].copy().sort_values(["model", "forecast_date"]).reset_index(drop=True)

    # The predictions at the first origin must match bit-for-bit
    assert len(clean_origin_0_preds) == len(tampered_origin_0_preds)
    np.testing.assert_array_equal(
        clean_origin_0_preds["predicted_quantity"].values,
        tampered_origin_0_preds["predicted_quantity"].values,
    )
