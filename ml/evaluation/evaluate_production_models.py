from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from ml.data.data_loader import (
    load_processed_data,
    aggregate_daily_data,
)
from ml.data.data_validation import (
    validate_processed_data,
    validate_daily_data,
)
from ml.evaluation.rolling_evaluator import (
    EvaluationConfig,
    RollingEvaluator,
)
from ml.features.feature_pipeline import (
    FEATURE_COLUMNS,
    build_feature_dataset,
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_daily_forecasting_features.csv"
)

ARTIFACTS_DIR = (
    PROJECT_ROOT
    / "ml"
    / "artifacts"
)

RESULTS_PATH = (
    ARTIFACTS_DIR
    / "production_model_evaluation.csv"
)

PREDICTIONS_PATH = (
    ARTIFACTS_DIR
    / "production_model_predictions.csv"
)

ORIGIN_METRICS_PATH = (
    ARTIFACTS_DIR
    / "production_model_origin_metrics.csv"
)


MODELS = [
    {
        "horizon": 7,
        "model_name": "Random Forest",
        "filename": "demand_7d_random_forest.joblib",
    },
    {
        "horizon": 30,
        "model_name": "HistGradientBoosting",
        "filename": "demand_30d_gradient_boosting.joblib",
    },
    {
        "horizon": 90,
        "model_name": "Linear Regression",
        "filename": "demand_90d_linear_regression.joblib",
    },
]


def main():
    print("=" * 70)
    print("SMART SALES FORECASTING - PRODUCTION ROLLING EVALUATION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------
    print("\n[1/4] Loading processed dataset...")
    df = load_processed_data(DATA_PATH)
    print(f"Rows loaded: {len(df):,}")

    validate_processed_data(df)
    print("Processed dataset validation: PASSED")

    # ---------------------------------------------------------
    # 2. Build daily dataset
    # ---------------------------------------------------------
    print("\n[2/4] Building canonical daily dataset...")
    daily = aggregate_daily_data(df)
    validate_daily_data(daily)

    print(f"Daily observations: {len(daily):,}")
    print(f"Date range: {daily.index.min().date()} -> {daily.index.max().date()}")

    # ---------------------------------------------------------
    # 3. Validate feature pipeline
    # ---------------------------------------------------------
    print("\n[3/4] Validating production feature pipeline...")
    features = build_feature_dataset(daily)

    missing_features = [
        column for column in FEATURE_COLUMNS if column not in features.columns
    ]
    if missing_features:
        raise ValueError(f"Missing production features: {missing_features}")

    print(f"Feature rows: {len(features):,}")
    print(f"Feature count: {len(FEATURE_COLUMNS)}")
    print("Production feature pipeline: PASSED")

    # ---------------------------------------------------------
    # 4. Evaluate production models via RollingEvaluator
    # ---------------------------------------------------------
    print("\n[4/4] Evaluating models with rolling-origin recursive framework...")

    config = EvaluationConfig(
        horizons=(7, 30, 90),
        evaluation_start="2025-01-01",
        evaluation_end="2025-12-31",
        baseline_model_name="Seasonal Naive",
        seasonal_period=7,
        run_id="prod_eval_2025",
    )

    evaluator = RollingEvaluator(
        config=config,
        artifacts_dir=ARTIFACTS_DIR,
    )

    result = evaluator.evaluate_series(
        models=MODELS,
        daily=daily,
        include_baseline=True,
    )

    # ---------------------------------------------------------
    # Save evaluation artifacts
    # ---------------------------------------------------------
    saved_paths = result.save(ARTIFACTS_DIR)

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("PRODUCTION MODEL EVALUATION COMPLETED")
    print("=" * 70)

    display_columns = [
        "model",
        "horizon_days",
        "number_of_origins",
        "total_forecast_days",
        "MAE",
        "RMSE",
        "MAPE",
        "WAPE",
        "Bias",
        "WAPE_delta_vs_baseline",
    ]

    print("\nSummary Results:")
    summary_display = result.summary_df[display_columns].copy()
    summary_display["MAPE"] = (summary_display["MAPE"] * 100).round(2).astype(str) + "%"
    summary_display["WAPE"] = (summary_display["WAPE"] * 100).round(2).astype(str) + "%"
    summary_display["MAE"] = summary_display["MAE"].round(2)
    summary_display["RMSE"] = summary_display["RMSE"].round(2)
    summary_display["Bias"] = summary_display["Bias"].round(2)
    summary_display["WAPE_delta_vs_baseline"] = (
        (summary_display["WAPE_delta_vs_baseline"] * 100).round(2).astype(str) + "%"
    )

    print(summary_display.to_string(index=False))

    print(f"\nSaved evaluation summary:\n{saved_paths['summary']}")
    print(f"Saved prediction-level observations:\n{saved_paths['predictions']}")
    print(f"Saved per-origin metrics:\n{saved_paths['origin_metrics']}")


if __name__ == "__main__":
    main()
