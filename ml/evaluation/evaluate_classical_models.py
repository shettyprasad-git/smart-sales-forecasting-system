from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from ml.data.data_loader import (
    aggregate_daily_data,
    load_processed_data,
)
from ml.data.data_validation import (
    validate_daily_data,
    validate_processed_data,
)
from ml.evaluation.rolling_evaluator import (
    EvaluationConfig,
    RollingEvaluator,
)
from ml.models.classical_forecasters import (
    DampedHoltWintersModel,
    HoltAdditiveModel,
    HoltWintersAdditiveModel,
)


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_daily_forecasting_features.csv"
)

CLASSICAL_ARTIFACTS_DIR = (
    PROJECT_ROOT
    / "ml"
    / "artifacts"
    / "classical_model_evaluation"
)


MODELS = [
    HoltAdditiveModel(),
    HoltWintersAdditiveModel(seasonal_periods=7),
    DampedHoltWintersModel(seasonal_periods=7),
]


def run_classical_evaluation() -> dict[str, Path]:
    """
    Execute formal rolling-origin evaluation for classical forecasting models
    (Holt Additive, Holt-Winters Additive Weekly, Damped Holt-Winters)
    benchmarked against Seasonal Naive on the 2025 canonical test period.
    """
    print("=" * 70)
    print("SMART SALES FORECASTING - CLASSICAL MODEL ROLLING EVALUATION")
    print("=" * 70)

    # 1. Load and validate data
    print("\n[1/3] Loading canonical processed dataset...")
    df = load_processed_data(DATA_PATH)
    validate_processed_data(df)
    print(f"Rows loaded: {len(df):,}")

    daily = aggregate_daily_data(df)
    validate_daily_data(daily)
    print(f"Canonical daily observations: {len(daily):,}")
    print(f"Date range: {daily.index.min().date()} -> {daily.index.max().date()}")

    # 2. Configure RollingEvaluator
    print("\n[2/3] Configuring rolling-origin evaluation (2025-01-01 to 2025-12-31)...")
    config = EvaluationConfig(
        horizons=(7, 30, 90),
        evaluation_start="2025-01-01",
        evaluation_end="2025-12-31",
        baseline_model_name="Seasonal Naive",
        seasonal_period=7,
        run_id="classical_eval_2025",
    )

    evaluator = RollingEvaluator(
        config=config,
        artifacts_dir=CLASSICAL_ARTIFACTS_DIR,
    )

    # 3. Evaluate classical models and Seasonal Naive baseline
    print("[3/3] Evaluating classical models across horizons [7, 30, 90]...")
    result = evaluator.evaluate_series(
        models=MODELS,
        daily=daily,
        include_baseline=True,
    )

    # 4. Save evaluation artifacts
    saved_paths = result.save(
        artifacts_dir=CLASSICAL_ARTIFACTS_DIR,
        prefix="classical_model",
    )

    # 5. Display results table
    print("\n" + "=" * 70)
    print("CLASSICAL MODEL EVALUATION COMPLETED")
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

    summary_display = result.summary_df[display_columns].copy()
    summary_display["MAPE"] = (summary_display["MAPE"] * 100).round(2).astype(str) + "%"
    summary_display["WAPE"] = (summary_display["WAPE"] * 100).round(2).astype(str) + "%"
    summary_display["MAE"] = summary_display["MAE"].round(2)
    summary_display["RMSE"] = summary_display["RMSE"].round(2)
    summary_display["Bias"] = summary_display["Bias"].round(2)
    summary_display["WAPE_delta_vs_baseline"] = (
        (summary_display["WAPE_delta_vs_baseline"] * 100).round(2).astype(str) + "%"
    )

    print("\nSummary Results (Sorted by Horizon and WAPE):")
    print(summary_display.to_string(index=False))

    print(f"\nSaved evaluation summary:\n  {saved_paths['summary']}")
    print(f"Saved prediction-level observations:\n  {saved_paths['predictions']}")
    print(f"Saved per-origin metrics:\n  {saved_paths['origin_metrics']}")

    return saved_paths


def main():
    run_classical_evaluation()


if __name__ == "__main__":
    main()
