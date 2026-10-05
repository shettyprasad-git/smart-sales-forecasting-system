from __future__ import annotations

import json
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
from ml.evaluation.select_sarima_order import (
    SELECTION_CUTOFF_DATE,
    select_sarima_order,
)
from ml.models.sarima_forecaster import SARIMAForecaster


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_daily_forecasting_features.csv"
)

SARIMA_ARTIFACTS_DIR = (
    PROJECT_ROOT
    / "ml"
    / "artifacts"
    / "sarima_evaluation"
)


def run_sarima_evaluation() -> dict[str, Path]:
    """
    Execute formal rolling-origin evaluation for the selected SARIMA model
    benchmarked against Seasonal Naive on the 2025 canonical test period.

    Evaluation Invariants:
      1. Order selection performed strictly on pre-2025 data (Date <= 2024-12-31).
      2. The selected SARIMA specification is locked prior to holdout evaluation.
      3. At each rolling origin, SARIMA is refit strictly on history prior to that origin.
      4. Evaluates across horizons 7, 30, and 90 days.
      5. Saves all artifacts to ml/artifacts/sarima_evaluation/ with prefix sarima_model_.
    """
    print("=" * 70)
    print("SMART SALES FORECASTING - SARIMA ROLLING EVALUATION")
    print("=" * 70)

    # 1. Load and validate canonical dataset
    print("\n[1/4] Loading canonical processed dataset...")
    df = load_processed_data(DATA_PATH)
    validate_processed_data(df)
    print(f"Rows loaded: {len(df):,}")

    daily = aggregate_daily_data(df)
    validate_daily_data(daily)
    print(f"Canonical daily observations: {len(daily):,}")
    print(f"Date range: {daily.index.min().date()} -> {daily.index.max().date()}")

    # 2. Select and lock SARIMA order strictly on pre-2025 data
    print(f"\n[2/4] Selecting optimal SARIMA order strictly on data <= {SELECTION_CUTOFF_DATE}...")
    selected_order, selected_s_order, selection_df, selection_meta = select_sarima_order(
        daily=daily,
        cutoff_date=SELECTION_CUTOFF_DATE,
        artifacts_dir=SARIMA_ARTIFACTS_DIR,
    )

    # Also persist to root ml/artifacts
    root_artifacts = PROJECT_ROOT / "ml" / "artifacts"
    selection_df.to_csv(root_artifacts / "sarima_order_selection.csv", index=False)
    with open(root_artifacts / "sarima_selected_order.json", "w", encoding="utf-8") as f:
        json.dump(selection_meta, f, indent=2)

    print(
        f"LOCKED SARIMA specification: {selected_order} x {selected_s_order} "
        f"(AIC={selection_meta['selected_aic']:.2f}, BIC={selection_meta['selected_bic']:.2f})"
    )

    # 3. Instantiate locked SARIMA model and configure RollingEvaluator
    print("\n[3/4] Configuring rolling-origin evaluation (2025-01-01 to 2025-12-31)...")
    sarima_model = SARIMAForecaster(
        order=selected_order,
        seasonal_order=selected_s_order,
    )

    config = EvaluationConfig(
        horizons=(7, 30, 90),
        evaluation_start="2025-01-01",
        evaluation_end="2025-12-31",
        baseline_model_name="Seasonal Naive",
        seasonal_period=7,
        run_id="sarima_eval_2025",
    )

    evaluator = RollingEvaluator(
        config=config,
        artifacts_dir=SARIMA_ARTIFACTS_DIR,
    )

    # 4. Evaluate SARIMA and Seasonal Naive baseline
    print(f"[4/4] Evaluating {sarima_model.name} and Seasonal Naive across horizons [7, 30, 90]...")
    result = evaluator.evaluate_series(
        models=[sarima_model],
        daily=daily,
        include_baseline=True,
    )

    # 5. Persist evaluation artifacts
    saved_paths = result.save(
        artifacts_dir=SARIMA_ARTIFACTS_DIR,
        prefix="sarima_model",
    )
    saved_paths["order_selection_csv"] = SARIMA_ARTIFACTS_DIR / "sarima_order_selection.csv"
    saved_paths["selected_order_json"] = SARIMA_ARTIFACTS_DIR / "sarima_selected_order.json"

    # 6. Display results table
    print("\n" + "=" * 70)
    print("SARIMA EVALUATION COMPLETED")
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
    print(f"Saved order selection CSV:\n  {saved_paths['order_selection_csv']}")
    print(f"Saved selected order JSON:\n  {saved_paths['selected_order_json']}")

    return saved_paths


if __name__ == "__main__":
    run_sarima_evaluation()
