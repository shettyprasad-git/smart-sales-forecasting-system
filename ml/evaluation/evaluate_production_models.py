from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.data.data_loader import (
    load_processed_data,
    aggregate_daily_data,
)
from ml.data.data_validation import (
    validate_processed_data,
    validate_daily_data,
)
from ml.evaluation.metrics import evaluate
from ml.evaluation.recursive_forecast import (
    forecast_block,
    rolling_origin_forecast,
)
from ml.features.feature_pipeline import (
    FEATURE_COLUMNS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


def evaluate_model(
    daily: pd.DataFrame,
    model_info: dict,
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate one production model using rolling-origin
    forecasting over the 2025 test period.
    """

    horizon = model_info["horizon"]
    model_name = model_info["model_name"]
    filename = model_info["filename"]

    print("\n" + "-" * 70)

    print(
        f"Evaluating {model_name} "
        f"({horizon}-day horizon)"
    )

    print("-" * 70)

    artifact_path = (
        ARTIFACTS_DIR
        / filename
    )

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: "
            f"{artifact_path}"
        )

    artifact = joblib.load(
        artifact_path
    )

    if not isinstance(
        artifact,
        dict,
    ):
        raise ValueError(
            f"Invalid model artifact: "
            f"{artifact_path}"
        )

    if "model" not in artifact:
        raise ValueError(
            f"Artifact does not contain "
            f"'model': {artifact_path}"
        )

    if "reference_start_date" not in artifact:
        raise ValueError(
            f"Artifact does not contain "
            f"'reference_start_date': "
            f"{artifact_path}"
        )

    model = artifact["model"]

    reference_start_date = pd.Timestamp(
        artifact[
            "reference_start_date"
        ]
    )

    # -----------------------------------------------------
    # Test period
    # -----------------------------------------------------

    test_start = pd.Timestamp(
        "2025-01-01"
    )

    test_end = pd.Timestamp(
        "2025-12-31"
    )

    test_series = daily.loc[
        test_start:test_end,
        "Quantity",
    ].astype(float)

    if test_series.empty:
        raise ValueError(
            "2025 test period is empty."
        )

    # -----------------------------------------------------
    # Historical data available before test period
    # -----------------------------------------------------

    history_series = daily.loc[
        daily.index < test_start,
        "Quantity",
    ].astype(float)

    if len(history_series) < 28:
        raise ValueError(
            "At least 28 historical observations "
            "are required before the test period."
        )

    # -----------------------------------------------------
    # Event data
    # -----------------------------------------------------

    event_columns = []

    if "Promotion" in daily.columns:
        event_columns.append(
            "Promotion"
        )

    if "Is_Holiday" in daily.columns:
        event_columns.append(
            "Is_Holiday"
        )

    if "Promotions" in daily.columns:
        event_columns.append(
            "Promotions"
        )

    if "Holiday_Flag" in daily.columns:
        event_columns.append(
            "Holiday_Flag"
        )

    events = daily[
        event_columns
    ].copy()

    # -----------------------------------------------------
    # Rolling-origin forecasting
    # -----------------------------------------------------

    actual, predicted = (
        rolling_origin_forecast(
            model=model,
            history_series=history_series,
            test_series=test_series,
            horizon=horizon,
            events=events,
            reference_start_date=reference_start_date,
        )
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    metrics = evaluate(
        y_true=actual,
        y_pred=predicted,
    )

    result = {
        "Model": model_name,
        "Horizon_Days": horizon,
        "Test_Start": str(
            test_start.date()
        ),
        "Test_End": str(
            test_end.date()
        ),
        "Test_Observations": len(actual),
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"],
        "WAPE": metrics["WAPE"],
        "Bias": metrics["Bias"],
    }

    # -----------------------------------------------------
    # Prediction dataframe
    # -----------------------------------------------------

    prediction_frame = pd.DataFrame(
        {
            "Date": test_series.index,
            "Actual_Quantity": actual,
            "Predicted_Quantity": predicted,
            "Model": model_name,
            "Horizon_Days": horizon,
        }
    )

    print(
        f"Test observations: "
        f"{len(actual):,}"
    )

    print(
        f"MAE:  {metrics['MAE']:,.2f}"
    )

    print(
        f"RMSE: {metrics['RMSE']:,.2f}"
    )

    print(
        f"MAPE: {metrics['MAPE'] * 100:.2f}%"
    )

    print(
        f"WAPE: {metrics['WAPE'] * 100:.2f}%"
    )

    print(
        f"Bias: {metrics['Bias']:,.2f}"
    )

    return (
        result,
        prediction_frame,
    )


def main():
    print("=" * 70)

    print(
        "SMART SALES FORECASTING - "
        "PRODUCTION MODEL EVALUATION"
    )

    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    print(
        "\n[1/4] Loading processed dataset..."
    )

    df = load_processed_data(
        DATA_PATH
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    validate_processed_data(
        df
    )

    print(
        "Processed dataset validation: PASSED"
    )

    # ---------------------------------------------------------
    # 2. Build daily dataset
    # ---------------------------------------------------------

    print(
        "\n[2/4] Building daily dataset..."
    )

    daily = aggregate_daily_data(
        df
    )

    validate_daily_data(
        daily
    )

    print(
        f"Daily observations: "
        f"{len(daily):,}"
    )

    print(
        f"Date range: "
        f"{daily.index.min().date()} → "
        f"{daily.index.max().date()}"
    )

    # ---------------------------------------------------------
    # 3. Validate feature pipeline
    # ---------------------------------------------------------

    print(
        "\n[3/4] Validating production feature pipeline..."
    )

    # Build the features to make sure the current
    # production feature pipeline remains valid.
    #
    # The evaluator itself uses the recursive feature
    # builder because the test must reproduce the actual
    # inference process.

    from ml.features.feature_pipeline import (
        build_feature_dataset,
    )

    features = build_feature_dataset(
        daily
    )

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing production features: "
            f"{missing_features}"
        )

    print(
        f"Feature rows: {len(features):,}"
    )

    print(
        f"Feature count: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        "Production feature pipeline: PASSED"
    )

    # ---------------------------------------------------------
    # 4. Evaluate production models
    # ---------------------------------------------------------

    print(
        "\n[4/4] Evaluating production models..."
    )

    results = []
    prediction_frames = []

    for model_info in MODELS:

        result, prediction_frame = (
            evaluate_model(
                daily=daily,
                model_info=model_info,
            )
        )

        results.append(
            result
        )

        prediction_frames.append(
            prediction_frame
        )

    # ---------------------------------------------------------
    # Results dataframe
    # ---------------------------------------------------------

    results_df = (
        pd.DataFrame(results)
        .sort_values(
            [
                "Horizon_Days",
                "WAPE",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "PRODUCTION MODEL EVALUATION COMPLETED"
    )

    print("=" * 70)

    print("\nResults:")

    display_columns = [
        "Model",
        "Horizon_Days",
        "MAE",
        "RMSE",
        "MAPE",
        "WAPE",
        "Bias",
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        f"\nSaved evaluation results:"
        f"\n{RESULTS_PATH}"
    )

    print(
        f"\nSaved predictions:"
        f"\n{PREDICTIONS_PATH}"
    )


if __name__ == "__main__":
    main()

