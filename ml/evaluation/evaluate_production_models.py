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
from ml.features.feature_pipeline import (
    build_recursive_feature_row,
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


def forecast_block(
    model,
    history: list[float],
    dates: pd.DatetimeIndex,
    events: pd.DataFrame,
    reference_start_date: pd.Timestamp,
) -> list[float]:
    """
    Forecast one complete block recursively.

    Predictions within the block are fed back into history,
    matching the recursive forecasting logic from Phase 3.
    """

    predictions = []

    for date in dates:

        promotion = 0
        holiday = 0

        if date in events.index:

            event = events.loc[date]

            if isinstance(
                event,
                pd.DataFrame,
            ):
                event = event.iloc[0]

            promotion = event.get(
                "Promotion",
                event.get(
                    "Promotions",
                    0,
                ),
            )

            holiday = event.get(
                "Is_Holiday",
                event.get(
                    "Holiday_Flag",
                    0,
                ),
            )

        feature_row = build_recursive_feature_row(
            history=history,
            future_date=date,
            promotion=promotion,
            holiday=holiday,
            reference_start_date=reference_start_date,
        )

        X = feature_row[
            FEATURE_COLUMNS
        ]

        prediction = float(
            model.predict(X)[0]
        )

        prediction = max(
            0.0,
            prediction,
        )

        predictions.append(
            prediction
        )

        # Recursive update within the forecast block.
        history.append(
            prediction
        )

    return predictions


def rolling_origin_forecast(
    model,
    history_series: pd.Series,
    test_series: pd.Series,
    horizon: int,
    events: pd.DataFrame,
    reference_start_date: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform rolling-origin recursive forecasting.

    Process:

        1. Start with all historical observations.
        2. Forecast one block of `horizon` days.
        3. Compare against the actual block.
        4. Add the actual observations to history.
        5. Forecast the next block.
        6. Repeat until the test period is complete.

    This matches the Phase 3 methodology.
    """

    history = list(
        history_series.astype(
            float
        ).values
    )

    actual = []
    predicted = []

    start = 0

    while start < len(test_series):

        block_size = min(
            horizon,
            len(test_series) - start,
        )

        dates = pd.DatetimeIndex(
            test_series.index[
                start:start + block_size
            ]
        )

        # ---------------------------------------------
        # Forecast this block recursively.
        # ---------------------------------------------

        block_predictions = forecast_block(
            model=model,
            history=history.copy(),
            dates=dates,
            events=events,
            reference_start_date=reference_start_date,
        )

        block_actual = (
            test_series
            .iloc[
                start:start + block_size
            ]
            .astype(float)
            .values
        )

        actual.extend(
            block_actual
        )

        predicted.extend(
            block_predictions
        )

        # ---------------------------------------------
        # Critical rolling-origin step:
        # reveal actual observations only after the
        # forecast block has been completed.
        # ---------------------------------------------

        history.extend(
            block_actual.tolist()
        )

        start += block_size

    return (
        np.asarray(actual),
        np.asarray(predicted),
    )


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

