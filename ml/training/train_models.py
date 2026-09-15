from pathlib import Path
import json

import joblib
import pandas as pd

from ml.data.data_loader import (
    load_processed_data,
    aggregate_daily_data,
)
from ml.data.data_validation import (
    validate_processed_data,
    validate_daily_data,
)
from ml.features.feature_pipeline import (
    build_feature_dataset,
    FEATURE_COLUMNS,
)
from ml.models.random_forest import create_model as create_rf
from ml.models.gradient_boosting import create_model as create_gb
from ml.models.linear_regression import create_model as create_lr


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

ARTIFACTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def train():
    print("=" * 70)
    print("SMART SALES FORECASTING - PRODUCTION MODEL TRAINING")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------
    print("\n[1/6] Loading processed dataset...")
    print(f"Dataset: {DATA_PATH}")

    df = load_processed_data(DATA_PATH)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # 2. Validate data
    # ---------------------------------------------------------
    print("\n[2/6] Validating processed dataset...")

    validate_processed_data(df)

    print("Processed dataset validation: PASSED")

    # ---------------------------------------------------------
    # 3. Aggregate daily data
    # ---------------------------------------------------------
    print("\n[3/6] Building daily demand dataset...")

    daily = aggregate_daily_data(df)

    validate_daily_data(daily)

    print(f"Daily observations: {len(daily):,}")
    print(
        f"Date range: "
        f"{daily.index.min().date()} → "
        f"{daily.index.max().date()}"
    )

    # ---------------------------------------------------------
    # Store the exact reference date used for
    # days_since_start during feature engineering.
    # ---------------------------------------------------------
    reference_start_date = pd.Timestamp(
        daily.index.min()
    ).normalize()

    print(
        f"Feature reference date: "
        f"{reference_start_date.date()}"
    )

    # ---------------------------------------------------------
    # 4. Build features
    # ---------------------------------------------------------
    print("\n[4/6] Building forecasting features...")

    features = build_feature_dataset(daily)

    print(f"Feature rows: {len(features):,}")
    print(f"Number of features: {len(FEATURE_COLUMNS)}")

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing feature columns: {missing_features}"
        )

    # ---------------------------------------------------------
    # Ensure Date is the index.
    # ---------------------------------------------------------
    if "Date" in features.columns:

        features["Date"] = pd.to_datetime(
            features["Date"],
            errors="raise",
        )

        features = (
            features
            .sort_values("Date")
            .set_index("Date")
        )

    elif isinstance(
        features.index,
        pd.DatetimeIndex,
    ):

        features.index = pd.to_datetime(
            features.index
        )

        features = features.sort_index()

    else:

        raise ValueError(
            "Feature dataset must contain a Date column "
            "or use a DatetimeIndex."
        )

    print(
        f"Feature date range: "
        f"{features.index.min().date()} → "
        f"{features.index.max().date()}"
    )

    # ---------------------------------------------------------
    # Verify that the first feature date is exactly 28 days
    # after the reference start date.
    # ---------------------------------------------------------
    expected_first_feature_day = (
        reference_start_date
        + pd.Timedelta(days=28)
    )

    actual_first_feature_day = (
        pd.Timestamp(features.index.min())
        .normalize()
    )

    if actual_first_feature_day != expected_first_feature_day:

        raise ValueError(
            "Feature reference date validation failed. "
            f"Expected first feature date "
            f"{expected_first_feature_day.date()}, "
            f"got {actual_first_feature_day.date()}."
        )

    print("Feature validation: PASSED")

    # ---------------------------------------------------------
    # 5. Time-based split
    # ---------------------------------------------------------
    print("\n[5/6] Creating time-based training split...")

    train_df = features.loc[
        features.index <= pd.Timestamp("2023-12-31")
    ].copy()

    validation_df = features.loc[
        (features.index >= pd.Timestamp("2024-01-01"))
        & (features.index <= pd.Timestamp("2024-12-31"))
    ].copy()

    test_df = features.loc[
        (features.index >= pd.Timestamp("2025-01-01"))
        & (features.index <= pd.Timestamp("2025-12-31"))
    ].copy()

    if train_df.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if validation_df.empty:
        raise ValueError(
            "Validation dataset is empty."
        )

    if test_df.empty:
        raise ValueError(
            "Test dataset is empty."
        )

    print(
        f"Training rows:   {len(train_df):,}"
    )

    print(
        f"Validation rows: {len(validation_df):,}"
    )

    print(
        f"Test rows:       {len(test_df):,}"
    )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["Quantity"]

    # ---------------------------------------------------------
    # 6. Train production models
    # ---------------------------------------------------------
    print("\n[6/6] Training production models...")

    jobs = [
        {
            "horizon": 7,
            "model_name": "Random Forest",
            "model": create_rf(),
            "filename": "demand_7d_random_forest.joblib",
        },
        {
            "horizon": 30,
            "model_name": "HistGradientBoosting",
            "model": create_gb(),
            "filename": "demand_30d_gradient_boosting.joblib",
        },
        {
            "horizon": 90,
            "model_name": "Linear Regression",
            "model": create_lr(),
            "filename": "demand_90d_linear_regression.joblib",
        },
    ]

    metadata = {
        "target": "Quantity",

        "reference_start_date": str(
            reference_start_date.date()
        ),

        "training_start": str(
            train_df.index.min().date()
        ),

        "training_end": str(
            train_df.index.max().date()
        ),

        "validation_start": str(
            validation_df.index.min().date()
        ),

        "validation_end": str(
            validation_df.index.max().date()
        ),

        "test_start": str(
            test_df.index.min().date()
        ),

        "test_end": str(
            test_df.index.max().date()
        ),

        "feature_count": len(FEATURE_COLUMNS),

        "features": FEATURE_COLUMNS,

        "models": {},
    }

    for job in jobs:

        print(
            f"\nTraining {job['model_name']} "
            f"for {job['horizon']}-day forecasting..."
        )

        model = job["model"]

        model.fit(
            X_train,
            y_train,
        )

        # -----------------------------------------------------
        # Save model artifact
        # -----------------------------------------------------
        artifact = {
            "model": model,

            "features": FEATURE_COLUMNS,

            "target": "Quantity",

            "horizon": job["horizon"],

            "model_name": job["model_name"],

            # Required by ForecastService for
            # days_since_start during inference.
            "reference_start_date": str(
                reference_start_date.date()
            ),
        }

        output_path = (
            ARTIFACTS_DIR
            / job["filename"]
        )

        joblib.dump(
            artifact,
            output_path,
        )

        metadata["models"][
            str(job["horizon"])
        ] = {
            "model_name": job["model_name"],

            "artifact": job["filename"],

            "reference_start_date": str(
                reference_start_date.date()
            ),
        }

        print(
            f"Saved: {output_path}"
        )

    # ---------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------
    metadata_path = (
        ARTIFACTS_DIR
        / "model_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"\nSaved metadata: {metadata_path}"
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nProduction models:")

    for job in jobs:

        print(
            f"  {job['horizon']:>3}-day → "
            f"{job['model_name']}"
        )

    print(
        f"\nArtifacts directory:\n"
        f"{ARTIFACTS_DIR}"
    )


if __name__ == "__main__":
    train()