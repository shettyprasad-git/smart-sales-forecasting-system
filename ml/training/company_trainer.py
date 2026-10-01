from __future__ import annotations

import io
import logging
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.data.canonical_series import build_canonical_daily_series
from ml.evaluation.metrics import evaluate
from ml.evaluation.recursive_forecast import forecast_block
from ml.features.feature_pipeline import (
    FEATURE_COLUMNS,
    build_feature_dataset,
    get_feature_columns,
)

logger = logging.getLogger(__name__)


class SeasonalNaiveModel:
    """
    Seasonal-Naive baseline predicting the value observed at the 7-day seasonal cycle.
    Compatible with scikit-learn estimator interface.
    """

    def __init__(self, lag_column: str = "quantity_lag_7") -> None:
        self.lag_column = lag_column
        self.fallback_value = 0.0

    def fit(self, X: Any, y: Any) -> "SeasonalNaiveModel":
        y_arr = np.asarray(y, dtype=float)
        self.fallback_value = float(np.nanmean(y_arr)) if len(y_arr) > 0 else 0.0
        return self

    def predict(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and self.lag_column in X.columns:
            preds = pd.to_numeric(X[self.lag_column], errors="coerce").fillna(self.fallback_value).to_numpy(dtype=float)
        elif hasattr(X, "iloc"):
            preds = np.full(len(X), self.fallback_value, dtype=float)
        else:
            preds = np.full(len(X), self.fallback_value, dtype=float)
        return np.maximum(0.0, preds)


def get_candidate_factories() -> dict[str, Callable[[], Any]]:
    """Return dictionary of factory callables for candidate models."""
    return {
        "Seasonal Naive": lambda: SeasonalNaiveModel(),
        "HistGradientBoosting": lambda: HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        ),
        "Ridge Regression": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0)),
            ]
        ),
    }


def calculate_min_required_days(horizon: int, min_train_days: int = 28) -> int:
    """
    Calculate minimum historical daily observations needed for horizon H:
    28 lag buffer + min_train_days + validation_window (H) + test_window (H).
    """
    return 28 + min_train_days + (2 * horizon)


def check_data_sufficiency(
    daily_df: pd.DataFrame,
    horizon: int,
    min_train_days: int = 28,
) -> tuple[bool, str | None]:
    """
    Verify whether the company daily aggregate has sufficient chronological history.
    """
    total_days = len(daily_df)
    min_needed = calculate_min_required_days(horizon, min_train_days=min_train_days)

    if total_days < min_needed:
        msg = (
            f"Dataset contains {total_days} daily observations; "
            f"{horizon}-day company model requires at least {min_needed} days "
            f"(28-day lag buffer + {min_train_days} training days + "
            f"{horizon}-day validation + {horizon}-day test). "
            f"Global pre-trained model will be used."
        )
        return False, msg

    return True, None


def split_chronological(
    df: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Strict chronological partition into Train, Validation, and Test sets.
    Test: trailing H rows
    Val: preceding H rows
    Train: all preceding rows
    Guarantees max(Train.Date) < min(Val.Date) <= max(Val.Date) < min(Test.Date).
    """
    total_rows = len(df)
    if total_rows < (2 * horizon + 1):
        raise ValueError(
            f"Not enough rows ({total_rows}) for chronological 3-way split with horizon {horizon}."
        )

    test_df = df.iloc[-horizon:].copy()
    val_df = df.iloc[-(2 * horizon) : -horizon].copy()
    train_df = df.iloc[: -(2 * horizon)].copy()

    # Invariant assertion
    if not train_df.empty and not val_df.empty and hasattr(df.index, "is_monotonic_increasing") and df.index.is_monotonic_increasing:
        assert train_df.index.max() < val_df.index.min(), "Lookahead leakage detected between Train and Validation!"
    if not val_df.empty and not test_df.empty and hasattr(df.index, "is_monotonic_increasing") and df.index.is_monotonic_increasing:
        assert val_df.index.max() < test_df.index.min(), "Lookahead leakage detected between Validation and Test!"

    return train_df, val_df, test_df


def benchmark_and_train_horizon(
    daily_df: pd.DataFrame,
    horizon: int,
    min_train_days: int = 28,
) -> dict[str, Any]:
    """
    Execute full feature engineering, chronological train/val/test splitting,
    validation-based candidate benchmarking (WAPE), test evaluation,
    refitting on Train + Validation, and artifact generation for a single horizon.
    """
    canonical_df, report = build_canonical_daily_series(daily_df)
    if not report.is_continuous:
        sample_str = f" Sample missing dates: {', '.join(report.missing_date_samples[:5])}." if report.missing_date_samples else ""
        gap_str = f" Longest gap: {report.longest_missing_gap} days." if report.longest_missing_gap > 0 else ""
        msg = (
            f"Dataset time series is discontinuous: {report.missing_days} calendar days are missing "
            f"out of {report.expected_days} expected days (coverage ratio: {report.coverage_ratio:.1%})."
            f"{gap_str}{sample_str} "
            f"Forecasting models require a complete daily calendar without gaps."
        )
        return {
            "horizon": horizon,
            "status": "insufficient_data",
            "status_message": msg,
            "model_type": None,
            "artifact_bytes": None,
            "validation_wape": None,
            "validation_mae": None,
            "validation_rmse": None,
            "test_wape": None,
            "test_mae": None,
            "test_rmse": None,
            "training_rows": None,
            "reference_start_date": None,
        }

    is_sufficient, insufficiency_msg = check_data_sufficiency(canonical_df, horizon, min_train_days)
    if not is_sufficient:
        return {
            "horizon": horizon,
            "status": "insufficient_data",
            "status_message": insufficiency_msg,
            "model_type": None,
            "artifact_bytes": None,
            "validation_wape": None,
            "validation_mae": None,
            "validation_rmse": None,
            "test_wape": None,
            "test_mae": None,
            "test_rmse": None,
            "training_rows": None,
            "reference_start_date": None,
        }

    df = canonical_df
    raw_dates = pd.to_datetime(df["Date"])
    reference_start_date = pd.Timestamp(raw_dates.min()).normalize()

    # Events lookup indexed by normalized Timestamp
    events_df = df.set_index(pd.to_datetime(df["Date"]).dt.normalize())

    # 2. Raw Chronological Partition
    train_daily = df.iloc[: -(2 * horizon)].copy()
    val_daily = df.iloc[-(2 * horizon) : -horizon].copy()
    test_daily = df.iloc[-horizon:].copy()

    train_history = list(train_daily["Quantity"].astype(float).values)
    val_dates = pd.DatetimeIndex(val_daily["Date"])
    val_actuals = val_daily["Quantity"].to_numpy(dtype=float)

    test_history = list(df.iloc[:-horizon]["Quantity"].astype(float).values)
    test_dates = pd.DatetimeIndex(test_daily["Date"])
    test_actuals = test_daily["Quantity"].to_numpy(dtype=float)

    # 3. Training Features (Built ONLY from historical training data)
    train_features = build_feature_dataset(train_daily)
    missing_features = [col for col in FEATURE_COLUMNS if col not in train_features.columns]
    if missing_features:
        raise ValueError(f"Feature pipeline missing required columns: {missing_features}")

    X_train = train_features[FEATURE_COLUMNS]
    y_train = train_features["Quantity"]

    # 4. Benchmark Candidates on Train -> Recursive Holdout Validation
    candidate_factories = get_candidate_factories()
    candidate_evaluations: list[dict[str, Any]] = []

    for name, factory in candidate_factories.items():
        try:
            model = factory()
            model.fit(X_train, y_train)

            val_preds = forecast_block(
                model=model,
                history=train_history,
                dates=val_dates,
                events=events_df,
                reference_start_date=reference_start_date,
                feature_columns=FEATURE_COLUMNS,
            )
            val_metrics = evaluate(val_actuals, np.asarray(val_preds, dtype=float))

            candidate_evaluations.append(
                {
                    "name": name,
                    "model": model,
                    "wape": val_metrics["WAPE"],
                    "mae": val_metrics["MAE"],
                    "rmse": val_metrics["RMSE"],
                }
            )
            logger.info(
                "Horizon %dD - Candidate %s: Val WAPE=%.4f, MAE=%.2f, RMSE=%.2f",
                horizon,
                name,
                val_metrics["WAPE"],
                val_metrics["MAE"],
                val_metrics["RMSE"],
            )
        except Exception as exc:
            logger.warning("Horizon %dD - Candidate %s failed evaluation: %s", horizon, name, exc)

    if not candidate_evaluations:
        raise RuntimeError(f"All candidate models failed during benchmarking for {horizon}-day horizon.")

    # 5. Model Selection: Lowest Validation WAPE (Test set strictly NOT used for selection)
    candidate_evaluations.sort(key=lambda x: (x["wape"], x["mae"], x["rmse"]))
    best_candidate = candidate_evaluations[0]
    winner_name = best_candidate["name"]
    best_val_wape = best_candidate["wape"]
    best_val_mae = best_candidate["mae"]
    best_val_rmse = best_candidate["rmse"]

    logger.info("Horizon %dD selected winning algorithm: %s (Val WAPE: %.4f)", horizon, winner_name, best_val_wape)

    # 6. Unbiased Test Set Evaluation (Using candidate model trained on Train only)
    trained_candidate_model = best_candidate["model"]
    test_preds = forecast_block(
        model=trained_candidate_model,
        history=test_history,
        dates=test_dates,
        events=events_df,
        reference_start_date=reference_start_date,
        feature_columns=FEATURE_COLUMNS,
    )
    test_metrics = evaluate(test_actuals, np.asarray(test_preds, dtype=float))

    logger.info(
        "Horizon %dD - Winning candidate %s Test WAPE=%.4f, MAE=%.2f, RMSE=%.2f",
        horizon,
        winner_name,
        test_metrics["WAPE"],
        test_metrics["MAE"],
        test_metrics["RMSE"],
    )

    # 7. Refit Winning Architecture on Train + Validation
    train_val_daily = df.iloc[:-horizon].copy()
    train_val_features = build_feature_dataset(train_val_daily)
    X_train_val = train_val_features[FEATURE_COLUMNS]
    y_train_val = train_val_features["Quantity"]

    final_model = candidate_factories[winner_name]()
    final_model.fit(X_train_val, y_train_val)

    # 8. Package Production Artifact
    artifact = {
        "model": final_model,
        "features": FEATURE_COLUMNS,
        "target": "Quantity",
        "horizon": horizon,
        "model_name": winner_name,
        "reference_start_date": str(reference_start_date.date()),
        "feature_version": "v1",
        "validation_wape": float(best_val_wape),
        "validation_mae": float(best_val_mae),
        "validation_rmse": float(best_val_rmse),
        "test_wape": float(test_metrics["WAPE"]),
        "test_mae": float(test_metrics["MAE"]),
        "test_rmse": float(test_metrics["RMSE"]),
        "training_rows": len(train_val_features),
    }

    # Serialize to memory bytes
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer)
    artifact_bytes = buffer.getvalue()

    return {
        "horizon": horizon,
        "status": "ready",
        "status_message": None,
        "model_type": winner_name,
        "artifact_bytes": artifact_bytes,
        "validation_wape": float(best_val_wape),
        "validation_mae": float(best_val_mae),
        "validation_rmse": float(best_val_rmse),
        "test_wape": float(test_metrics["WAPE"]),
        "test_mae": float(test_metrics["MAE"]),
        "test_rmse": float(test_metrics["RMSE"]),
        "training_rows": len(train_val_features),
        "reference_start_date": str(reference_start_date.date()),
    }
