from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence
import uuid

import joblib
import numpy as np
import pandas as pd

from ml.data.canonical_series import assess_time_series_quality
from ml.evaluation.metrics import evaluate, wape
from ml.evaluation.recursive_forecast import extract_event_flags, forecast_block
from ml.features.feature_pipeline import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"


# =====================================================================
# 1. Seasonal Naive Baseline Model
# =====================================================================

class SeasonalNaiveModel:
    """
    Deterministic seasonal naive forecasting baseline.

    Predicts the observation from `seasonal_period` days prior (default: 7 days).
    Recursively feeds prior predictions back into history for multi-step horizons.
    Clips predictions to non-negative values: max(0.0, prediction).
    """

    def __init__(self, seasonal_period: int = 7):
        if seasonal_period <= 0:
            raise ValueError(f"seasonal_period must be positive, got {seasonal_period}.")
        self.seasonal_period = seasonal_period
        self.name = "Seasonal Naive"

    def forecast(
        self,
        history: Sequence[float] | np.ndarray | pd.Series,
        steps: int,
    ) -> list[float]:
        """
        Produce a deterministic multi-step forecast using recursive seasonal lags.
        """
        if steps <= 0:
            return []

        hist = list(pd.Series(history, dtype=float).values)
        if len(hist) < self.seasonal_period:
            raise ValueError(
                f"History has {len(hist)} observations, but seasonal_period requires at least {self.seasonal_period}."
            )

        preds: list[float] = []
        for _ in range(steps):
            lag_val = hist[-self.seasonal_period]
            clipped = max(0.0, float(lag_val))
            preds.append(clipped)
            hist.append(clipped)

        return preds

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Scikit-learn compatible predict interface.
        Extracts seasonal lag column from the feature matrix X.
        """
        col_name = f"quantity_lag_{self.seasonal_period}"
        if isinstance(X, pd.DataFrame):
            if col_name in X.columns:
                return np.maximum(0.0, X[col_name].to_numpy(dtype=float))
            if "quantity_lag_7" in X.columns:
                return np.maximum(0.0, X["quantity_lag_7"].to_numpy(dtype=float))
            # Fallback to first numeric column if named lag is absent
            return np.maximum(0.0, X.iloc[:, 0].to_numpy(dtype=float))

        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            return np.maximum(0.0, arr)
        return np.maximum(0.0, arr[:, 0])

    def __repr__(self) -> str:
        return f"SeasonalNaiveModel(seasonal_period={self.seasonal_period})"


def forecast_seasonal_naive(
    history: Sequence[float] | np.ndarray | pd.Series,
    steps: int,
    seasonal_period: int = 7,
) -> list[float]:
    """
    Convenience function for deterministic Seasonal Naive multi-step forecasting.
    """
    model = SeasonalNaiveModel(seasonal_period=seasonal_period)
    return model.forecast(history=history, steps=steps)


# =====================================================================
# 2. Evaluation Configuration and Results
# =====================================================================

@dataclass(frozen=True)
class EvaluationConfig:
    """
    Configuration parameters for rolling-origin evaluation.
    """
    horizons: tuple[int, ...] = (7, 30, 90)
    evaluation_start: Optional[str | pd.Timestamp] = None
    evaluation_end: Optional[str | pd.Timestamp] = None
    origin_step: Optional[int] = None  # None = non-overlapping block steps equal to horizon
    min_history_days: int = 28
    baseline_model_name: str = "Seasonal Naive"
    seasonal_period: int = 7
    dataset_id: Optional[str] = None
    run_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.evaluation_start is not None:
            d["evaluation_start"] = str(pd.Timestamp(self.evaluation_start).date())
        if self.evaluation_end is not None:
            d["evaluation_end"] = str(pd.Timestamp(self.evaluation_end).date())
        return d


@dataclass
class EvaluationResult:
    """
    Container for rolling-origin evaluation output artifacts.
    """
    config: EvaluationConfig
    summary_df: pd.DataFrame
    predictions_df: pd.DataFrame
    origin_metrics_df: pd.DataFrame

    def save(
        self,
        artifacts_dir: Path | str,
        prefix: Optional[str] = None,
    ) -> dict[str, Path]:
        """
        Persist evaluation artifacts to disk.
        """
        out_dir = Path(artifacts_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        prefix_str = f"{prefix}_" if prefix else "production_model_"
        summary_path = out_dir / f"{prefix_str}evaluation.csv"
        predictions_path = out_dir / f"{prefix_str}predictions.csv"
        origin_metrics_path = out_dir / f"{prefix_str}origin_metrics.csv"

        self.summary_df.to_csv(summary_path, index=False)
        self.predictions_df.to_csv(predictions_path, index=False)
        self.origin_metrics_df.to_csv(origin_metrics_path, index=False)

        return {
            "summary": summary_path,
            "predictions": predictions_path,
            "origin_metrics": origin_metrics_path,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "summary": self.summary_df.to_dict(orient="records"),
            "origin_count": len(self.origin_metrics_df),
            "prediction_count": len(self.predictions_df),
        }


# =====================================================================
# 3. Model Specification Helper
# =====================================================================

@dataclass
class ModelSpec:
    name: str
    model: Any
    horizon: int
    reference_start_date: Optional[pd.Timestamp] = None
    feature_columns: Optional[list[str]] = None


# =====================================================================
# 4. Rolling Evaluator Framework
# =====================================================================

class RollingEvaluator:
    """
    Formal rolling-origin recursive forecasting evaluator.

    Enforces:
      - Zero target leakage through recursive multi-step forecasting.
      - Canonical calendar continuity with strict quality verification.
      - Deterministic baseline comparison (Seasonal Naive).
      - Transparent tracking of origin-level metric distributions and bias.
    """

    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        artifacts_dir: Optional[Path | str] = None,
    ):
        self.config = config or EvaluationConfig()
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else DEFAULT_ARTIFACTS_DIR
        self.run_id = self.config.run_id or str(uuid.uuid4())[:8]

    def validate_time_series(self, daily: pd.DataFrame) -> pd.DataFrame:
        """
        Validate time-series format and enforce canonical daily continuity.
        Raises ValueError if calendar gaps or discontinuities are present.
        """
        if daily is None or daily.empty:
            raise ValueError("Input daily dataset is empty or None.")

        work_df = daily.copy()
        if "Date" in work_df.columns:
            work_df["Date"] = pd.to_datetime(work_df["Date"]).dt.normalize()
            work_df = work_df.set_index("Date")
        elif isinstance(work_df.index, pd.DatetimeIndex):
            work_df.index = work_df.index.normalize()
        else:
            raise ValueError("Input data must contain a 'Date' column or DatetimeIndex.")

        work_df = work_df.sort_index()

        if "Quantity" not in work_df.columns:
            raise ValueError("Input data must contain a 'Quantity' column.")

        # Check explicit missing date indicator if provided
        if "is_missing" in work_df.columns:
            missing_count = int(work_df["is_missing"].astype(bool).sum())
            if missing_count > 0:
                raise ValueError(
                    f"Input time series contains {missing_count} missing calendar day indicators. "
                    "Canonical daily continuity is required for evaluation."
                )

        # Enforce canonical calendar continuity
        report = assess_time_series_quality(work_df.reset_index())
        if not report.is_continuous or report.missing_days > 0:
            raise ValueError(
                f"Input time series has continuity defects: {report.missing_days} missing calendar days detected "
                f"between {report.min_date} and {report.max_date}. Canonical daily continuity is required."
            )

        return work_df

    def generate_origins(
        self,
        series_index: pd.DatetimeIndex,
        horizon: int,
    ) -> list[tuple[pd.Timestamp, pd.DatetimeIndex]]:
        """
        Generate (origin_date, block_dates) pairs for a given horizon.

        When config.origin_step is None (default):
          Partitions the evaluation holdout period into consecutive, non-overlapping
          blocks of length `min(horizon, remaining_dates)`.
        When config.origin_step is set:
          Advances origin by `origin_step` days while origin <= evaluation_end.
        """
        if horizon <= 0:
            raise ValueError(f"Horizon must be positive, got {horizon}.")

        all_dates = series_index.sort_values().drop_duplicates()

        eval_start = (
            pd.Timestamp(self.config.evaluation_start).normalize()
            if self.config.evaluation_start is not None
            else all_dates[0] + pd.Timedelta(days=self.config.min_history_days)
        )

        eval_end = (
            pd.Timestamp(self.config.evaluation_end).normalize()
            if self.config.evaluation_end is not None
            else all_dates[-1]
        )

        if eval_start > eval_end:
            raise ValueError(
                f"evaluation_start ({eval_start.date()}) cannot be after evaluation_end ({eval_end.date()})."
            )

        prior_history = all_dates[all_dates < eval_start]
        if len(prior_history) < self.config.min_history_days:
            raise ValueError(
                f"Insufficient historical observations before evaluation_start ({eval_start.date()}): "
                f"required at least {self.config.min_history_days}, found {len(prior_history)}."
            )

        test_dates = all_dates[(all_dates >= eval_start) & (all_dates <= eval_end)]
        if len(test_dates) == 0:
            raise ValueError(
                f"No test observations found between {eval_start.date()} and {eval_end.date()}."
            )

        origins: list[tuple[pd.Timestamp, pd.DatetimeIndex]] = []

        if self.config.origin_step is None:
            # Non-overlapping partitions covering test_dates exactly
            start_idx = 0
            total_test = len(test_dates)
            while start_idx < total_test:
                block_size = min(horizon, total_test - start_idx)
                origin_date = test_dates[start_idx]
                block_dates = pd.DatetimeIndex(test_dates[start_idx : start_idx + block_size])
                origins.append((origin_date, block_dates))
                start_idx += block_size
        else:
            # Stepped rolling origins
            step = self.config.origin_step
            curr_origin = eval_start
            while curr_origin <= eval_end:
                block_end = curr_origin + pd.Timedelta(days=horizon)
                block_dates = pd.DatetimeIndex(
                    test_dates[(test_dates >= curr_origin) & (test_dates < block_end)]
                )
                if len(block_dates) > 0:
                    origins.append((curr_origin, block_dates))
                curr_origin = curr_origin + pd.Timedelta(days=step)

        return origins

    def evaluate_model(
        self,
        model: Any,
        daily: pd.DataFrame,
        horizon: int,
        model_name: str,
        reference_start_date: Optional[pd.Timestamp | str] = None,
        feature_columns: Optional[list[str]] = None,
        events: Optional[pd.DataFrame] = None,
    ) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
        """
        Evaluate a single model for a specific horizon across all rolling origins.
        """
        valid_df = self.validate_time_series(daily)
        origins = self.generate_origins(valid_df.index, horizon)

        all_actuals: list[float] = []
        all_preds: list[float] = []
        prediction_rows: list[dict[str, Any]] = []
        origin_rows: list[dict[str, Any]] = []

        ref_start = (
            pd.Timestamp(reference_start_date).normalize()
            if reference_start_date is not None
            else valid_df.index.min()
        )

        for origin_date, block_dates in origins:
            # Historical observations available STRICTLY BEFORE origin_date
            history_series = valid_df.loc[valid_df.index < origin_date, "Quantity"].astype(float)
            if len(history_series) < self.config.min_history_days:
                raise ValueError(
                    f"Insufficient history ({len(history_series)} days) before origin {origin_date.date()}."
                )

            history_list = history_series.tolist()

            # Recursive forecast for the block
            if hasattr(model, "forecast") and callable(getattr(model, "forecast")):
                try:
                    block_predictions = model.forecast(
                        history=history_list,
                        steps=len(block_dates),
                    )
                except TypeError:
                    block_predictions = model.forecast(
                        history_list,
                        len(block_dates),
                    )
            else:
                block_predictions = forecast_block(
                    model=model,
                    history=history_list,
                    dates=block_dates,
                    events=events,
                    reference_start_date=ref_start,
                    feature_columns=feature_columns,
                )

            # Ensure non-negative demand predictions
            block_predictions = [max(0.0, float(p)) for p in block_predictions]
            block_actuals = valid_df.loc[block_dates, "Quantity"].astype(float).values

            all_actuals.extend(block_actuals.tolist())
            all_preds.extend(block_predictions)

            # Compute origin-level metrics
            origin_metrics = evaluate(
                y_true=block_actuals,
                y_pred=np.asarray(block_predictions, dtype=float),
            )

            origin_row = {
                "origin_date": origin_date.strftime("%Y-%m-%d"),
                "horizon_days": horizon,
                "model": model_name,
                "test_observations": len(block_actuals),
                "MAE": origin_metrics["MAE"],
                "RMSE": origin_metrics["RMSE"],
                "MAPE": origin_metrics["MAPE"],
                "WAPE": origin_metrics["WAPE"],
                "Bias": origin_metrics["Bias"],
                "dataset_id": self.config.dataset_id,
                "evaluation_run_id": self.run_id,
            }
            origin_rows.append(origin_row)

            # Record per-prediction observations
            for idx, date_val in enumerate(block_dates):
                act = float(block_actuals[idx])
                pred = float(block_predictions[idx])
                err = act - pred
                abs_err = abs(err)
                sq_err = err ** 2
                ape = (abs_err / act) if act != 0.0 else np.nan
                is_zero = (act == 0.0)

                pred_row = {
                    "origin_date": origin_date.strftime("%Y-%m-%d"),
                    "forecast_date": date_val.strftime("%Y-%m-%d"),
                    "horizon_days": horizon,
                    "model": model_name,
                    "actual_quantity": act,
                    "predicted_quantity": pred,
                    "error": err,
                    "absolute_error": abs_err,
                    "squared_error": sq_err,
                    "absolute_percentage_error": ape,
                    "is_zero_actual": is_zero,
                    "dataset_id": self.config.dataset_id,
                    "evaluation_run_id": self.run_id,
                }
                prediction_rows.append(pred_row)

        # Pooled metrics across all origins
        y_true_all = np.asarray(all_actuals, dtype=float)
        y_pred_all = np.asarray(all_preds, dtype=float)
        pooled_metrics = evaluate(y_true=y_true_all, y_pred=y_pred_all)

        # Origin distribution statistics
        origin_wapes = [r["WAPE"] for r in origin_rows if not np.isnan(r["WAPE"])]
        origin_biases = [r["Bias"] for r in origin_rows if not np.isnan(r["Bias"])]

        mean_origin_wape = float(np.mean(origin_wapes)) if origin_wapes else np.nan
        median_origin_wape = float(np.median(origin_wapes)) if origin_wapes else np.nan
        std_origin_wape = float(np.std(origin_wapes, ddof=1)) if len(origin_wapes) > 1 else 0.0
        min_origin_wape = float(np.min(origin_wapes)) if origin_wapes else np.nan
        max_origin_wape = float(np.max(origin_wapes)) if origin_wapes else np.nan

        mean_origin_bias = float(np.mean(origin_biases)) if origin_biases else np.nan
        std_origin_bias = float(np.std(origin_biases, ddof=1)) if len(origin_biases) > 1 else 0.0

        summary = {
            "model": model_name,
            "horizon_days": horizon,
            "number_of_origins": len(origins),
            "total_forecast_days": len(all_actuals),
            "MAE": pooled_metrics["MAE"],
            "RMSE": pooled_metrics["RMSE"],
            "MAPE": pooled_metrics["MAPE"],
            "WAPE": pooled_metrics["WAPE"],
            "Bias": pooled_metrics["Bias"],
            "mean_origin_WAPE": mean_origin_wape,
            "median_origin_WAPE": median_origin_wape,
            "std_origin_WAPE": std_origin_wape,
            "min_origin_WAPE": min_origin_wape,
            "max_origin_WAPE": max_origin_wape,
            "mean_origin_Bias": mean_origin_bias,
            "std_origin_Bias": std_origin_bias,
        }

        preds_df = pd.DataFrame(prediction_rows)
        origin_metrics_df = pd.DataFrame(origin_rows)

        return summary, preds_df, origin_metrics_df

    def _normalize_model_specs(
        self,
        models: list[dict | tuple[str, Any] | Any],
    ) -> list[ModelSpec]:
        """
        Normalize incoming model representations into standard ModelSpec objects.
        """
        specs: list[ModelSpec] = []

        for item in models:
            if isinstance(item, dict):
                model_name = item.get("model_name", item.get("name", "Model"))
                horizon = item.get("horizon", item.get("horizon_days"))
                model_obj = item.get("model")
                ref_start = item.get("reference_start_date")
                feature_cols = item.get("feature_columns")

                if model_obj is None and "filename" in item:
                    filename = item["filename"]
                    filepath = self.artifacts_dir / filename
                    if not filepath.exists():
                        raise FileNotFoundError(f"Model artifact not found: {filepath}")
                    artifact = joblib.load(filepath)
                    if isinstance(artifact, dict):
                        model_obj = artifact.get("model")
                        ref_start = artifact.get("reference_start_date", ref_start)
                    else:
                        model_obj = artifact

                horizons_to_add = [horizon] if horizon is not None else list(self.config.horizons)
                for h in horizons_to_add:
                    specs.append(
                        ModelSpec(
                            name=model_name,
                            model=model_obj,
                            horizon=int(h),
                            reference_start_date=pd.Timestamp(ref_start) if ref_start else None,
                            feature_columns=feature_cols,
                        )
                    )

            elif isinstance(item, (tuple, list)):
                if len(item) == 2:
                    name, model_obj = item
                    for h in self.config.horizons:
                        specs.append(ModelSpec(name=name, model=model_obj, horizon=int(h)))
                elif len(item) >= 3:
                    name, model_obj, h = item[:3]
                    specs.append(ModelSpec(name=name, model=model_obj, horizon=int(h)))

            elif isinstance(item, SeasonalNaiveModel):
                for h in self.config.horizons:
                    specs.append(ModelSpec(name=item.name, model=item, horizon=int(h)))

            else:
                name = getattr(item, "name", item.__class__.__name__)
                for h in self.config.horizons:
                    specs.append(ModelSpec(name=name, model=item, horizon=int(h)))

        return specs

    def evaluate_series(
        self,
        models: list[dict | tuple[str, Any] | Any],
        daily: pd.DataFrame,
        events: Optional[pd.DataFrame] = None,
        include_baseline: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a suite of models and optional Seasonal Naive baseline over a daily series.
        """
        valid_df = self.validate_time_series(daily)

        # Extract calendar/event flags if not explicitly provided
        if events is None:
            event_cols = [c for c in ["Promotion", "Is_Holiday", "Promotions", "Holiday_Flag"] if c in valid_df.columns]
            events = valid_df[event_cols].copy() if event_cols else None

        specs = self._normalize_model_specs(models)

        # Automatically inject Seasonal Naive baseline for all evaluated horizons if requested
        if include_baseline:
            evaluated_horizons = sorted({spec.horizon for spec in specs})
            baseline_name = self.config.baseline_model_name
            for h in evaluated_horizons:
                has_baseline = any(s.name == baseline_name and s.horizon == h for s in specs)
                if not has_baseline:
                    specs.append(
                        ModelSpec(
                            name=baseline_name,
                            model=SeasonalNaiveModel(seasonal_period=self.config.seasonal_period),
                            horizon=h,
                        )
                    )

        summary_list: list[dict[str, Any]] = []
        all_pred_frames: list[pd.DataFrame] = []
        all_origin_frames: list[pd.DataFrame] = []

        for spec in specs:
            summary, preds_df, origin_metrics_df = self.evaluate_model(
                model=spec.model,
                daily=valid_df,
                horizon=spec.horizon,
                model_name=spec.name,
                reference_start_date=spec.reference_start_date,
                feature_columns=spec.feature_columns,
                events=events,
            )
            summary_list.append(summary)
            all_pred_frames.append(preds_df)
            all_origin_frames.append(origin_metrics_df)

        # Calculate deltas relative to baseline
        baseline_name = self.config.baseline_model_name
        baseline_by_horizon: dict[int, dict[str, Any]] = {}
        for s in summary_list:
            if s["model"] == baseline_name:
                baseline_by_horizon[s["horizon_days"]] = s

        for s in summary_list:
            h = s["horizon_days"]
            base = baseline_by_horizon.get(h)
            if base is not None:
                s["WAPE_delta_vs_baseline"] = round(s["WAPE"] - base["WAPE"], 6)
                s["MAE_delta_vs_baseline"] = round(s["MAE"] - base["MAE"], 6)
                s["RMSE_delta_vs_baseline"] = round(s["RMSE"] - base["RMSE"], 6)
            else:
                s["WAPE_delta_vs_baseline"] = np.nan
                s["MAE_delta_vs_baseline"] = np.nan
                s["RMSE_delta_vs_baseline"] = np.nan

        summary_df = (
            pd.DataFrame(summary_list)
            .sort_values(["horizon_days", "WAPE"])
            .reset_index(drop=True)
        )

        combined_predictions = (
            pd.concat(all_pred_frames, ignore_index=True)
            if all_pred_frames
            else pd.DataFrame()
        )

        combined_origin_metrics = (
            pd.concat(all_origin_frames, ignore_index=True)
            if all_origin_frames
            else pd.DataFrame()
        )

        return EvaluationResult(
            config=self.config,
            summary_df=summary_df,
            predictions_df=combined_predictions,
            origin_metrics_df=combined_origin_metrics,
        )
