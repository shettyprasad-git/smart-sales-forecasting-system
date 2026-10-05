from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts" / "residual_diagnostics"


# ==============================================================================
# 1. Data Structures & Result Containers
# ==============================================================================

@dataclass
class LjungBoxRecord:
    lag: int
    statistic: float
    p_value: float
    significant_at_0_05: bool


@dataclass
class OutlierRecord:
    date: str
    actual: float
    predicted: float
    residual: float
    z_score: float


@dataclass
class SeriesResidualDiagnostics:
    model: str
    horizon_days: int
    count: int
    mean_residual: float
    median_residual: float
    std_residual: float
    mae: float
    rmse: float
    bias: float
    skewness: float
    kurtosis: float
    acf_lag_1: float
    acf_lag_7: float
    acf_lag_14: float
    acf_lag_28: float
    pct_positive_residuals: float
    pct_negative_residuals: float
    max_positive_residual: float
    max_negative_residual: float
    first_half_std: float
    second_half_std: float
    variance_ratio: float
    outlier_count_3sigma: int
    outlier_pct_3sigma: float
    ljung_box: list[LjungBoxRecord]
    acf_values: list[float]
    pacf_values: list[float]
    rolling_std_values: list[float]
    top_outliers: list[OutlierRecord]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "horizon_days": self.horizon_days,
            "count": self.count,
            "mean_residual": self.mean_residual,
            "median_residual": self.median_residual,
            "std_residual": self.std_residual,
            "MAE": self.mae,
            "RMSE": self.rmse,
            "bias": self.bias,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "acf_lag_1": self.acf_lag_1,
            "acf_lag_7": self.acf_lag_7,
            "acf_lag_14": self.acf_lag_14,
            "acf_lag_28": self.acf_lag_28,
            "pct_positive": self.pct_positive_residuals,
            "pct_negative": self.pct_negative_residuals,
            "max_positive_residual": self.max_positive_residual,
            "max_negative_residual": self.max_negative_residual,
            "first_half_std": self.first_half_std,
            "second_half_std": self.second_half_std,
            "variance_ratio": self.variance_ratio,
            "outlier_count_3sigma": self.outlier_count_3sigma,
            "outlier_pct_3sigma": self.outlier_pct_3sigma,
        }


# ==============================================================================
# 2. Input Validation and Calculations
# ==============================================================================

def validate_residual_input(
    residuals: Sequence[float] | np.ndarray | pd.Series,
    min_observations: int = 29,
    series_name: str = "Residual series",
) -> np.ndarray:
    """
    Validate input residual observations ensuring finite numbers and sufficient length.
    Guarantees that input caller data structures are not mutated in-place.
    """
    if residuals is None:
        raise ValueError(f"{series_name} requires non-None observations.")

    if isinstance(residuals, pd.Series):
        arr = residuals.to_numpy(dtype=float, copy=True)
    elif isinstance(residuals, np.ndarray):
        arr = residuals.astype(float, copy=True).flatten()
    else:
        try:
            arr = np.array(list(residuals), dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{series_name} contains non-numeric elements: {exc}") from exc

    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError(f"{series_name} cannot be empty and must be 1-dimensional.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{series_name} contains NaN or infinite values.")

    if len(arr) < min_observations:
        raise ValueError(
            f"{series_name} requires at least {min_observations} observations, "
            f"but received only {len(arr)}."
        )

    return arr


def compute_residuals(
    actual: Sequence[float] | np.ndarray | pd.Series,
    predicted: Sequence[float] | np.ndarray | pd.Series,
) -> np.ndarray:
    """
    Calculate residuals: residual = actual - predicted.
    Enforces non-mutation of inputs.
    """
    y_act = validate_residual_input(actual, min_observations=1, series_name="Actuals")
    y_pred = validate_residual_input(predicted, min_observations=1, series_name="Predictions")

    if len(y_act) != len(y_pred):
        raise ValueError(
            f"Length mismatch: actuals has {len(y_act)} rows, predictions has {len(y_pred)} rows."
        )

    return y_act - y_pred


def analyze_residual_series(
    residuals: Sequence[float] | np.ndarray | pd.Series,
    model_name: str,
    horizon_days: int,
    dates: Optional[Sequence[str] | pd.Series] = None,
    actuals: Optional[Sequence[float]] = None,
    predictions: Optional[Sequence[float]] = None,
    max_lag: int = 28,
    rolling_window: int = 7,
    outlier_z_threshold: float = 3.0,
) -> SeriesResidualDiagnostics:
    """
    Compute comprehensive statistical and econometric diagnostics for a residual series.
    """
    e = validate_residual_input(
        residuals,
        min_observations=max_lag + 1,
        series_name=f"Residuals for {model_name} (horizon {horizon_days})",
    )
    n = len(e)

    # 1. Basic descriptive statistics
    mean_val = float(np.mean(e))
    median_val = float(np.median(e))
    std_val = float(np.std(e, ddof=1)) if n > 1 else 0.0
    mae_val = float(np.mean(np.abs(e)))
    rmse_val = float(np.sqrt(np.mean(e ** 2)))
    bias_val = mean_val  # Bias is defined as mean(actual - predicted) = mean(residual)

    # 2. Higher order moments
    # Constant residual series handles zero-variance safely
    if std_val < 1e-12:
        skew_val = 0.0
        kurt_val = 0.0
    else:
        skew_val = float(stats.skew(e, bias=False))
        kurt_val = float(stats.kurtosis(e, bias=False))

    # 3. Autocorrelation and Partial Autocorrelation
    if std_val < 1e-12:
        acf_arr = np.zeros(max_lag + 1)
        acf_arr[0] = 1.0
        pacf_arr = np.zeros(max_lag + 1)
        pacf_arr[0] = 1.0
    else:
        acf_arr = acf(e, nlags=max_lag, fft=True)
        # pacf requires nlags < n // 2
        pacf_lags = min(max_lag, (n // 2) - 1)
        pacf_arr = pacf(e, nlags=pacf_lags, method="ywm")
        # Pad pacf_arr if truncated
        if len(pacf_arr) < max_lag + 1:
            pacf_arr = np.pad(pacf_arr, (0, max_lag + 1 - len(pacf_arr)), constant_values=0.0)

    acf_1 = float(acf_arr[1]) if len(acf_arr) > 1 else 0.0
    acf_7 = float(acf_arr[7]) if len(acf_arr) > 7 else 0.0
    acf_14 = float(acf_arr[14]) if len(acf_arr) > 14 else 0.0
    acf_28 = float(acf_arr[28]) if len(acf_arr) > 28 else 0.0

    # 4. Ljung-Box Portmanteau Test
    lb_lags = [7, 14, 28]
    lb_records: list[LjungBoxRecord] = []
    if std_val < 1e-12:
        for lag in lb_lags:
            lb_records.append(LjungBoxRecord(lag=lag, statistic=0.0, p_value=1.0, significant_at_0_05=False))
    else:
        try:
            lb_df = acorr_ljungbox(e, lags=lb_lags, return_df=True)
            for lag in lb_lags:
                stat = float(lb_df.loc[lag, "lb_stat"])
                pval = float(lb_df.loc[lag, "lb_pvalue"])
                sig = bool(pval < 0.05)
                lb_records.append(
                    LjungBoxRecord(
                        lag=int(lag),
                        statistic=stat,
                        p_value=pval,
                        significant_at_0_05=sig,
                    )
                )
        except Exception:
            for lag in lb_lags:
                lb_records.append(LjungBoxRecord(lag=lag, statistic=np.nan, p_value=np.nan, significant_at_0_05=False))

    # 5. Sign distribution and extremes
    pct_pos = float(np.mean(e > 0.0) * 100.0)
    pct_neg = float(np.mean(e < 0.0) * 100.0)
    max_pos = float(np.max(e))
    max_neg = float(np.min(e))

    # 6. Variance Stability & Rolling Standard Deviation
    half_idx = n // 2
    first_half = e[:half_idx]
    second_half = e[half_idx:]
    first_std = float(np.std(first_half, ddof=1)) if len(first_half) > 1 else 0.0
    second_std = float(np.std(second_half, ddof=1)) if len(second_half) > 1 else 0.0
    if first_std > 1e-9:
        var_ratio = float((second_std ** 2) / (first_std ** 2))
    else:
        var_ratio = np.nan

    # Rolling standard deviation (7-day window)
    e_series = pd.Series(e)
    rolling_std = e_series.rolling(window=rolling_window, min_periods=1).std(ddof=1).fillna(0.0).tolist()

    # 7. Outlier Detection (|z| > 3.0)
    if std_val > 1e-12:
        z_scores = (e - mean_val) / std_val
    else:
        z_scores = np.zeros(n)

    outlier_mask = np.abs(z_scores) > outlier_z_threshold
    outlier_count = int(np.sum(outlier_mask))
    outlier_pct = float(outlier_count / n * 100.0)

    # Collect detailed outlier records
    date_list = [str(d) for d in dates] if dates is not None else [str(i) for i in range(n)]
    act_list = [float(a) for a in actuals] if actuals is not None else [np.nan] * n
    pred_list = [float(p) for p in predictions] if predictions is not None else [np.nan] * n

    outlier_records: list[OutlierRecord] = []
    outlier_indices = np.where(outlier_mask)[0]
    # Sort outliers by absolute z-score descending
    sorted_outlier_indices = sorted(outlier_indices, key=lambda i: abs(z_scores[i]), reverse=True)

    for idx in sorted_outlier_indices:
        outlier_records.append(
            OutlierRecord(
                date=date_list[idx],
                actual=act_list[idx],
                predicted=pred_list[idx],
                residual=float(e[idx]),
                z_score=float(z_scores[idx]),
            )
        )

    return SeriesResidualDiagnostics(
        model=model_name,
        horizon_days=horizon_days,
        count=n,
        mean_residual=mean_val,
        median_residual=median_val,
        std_residual=std_val,
        mae=mae_val,
        rmse=rmse_val,
        bias=bias_val,
        skewness=skew_val,
        kurtosis=kurt_val,
        acf_lag_1=acf_1,
        acf_lag_7=acf_7,
        acf_lag_14=acf_14,
        acf_lag_28=acf_28,
        pct_positive_residuals=pct_pos,
        pct_negative_residuals=pct_neg,
        max_positive_residual=max_pos,
        max_negative_residual=max_neg,
        first_half_std=first_std,
        second_half_std=second_std,
        variance_ratio=var_ratio,
        outlier_count_3sigma=outlier_count,
        outlier_pct_3sigma=outlier_pct,
        ljung_box=lb_records,
        acf_values=[float(x) for x in acf_arr],
        pacf_values=[float(x) for x in pacf_arr],
        rolling_std_values=[float(x) for x in rolling_std],
        top_outliers=outlier_records,
    )


# ==============================================================================
# 3. Artifact Generation & Plotting
# ==============================================================================

def export_diagnostics_artifacts(
    diagnostics_list: Sequence[SeriesResidualDiagnostics],
    output_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> dict[str, Path]:
    """
    Persist all tabular, json, and figure residual diagnostics artifacts.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. residual_summary.csv
    summary_rows = [d.to_summary_dict() for d in diagnostics_list]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_path / "residual_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # 2. residual_ljung_box.csv
    lb_rows: list[dict[str, Any]] = []
    for d in diagnostics_list:
        for lb in d.ljung_box:
            lb_rows.append(
                {
                    "model": d.model,
                    "horizon_days": d.horizon_days,
                    "lag": lb.lag,
                    "statistic": lb.statistic,
                    "p_value": lb.p_value,
                    "significant_at_0_05": lb.significant_at_0_05,
                }
            )
    lb_df = pd.DataFrame(lb_rows)
    lb_csv = out_path / "residual_ljung_box.csv"
    lb_df.to_csv(lb_csv, index=False)

    # 3. residual_autocorrelation.csv
    acf_rows: list[dict[str, Any]] = []
    for d in diagnostics_list:
        for lag_idx in range(len(d.acf_values)):
            pacf_val = d.pacf_values[lag_idx] if lag_idx < len(d.pacf_values) else np.nan
            acf_rows.append(
                {
                    "model": d.model,
                    "horizon_days": d.horizon_days,
                    "lag": lag_idx,
                    "ACF": d.acf_values[lag_idx],
                    "PACF": pacf_val,
                }
            )
    acf_df = pd.DataFrame(acf_rows)
    acf_csv = out_path / "residual_autocorrelation.csv"
    acf_df.to_csv(acf_csv, index=False)

    # 4. residual_outliers.csv
    outlier_rows: list[dict[str, Any]] = []
    for d in diagnostics_list:
        for rec in d.top_outliers:
            outlier_rows.append(
                {
                    "model": d.model,
                    "horizon_days": d.horizon_days,
                    "forecast_date": rec.date,
                    "actual_quantity": rec.actual,
                    "predicted_quantity": rec.predicted,
                    "residual": rec.residual,
                    "z_score": rec.z_score,
                }
            )
    outlier_df = pd.DataFrame(outlier_rows)
    outlier_csv = out_path / "residual_outliers.csv"
    outlier_df.to_csv(outlier_csv, index=False)

    # 5. residual_diagnostics.json
    json_data: dict[str, Any] = {
        "analysis_version": "P1.0.2d",
        "series_count": len(diagnostics_list),
        "models": sorted(list({d.model for d in diagnostics_list})),
        "horizons": sorted(list({d.horizon_days for d in diagnostics_list})),
        "summaries": summary_rows,
    }
    json_path = out_path / "residual_diagnostics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    return {
        "summary": summary_csv,
        "ljung_box": lb_csv,
        "autocorrelation": acf_csv,
        "outliers": outlier_csv,
        "json": json_path,
    }


def generate_horizon_diagnostic_plots(
    horizon_predictions: dict[str, pd.DataFrame],
    horizon: int,
    output_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    primary_models: Sequence[str] = (
        "SARIMA(1,1,1)(0,1,1,7)",
        "Holt-Winters Additive Weekly",
        "ARIMA(2,1,2)",
        "Seasonal Naive",
    ),
) -> dict[str, Path]:
    """
    Generate the 3 required diagnostic figures for a specific horizon:
      1. residual_timeseries_{horizon}d.png
      2. residual_acf_{horizon}d.png
      3. residual_pacf_{horizon}d.png
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Filter to requested models present in the data
    available_models = [m for m in primary_models if m in horizon_predictions]
    if not available_models:
        available_models = list(horizon_predictions.keys())[:4]

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    palette = {m: colors[i % len(colors)] for i, m in enumerate(available_models)}

    # Figure 1: Residual Time Series
    fig_ts, ax_ts = plt.subplots(figsize=(12, 5), dpi=150)
    for model_name in available_models:
        m_df = horizon_predictions[model_name].copy()
        m_df["forecast_date"] = pd.to_datetime(m_df["forecast_date"])
        m_df = m_df.sort_values("forecast_date")
        residuals = m_df["actual_quantity"] - m_df["predicted_quantity"]
        ax_ts.plot(
            m_df["forecast_date"],
            residuals,
            label=model_name,
            color=palette[model_name],
            alpha=0.75,
            linewidth=1.2,
        )

    ax_ts.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_ts.set_title(f"{horizon}-Day Forecast Residual Time Series (2025 Holdout)", fontsize=13, fontweight="bold")
    ax_ts.set_xlabel("Forecast Date", fontsize=11)
    ax_ts.set_ylabel("Residual (Actual - Prediction)", fontsize=11)
    ax_ts.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax_ts.grid(True, linestyle=":", alpha=0.6)
    fig_ts.tight_layout()

    ts_path = out_path / f"residual_timeseries_{horizon}d.png"
    fig_ts.savefig(ts_path)
    plt.close(fig_ts)

    # Figure 2: Residual ACF
    fig_acf, ax_acf = plt.subplots(figsize=(10, 4.5), dpi=150)
    lags = np.arange(29)
    width = 0.8 / len(available_models)

    for i, model_name in enumerate(available_models):
        m_df = horizon_predictions[model_name]
        residuals = (m_df["actual_quantity"] - m_df["predicted_quantity"]).values
        acf_vals = acf(residuals, nlags=28, fft=True)
        offset = (i - len(available_models) / 2 + 0.5) * width
        ax_acf.bar(
            lags[1:] + offset,
            acf_vals[1:],
            width=width,
            label=model_name,
            color=palette[model_name],
            alpha=0.85,
        )

    # 95% Bartlett confidence bands for white noise: +/- 1.96 / sqrt(N)
    n_obs = len(horizon_predictions[available_models[0]])
    ci = 1.96 / np.sqrt(n_obs)
    ax_acf.axhline(ci, color="gray", linestyle="--", linewidth=0.9, label="95% White Noise Bound")
    ax_acf.axhline(-ci, color="gray", linestyle="--", linewidth=0.9)
    ax_acf.axhline(0.0, color="black", linestyle="-", linewidth=0.6)
    ax_acf.set_title(f"{horizon}-Day Residual Autocorrelation Function (ACF)", fontsize=13, fontweight="bold")
    ax_acf.set_xlabel("Lag (Days)", fontsize=11)
    ax_acf.set_ylabel("Autocorrelation", fontsize=11)
    ax_acf.set_xticks([1, 7, 14, 21, 28])
    ax_acf.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax_acf.grid(True, linestyle=":", alpha=0.6)
    fig_acf.tight_layout()

    acf_path = out_path / f"residual_acf_{horizon}d.png"
    fig_acf.savefig(acf_path)
    plt.close(fig_acf)

    # Figure 3: Residual PACF
    fig_pacf, ax_pacf = plt.subplots(figsize=(10, 4.5), dpi=150)
    for i, model_name in enumerate(available_models):
        m_df = horizon_predictions[model_name]
        residuals = (m_df["actual_quantity"] - m_df["predicted_quantity"]).values
        pacf_vals = pacf(residuals, nlags=28, method="ywm")
        offset = (i - len(available_models) / 2 + 0.5) * width
        ax_pacf.bar(
            lags[1:] + offset,
            pacf_vals[1:],
            width=width,
            label=model_name,
            color=palette[model_name],
            alpha=0.85,
        )

    ax_pacf.axhline(ci, color="gray", linestyle="--", linewidth=0.9, label="95% White Noise Bound")
    ax_pacf.axhline(-ci, color="gray", linestyle="--", linewidth=0.9)
    ax_pacf.axhline(0.0, color="black", linestyle="-", linewidth=0.6)
    ax_pacf.set_title(f"{horizon}-Day Residual Partial Autocorrelation (PACF)", fontsize=13, fontweight="bold")
    ax_pacf.set_xlabel("Lag (Days)", fontsize=11)
    ax_pacf.set_ylabel("Partial Autocorrelation", fontsize=11)
    ax_pacf.set_xticks([1, 7, 14, 21, 28])
    ax_pacf.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax_pacf.grid(True, linestyle=":", alpha=0.6)
    fig_pacf.tight_layout()

    pacf_path = out_path / f"residual_pacf_{horizon}d.png"
    fig_pacf.savefig(pacf_path)
    plt.close(fig_pacf)

    return {
        f"timeseries_{horizon}d": ts_path,
        f"acf_{horizon}d": acf_path,
        f"pacf_{horizon}d": pacf_path,
    }
