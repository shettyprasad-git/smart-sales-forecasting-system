from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from ml.analysis.residual_diagnostics import (
    DEFAULT_ARTIFACTS_DIR,
    SeriesResidualDiagnostics,
    analyze_residual_series,
    compute_residuals,
    export_diagnostics_artifacts,
    generate_horizon_diagnostic_plots,
)


PREDICTION_FILES = {
    "production": PROJECT_ROOT / "ml" / "artifacts" / "production_model_predictions.csv",
    "classical": (
        PROJECT_ROOT
        / "ml"
        / "artifacts"
        / "classical_model_evaluation"
        / "classical_model_predictions.csv"
    ),
    "arima": (
        PROJECT_ROOT
        / "ml"
        / "artifacts"
        / "arima_evaluation"
        / "arima_model_predictions.csv"
    ),
    "sarima": (
        PROJECT_ROOT
        / "ml"
        / "artifacts"
        / "sarima_evaluation"
        / "sarima_model_predictions.csv"
    ),
}


def load_all_prediction_artifacts() -> pd.DataFrame:
    """
    Load all available prediction artifacts across production, classical ETS, ARIMA, and SARIMA.
    Deduplicates repeated Seasonal Naive baseline records across runs.
    """
    dfs: list[pd.DataFrame] = []

    for name, path in PREDICTION_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Required prediction artifact missing: {path}")

        sub_df = pd.read_csv(path)
        sub_df["source_artifact"] = name
        dfs.append(sub_df)

    combined = pd.concat(dfs, ignore_index=True)

    # Standardize types
    combined["horizon_days"] = combined["horizon_days"].astype(int)
    combined["forecast_date"] = pd.to_datetime(combined["forecast_date"]).dt.strftime("%Y-%m-%d")
    combined["actual_quantity"] = combined["actual_quantity"].astype(float)
    combined["predicted_quantity"] = combined["predicted_quantity"].astype(float)

    # Deduplicate repeated predictions for same model, horizon, and forecast_date
    dedup = combined.drop_duplicates(subset=["model", "horizon_days", "forecast_date"]).copy()
    dedup = dedup.sort_values(["model", "horizon_days", "forecast_date"]).reset_index(drop=True)

    return dedup


def execute_residual_diagnostics(
    output_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> dict[str, Any]:
    """
    Main orchestration routine for Phase P1.0.2d Residual Diagnostics.
    """
    print("=" * 75)
    print("SMART SALES FORECASTING - RESIDUAL DIAGNOSTICS (P1.0.2d)")
    print("=" * 75)

    # 1. Load predictions
    print("\n[1/4] Loading prediction-level holdout artifacts...")
    preds_df = load_all_prediction_artifacts()
    print(f"Total deduplicated prediction rows: {len(preds_df):,}")

    # Discover model-horizon combinations
    series_groups = preds_df.groupby(["model", "horizon_days"])
    print(f"Total distinct (model, horizon) series discovered: {len(series_groups)}")

    # 2. Compute diagnostics for each series
    print("\n[2/4] Computing econometric and statistical residual diagnostics...")
    diagnostics_list: list[SeriesResidualDiagnostics] = []
    horizon_predictions: dict[int, dict[str, pd.DataFrame]] = {7: {}, 30: {}, 90: {}}

    for (model_name, horizon), group in series_groups:
        group_sorted = group.sort_values("forecast_date").reset_index(drop=True)
        residuals = compute_residuals(
            actual=group_sorted["actual_quantity"].values,
            predicted=group_sorted["predicted_quantity"].values,
        )

        diag = analyze_residual_series(
            residuals=residuals,
            model_name=model_name,
            horizon_days=horizon,
            dates=group_sorted["forecast_date"].tolist(),
            actuals=group_sorted["actual_quantity"].tolist(),
            predictions=group_sorted["predicted_quantity"].tolist(),
            max_lag=28,
            rolling_window=7,
        )
        diagnostics_list.append(diag)

        if horizon in horizon_predictions:
            horizon_predictions[horizon][model_name] = group_sorted

    # Sort diagnostics by horizon then MAE
    diagnostics_list.sort(key=lambda d: (d.horizon_days, d.mae))

    # 3. Export tabular and json artifacts
    print(f"\n[3/4] Persisting residual diagnostic artifacts to:\n  {output_dir}")
    artifact_paths = export_diagnostics_artifacts(diagnostics_list, output_dir=output_dir)

    # 4. Generate diagnostic plots for horizons 7, 30, and 90
    print("\n[4/4] Generating publication-quality diagnostic figures...")
    plot_paths: dict[str, Path] = {}
    for h in [7, 30, 90]:
        if horizon_predictions[h]:
            h_plots = generate_horizon_diagnostic_plots(
                horizon_predictions=horizon_predictions[h],
                horizon=h,
                output_dir=output_dir,
            )
            plot_paths.update(h_plots)

    # 5. Display compact results
    print("\n" + "=" * 75)
    print("RESIDUAL DIAGNOSTICS SUMMARY TABLE (2025 HOLDOUT)")
    print("=" * 75)

    display_cols = [
        "model",
        "horizon_days",
        "MAE",
        "RMSE",
        "bias",
        "acf_lag_1",
        "acf_lag_7",
        "acf_lag_14",
        "variance_ratio",
        "outlier_count_3sigma",
    ]
    summary_df = pd.DataFrame([d.to_summary_dict() for d in diagnostics_list])[display_cols]
    summary_df["MAE"] = summary_df["MAE"].round(2)
    summary_df["RMSE"] = summary_df["RMSE"].round(2)
    summary_df["bias"] = summary_df["bias"].round(2)
    summary_df["acf_lag_1"] = summary_df["acf_lag_1"].round(3)
    summary_df["acf_lag_7"] = summary_df["acf_lag_7"].round(3)
    summary_df["acf_lag_14"] = summary_df["acf_lag_14"].round(3)
    summary_df["variance_ratio"] = summary_df["variance_ratio"].round(3)

    print(summary_df.to_string(index=False))

    print("\n" + "-" * 75)
    print("LJUNG-BOX PORTMANTEAU TEST AT LAG 7, 14, 28 (p-value < 0.05 => Autocorrelation Detected):")
    print("-" * 75)
    lb_rows = []
    for d in diagnostics_list:
        for lb in d.ljung_box:
            lb_rows.append(
                {
                    "model": d.model,
                    "horizon": d.horizon_days,
                    "lag": lb.lag,
                    "stat": round(lb.statistic, 2),
                    "p_val": f"{lb.p_value:.4f}",
                    "significant": lb.significant_at_0_05,
                }
            )
    lb_df = pd.DataFrame(lb_rows)
    print(lb_df.to_string(index=False))

    print("\n" + "=" * 75)
    print("Artifacts generated:")
    for k, p in artifact_paths.items():
        print(f"  {k:15}: {p}")
    for k, p in plot_paths.items():
        print(f"  {k:15}: {p}")
    print("=" * 75)

    return {
        "diagnostics": diagnostics_list,
        "artifact_paths": artifact_paths,
        "plot_paths": plot_paths,
        "summary_df": summary_df,
    }


if __name__ == "__main__":
    execute_residual_diagnostics()
