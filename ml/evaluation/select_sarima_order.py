from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ml.data.canonical_series import assess_time_series_quality
from ml.data.data_loader import aggregate_daily_data, load_processed_data
from ml.data.data_validation import validate_daily_data, validate_processed_data


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_daily_forecasting_features.csv"
)

DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"

# Bounded candidate set: 16 parsimonious weekly SARIMA specifications (d=1, s=7)
# Combining established low-order non-seasonal patterns with seasonal AR, MA, and differencing
DEFAULT_SARIMA_CANDIDATE_GRID: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = [
    # Baseline & Simple
    ((0, 1, 0), (1, 0, 0, 7)),
    ((0, 1, 0), (0, 1, 1, 7)),
    # Pure seasonal AR (P=1, D=0)
    ((0, 1, 1), (1, 0, 0, 7)),
    ((1, 1, 0), (1, 0, 0, 7)),
    ((1, 1, 1), (1, 0, 0, 7)),
    # Pure seasonal MA (Q=1, D=0)
    ((0, 1, 1), (0, 0, 1, 7)),
    ((1, 1, 0), (0, 0, 1, 7)),
    ((1, 1, 1), (0, 0, 1, 7)),
    # Mixed seasonal ARMA without seasonal differencing (P=1, Q=1, D=0)
    ((0, 1, 1), (1, 0, 1, 7)),
    ((1, 1, 0), (1, 0, 1, 7)),
    ((1, 1, 1), (1, 0, 1, 7)),
    # Seasonal differenced MA (D=1, Q=1) - Airline Model family
    ((0, 1, 1), (0, 1, 1, 7)),
    ((1, 1, 0), (0, 1, 1, 7)),
    ((1, 1, 1), (0, 1, 1, 7)),
    # Seasonal differenced AR (D=1, P=1)
    ((0, 1, 1), (1, 1, 0, 7)),
    ((1, 1, 0), (1, 1, 0, 7)),
]

SELECTION_CUTOFF_DATE: str = "2024-12-31"


def select_sarima_order(
    daily: pd.DataFrame,
    candidate_grid: Sequence[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = DEFAULT_SARIMA_CANDIDATE_GRID,
    cutoff_date: str | pd.Timestamp = SELECTION_CUTOFF_DATE,
    artifacts_dir: Optional[Path | str] = None,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int], pd.DataFrame, dict[str, Any]]:
    """
    Select the optimal SARIMA(p, d, q)(P, D, Q, s) order strictly on pre-holdout data.

    Invariants Enforced:
      - Uses data strictly on or before cutoff_date (default: 2024-12-31).
      - Zero exposure to 2025 holdout observations during order selection.
      - Objective function: minimum AIC among converged candidates.
      - Tracks BIC as secondary diagnostic.
      - Records convergence status for each candidate fit.
      - Persists sarima_order_selection.csv and sarima_selected_order.json.

    Args:
        daily: Canonical daily time series with Date and Quantity.
        candidate_grid: List of ((p, d, q), (P, D, Q, s)) candidate tuples.
        cutoff_date: Maximum historical date allowed for order selection (<= 2024-12-31).
        artifacts_dir: Target directory for artifacts.

    Returns:
        tuple of (selected_order, selected_seasonal_order, results_df, metadata_dict)
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

    # Restrict data strictly on or before cutoff_date
    cutoff_ts = pd.Timestamp(cutoff_date).normalize()
    holdout_start = pd.Timestamp("2025-01-01").normalize()
    if cutoff_ts >= holdout_start:
        raise ValueError(
            f"cutoff_date ({cutoff_ts.date()}) violates the holdout rule: "
            f"selection must use data strictly before {holdout_start.date()}."
        )

    pre_holdout = work_df[work_df.index <= cutoff_ts]
    if len(pre_holdout) == 0:
        raise ValueError(f"No observations found on or before cutoff_date ({cutoff_ts.date()}).")

    # Verify canonical daily continuity
    report = assess_time_series_quality(pre_holdout.reset_index())
    if not report.is_continuous or report.missing_days > 0:
        raise ValueError(
            f"Pre-holdout time series has continuity defects: {report.missing_days} missing days. "
            "Canonical daily continuity is required for order selection."
        )

    y_train = pre_holdout["Quantity"].to_numpy(dtype=float)

    results: list[dict[str, Any]] = []

    for order, s_order in candidate_grid:
        p, d, q = order
        P, D, Q, s = s_order
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y_train,
                    order=order,
                    seasonal_order=s_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False)

                aic_val = float(fitted.aic)
                bic_val = float(fitted.bic)
                ll_val = float(fitted.llf)
                converged = (
                    bool(getattr(fitted.mle_retvals, "converged", True))
                    if hasattr(fitted, "mle_retvals")
                    else True
                )
        except Exception:
            aic_val = np.nan
            bic_val = np.nan
            ll_val = np.nan
            converged = False

        results.append(
            {
                "order": f"({p}, {d}, {q})",
                "p": int(p),
                "d": int(d),
                "q": int(q),
                "seasonal_order": f"({P}, {D}, {Q}, {s})",
                "P": int(P),
                "D": int(D),
                "Q": int(Q),
                "seasonal_period": int(s),
                "AIC": aic_val,
                "BIC": bic_val,
                "log_likelihood": ll_val,
                "converged": converged,
            }
        )

    results_df = pd.DataFrame(results)

    # Selection rule: minimum AIC among converged candidates
    valid_candidates = results_df[results_df["converged"] & results_df["AIC"].notna()]
    if valid_candidates.empty:
        valid_candidates = results_df[results_df["AIC"].notna()]
    if valid_candidates.empty:
        raise RuntimeError("No candidate SARIMA model produced a valid AIC metric.")

    best_idx = valid_candidates["AIC"].idxmin()
    best_row = results_df.loc[best_idx]
    selected_order = (int(best_row["p"]), int(best_row["d"]), int(best_row["q"]))
    selected_seasonal_order = (
        int(best_row["P"]),
        int(best_row["D"]),
        int(best_row["Q"]),
        int(best_row["seasonal_period"]),
    )

    metadata: dict[str, Any] = {
        "selection_cutoff": SELECTION_CUTOFF_DATE,
        "selected_order": list(selected_order),
        "selected_seasonal_order": list(selected_seasonal_order),
        "seasonal_period": int(best_row["seasonal_period"]),
        "selection_metric": "AIC",
        "selected_aic": float(best_row["AIC"]),
        "selected_bic": float(best_row["BIC"]),
        "candidate_count": len(candidate_grid),
        "training_observations": len(y_train),
        "training_start_date": str(pre_holdout.index.min().date()),
        "training_end_date": str(pre_holdout.index.max().date()),
    }

    # Persist artifacts if directory is provided
    if artifacts_dir is not None:
        target_dir = Path(artifacts_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        csv_path = target_dir / "sarima_order_selection.csv"
        json_path = target_dir / "sarima_selected_order.json"

        results_df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return selected_order, selected_seasonal_order, results_df, metadata


def main() -> None:
    print("=" * 70)
    print("SMART SALES FORECASTING - SARIMA ORDER SELECTION (PRE-2025 HOLDOUT)")
    print("=" * 70)

    # 1. Load canonical production data
    print("\n[1/3] Loading canonical processed dataset...")
    df = load_processed_data(DATA_PATH)
    validate_processed_data(df)
    daily = aggregate_daily_data(df)
    validate_daily_data(daily)
    print(f"Total dataset range: {daily.index.min().date()} -> {daily.index.max().date()} ({len(daily):,} days)")

    # 2. Run selection strictly on pre-2025 data (cutoff 2024-12-31)
    print(f"\n[2/3] Evaluating candidate SARIMA grid on data strictly <= {SELECTION_CUTOFF_DATE}...")
    selected_order, selected_s_order, results_df, metadata = select_sarima_order(
        daily=daily,
        cutoff_date=SELECTION_CUTOFF_DATE,
        artifacts_dir=DEFAULT_ARTIFACTS_DIR,
    )

    # Also persist directly in ml/artifacts/sarima_evaluation/
    eval_artifacts_dir = DEFAULT_ARTIFACTS_DIR / "sarima_evaluation"
    eval_artifacts_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(eval_artifacts_dir / "sarima_order_selection.csv", index=False)
    with open(eval_artifacts_dir / "sarima_selected_order.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 3. Display selection table
    print("\n[3/3] Candidate Grid Evaluation Results:")
    print("-" * 70)
    display_df = results_df.copy()
    display_df["AIC"] = display_df["AIC"].round(2)
    display_df["BIC"] = display_df["BIC"].round(2)
    display_df["log_likelihood"] = display_df["log_likelihood"].round(2)
    print(display_df.to_string(index=False))

    print("-" * 70)
    print(f"Selected SARIMA Order         : {selected_order}")
    print(f"Selected Seasonal Order       : {selected_s_order}")
    print(f"Selection Metric              : AIC = {metadata['selected_aic']:.2f} (BIC = {metadata['selected_bic']:.2f})")
    print(f"Selection Cutoff              : {metadata['selection_cutoff']}")
    print(f"Pre-Holdout Range             : {metadata['training_start_date']} to {metadata['training_end_date']} (N={metadata['training_observations']:,})")
    print(f"Artifacts Persisted           :\n  {DEFAULT_ARTIFACTS_DIR / 'sarima_order_selection.csv'}\n  {DEFAULT_ARTIFACTS_DIR / 'sarima_selected_order.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
