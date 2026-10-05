from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import functools
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ml" / "artifacts" / "model_selection"

# Canonical Evaluation Artifact Locations
DEFAULT_EVAL_FILES = {
    "production": PROJECT_ROOT / "ml" / "artifacts" / "production_model_evaluation.csv",
    "classical": (
        PROJECT_ROOT
        / "ml"
        / "artifacts"
        / "classical_model_evaluation"
        / "classical_model_evaluation.csv"
    ),
    "arima": (
        PROJECT_ROOT
        / "ml"
        / "artifacts"
        / "arima_evaluation"
        / "arima_model_evaluation.csv"
    ),
    "sarima": (
        PROJECT_ROOT
        / "ml"
        / "artifacts"
        / "sarima_evaluation"
        / "sarima_model_evaluation.csv"
    ),
}

DEFAULT_RESIDUAL_SUMMARY_FILE = (
    PROJECT_ROOT / "ml" / "artifacts" / "residual_diagnostics" / "residual_summary.csv"
)
DEFAULT_RESIDUAL_LJUNGBOX_FILE = (
    PROJECT_ROOT / "ml" / "artifacts" / "residual_diagnostics" / "residual_ljung_box.csv"
)

# Designated Primary Candidate Models per Forecast Horizon
PRIMARY_CANDIDATES: dict[int, list[str]] = {
    7: [
        "Random Forest",
        "Holt-Winters Additive Weekly",
        "Damped Holt-Winters Additive Weekly",
        "SARIMA(1,1,1)(0,1,1,7)",
        "Seasonal Naive",
    ],
    30: [
        "HistGradientBoosting",
        "Damped Holt-Winters Additive Weekly",
        "Holt-Winters Additive Weekly",
        "SARIMA(1,1,1)(0,1,1,7)",
        "Seasonal Naive",
    ],
    90: [
        "Linear Regression",
        "Damped Holt-Winters Additive Weekly",
        "Holt-Winters Additive Weekly",
        "SARIMA(1,1,1)(0,1,1,7)",
        "Seasonal Naive",
    ],
}

# Explicit Policy Tie-Breaking Thresholds
WAPE_TIE_THRESHOLD_PP: float = 0.10  # 0.10 percentage points = 0.0010 decimal ratio
WAPE_TIE_THRESHOLD_DECIMAL: float = 0.0010
MAE_TIE_THRESHOLD: float = 0.10      # 0.10 units
RMSE_TIE_THRESHOLD: float = 0.10     # 0.10 units
BIAS_TIE_THRESHOLD: float = 0.10     # 0.10 units
VAR_RATIO_TIE_THRESHOLD: float = 0.05  # Distance from 1.0 within 0.05

# Deterministic Simplicity Ranking (lower number = simpler model)
SIMPLICITY_RANK: dict[str, int] = {
    "Seasonal Naive": 1,
    "Linear Regression": 2,
    "Holt-Winters Additive Weekly": 3,
    "Damped Holt-Winters Additive Weekly": 4,
    "SARIMA(1,1,1)(0,1,1,7)": 5,
    "Random Forest": 6,
    "HistGradientBoosting": 7,
}


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class StructuralRiskAssessment:
    level: str  # "LOW", "MEDIUM", "HIGH"
    reasons: list[str]


@dataclass
class CandidateMetrics:
    horizon_days: int
    model: str
    mae: float
    rmse: float
    mape: float
    wape: float
    bias: float
    wape_delta_vs_baseline: float
    lag1_acf: float
    lag7_acf: float
    lag14_acf: float
    lag28_acf: float
    ljung_box_significant: bool
    variance_ratio: float
    outlier_percentage: float
    structural_risk: str
    structural_risk_reasons: list[str] = field(default_factory=list)
    accuracy_rank: int = 0
    selected: bool = False


@dataclass
class HorizonSelectionResult:
    horizon_days: int
    selected_model: str
    wape: float
    mae: float
    rmse: float
    bias: float
    structural_risk: str
    selection_reason: str
    diagnostic_concerns: list[str]
    candidates_ranked: list[CandidateMetrics]


# ==============================================================================
# Helper & Loading Functions
# ==============================================================================

def normalize_model_name(name: str) -> str:
    """Normalize model string removing enclosing quotation marks and extra whitespace."""
    if not isinstance(name, str):
        return str(name)
    return name.strip().strip('"').strip("'")


def load_evaluation_artifacts(
    eval_files: Optional[dict[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load and concatenate holdout evaluation metrics from all four model families.
    Deduplicates repeated entries (e.g., Seasonal Naive benchmark).
    """
    files_to_load = eval_files or DEFAULT_EVAL_FILES
    dfs: list[pd.DataFrame] = []

    for source_name, path in files_to_load.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation artifact: {path}")
        df = pd.read_csv(path)
        df["model"] = df["model"].apply(normalize_model_name)
        df["horizon_days"] = df["horizon_days"].astype(int)
        df["source_artifact"] = source_name
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    # Deduplicate by model and horizon_days keeping first encountered
    dedup = combined.drop_duplicates(subset=["model", "horizon_days"]).copy()
    return dedup


def load_residual_diagnostics(
    summary_path: Optional[Path] = None,
    ljung_box_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load residual diagnostics summary and Ljung-Box test results.
    """
    sum_path = summary_path or DEFAULT_RESIDUAL_SUMMARY_FILE
    lb_path = ljung_box_path or DEFAULT_RESIDUAL_LJUNGBOX_FILE

    if not sum_path.exists():
        raise FileNotFoundError(f"Missing residual summary artifact: {sum_path}")
    if not lb_path.exists():
        raise FileNotFoundError(f"Missing residual Ljung-Box artifact: {lb_path}")

    sum_df = pd.read_csv(sum_path)
    sum_df["model"] = sum_df["model"].apply(normalize_model_name)
    sum_df["horizon_days"] = sum_df["horizon_days"].astype(int)

    lb_df = pd.read_csv(lb_path)
    lb_df["model"] = lb_df["model"].apply(normalize_model_name)
    lb_df["horizon_days"] = lb_df["horizon_days"].astype(int)

    # Compute joint Ljung-Box significance across lags 7, 14, 28
    lb_summary = (
        lb_df.groupby(["model", "horizon_days"])["significant_at_0_05"]
        .any()
        .reset_index()
        .rename(columns={"significant_at_0_05": "ljung_box_significant"})
    )

    merged_diag = pd.merge(sum_df, lb_summary, on=["model", "horizon_days"], how="left")
    merged_diag["ljung_box_significant"] = merged_diag["ljung_box_significant"].fillna(True)

    return merged_diag


def merge_metrics_and_diagnostics(
    eval_df: pd.DataFrame,
    diag_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join evaluation accuracy metrics with residual diagnostics on (model, horizon_days).
    """
    merged = pd.merge(
        eval_df,
        diag_df,
        on=["model", "horizon_days"],
        how="inner",
        suffixes=("", "_diag"),
    )
    return merged


# ==============================================================================
# Structural Risk Assessment
# ==============================================================================

def assess_structural_risk(
    variance_ratio: float,
    bias: float,
    lag1_acf: float,
    lag7_acf: float,
    outlier_pct: float,
) -> StructuralRiskAssessment:
    """
    Evaluate structural risk (LOW, MEDIUM, HIGH) deterministically from residual diagnostics.

    Threshold Definitions:
    - HIGH:
      * Extreme variance instability: variance_ratio > 3.0 or < 0.33
      * Large systematic bias: |bias| >= 150.0
      * Severe persistent autocorrelation: |lag1_acf| >= 0.90 AND |lag7_acf| >= 0.70
      * Extreme outlier frequency: outlier_pct >= 7.0%
    - MEDIUM:
      * Moderate variance expansion: 1.5 <= variance_ratio <= 3.0 or 0.33 <= variance_ratio <= 0.67
      * Meaningful residual persistence: |lag1_acf| >= 0.70 OR |lag7_acf| >= 0.50
      * Moderate bias: 50.0 <= |bias| < 150.0
      * Moderate outlier frequency: 3.0% <= outlier_pct < 7.0%
    - LOW:
      * Stable variance (0.67 < variance_ratio < 1.5), low bias (< 50),
        low persistence (|lag1_acf| < 0.70, |lag7_acf| < 0.50), low outliers (< 3.0%).
    """
    reasons_high: list[str] = []
    if variance_ratio > 3.0 or variance_ratio < 0.33:
        reasons_high.append(f"Extreme variance instability (VR={variance_ratio:.2f})")
    if abs(bias) >= 150.0:
        reasons_high.append(f"Large systematic bias (|Bias|={abs(bias):.1f})")
    if abs(lag1_acf) >= 0.90 and abs(lag7_acf) >= 0.70:
        reasons_high.append(
            f"Severe persistent autocorrelation (rho1={lag1_acf:.2f}, rho7={lag7_acf:.2f})"
        )
    if outlier_pct >= 7.0:
        reasons_high.append(f"Extreme outlier frequency ({outlier_pct:.1f}%)")

    if reasons_high:
        return StructuralRiskAssessment(level="HIGH", reasons=reasons_high)

    reasons_med: list[str] = []
    if (1.5 <= variance_ratio <= 3.0) or (0.33 <= variance_ratio <= 0.67):
        reasons_med.append(f"Moderate variance expansion (VR={variance_ratio:.2f})")
    if abs(lag1_acf) >= 0.70 or abs(lag7_acf) >= 0.50:
        reasons_med.append(
            f"Meaningful residual persistence (rho1={lag1_acf:.2f}, rho7={lag7_acf:.2f})"
        )
    if 50.0 <= abs(bias) < 150.0:
        reasons_med.append(f"Moderate bias (|Bias|={abs(bias):.1f})")
    if 3.0 <= outlier_pct < 7.0:
        reasons_med.append(f"Moderate outlier frequency ({outlier_pct:.1f}%)")

    if reasons_med:
        return StructuralRiskAssessment(level="MEDIUM", reasons=reasons_med)

    return StructuralRiskAssessment(
        level="LOW",
        reasons=["Stable variance ratio", "Low residual persistence", "Low bias"],
    )


# ==============================================================================
# Model Ranking and Deterministic Selection Policy
# ==============================================================================

def compare_candidate_metrics(a: CandidateMetrics, b: CandidateMetrics) -> int:
    """
    Total ordering comparator between two models implementing the deterministic selection policy:
    1. Lowest WAPE
    2. If WAPE diff <= 0.10 percentage points (0.0010 decimal): compare MAE
    3. If MAE tied within 0.10 units: compare RMSE
    4. If RMSE tied within 0.10 units: prefer lower |Bias|
    5. If |Bias| tied within 0.10 units: prefer lower variance ratio distance from 1.0
    6. If still tied: prefer simpler model by SIMPLICITY_RANK

    Returns:
      -1 if a is preferred over b
      +1 if b is preferred over a
       0 if exactly tied
    """
    # 1. WAPE
    wape_diff = a.wape - b.wape
    if abs(wape_diff) > WAPE_TIE_THRESHOLD_DECIMAL:
        return -1 if wape_diff < 0 else 1

    # 2. MAE
    mae_diff = a.mae - b.mae
    if abs(mae_diff) > MAE_TIE_THRESHOLD:
        return -1 if mae_diff < 0 else 1

    # 3. RMSE
    rmse_diff = a.rmse - b.rmse
    if abs(rmse_diff) > RMSE_TIE_THRESHOLD:
        return -1 if rmse_diff < 0 else 1

    # 4. Absolute Bias
    bias_diff = abs(a.bias) - abs(b.bias)
    if abs(bias_diff) > BIAS_TIE_THRESHOLD:
        return -1 if bias_diff < 0 else 1

    # 5. Variance ratio distance from 1.0
    vr_dist_a = abs(a.variance_ratio - 1.0)
    vr_dist_b = abs(b.variance_ratio - 1.0)
    vr_diff = vr_dist_a - vr_dist_b
    if abs(vr_diff) > VAR_RATIO_TIE_THRESHOLD:
        return -1 if vr_diff < 0 else 1

    # 6. Simplicity hierarchy
    simp_a = SIMPLICITY_RANK.get(a.model, 99)
    simp_b = SIMPLICITY_RANK.get(b.model, 99)
    if simp_a != simp_b:
        return -1 if simp_a < simp_b else 1

    return 0


def generate_selection_reason(
    winner: CandidateMetrics,
    runner_up: Optional[CandidateMetrics],
) -> str:
    """Generate clear, audit-compliant explanation for model selection relative to runner-up."""
    if runner_up is None:
        return f"Sole designated candidate for horizon {winner.horizon_days}d."

    wape_diff_pp = (runner_up.wape - winner.wape) * 100.0
    if abs(runner_up.wape - winner.wape) > WAPE_TIE_THRESHOLD_DECIMAL:
        return (
            f"Selected by lowest WAPE ({winner.wape * 100.0:.2f}% vs "
            f"{runner_up.wape * 100.0:.2f}% for {runner_up.model}; delta = -{wape_diff_pp:.2f} pp)."
        )

    # Tied on WAPE
    mae_diff = runner_up.mae - winner.mae
    if abs(mae_diff) > MAE_TIE_THRESHOLD:
        return (
            f"Tied on WAPE within 0.10 pp ({winner.wape * 100.0:.3f}% vs {runner_up.wape * 100.0:.3f}% "
            f"for {runner_up.model}); selected by lower MAE ({winner.mae:.2f} vs {runner_up.mae:.2f}; "
            f"diff = -{mae_diff:.2f})."
        )

    # Tied on MAE
    rmse_diff = runner_up.rmse - winner.rmse
    if abs(rmse_diff) > RMSE_TIE_THRESHOLD:
        return (
            f"Tied on WAPE within 0.10 pp ({winner.wape * 100.0:.3f}% vs {runner_up.wape * 100.0:.3f}%) "
            f"and MAE within 0.10 units ({winner.mae:.2f} vs {runner_up.mae:.2f}); "
            f"selected by lower RMSE ({winner.rmse:.2f} vs {runner_up.rmse:.2f}; diff = -{rmse_diff:.2f})."
        )

    # Tied on RMSE
    bias_diff = abs(runner_up.bias) - abs(winner.bias)
    if abs(bias_diff) > BIAS_TIE_THRESHOLD:
        return (
            f"Tied on WAPE, MAE, and RMSE within thresholds; selected by lower absolute Bias "
            f"(|{winner.bias:.2f}| vs |{runner_up.bias:.2f}| for {runner_up.model})."
        )

    # Tied on Bias
    vr_dist_winner = abs(winner.variance_ratio - 1.0)
    vr_dist_runner = abs(runner_up.variance_ratio - 1.0)
    if abs(vr_dist_runner - vr_dist_winner) > VAR_RATIO_TIE_THRESHOLD:
        return (
            f"Tied on accuracy metrics within thresholds; selected by variance ratio closer to 1.0 "
            f"({winner.variance_ratio:.2f} vs {runner_up.variance_ratio:.2f} for {runner_up.model})."
        )

    # Simplicity
    return (
        f"Tied across all accuracy and variance metrics; selected by simplicity hierarchy "
        f"({winner.model} rank {SIMPLICITY_RANK.get(winner.model)} vs "
        f"{runner_up.model} rank {SIMPLICITY_RANK.get(runner_up.model)})."
    )


def select_model_for_horizon(
    merged_df: pd.DataFrame,
    horizon: int,
    candidate_names: Optional[Sequence[str]] = None,
) -> HorizonSelectionResult:
    """
    Filter candidate models for a given horizon, rank them via policy, assess structural risk,
    and select the optimal model.
    """
    target_candidates = candidate_names or PRIMARY_CANDIDATES.get(horizon)
    if not target_candidates:
        raise ValueError(f"No candidate models defined for horizon {horizon} days.")

    horizon_df = merged_df[merged_df["horizon_days"] == horizon].copy()
    candidate_set = set(target_candidates)
    filtered_df = horizon_df[horizon_df["model"].isin(candidate_set)].copy()

    if filtered_df.empty:
        raise ValueError(
            f"None of the candidate models {target_candidates} found in evaluation data for horizon {horizon}."
        )

    candidate_records: list[CandidateMetrics] = []
    for _, row in filtered_df.iterrows():
        model_name = str(row["model"])
        vr = float(row["variance_ratio"])
        bias_val = float(row["Bias"])
        lag1 = float(row["acf_lag_1"])
        lag7 = float(row["acf_lag_7"])
        lag14 = float(row["acf_lag_14"])
        lag28 = float(row["acf_lag_28"])
        out_pct = float(row["outlier_pct_3sigma"])
        lb_sig = bool(row["ljung_box_significant"])

        risk_assess = assess_structural_risk(
            variance_ratio=vr,
            bias=bias_val,
            lag1_acf=lag1,
            lag7_acf=lag7,
            outlier_pct=out_pct,
        )

        rec = CandidateMetrics(
            horizon_days=horizon,
            model=model_name,
            mae=float(row["MAE"]),
            rmse=float(row["RMSE"]),
            mape=float(row["MAPE"]),
            wape=float(row["WAPE"]),
            bias=bias_val,
            wape_delta_vs_baseline=float(row["WAPE_delta_vs_baseline"]),
            lag1_acf=lag1,
            lag7_acf=lag7,
            lag14_acf=lag14,
            lag28_acf=lag28,
            ljung_box_significant=lb_sig,
            variance_ratio=vr,
            outlier_percentage=out_pct,
            structural_risk=risk_assess.level,
            structural_risk_reasons=risk_assess.reasons,
        )
        candidate_records.append(rec)

    # Sort candidates using total ordering comparator
    candidate_records.sort(key=functools.cmp_to_key(compare_candidate_metrics))

    # Assign ranks and selection flag
    for rank_idx, cand in enumerate(candidate_records, start=1):
        cand.accuracy_rank = rank_idx
        cand.selected = (rank_idx == 1)

    winner = candidate_records[0]
    runner_up = candidate_records[1] if len(candidate_records) > 1 else None
    reason = generate_selection_reason(winner, runner_up)

    return HorizonSelectionResult(
        horizon_days=horizon,
        selected_model=winner.model,
        wape=winner.wape,
        mae=winner.mae,
        rmse=winner.rmse,
        bias=winner.bias,
        structural_risk=winner.structural_risk,
        selection_reason=reason,
        diagnostic_concerns=winner.structural_risk_reasons,
        candidates_ranked=candidate_records,
    )


# ==============================================================================
# Pipeline Execution & Artifact Export
# ==============================================================================

def execute_model_selection(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    eval_files: Optional[dict[str, Path]] = None,
    summary_path: Optional[Path] = None,
    ljung_box_path: Optional[Path] = None,
    horizons: Sequence[int] = (7, 30, 90),
) -> dict[str, Any]:
    """
    Main orchestration routine for unified model comparison and selection.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load artifacts
    eval_df = load_evaluation_artifacts(eval_files)
    diag_df = load_residual_diagnostics(summary_path, ljung_box_path)
    merged_df = merge_metrics_and_diagnostics(eval_df, diag_df)

    # 2. Select per horizon
    horizon_results: list[HorizonSelectionResult] = []
    all_ranked_candidates: list[CandidateMetrics] = []

    for h in horizons:
        h_res = select_model_for_horizon(merged_df, horizon=h)
        horizon_results.append(h_res)
        all_ranked_candidates.extend(h_res.candidates_ranked)

    # 3. Build unified_model_comparison.csv
    unified_rows: list[dict[str, Any]] = []
    for cand in all_ranked_candidates:
        unified_rows.append(
            {
                "horizon_days": cand.horizon_days,
                "model": cand.model,
                "MAE": cand.mae,
                "RMSE": cand.rmse,
                "MAPE": cand.mape,
                "WAPE": cand.wape,
                "Bias": cand.bias,
                "WAPE_delta_vs_baseline": cand.wape_delta_vs_baseline,
                "lag1_acf": cand.lag1_acf,
                "lag7_acf": cand.lag7_acf,
                "lag14_acf": cand.lag14_acf,
                "lag28_acf": cand.lag28_acf,
                "ljung_box_significant": cand.ljung_box_significant,
                "variance_ratio": cand.variance_ratio,
                "outlier_percentage": cand.outlier_percentage,
                "structural_risk": cand.structural_risk,
                "accuracy_rank": cand.accuracy_rank,
                "selected": cand.selected,
            }
        )
    unified_df = pd.DataFrame(unified_rows)
    unified_csv_path = out_path / "unified_model_comparison.csv"
    unified_df.to_csv(unified_csv_path, index=False)

    # 4. Build horizon_model_selection.csv
    selection_rows: list[dict[str, Any]] = []
    for hr in horizon_results:
        selection_rows.append(
            {
                "horizon_days": hr.horizon_days,
                "selected_model": hr.selected_model,
                "WAPE": hr.wape,
                "MAE": hr.mae,
                "RMSE": hr.rmse,
                "Bias": hr.bias,
                "structural_risk": hr.structural_risk,
                "selection_reason": hr.selection_reason,
            }
        )
    selection_df = pd.DataFrame(selection_rows)
    selection_csv_path = out_path / "horizon_model_selection.csv"
    selection_df.to_csv(selection_csv_path, index=False)

    # 5. Build model_selection_rationale.json
    now_iso = datetime.now(timezone.utc).isoformat()
    json_data: dict[str, Any] = {
        "selection_date": now_iso,
        "candidate_models": PRIMARY_CANDIDATES,
        "selection_policy": {
            "primary_metric": "WAPE (lower is better)",
            "wape_tie_threshold_pp": WAPE_TIE_THRESHOLD_PP,
            "wape_tie_threshold_decimal": WAPE_TIE_THRESHOLD_DECIMAL,
            "mae_tie_threshold_units": MAE_TIE_THRESHOLD,
            "rmse_tie_threshold_units": RMSE_TIE_THRESHOLD,
            "bias_tie_threshold_units": BIAS_TIE_THRESHOLD,
            "variance_ratio_tie_threshold": VAR_RATIO_TIE_THRESHOLD,
            "simplicity_hierarchy": SIMPLICITY_RANK,
            "structural_risk_definitions": {
                "HIGH": [
                    "variance_ratio > 3.0 or < 0.33",
                    "abs(bias) >= 150.0",
                    "abs(lag1_acf) >= 0.90 and abs(lag7_acf) >= 0.70",
                    "outlier_pct >= 7.0%",
                ],
                "MEDIUM": [
                    "1.5 <= variance_ratio <= 3.0 or 0.33 <= variance_ratio <= 0.67",
                    "abs(lag1_acf) >= 0.70 or abs(lag7_acf) >= 0.50",
                    "50.0 <= abs(bias) < 150.0",
                    "3.0% <= outlier_pct < 7.0%",
                ],
                "LOW": [
                    "0.67 < variance_ratio < 1.5",
                    "abs(lag1_acf) < 0.70 and abs(lag7_acf) < 0.50",
                    "abs(bias) < 50.0",
                    "outlier_pct < 3.0%",
                ],
            },
        },
        "per_horizon_results": {
            str(hr.horizon_days): {
                "selected_model": hr.selected_model,
                "WAPE": hr.wape,
                "MAE": hr.mae,
                "RMSE": hr.rmse,
                "Bias": hr.bias,
                "structural_risk": hr.structural_risk,
                "selection_reason": hr.selection_reason,
                "diagnostic_concerns": hr.diagnostic_concerns,
                "ranked_candidates": [
                    {
                        "rank": c.accuracy_rank,
                        "model": c.model,
                        "WAPE": c.wape,
                        "MAE": c.mae,
                        "RMSE": c.rmse,
                        "Bias": c.bias,
                        "structural_risk": c.structural_risk,
                        "concerns": c.structural_risk_reasons,
                    }
                    for c in hr.candidates_ranked
                ],
            }
            for hr in horizon_results
        },
        "selected_models": {
            str(hr.horizon_days): hr.selected_model for hr in horizon_results
        },
        "structural_risk_flags": {
            str(hr.horizon_days): {
                "level": hr.structural_risk,
                "reasons": hr.diagnostic_concerns,
            }
            for hr in horizon_results
        },
    }
    json_path = out_path / "model_selection_rationale.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)

    return {
        "unified_df": unified_df,
        "selection_df": selection_df,
        "horizon_results": horizon_results,
        "artifacts": {
            "unified_model_comparison": unified_csv_path,
            "horizon_model_selection": selection_csv_path,
            "model_selection_rationale": json_path,
        },
    }
