from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.analysis.model_selection import (
    BIAS_TIE_THRESHOLD,
    MAE_TIE_THRESHOLD,
    PRIMARY_CANDIDATES,
    RMSE_TIE_THRESHOLD,
    SIMPLICITY_RANK,
    VAR_RATIO_TIE_THRESHOLD,
    WAPE_TIE_THRESHOLD_DECIMAL,
    CandidateMetrics,
    HorizonSelectionResult,
    assess_structural_risk,
    compare_candidate_metrics,
    execute_model_selection,
    generate_selection_reason,
    load_evaluation_artifacts,
    load_residual_diagnostics,
    merge_metrics_and_diagnostics,
    normalize_model_name,
    select_model_for_horizon,
)


# ==============================================================================
# 1. Normalization & Loading Tests
# ==============================================================================

def test_normalize_model_name():
    """Verify string normalization strips whitespace, single quotes, and double quotes."""
    assert normalize_model_name('  "Random Forest"  ') == "Random Forest"
    assert normalize_model_name("'SARIMA(1,1,1)(0,1,1,7)'") == "SARIMA(1,1,1)(0,1,1,7)"
    assert normalize_model_name("HistGradientBoosting") == "HistGradientBoosting"
    assert normalize_model_name(123) == "123"


def test_load_evaluation_artifacts_real():
    """Verify loading real evaluation artifacts returns expected schema and deduplicates."""
    df = load_evaluation_artifacts()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = {"model", "horizon_days", "MAE", "RMSE", "MAPE", "WAPE", "Bias"}
    assert expected_cols.issubset(set(df.columns))

    # Verify deduplication: exactly 1 row per (model, horizon_days)
    dups = df.duplicated(subset=["model", "horizon_days"]).sum()
    assert dups == 0


def test_load_residual_diagnostics_real():
    """Verify loading residual diagnostics returns expected summary and Ljung-Box columns."""
    df = load_residual_diagnostics()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_cols = {
        "model",
        "horizon_days",
        "variance_ratio",
        "acf_lag_1",
        "acf_lag_7",
        "acf_lag_14",
        "acf_lag_28",
        "outlier_pct_3sigma",
        "ljung_box_significant",
    }
    assert expected_cols.issubset(set(df.columns))


def test_merge_metrics_and_diagnostics():
    """Verify join preserves matching records across evaluation and residual diagnostics."""
    eval_df = load_evaluation_artifacts()
    diag_df = load_residual_diagnostics()
    merged = merge_metrics_and_diagnostics(eval_df, diag_df)

    assert not merged.empty
    assert "WAPE" in merged.columns
    assert "variance_ratio" in merged.columns
    assert "ljung_box_significant" in merged.columns


def test_missing_artifact_handling(tmp_path: Path):
    """Verify missing artifact paths raise FileNotFoundError."""
    fake_path = tmp_path / "nonexistent.csv"
    with pytest.raises(FileNotFoundError, match="Missing evaluation artifact"):
        load_evaluation_artifacts({"fake": fake_path})

    with pytest.raises(FileNotFoundError, match="Missing residual summary artifact"):
        load_residual_diagnostics(summary_path=fake_path)


# ==============================================================================
# 2. Candidate Filtering & Horizon Sets
# ==============================================================================

def test_horizon_specific_candidate_sets():
    """Verify designated candidate sets per horizon exclude dominated models (Holt Additive, ARIMA)."""
    assert 7 in PRIMARY_CANDIDATES
    assert 30 in PRIMARY_CANDIDATES
    assert 90 in PRIMARY_CANDIDATES

    # 7-day
    c7 = PRIMARY_CANDIDATES[7]
    assert "Random Forest" in c7
    assert "Holt-Winters Additive Weekly" in c7
    assert "Damped Holt-Winters Additive Weekly" in c7
    assert "SARIMA(1,1,1)(0,1,1,7)" in c7
    assert "Seasonal Naive" in c7
    assert "Holt Additive" not in c7
    assert "ARIMA(2,1,2)" not in c7

    # 30-day
    c30 = PRIMARY_CANDIDATES[30]
    assert "HistGradientBoosting" in c30
    assert "Holt Additive" not in c30
    assert "ARIMA(2,1,2)" not in c30

    # 90-day
    c90 = PRIMARY_CANDIDATES[90]
    assert "Linear Regression" in c90
    assert "Holt Additive" not in c90
    assert "ARIMA(2,1,2)" not in c90


# ==============================================================================
# 3. Deterministic Selection Policy & Tie-Breaking
# ==============================================================================

def test_wape_ranking_primary_selection():
    """Primary selection prefers lowest WAPE when difference > 0.10 percentage points."""
    cand_a = CandidateMetrics(
        horizon_days=7,
        model="ModelA",
        mae=100.0,
        rmse=150.0,
        mape=0.03,
        wape=0.025,  # 2.5%
        bias=10.0,
        wape_delta_vs_baseline=-0.01,
        lag1_acf=0.5,
        lag7_acf=0.2,
        lag14_acf=0.1,
        lag28_acf=0.05,
        ljung_box_significant=True,
        variance_ratio=1.1,
        outlier_percentage=1.0,
        structural_risk="LOW",
    )
    cand_b = CandidateMetrics(
        horizon_days=7,
        model="ModelB",
        mae=90.0,  # Lower MAE
        rmse=140.0,  # Lower RMSE
        mape=0.035,
        wape=0.028,  # 2.8% (diff = 0.3 pp > 0.10 pp)
        bias=5.0,
        wape_delta_vs_baseline=-0.007,
        lag1_acf=0.5,
        lag7_acf=0.2,
        lag14_acf=0.1,
        lag28_acf=0.05,
        ljung_box_significant=True,
        variance_ratio=1.1,
        outlier_percentage=1.0,
        structural_risk="LOW",
    )

    # Cand A must beat Cand B because WAPE is lower by > 0.10 pp, despite Cand B having lower MAE/RMSE
    assert compare_candidate_metrics(cand_a, cand_b) == -1
    assert compare_candidate_metrics(cand_b, cand_a) == 1


def test_tie_breaking_by_mae():
    """When WAPE difference <= 0.10 pp, tie is broken by MAE."""
    # WAPE difference = 0.0004 (0.04 pp <= 0.10 pp)
    cand_a = CandidateMetrics(
        horizon_days=7,
        model="ModelA",
        mae=120.0,
        rmse=200.0,
        mape=0.02,
        wape=0.0254,
        bias=10.0,
        wape_delta_vs_baseline=-0.01,
        lag1_acf=0.5,
        lag7_acf=0.2,
        lag14_acf=0.1,
        lag28_acf=0.05,
        ljung_box_significant=True,
        variance_ratio=1.1,
        outlier_percentage=1.0,
        structural_risk="LOW",
    )
    cand_b = CandidateMetrics(
        horizon_days=7,
        model="ModelB",
        mae=110.0,  # MAE lower by 10 units (> 0.10 units)
        rmse=205.0,
        mape=0.02,
        wape=0.0250,
        bias=10.0,
        wape_delta_vs_baseline=-0.01,
        lag1_acf=0.5,
        lag7_acf=0.2,
        lag14_acf=0.1,
        lag28_acf=0.05,
        ljung_box_significant=True,
        variance_ratio=1.1,
        outlier_percentage=1.0,
        structural_risk="LOW",
    )

    assert compare_candidate_metrics(cand_b, cand_a) == -1


def test_tie_breaking_by_rmse():
    """When WAPE and MAE are tied within 0.10 thresholds, tie is broken by RMSE."""
    cand_a = CandidateMetrics(
        horizon_days=7,
        model="ModelA",
        mae=150.00,
        rmse=316.50,  # Lower RMSE by 0.50 units (> 0.10)
        mape=0.025,
        wape=0.026218,
        bias=22.0,
        wape_delta_vs_baseline=-0.01,
        lag1_acf=0.75,
        lag7_acf=-0.36,
        lag14_acf=0.03,
        lag28_acf=0.01,
        ljung_box_significant=True,
        variance_ratio=2.0,
        outlier_percentage=6.0,
        structural_risk="MEDIUM",
    )
    cand_b = CandidateMetrics(
        horizon_days=7,
        model="ModelB",
        mae=150.02,  # MAE diff = 0.02 <= 0.10 (tied)
        rmse=317.00,
        mape=0.025,
        wape=0.026220,  # WAPE diff = 0.000002 <= 0.0010 (tied)
        bias=20.0,
        wape_delta_vs_baseline=-0.01,
        lag1_acf=0.75,
        lag7_acf=-0.36,
        lag14_acf=0.03,
        lag28_acf=0.01,
        ljung_box_significant=True,
        variance_ratio=2.0,
        outlier_percentage=6.0,
        structural_risk="MEDIUM",
    )

    assert compare_candidate_metrics(cand_a, cand_b) == -1


def test_tie_breaking_by_absolute_bias():
    """When WAPE, MAE, and RMSE are tied within 0.10 thresholds, tie is broken by |Bias|."""
    cand_a = CandidateMetrics(
        horizon_days=7,
        model="ModelA",
        mae=150.00,
        rmse=300.00,
        mape=0.025,
        wape=0.0260,
        bias=30.0,  # |Bias| = 30.0
        wape_delta_vs_baseline=-0.01,
        lag1_acf=0.5,
        lag7_acf=0.2,
        lag14_acf=0.1,
        lag28_acf=0.05,
        ljung_box_significant=True,
        variance_ratio=1.1,
        outlier_percentage=1.0,
        structural_risk="LOW",
    )
    cand_b = CandidateMetrics(
        horizon_days=7,
        model="ModelB",
        mae=150.02,
        rmse=300.04,
        mape=0.025,
        wape=0.0260,
        bias=10.0,  # |Bias| = 10.0 (lower by 20.0 units)
        wape_delta_vs_baseline=-0.01,
        lag1_acf=0.5,
        lag7_acf=0.2,
        lag14_acf=0.1,
        lag28_acf=0.05,
        ljung_box_significant=True,
        variance_ratio=1.1,
        outlier_percentage=1.0,
        structural_risk="LOW",
    )

    assert compare_candidate_metrics(cand_b, cand_a) == -1


def test_tie_breaking_by_variance_ratio():
    """When accuracy and bias are tied, tie is broken by variance ratio distance from 1.0."""
    cand_a = CandidateMetrics(
        horizon_days=7,
        model="ModelA",
        mae=150.0,
        rmse=300.0,
        mape=0.025,
        wape=0.0260,
        bias=10.0,
        wape_delta_vs_baseline=-0.01,
        lag1_acf=0.5,
        lag7_acf=0.2,
        lag14_acf=0.1,
        lag28_acf=0.05,
        ljung_box_significant=True,
        variance_ratio=1.10,  # Dist = 0.10
        outlier_percentage=1.0,
        structural_risk="LOW",
    )
    cand_b = CandidateMetrics(
        horizon_days=7,
        model="ModelB",
        mae=150.0,
        rmse=300.0,
        mape=0.025,
        wape=0.0260,
        bias=10.0,
        wape_delta_vs_baseline=-0.01,
        lag1_acf=0.5,
        lag7_acf=0.2,
        lag14_acf=0.1,
        lag28_acf=0.05,
        ljung_box_significant=True,
        variance_ratio=1.50,  # Dist = 0.50 (further from 1.0)
        outlier_percentage=1.0,
        structural_risk="LOW",
    )

    assert compare_candidate_metrics(cand_a, cand_b) == -1


def test_tie_breaking_by_simplicity_hierarchy():
    """When all numerical metrics are tied, simpler model wins by SIMPLICITY_RANK."""
    cand_naive = CandidateMetrics(
        horizon_days=7,
        model="Seasonal Naive",  # Simplicity rank 1
        mae=150.0,
        rmse=300.0,
        mape=0.025,
        wape=0.0260,
        bias=10.0,
        wape_delta_vs_baseline=0.0,
        lag1_acf=0.5,
        lag7_acf=0.2,
        lag14_acf=0.1,
        lag28_acf=0.05,
        ljung_box_significant=True,
        variance_ratio=1.10,
        outlier_percentage=1.0,
        structural_risk="LOW",
    )
    cand_sarima = CandidateMetrics(
        horizon_days=7,
        model="SARIMA(1,1,1)(0,1,1,7)",  # Simplicity rank 5
        mae=150.0,
        rmse=300.0,
        mape=0.025,
        wape=0.0260,
        bias=10.0,
        wape_delta_vs_baseline=-0.01,
        lag1_acf=0.5,
        lag7_acf=0.2,
        lag14_acf=0.1,
        lag28_acf=0.05,
        ljung_box_significant=True,
        variance_ratio=1.10,
        outlier_percentage=1.0,
        structural_risk="LOW",
    )

    assert compare_candidate_metrics(cand_naive, cand_sarima) == -1


# ==============================================================================
# 4. Structural Risk Flag Classification Tests
# ==============================================================================

def test_structural_risk_flag_levels():
    """Verify deterministic structural risk level assignment."""
    # 1. LOW: All clean
    risk_low = assess_structural_risk(
        variance_ratio=1.05,
        bias=12.0,
        lag1_acf=0.45,
        lag7_acf=0.30,
        outlier_pct=1.2,
    )
    assert risk_low.level == "LOW"

    # 2. MEDIUM: Triggered by variance expansion (VR = 2.0)
    risk_med1 = assess_structural_risk(
        variance_ratio=2.0,
        bias=15.0,
        lag1_acf=0.50,
        lag7_acf=0.30,
        outlier_pct=1.0,
    )
    assert risk_med1.level == "MEDIUM"

    # 3. MEDIUM: Triggered by moderate bias (|Bias| = 80.0)
    risk_med2 = assess_structural_risk(
        variance_ratio=1.1,
        bias=80.0,
        lag1_acf=0.50,
        lag7_acf=0.30,
        outlier_pct=1.0,
    )
    assert risk_med2.level == "MEDIUM"

    # 4. HIGH: Triggered by extreme variance ratio (VR = 6.14)
    risk_high1 = assess_structural_risk(
        variance_ratio=6.14,
        bias=20.0,
        lag1_acf=0.50,
        lag7_acf=0.30,
        outlier_pct=1.0,
    )
    assert risk_high1.level == "HIGH"

    # 5. HIGH: Triggered by severe bias (|Bias| = 220.0)
    risk_high2 = assess_structural_risk(
        variance_ratio=1.1,
        bias=-220.0,
        lag1_acf=0.50,
        lag7_acf=0.30,
        outlier_pct=1.0,
    )
    assert risk_high2.level == "HIGH"

    # 6. HIGH: Triggered by severe persistent autocorrelation (rho1 >= 0.90 and rho7 >= 0.70)
    risk_high3 = assess_structural_risk(
        variance_ratio=1.1,
        bias=20.0,
        lag1_acf=0.92,
        lag7_acf=0.76,
        outlier_pct=1.0,
    )
    assert risk_high3.level == "HIGH"


# ==============================================================================
# 5. Adversarial Synthetic Selection Tests
# ==============================================================================

def test_adversarial_selection_lower_wape_wins_despite_high_risk():
    """
    CRITICAL POLICY TEST:
    A model with lower WAPE must be selected even if it has HIGH structural risk.
    Structural risk must NOT silently change model ranking.
    """
    model_high_risk = CandidateMetrics(
        horizon_days=90,
        model="Linear Regression",
        mae=253.74,
        rmse=376.30,
        mape=0.045,
        wape=0.0440,  # 4.40%
        bias=-221.23,
        wape_delta_vs_baseline=-0.02,
        lag1_acf=0.92,
        lag7_acf=0.77,
        lag14_acf=0.63,
        lag28_acf=0.42,
        ljung_box_significant=True,
        variance_ratio=6.14,
        outlier_percentage=0.0,
        structural_risk="HIGH",
        structural_risk_reasons=["Extreme variance instability"],
    )
    model_low_risk = CandidateMetrics(
        horizon_days=90,
        model="Seasonal Naive",
        mae=377.63,
        rmse=541.72,
        mape=0.065,
        wape=0.0656,  # 6.56% (clearly worse WAPE)
        bias=79.09,
        wape_delta_vs_baseline=0.0,
        lag1_acf=0.89,
        lag7_acf=0.65,
        lag14_acf=0.56,
        lag28_acf=0.41,
        ljung_box_significant=True,
        variance_ratio=1.22,
        outlier_percentage=1.6,
        structural_risk="LOW",
        structural_risk_reasons=[],
    )

    # Linear Regression must be ranked #1
    assert compare_candidate_metrics(model_high_risk, model_low_risk) == -1


def test_adversarial_selection_higher_wape_does_not_win():
    """A model with higher WAPE cannot win simply because it has lower structural risk."""
    cand_worse_accuracy = CandidateMetrics(
        horizon_days=30,
        model="ModelClean",
        mae=280.0,
        rmse=430.0,
        mape=0.048,
        wape=0.0485,  # 4.85%
        bias=10.0,
        wape_delta_vs_baseline=0.0,
        lag1_acf=0.5,
        lag7_acf=0.2,
        lag14_acf=0.1,
        lag28_acf=0.05,
        ljung_box_significant=True,
        variance_ratio=1.05,
        outlier_percentage=0.8,
        structural_risk="LOW",
    )
    cand_better_accuracy = CandidateMetrics(
        horizon_days=30,
        model="ModelAccurate",
        mae=168.0,
        rmse=238.0,
        mape=0.027,
        wape=0.0293,  # 2.93%
        bias=105.0,
        wape_delta_vs_baseline=-0.019,
        lag1_acf=0.65,
        lag7_acf=0.49,
        lag14_acf=0.35,
        lag28_acf=0.27,
        ljung_box_significant=True,
        variance_ratio=0.85,
        outlier_percentage=0.8,
        structural_risk="MEDIUM",
    )

    assert compare_candidate_metrics(cand_better_accuracy, cand_worse_accuracy) == -1


def test_adversarial_tied_wape_different_risks():
    """
    When WAPE is tied within 0.10 pp, lower MAE wins even if it has higher risk.
    """
    cand_tied_wape_better_mae = CandidateMetrics(
        horizon_days=7,
        model="ModelRiskA",
        mae=150.0,  # Lower MAE
        rmse=316.0,
        mape=0.025,
        wape=0.026210,
        bias=22.0,
        wape_delta_vs_baseline=-0.01,
        lag1_acf=0.75,
        lag7_acf=-0.36,
        lag14_acf=0.03,
        lag28_acf=0.01,
        ljung_box_significant=True,
        variance_ratio=2.1,
        outlier_percentage=6.3,
        structural_risk="HIGH",
    )
    cand_tied_wape_worse_mae = CandidateMetrics(
        horizon_days=7,
        model="ModelRiskB",
        mae=165.0,  # Worse MAE by 15 units
        rmse=318.0,
        mape=0.025,
        wape=0.026215,  # Diff = 0.000005 (tied within 0.0010)
        bias=5.0,
        wape_delta_vs_baseline=-0.01,
        lag1_acf=0.50,
        lag7_acf=0.20,
        lag14_acf=0.05,
        lag28_acf=0.01,
        ljung_box_significant=True,
        variance_ratio=1.05,
        outlier_percentage=1.0,
        structural_risk="LOW",
    )

    assert compare_candidate_metrics(cand_tied_wape_better_mae, cand_tied_wape_worse_mae) == -1


# ==============================================================================
# 6. End-to-End Pipeline & Reproducibility Tests
# ==============================================================================

def test_deterministic_repeated_selection(tmp_path: Path):
    """Running selection multiple times produces identical selections, metrics, and rationales."""
    res1 = execute_model_selection(output_dir=tmp_path / "run1")
    res2 = execute_model_selection(output_dir=tmp_path / "run2")

    hr1 = res1["horizon_results"]
    hr2 = res2["horizon_results"]

    assert len(hr1) == len(hr2) == 3
    for h_a, h_b in zip(hr1, hr2):
        assert h_a.horizon_days == h_b.horizon_days
        assert h_a.selected_model == h_b.selected_model
        assert h_a.wape == h_b.wape
        assert h_a.mae == h_b.mae
        assert h_a.rmse == h_b.rmse
        assert h_a.bias == h_b.bias
        assert h_a.structural_risk == h_b.structural_risk
        assert h_a.selection_reason == h_b.selection_reason


def test_end_to_end_artifact_generation(tmp_path: Path):
    """Verify execution writes all 3 artifacts to disk with complete, expected schemas."""
    res = execute_model_selection(output_dir=tmp_path)
    artifacts = res["artifacts"]

    # 1. unified_model_comparison.csv
    unified_p = artifacts["unified_model_comparison"]
    assert unified_p.exists()
    assert unified_p.stat().st_size > 0
    df_uni = pd.read_csv(unified_p)
    assert len(df_uni) == 15  # 3 horizons * 5 candidates
    assert "accuracy_rank" in df_uni.columns
    assert "selected" in df_uni.columns
    assert df_uni["selected"].sum() == 3  # Exactly 1 winner per horizon

    # 2. horizon_model_selection.csv
    selection_p = artifacts["horizon_model_selection"]
    assert selection_p.exists()
    assert selection_p.stat().st_size > 0
    df_sel = pd.read_csv(selection_p)
    assert len(df_sel) == 3
    assert set(df_sel["horizon_days"]) == {7, 30, 90}
    assert "selection_reason" in df_sel.columns

    # Verify expected winners on actual data:
    # 7-day: Damped Holt-Winters
    # 30-day: HistGradientBoosting
    # 90-day: Linear Regression (with HIGH risk)
    sel_7 = df_sel[df_sel["horizon_days"] == 7].iloc[0]
    assert sel_7["selected_model"] == "Damped Holt-Winters Additive Weekly"

    sel_30 = df_sel[df_sel["horizon_days"] == 30].iloc[0]
    assert sel_30["selected_model"] == "HistGradientBoosting"

    sel_90 = df_sel[df_sel["horizon_days"] == 90].iloc[0]
    assert sel_90["selected_model"] == "Linear Regression"
    assert sel_90["structural_risk"] == "HIGH"

    # 3. model_selection_rationale.json
    json_p = artifacts["model_selection_rationale"]
    assert json_p.exists()
    with open(json_p, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert "selection_date" in meta
    assert "candidate_models" in meta
    assert "selection_policy" in meta
    assert "per_horizon_results" in meta
    assert "selected_models" in meta
    assert "structural_risk_flags" in meta
