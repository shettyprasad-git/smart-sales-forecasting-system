from __future__ import annotations

from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.analysis.model_selection import (
    DEFAULT_OUTPUT_DIR,
    HorizonSelectionResult,
    execute_model_selection,
)


def run() -> dict[str, Any]:
    """
    Execute Phase P1.0.2e Unified Model Comparison and Horizon-Specific Selection.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    print("=" * 80)
    print("SMART SALES FORECASTING - UNIFIED MODEL COMPARISON & SELECTION (P1.0.2e)")
    print("=" * 80)
    print(f"Execution Timestamp: {timestamp}")
    print("Methodology: Single Auditable Holdout Evaluation (2025 holdout, N=365 days)\n")

    result = execute_model_selection(output_dir=DEFAULT_OUTPUT_DIR)
    horizon_results: list[HorizonSelectionResult] = result["horizon_results"]

    print("-" * 80)
    print("HORIZON-SPECIFIC MODEL SELECTION RESULTS")
    print("-" * 80)

    # Header
    header = f"{'Horizon':<9} | {'Selected Model':<35} | {'WAPE':<8} | {'MAE':<8} | {'RMSE':<8} | {'Bias':<9} | {'Structural Risk':<15}"
    print(header)
    print("-" * len(header))

    for hr in horizon_results:
        wape_pct = f"{hr.wape * 100:.2f}%"
        mae_str = f"{hr.mae:.2f}"
        rmse_str = f"{hr.rmse:.2f}"
        bias_str = f"{hr.bias:+.2f}"
        row = (
            f"{str(hr.horizon_days) + ' days':<9} | "
            f"{hr.selected_model:<35} | "
            f"{wape_pct:<8} | "
            f"{mae_str:<8} | "
            f"{rmse_str:<8} | "
            f"{bias_str:<9} | "
            f"{hr.structural_risk:<15}"
        )
        print(row)

    print("-" * len(header))
    print("\n" + "=" * 80)
    print("SELECTION RATIONALE & DIAGNOSTIC CONCERNS")
    print("=" * 80)

    for hr in horizon_results:
        print(f"\n[Horizon {hr.horizon_days} Days] Selected: {hr.selected_model}")
        print(f"  * Selection Reason    : {hr.selection_reason}")
        print(f"  * Structural Risk     : {hr.structural_risk}")
        if hr.diagnostic_concerns:
            print("  * Diagnostic Concerns :")
            for c in hr.diagnostic_concerns:
                print(f"      - {c}")
        else:
            print("  * Diagnostic Concerns : None (clean residual properties)")

        print("  * Full Candidate Rankings (by policy):")
        for cand in hr.candidates_ranked:
            sel_tag = " [SELECTED]" if cand.selected else ""
            print(
                f"      Rank {cand.accuracy_rank}: {cand.model:<35} | "
                f"WAPE={cand.wape * 100:.3f}% | MAE={cand.mae:.2f} | RMSE={cand.rmse:.2f} | "
                f"Risk={cand.structural_risk}{sel_tag}"
            )

    print("\n" + "=" * 80)
    print("PERSISTED SELECTION ARTIFACTS")
    print("=" * 80)
    for name, path in result["artifacts"].items():
        print(f"  * {name:<26}: {path}")
    print("=" * 80)

    return result


if __name__ == "__main__":
    run()
