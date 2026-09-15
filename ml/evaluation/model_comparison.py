from __future__ import annotations

import pandas as pd


def best_model_by_horizon(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the best-performing model for each
    forecasting horizon based on WAPE.

    Lower WAPE is better.
    """

    required_columns = {
        "Horizon_Days",
        "WAPE",
    }

    missing_columns = (
        required_columns
        - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Results are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return (
        results
        .sort_values(
            [
                "Horizon_Days",
                "WAPE",
            ]
        )
        .groupby(
            "Horizon_Days",
            as_index=False,
        )
        .first()
    )


def rank_models(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank models within each forecasting horizon.

    Rank 1 represents the best model.
    """

    required_columns = {
        "Horizon_Days",
        "Model",
        "WAPE",
    }

    missing_columns = (
        required_columns
        - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Results are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    ranked = results.copy()

    ranked["Rank"] = (
        ranked
        .groupby("Horizon_Days")["WAPE"]
        .rank(
            method="min",
            ascending=True,
        )
        .astype(int)
    )

    return (
        ranked
        .sort_values(
            [
                "Horizon_Days",
                "Rank",
            ]
        )
        .reset_index(drop=True)
    )

