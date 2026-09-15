from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_daily_forecasting_features.csv"
)


class HistoryService:
    """
    Provides historical demand data for dashboard visualization.

    The current production forecasting models are trained on the
    processed historical forecasting dataset, so this service reads
    from the same dataset.
    """

    def get_history(
        self,
        limit: int = 365,
    ) -> pd.DataFrame:

        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {DATASET_PATH}"
            )

        if limit < 1:
            raise ValueError(
                "limit must be greater than 0."
            )

        df = pd.read_csv(
            DATASET_PATH,
            parse_dates=["Date"],
        )

        if df.empty:
            raise ValueError(
                "Processed forecasting dataset is empty."
            )

        required_columns = [
            "Date",
            "Quantity",
            "Sales_Amount",
            "Profit",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Required historical columns are missing: "
                + ", ".join(missing_columns)
            )

        df = (
            df[
                required_columns
            ]
            .sort_values("Date")
            .tail(limit)
            .copy()
        )

        return df
