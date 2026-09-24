from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed_daily_forecasting_features.csv"
)


from typing import Any
import pandas as pd

from backend.app.services.dataset_runtime_service import (
    dataset_runtime_service,
)


class HistoryService:
    """
    Provides historical demand, revenue, and profit data for dashboard visualization.
    Loads active user dataset via DatasetRuntimeService with user-isolation.
    """

    def get_history(
        self,
        limit: int = 365,
        user_id: int | None = None,
        db: Any = None,
    ) -> pd.DataFrame:

        if limit < 1:
            raise ValueError(
                "limit must be greater than 0."
            )

        df = dataset_runtime_service.get_daily_aggregate(
            user_id=user_id,
            db=db,
        )

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "Date",
                    "Quantity",
                    "Sales_Amount",
                    "Profit",
                ]
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
