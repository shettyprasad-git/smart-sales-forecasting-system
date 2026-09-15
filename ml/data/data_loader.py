from pathlib import Path

import pandas as pd

from ml.data.data_validation import validate_processed_data


def load_processed_data(path: str | Path) -> pd.DataFrame:
    """
    Load the processed forecasting dataset and validate its schema.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    validate_processed_data(df)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


def aggregate_daily_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate product-level data into daily forecasting data.

    The source dataset uses:
        Promotions
        Holiday_Flag
    """

    required = {
        "Date",
        "Quantity",
        "Sales_Amount",
        "Profit",
        "Promotions",
        "Holiday_Flag",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns required for daily aggregation: {sorted(missing)}"
        )

    daily = (
        df.groupby("Date", as_index=True)
        .agg(
            Quantity=("Quantity", "sum"),
            Sales_Amount=("Sales_Amount", "sum"),
            Profit=("Profit", "sum"),
            Promotions=("Promotions", "max"),
            Holiday_Flag=("Holiday_Flag", "max"),
        )
        .sort_index()
    )

    daily["Promotions"] = daily["Promotions"].fillna(0)
    daily["Holiday_Flag"] = daily["Holiday_Flag"].fillna(0)

    return daily


def load_daily_data(path: str | Path) -> pd.DataFrame:
    """
    Load processed data and return daily aggregated data.
    """
    df = load_processed_data(path)
    return aggregate_daily_data(df)