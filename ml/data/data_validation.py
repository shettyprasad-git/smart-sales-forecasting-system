import pandas as pd


REQUIRED_COLUMNS = {
    "Date",
    "Quantity",
    "Promotions",
    "Holiday_Flag",
}


def validate_processed_data(df: pd.DataFrame) -> None:
    """
    Validate the processed forecasting dataset.

    Expected event columns:
        Promotions
        Holiday_Flag
    """

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["Date"].isna().any():
        raise ValueError("Date contains missing values.")

    if df["Quantity"].isna().any():
        raise ValueError("Quantity contains missing values.")

    if (df["Quantity"] < 0).any():
        raise ValueError("Quantity contains negative values.")

    if df["Date"].duplicated().any():
        # Duplicate dates are valid at product-day level.
        pass


def validate_daily_data(daily: pd.DataFrame) -> None:
    """
    Validate aggregated daily forecasting data.
    """

    if not isinstance(daily.index, pd.DatetimeIndex):
        raise ValueError("Daily data must use a DatetimeIndex.")

    if daily.index.duplicated().any():
        raise ValueError("Daily data contains duplicate dates.")

    if "Quantity" not in daily.columns:
        raise ValueError("Daily data must contain Quantity.")

    if daily["Quantity"].isna().any():
        raise ValueError("Daily Quantity contains missing values.")

    if (daily["Quantity"] < 0).any():
        raise ValueError("Daily Quantity contains negative values.")