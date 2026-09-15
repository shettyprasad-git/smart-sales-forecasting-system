from __future__ import annotations

from typing import Iterable

import pandas as pd


FEATURE_COLUMNS = [
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "is_weekend",
    "days_since_start",
    "quantity_lag_1",
    "quantity_lag_7",
    "quantity_lag_14",
    "quantity_lag_28",
    "quantity_rolling_mean_7",
    "quantity_rolling_std_7",
    "quantity_rolling_mean_14",
    "quantity_rolling_std_14",
    "quantity_rolling_mean_28",
    "quantity_rolling_std_28",
    "Promotions",
    "Holiday_Flag",
]


def _get_dates(df: pd.DataFrame) -> pd.Series:
    """Return dates as a pandas Series."""

    if "Date" in df.columns:
        return pd.to_datetime(df["Date"])

    if isinstance(df.index, pd.DatetimeIndex):
        return pd.Series(
            df.index,
            index=df.index,
        )

    raise ValueError(
        "Data must contain a Date column or DatetimeIndex."
    )


def add_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add calendar-based forecasting features."""

    result = df.copy()

    dates = _get_dates(result)

    result["day_of_week"] = dates.dt.dayofweek
    result["day_of_month"] = dates.dt.day

    result["week_of_year"] = (
        dates.dt.isocalendar()
        .week
        .astype(int)
    )

    result["month"] = dates.dt.month
    result["quarter"] = dates.dt.quarter

    result["is_weekend"] = (
        dates.dt.dayofweek >= 5
    ).astype(int)

    first_date = dates.min()

    result["days_since_start"] = (
        dates - first_date
    ).dt.days

    return result


def add_lag_features(
    df: pd.DataFrame,
    target_column: str = "Quantity",
    lags: Iterable[int] = (
        1,
        7,
        14,
        28,
    ),
) -> pd.DataFrame:
    """Add lag features for the target variable."""

    result = df.copy()

    for lag in lags:
        result[
            f"{target_column.lower()}_lag_{lag}"
        ] = result[target_column].shift(lag)

    return result


def add_rolling_features(
    df: pd.DataFrame,
    target_column: str = "Quantity",
    windows: Iterable[int] = (
        7,
        14,
        28,
    ),
) -> pd.DataFrame:
    """
    Add rolling mean and standard deviation.

    Shift by one day to prevent target leakage.
    """

    result = df.copy()

    for window in windows:

        shifted = result[
            target_column
        ].shift(1)

        result[
            f"{target_column.lower()}_rolling_mean_{window}"
        ] = shifted.rolling(window).mean()

        result[
            f"{target_column.lower()}_rolling_std_{window}"
        ] = shifted.rolling(window).std()

    return result


def build_feature_dataset(
    df: pd.DataFrame,
    target_column: str = "Quantity",
) -> pd.DataFrame:
    """
    Build the complete production feature dataset.

    The Date index is preserved so that downstream
    time-based train/validation/test splitting works.
    """

    result = df.copy()

    # ---------------------------------------------------------
    # Preserve Date information
    # ---------------------------------------------------------

    if "Date" in result.columns:

        result["Date"] = pd.to_datetime(
            result["Date"],
            errors="raise",
        )

        result = (
            result
            .sort_values("Date")
            .set_index("Date")
        )

    elif isinstance(
        result.index,
        pd.DatetimeIndex,
    ):

        result.index = pd.to_datetime(
            result.index,
            errors="raise",
        )

        result = result.sort_index()

    else:

        raise ValueError(
            "Input data must contain a Date column "
            "or use a DatetimeIndex."
        )

    # ---------------------------------------------------------
    # Calendar features
    # ---------------------------------------------------------

    result = add_calendar_features(result)

    # ---------------------------------------------------------
    # Lag features
    # ---------------------------------------------------------

    result = add_lag_features(
        result,
        target_column=target_column,
    )

    # ---------------------------------------------------------
    # Rolling features
    # ---------------------------------------------------------

    result = add_rolling_features(
        result,
        target_column=target_column,
    )

    # ---------------------------------------------------------
    # Event features
    # ---------------------------------------------------------

    if "Promotions" not in result.columns:
        result["Promotions"] = 0

    if "Holiday_Flag" not in result.columns:
        result["Holiday_Flag"] = 0

    result["Promotions"] = pd.to_numeric(
        result["Promotions"],
        errors="coerce",
    ).fillna(0)

    result["Holiday_Flag"] = pd.to_numeric(
        result["Holiday_Flag"],
        errors="coerce",
    ).fillna(0)

    # ---------------------------------------------------------
    # Remove rows without enough historical data
    # ---------------------------------------------------------

    required_history_features = [
        "quantity_lag_1",
        "quantity_lag_7",
        "quantity_lag_14",
        "quantity_lag_28",
        "quantity_rolling_mean_7",
        "quantity_rolling_std_7",
        "quantity_rolling_mean_14",
        "quantity_rolling_std_14",
        "quantity_rolling_mean_28",
        "quantity_rolling_std_28",
    ]

    result = result.dropna(
        subset=required_history_features
    )

    return result


def get_feature_columns() -> list[str]:
    """Return the canonical production feature list."""

    return FEATURE_COLUMNS.copy()


def build_recursive_feature_row(
    history: pd.Series | list[float],
    future_date: pd.Timestamp,
    promotion: int = 0,
    holiday: int = 0,
    reference_start_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """
    Build one feature row for recursive forecasting.

    Parameters
    ----------
    history:
        Historical demand observations.

    future_date:
        Date being forecast.

    promotion:
        Promotion indicator for the forecast date.

    holiday:
        Holiday indicator for the forecast date.

    reference_start_date:
        The same starting date used when the model's
        training features were generated.

        This is required for a production-consistent
        days_since_start feature.
    """

    if not isinstance(
        future_date,
        pd.Timestamp,
    ):
        future_date = pd.Timestamp(
            future_date
        )

    history = pd.Series(
        history,
        dtype=float,
    ).reset_index(drop=True)

    if len(history) < 28:
        raise ValueError(
            "At least 28 historical observations "
            "are required."
        )

    if reference_start_date is None:
        raise ValueError(
            "reference_start_date is required for "
            "production inference so that "
            "days_since_start matches training."
        )

    reference_start_date = pd.Timestamp(
        reference_start_date
    )

    if future_date < reference_start_date:
        raise ValueError(
            "future_date cannot be earlier than "
            "reference_start_date."
        )

    day_of_week = future_date.dayofweek

    days_since_start = (
        future_date - reference_start_date
    ).days

    row = {
        "day_of_week": day_of_week,

        "day_of_month": future_date.day,

        "week_of_year": int(
            future_date.isocalendar().week
        ),

        "month": future_date.month,

        "quarter": future_date.quarter,

        "is_weekend": int(
            day_of_week >= 5
        ),

        "days_since_start": days_since_start,

        "quantity_lag_1": history.iloc[-1],

        "quantity_lag_7": history.iloc[-7],

        "quantity_lag_14": history.iloc[-14],

        "quantity_lag_28": history.iloc[-28],

        "quantity_rolling_mean_7": (
            history.iloc[-7:].mean()
        ),

        "quantity_rolling_std_7": (
            history.iloc[-7:].std()
        ),

        "quantity_rolling_mean_14": (
            history.iloc[-14:].mean()
        ),

        "quantity_rolling_std_14": (
            history.iloc[-14:].std()
        ),

        "quantity_rolling_mean_28": (
            history.iloc[-28:].mean()
        ),

        "quantity_rolling_std_28": (
            history.iloc[-28:].std()
        ),

        "Promotions": promotion,

        "Holiday_Flag": holiday,
    }

    return pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS,
    )