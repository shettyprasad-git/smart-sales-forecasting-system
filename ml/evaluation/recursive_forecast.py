from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ml.evaluation.metrics import evaluate
from ml.features.feature_pipeline import (
    FEATURE_COLUMNS,
    build_recursive_feature_row,
)


def extract_event_flags(
    events: Optional[pd.DataFrame | dict | pd.Series],
    date: pd.Timestamp,
) -> tuple[int, int]:
    """
    Extract (promotion, holiday) flags for a given date.

    Supports and normalizes alternative naming conventions:
      Promotions / Holiday_Flag
      Promotion / Is_Holiday

    Defaults to (0, 0) if absent, missing, or null.
    """
    if events is None:
        return 0, 0

    date = pd.Timestamp(date).normalize()

    # If events is a DataFrame
    if isinstance(events, pd.DataFrame):
        if not isinstance(events.index, pd.DatetimeIndex):
            if "Date" in events.columns:
                events = events.set_index(pd.to_datetime(events["Date"]).dt.normalize())
            else:
                try:
                    events = events.copy()
                    events.index = pd.to_datetime(events.index).normalize()
                except Exception:
                    pass

        if date not in events.index:
            return 0, 0

        event_entry = events.loc[date]
        if isinstance(event_entry, pd.DataFrame):
            event_entry = event_entry.iloc[0]

        promo_raw = event_entry.get("Promotions", event_entry.get("Promotion", 0))
        holiday_raw = event_entry.get("Holiday_Flag", event_entry.get("Is_Holiday", 0))

        try:
            promo = 0 if pd.isna(promo_raw) else int(promo_raw)
        except (ValueError, TypeError):
            promo = 0

        try:
            holiday = 0 if pd.isna(holiday_raw) else int(holiday_raw)
        except (ValueError, TypeError):
            holiday = 0

        return promo, holiday

    # If events is a dict mapping date -> (promo, holiday) or date -> dict
    if isinstance(events, dict):
        entry = events.get(date)
        if entry is None:
            return 0, 0
        if isinstance(entry, (tuple, list)) and len(entry) >= 2:
            return int(entry[0] or 0), int(entry[1] or 0)
        if isinstance(entry, dict):
            promo_raw = entry.get("Promotions", entry.get("Promotion", 0))
            holiday_raw = entry.get("Holiday_Flag", entry.get("Is_Holiday", 0))
            return int(promo_raw or 0), int(holiday_raw or 0)

    # If events is a Series
    if isinstance(events, pd.Series):
        val = events.get(date, 0)
        return int(val or 0), 0

    return 0, 0


def forecast_block(
    model: Any,
    history: Sequence[float] | np.ndarray | pd.Series,
    dates: Iterable[pd.Timestamp] | pd.DatetimeIndex,
    events: Optional[pd.DataFrame | dict | pd.Series] = None,
    reference_start_date: Optional[pd.Timestamp | str] = None,
    feature_columns: Optional[list[str]] = None,
) -> list[float]:
    """
    Forecast one complete block of dates recursively.

    Predictions within the block are appended back to history
    to compute lag and rolling features for subsequent days.
    Future target values are never used during the forecast block.

    All predictions are constrained to non-negative demand: max(0.0, prediction).
    """
    # Clone history to avoid mutating caller's data
    current_history = list(pd.Series(history, dtype=float).values)

    if len(current_history) < 28:
        raise ValueError(
            f"At least 28 historical observations are required for recursive forecasting; got {len(current_history)}."
        )

    if reference_start_date is None:
        raise ValueError(
            "reference_start_date is required so days_since_start matches training."
        )

    ref_date = pd.Timestamp(reference_start_date).normalize()
    active_features = feature_columns if feature_columns is not None else FEATURE_COLUMNS

    predictions: list[float] = []

    for raw_date in dates:
        date = pd.Timestamp(raw_date).normalize()
        promotion, holiday = extract_event_flags(events, date)

        feature_row = build_recursive_feature_row(
            history=current_history,
            future_date=date,
            promotion=promotion,
            holiday=holiday,
            reference_start_date=ref_date,
        )

        X = feature_row[active_features]
        raw_pred = float(model.predict(X)[0])
        prediction = max(0.0, raw_pred)

        predictions.append(prediction)
        # Feed prediction back into history for next step's lags/rolling statistics
        current_history.append(prediction)

    return predictions


def recursive_forecast(
    model: Any,
    history: Sequence[float] | np.ndarray | pd.Series,
    dates_or_start_date: pd.Timestamp | str | Iterable[pd.Timestamp] | pd.DatetimeIndex,
    horizon: Optional[int] = None,
    events: Optional[pd.DataFrame | dict | pd.Series] = None,
    reference_start_date: Optional[pd.Timestamp | str] = None,
    feature_columns: Optional[list[str]] = None,
    as_dataframe: bool = True,
) -> pd.DataFrame | list[float]:
    """
    Generate recursive multi-step demand predictions for a specified horizon
    or a sequence of dates.
    """
    if horizon is not None and (isinstance(dates_or_start_date, (str, pd.Timestamp)) or not hasattr(dates_or_start_date, "__len__")):
        if horizon <= 0:
            raise ValueError("Horizon must be greater than zero.")
        start_date = pd.Timestamp(dates_or_start_date).normalize()
        dates = pd.date_range(start=start_date, periods=horizon, freq="D")
    else:
        dates = pd.DatetimeIndex(dates_or_start_date)
        if len(dates) == 0:
            raise ValueError("At least one date is required for recursive forecasting.")

    preds = forecast_block(
        model=model,
        history=history,
        dates=dates,
        events=events,
        reference_start_date=reference_start_date,
        feature_columns=feature_columns,
    )

    if as_dataframe:
        return pd.DataFrame(
            {
                "Date": dates,
                "Predicted_Quantity": preds,
            }
        )
    return preds


def rolling_origin_forecast(
    model: Any,
    history_series: Sequence[float] | np.ndarray | pd.Series,
    test_series: pd.Series,
    horizon: int,
    events: Optional[pd.DataFrame | dict | pd.Series] = None,
    reference_start_date: Optional[pd.Timestamp | str] = None,
    feature_columns: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform rolling-origin recursive forecasting over an evaluation series.

    Protocol:
      1. Start with initial historical observations.
      2. Recursively forecast a block of up to `horizon` days.
      3. Compare predictions against the actual block.
      4. Reveal actual observations ONLY after the forecast block has completed.
      5. Add the actual block observations to history.
      6. Move origin forward and repeat until the evaluation series is exhausted.
    """
    if horizon <= 0:
        raise ValueError("Horizon must be greater than zero.")

    history = list(pd.Series(history_series, dtype=float).values)
    actual_list: list[float] = []
    predicted_list: list[float] = []

    start = 0
    total_test = len(test_series)

    while start < total_test:
        block_size = min(horizon, total_test - start)
        dates = pd.DatetimeIndex(test_series.index[start : start + block_size])

        block_predictions = forecast_block(
            model=model,
            history=history.copy(),
            dates=dates,
            events=events,
            reference_start_date=reference_start_date,
            feature_columns=feature_columns,
        )

        block_actual = test_series.iloc[start : start + block_size].astype(float).values

        actual_list.extend(block_actual)
        predicted_list.extend(block_predictions)

        # Critical rolling-origin step:
        # Reveal actual observations only after the forecast block is complete.
        history.extend(block_actual.tolist())

        start += block_size

    return (
        np.asarray(actual_list, dtype=float),
        np.asarray(predicted_list, dtype=float),
    )


def evaluate_recursive_forecast(
    actual_or_model: Any,
    predicted_or_history: Any = None,
    *,
    dates: Optional[Iterable[pd.Timestamp]] = None,
    actuals: Optional[Sequence[float] | np.ndarray | pd.Series] = None,
    events: Optional[pd.DataFrame | dict | pd.Series] = None,
    reference_start_date: Optional[pd.Timestamp | str] = None,
    feature_columns: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Calculate forecasting evaluation metrics (WAPE, MAE, RMSE, MAPE, Bias).

    Supports two invocation styles:
      1. Direct evaluation of true vs predicted values:
         evaluate_recursive_forecast(y_true, y_pred)
      2. Full recursive evaluation of a model over holdout dates:
         evaluate_recursive_forecast(
             model=model,
             history=history,
             dates=val_dates,
             actuals=val_actuals,
             reference_start_date=ref_date,
         )
    """
    if predicted_or_history is not None and not hasattr(actual_or_model, "predict"):
        # Style 1: Direct metric evaluation
        return evaluate(actual_or_model, predicted_or_history)

    # Style 2: Model + history + holdout dates
    model = actual_or_model
    history = predicted_or_history
    if dates is None or actuals is None:
        raise ValueError("Both 'dates' and 'actuals' are required when evaluating a model directly.")

    preds = forecast_block(
        model=model,
        history=history,
        dates=dates,
        events=events,
        reference_start_date=reference_start_date,
        feature_columns=feature_columns,
    )
    return evaluate(actuals, np.asarray(preds, dtype=float))
