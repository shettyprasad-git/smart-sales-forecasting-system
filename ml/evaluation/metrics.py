from __future__ import annotations

import numpy as np


def wape(
    y_true,
    y_pred,
) -> float:
    """
    Weighted Absolute Percentage Error.

    WAPE = sum(|actual - prediction|)
           / sum(|actual|)
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    denominator = np.sum(
        np.abs(y_true)
    )

    if denominator == 0:
        return np.nan

    return float(
        np.sum(
            np.abs(y_true - y_pred)
        )
        / denominator
    )


def evaluate(
    y_true,
    y_pred,
) -> dict[str, float]:
    """
    Calculate forecasting evaluation metrics.

    Metrics:
        MAE
        RMSE
        MAPE
        WAPE
        Bias
    """

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if y_true.size == 0:
        raise ValueError(
            "Cannot evaluate empty arrays."
        )

    errors = y_true - y_pred

    # ---------------------------------------------------------
    # MAE
    # ---------------------------------------------------------

    mae = np.mean(
        np.abs(errors)
    )

    # ---------------------------------------------------------
    # RMSE
    # ---------------------------------------------------------

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    # ---------------------------------------------------------
    # MAPE
    #
    # Ignore observations where actual demand is zero.
    # ---------------------------------------------------------

    non_zero_mask = y_true != 0

    if np.any(non_zero_mask):

        mape = np.mean(
            np.abs(
                (
                    y_true[non_zero_mask]
                    - y_pred[non_zero_mask]
                )
                / y_true[non_zero_mask]
            )
        )

    else:

        mape = np.nan

    # ---------------------------------------------------------
    # WAPE
    # ---------------------------------------------------------

    wape_value = wape(
        y_true,
        y_pred,
    )

    # ---------------------------------------------------------
    # Bias
    #
    # Positive bias  = model tends to under-predict.
    # Negative bias  = model tends to over-predict.
    #
    # Defined as:
    # actual - predicted
    # ---------------------------------------------------------

    bias = np.mean(
        errors
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "WAPE": float(wape_value),
        "Bias": float(bias),
    }
