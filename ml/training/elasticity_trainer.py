"""
Dedicated Company Demand Sensitivity & Elasticity Trainer.

Estimates price elasticity of demand and discount sensitivity from historical
sales records using regularized log-log regression (Ridge).

Econometric Concept:
    log(quantity + small_offset) ~ beta_0
                                   + beta_p * log(unit_price)
                                   + beta_d * (discount_percent / 100)
                                   + beta_promo * promotion
                                   + beta_hol * holiday_flag
                                   + calendar seasonality
                                   + product/category fixed effects

Safeguards & Invariants:
- Deterministic ML only (zero LLM numerical hallucination).
- Strict data sufficiency checks:
  * Minimum observations threshold (>= 30 non-zero quantity records).
  * Price variation check (>= 2 distinct prices, std > 1e-4).
  * Discount variation check (>= 2 distinct discounts, std > 1e-4).
- Safe fail-closed: if variation is insufficient, marks model UNAVAILABLE with clear diagnostics.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

MINIMUM_OBSERVATIONS = 30
VARIATION_STD_THRESHOLD = 1e-4


def train_company_elasticity(df: pd.DataFrame | None) -> dict[str, Any]:
    """
    Train a company-specific demand elasticity and discount sensitivity model.

    Args:
        df: Historical product-level sales records containing quantity, price, and discount.

    Returns:
        Dictionary containing model status ('ready', 'insufficient_data', or 'unavailable'),
        estimated coefficients, regression metrics, diagnostics, and serialized artifact bytes.
    """
    if df is None or df.empty:
        return {
            "status": "insufficient_data",
            "status_message": "Historical transaction dataset is empty.",
            "price_elasticity": None,
            "discount_sensitivity": None,
            "r2_score": None,
            "mae": None,
            "rmse": None,
            "training_rows": 0,
            "price_supported": False,
            "discount_supported": False,
            "price_reason": "Historical dataset is empty.",
            "discount_reason": "Historical dataset is empty.",
            "diagnostics": {"rows": 0},
            "artifact_bytes": None,
            "feature_columns": [],
        }

    # Normalize column names (case-insensitive lookup with common aliases)
    col_map = {str(c).lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}

    qty_col = (
        col_map.get("quantity")
        or col_map.get("qty")
        or col_map.get("units_sold")
        or col_map.get("units")
        or col_map.get("volume")
    )
    price_col = (
        col_map.get("unit_price")
        or col_map.get("unitprice")
        or col_map.get("price")
        or col_map.get("item_price")
    )
    disc_col = (
        col_map.get("discount_percent")
        or col_map.get("discount")
        or col_map.get("discount_pct")
        or col_map.get("discount_rate")
    )
    date_col = (
        col_map.get("date")
        or col_map.get("sale_date")
        or col_map.get("saledate")
        or col_map.get("order_date")
        or col_map.get("transaction_date")
    )
    promo_col = (
        col_map.get("promotion")
        or col_map.get("promotions")
        or col_map.get("promo")
        or col_map.get("is_promotion")
    )
    holiday_col = (
        col_map.get("is_holiday")
        or col_map.get("holiday_flag")
        or col_map.get("holiday")
    )
    cat_col = (
        col_map.get("category_id")
        or col_map.get("category_name")
        or col_map.get("category")
        or col_map.get("product_category")
    )

    if not qty_col:
        return {
            "status": "insufficient_data",
            "status_message": "Required field 'Quantity' is missing from the dataset.",
            "price_elasticity": None,
            "discount_sensitivity": None,
            "r2_score": None,
            "mae": None,
            "rmse": None,
            "training_rows": 0,
            "price_supported": False,
            "discount_supported": False,
            "price_reason": "Quantity field missing.",
            "discount_reason": "Quantity field missing.",
            "diagnostics": {"missing_column": "Quantity"},
            "artifact_bytes": None,
            "feature_columns": [],
        }

    # 1. Filter to positive finite quantities
    qty_series = pd.to_numeric(df[qty_col], errors="coerce")
    clean_df = df[qty_series > 0].copy()
    clean_df["_clean_qty"] = qty_series[qty_series > 0]

    if len(clean_df) < MINIMUM_OBSERVATIONS:
        return {
            "status": "insufficient_data",
            "status_message": (
                f"Insufficient positive quantity observations for demand sensitivity modeling: "
                f"found {len(clean_df)}, minimum {MINIMUM_OBSERVATIONS} required."
            ),
            "price_elasticity": None,
            "discount_sensitivity": None,
            "r2_score": None,
            "mae": None,
            "rmse": None,
            "training_rows": len(clean_df),
            "price_supported": False,
            "discount_supported": False,
            "price_reason": "Insufficient positive quantity observations.",
            "discount_reason": "Insufficient positive quantity observations.",
            "diagnostics": {"rows": len(clean_df), "min_required": MINIMUM_OBSERVATIONS},
            "artifact_bytes": None,
            "feature_columns": [],
        }

    # 2. Check Price Sufficiency
    price_supported = False
    price_reason: str | None = None
    distinct_prices = 0
    price_std = 0.0

    if not price_col:
        price_supported = False
        price_reason = "Required field 'unit_price' is missing from the dataset."
    else:
        price_series = pd.to_numeric(clean_df[price_col], errors="coerce")
        valid_price_mask = (price_series > 0) & np.isfinite(price_series)
        valid_prices = price_series[valid_price_mask]

        if len(valid_prices) < MINIMUM_OBSERVATIONS:
            price_supported = False
            price_reason = "Insufficient valid positive price observations."
        else:
            distinct_prices = int(valid_prices.nunique())
            price_std = float(valid_prices.std()) if len(valid_prices) > 1 else 0.0

            if distinct_prices < 2 or price_std <= VARIATION_STD_THRESHOLD:
                price_supported = False
                price_reason = (
                    "Active dataset does not contain sufficient historical price variation "
                    "to estimate demand sensitivity."
                )
            else:
                price_supported = True
                clean_df["_clean_price"] = price_series

    # 3. Check Discount Sufficiency
    discount_supported = False
    discount_reason: str | None = None
    distinct_discounts = 0
    discount_std = 0.0

    if not disc_col:
        discount_supported = False
        discount_reason = "Required field 'discount_percent' is missing from the dataset."
    else:
        disc_series = pd.to_numeric(clean_df[disc_col], errors="coerce").fillna(0.0)
        # Discount must be finite and within [0, 100]
        valid_disc_mask = (disc_series >= 0) & (disc_series <= 100) & np.isfinite(disc_series)
        valid_discs = disc_series[valid_disc_mask]

        if len(valid_discs) < MINIMUM_OBSERVATIONS:
            discount_supported = False
            discount_reason = "Insufficient valid discount observations."
        else:
            distinct_discounts = int(valid_discs.nunique())
            discount_std = float(valid_discs.std()) if len(valid_discs) > 1 else 0.0

            if distinct_discounts < 2 or discount_std <= VARIATION_STD_THRESHOLD:
                discount_supported = False
                discount_reason = (
                    "Active dataset does not contain sufficient historical discount variation "
                    "to estimate discount sensitivity."
                )
            else:
                discount_supported = True
                clean_df["_clean_discount"] = disc_series

    diagnostics = {
        "training_rows": len(clean_df),
        "distinct_prices": distinct_prices,
        "price_std": round(price_std, 4),
        "price_supported": price_supported,
        "price_reason": price_reason,
        "distinct_discounts": distinct_discounts,
        "discount_std": round(discount_std, 4),
        "discount_supported": discount_supported,
        "discount_reason": discount_reason,
    }

    # 4. If neither price nor discount is supported, return UNAVAILABLE
    if not price_supported and not discount_supported:
        reasons = [r for r in (price_reason, discount_reason) if r]
        return {
            "status": "unavailable",
            "status_message": " ; ".join(reasons) if reasons else "Insufficient price and discount variation.",
            "price_elasticity": None,
            "discount_sensitivity": None,
            "r2_score": None,
            "mae": None,
            "rmse": None,
            "training_rows": len(clean_df),
            "price_supported": False,
            "discount_supported": False,
            "price_reason": price_reason,
            "discount_reason": discount_reason,
            "diagnostics": diagnostics,
            "artifact_bytes": None,
            "feature_columns": [],
        }

    # 5. Build feature matrix X and target y
    # Drop rows that have invalid price if price is modeled, or invalid discount if discount is modeled
    fit_df = clean_df.copy()
    if price_supported:
        fit_df = fit_df[fit_df["_clean_price"] > 0]
    if discount_supported:
        fit_df = fit_df[fit_df["_clean_discount"].between(0, 100)]

    if len(fit_df) < MINIMUM_OBSERVATIONS:
        return {
            "status": "insufficient_data",
            "status_message": f"Fewer than {MINIMUM_OBSERVATIONS} clean rows available for regression.",
            "price_elasticity": None,
            "discount_sensitivity": None,
            "r2_score": None,
            "mae": None,
            "rmse": None,
            "training_rows": len(fit_df),
            "price_supported": False,
            "discount_supported": False,
            "price_reason": price_reason,
            "discount_reason": discount_reason,
            "diagnostics": diagnostics,
            "artifact_bytes": None,
            "feature_columns": [],
        }

    # Target: log(Quantity)
    y = np.log(fit_df["_clean_qty"].astype(float).values)

    feature_dict: dict[str, np.ndarray] = {}

    if price_supported:
        feature_dict["log_price"] = np.log(fit_df["_clean_price"].astype(float).values)

    if discount_supported:
        feature_dict["norm_discount"] = (fit_df["_clean_discount"].astype(float).values) / 100.0

    # Controls
    if promo_col and promo_col in fit_df.columns:
        feature_dict["promo"] = pd.to_numeric(fit_df[promo_col], errors="coerce").fillna(0.0).astype(float).values

    if holiday_col and holiday_col in fit_df.columns:
        feature_dict["holiday"] = pd.to_numeric(fit_df[holiday_col], errors="coerce").fillna(0.0).astype(float).values

    # Calendar seasonality control (day of week)
    if date_col and date_col in fit_df.columns:
        dates = pd.to_datetime(fit_df[date_col], errors="coerce")
        if dates.notna().all():
            dow = dates.dt.dayofweek.values
            feature_dict["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
            feature_dict["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    X_df = pd.DataFrame(feature_dict, index=fit_df.index)

    # Optional Category fixed effects if 2-15 categories exist
    if cat_col and cat_col in fit_df.columns:
        cats = fit_df[cat_col].astype(str)
        if 1 < cats.nunique() <= 15:
            cat_dummies = pd.get_dummies(cats, prefix="cat", drop_first=True, dtype=float)
            X_df = pd.concat([X_df, cat_dummies], axis=1)

    feature_names = list(X_df.columns)

    # 6. Fit regularized Ridge regression
    try:
        model = Ridge(alpha=1.0, fit_intercept=True)
        model.fit(X_df.values, y)

        y_pred = model.predict(X_df.values)
        r2 = float(r2_score(y, y_pred))
        mae = float(mean_absolute_error(np.exp(y), np.exp(y_pred)))
        rmse = float(np.sqrt(mean_squared_error(np.exp(y), np.exp(y_pred))))

        price_coef: float | None = None
        if price_supported and "log_price" in feature_names:
            idx = feature_names.index("log_price")
            raw_beta_p = float(model.coef_[idx])
            price_coef = round(raw_beta_p, 4)

        discount_coef: float | None = None
        if discount_supported and "norm_discount" in feature_names:
            idx = feature_names.index("norm_discount")
            raw_beta_d = float(model.coef_[idx])
            discount_coef = round(raw_beta_d, 4)

        # Diagnostics enhancement
        diagnostics["r2_score"] = round(r2, 4)
        diagnostics["features_used"] = feature_names

        # 7. Serialize Artifact
        artifact = {
            "model": model,
            "feature_columns": feature_names,
            "feature_version": "v1",
            "model_type": "RidgeLogLog",
            "price_elasticity": price_coef,
            "discount_sensitivity": discount_coef,
            "r2_score": round(r2, 4),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "training_rows": len(fit_df),
            "price_supported": price_supported,
            "discount_supported": discount_supported,
            "price_reason": price_reason,
            "discount_reason": discount_reason,
            "diagnostics": diagnostics,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }

        buf = io.BytesIO()
        joblib.dump(artifact, buf)
        artifact_bytes = buf.getvalue()

        return {
            "status": "ready",
            "status_message": None,
            "price_elasticity": price_coef,
            "discount_sensitivity": discount_coef,
            "r2_score": round(r2, 4),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "training_rows": len(fit_df),
            "price_supported": price_supported,
            "discount_supported": discount_supported,
            "price_reason": price_reason,
            "discount_reason": discount_reason,
            "diagnostics": diagnostics,
            "artifact_bytes": artifact_bytes,
            "feature_columns": feature_names,
        }

    except Exception as exc:
        logger.exception("Failed to fit elasticity Ridge regression: %s", exc)
        return {
            "status": "failed",
            "status_message": f"Model fitting failed: {exc}",
            "price_elasticity": None,
            "discount_sensitivity": None,
            "r2_score": None,
            "mae": None,
            "rmse": None,
            "training_rows": len(fit_df),
            "price_supported": False,
            "discount_supported": False,
            "price_reason": "Model fitting failed.",
            "discount_reason": "Model fitting failed.",
            "diagnostics": {"error": str(exc)},
            "artifact_bytes": None,
            "feature_columns": [],
        }
