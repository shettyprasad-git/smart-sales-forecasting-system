from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Literal
import uuid

import numpy as np
import pandas as pd

from backend.app.schemas.anomalies import (
    AnomalyItem,
    AnomalyListResponse,
    AnomalySummary,
    SeverityLevel,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DAILY_DATASET_PATH = PROJECT_ROOT / "data" / "synthetic_daily_forecasting.csv"
PRODUCT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "synthetic_product_daily_forecasting.csv"
)

SEVERITY_THRESHOLDS = [
    (4.0, "critical"),
    (3.0, "high"),
    (2.5, "medium"),
    (2.0, "low"),
]

SEVERITY_RANKS = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def score_to_severity(abs_score: float) -> SeverityLevel | None:
    """Map absolute z-score to statistically grounded severity level."""
    for threshold, severity in SEVERITY_THRESHOLDS:
        if abs_score >= threshold:
            return severity
    return None


from backend.app.services.dataset_runtime_service import (
    dataset_runtime_service,
)


class AnomalyDetectionService:
    """
    Production service for detecting sales and demand anomalies
    using causal rolling baselines and robust statistics (Median + MAD).
    Integrates with DatasetRuntimeService for user-isolated multi-tenant data.
    """

    def __init__(self) -> None:
        self._daily_cache: dict[tuple, pd.DataFrame] = {}
        self._category_cache: dict[tuple, pd.DataFrame] = {}

    def clear_cache(self, user_id: int | None = None) -> None:
        """Invalidate caches for a specific user, or all users if user_id is None."""
        if user_id is None:
            self._daily_cache.clear()
            self._category_cache.clear()
        else:
            self._daily_cache = {k: v for k, v in self._daily_cache.items() if k[0] != user_id}
            self._category_cache = {k: v for k, v in self._category_cache.items() if k[0] != user_id}

    def _load_daily_data(self, user_id: int | None = None, db: Any = None) -> pd.DataFrame:
        """Load and cache the daily aggregated sales dataset for user_id."""
        cache_key = (user_id,)
        if cache_key not in self._daily_cache:
            df = dataset_runtime_service.get_daily_aggregate(user_id=user_id, db=db)
            self._daily_cache[cache_key] = df
        return self._daily_cache[cache_key].copy()

    def _load_category_data(self, user_id: int | None = None, db: Any = None) -> pd.DataFrame:
        """Load and aggregate category-level daily sales for user_id."""
        cache_key = (user_id,)
        if cache_key not in self._category_cache:
            df = dataset_runtime_service.get_product_daily(user_id=user_id, db=db)
            if df.empty:
                cat_df = pd.DataFrame(
                    columns=[
                        "Date",
                        "Category_ID",
                        "Category_Name",
                        "Quantity",
                        "Sales_Amount",
                        "Promotion",
                        "Is_Holiday",
                    ]
                )
            else:
                cat_df = (
                    df.groupby(
                        ["Date", "Category_ID", "Category_Name"], as_index=False
                    )
                    .agg(
                        {
                            "Quantity": "sum",
                            "Sales_Amount": "sum",
                            "Promotion": "max",
                            "Is_Holiday": "max",
                        }
                    )
                    .sort_values("Date")
                    .reset_index(drop=True)
                )
            self._category_cache[cache_key] = cat_df
        return self._category_cache[cache_key].copy()

    @staticmethod
    def _compute_series_anomalies(
        dates: pd.Series,
        values: pd.Series,
        metric: Literal["quantity", "sales_amount"],
        entity_type: Literal["aggregate", "product", "category"] = "aggregate",
        entity_id: str | None = None,
        entity_name: str | None = None,
        promotions: pd.Series | None = None,
        holidays: pd.Series | None = None,
        window: int = 28,
        min_periods: int = 7,
    ) -> list[AnomalyItem]:
        """
        Compute anomalies on a 1D time series using strict causal rolling window.
        No data leakage: day t is evaluated using observations prior to t.
        """
        if len(values) < min_periods:
            return []

        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(dates),
                "Value": values.astype(float),
                "Promotion": (
                    promotions.astype(int)
                    if promotions is not None
                    else np.zeros(len(values), dtype=int)
                ),
                "Holiday": (
                    holidays.astype(int)
                    if holidays is not None
                    else np.zeros(len(values), dtype=int)
                ),
            }
        ).sort_values("Date").reset_index(drop=True)

        # Causal shift to guarantee zero lookahead leakage
        shifted = df["Value"].shift(1)

        rolling_median = shifted.rolling(
            window=window, min_periods=min_periods
        ).median()

        def _calc_mad(arr: np.ndarray) -> float:
            m = np.median(arr)
            return float(np.median(np.abs(arr - m)))

        rolling_mad = shifted.rolling(
            window=window, min_periods=min_periods
        ).apply(_calc_mad, raw=True)

        scale = 1.4826 * rolling_mad

        # Robust fallbacks for scale if MAD is zero
        rolling_std = shifted.rolling(
            window=window, min_periods=min_periods
        ).std()
        scale = scale.where(scale > 1e-6, rolling_std)
        fallback_scale = np.maximum(1.0, 0.01 * rolling_median.abs())
        scale = scale.where(scale > 1e-6, fallback_scale).fillna(fallback_scale)

        deviations = df["Value"] - rolling_median
        z_scores = deviations / scale

        anomalies: list[AnomalyItem] = []

        for idx, row in df.iterrows():
            if pd.isna(rolling_median.iloc[idx]) or pd.isna(z_scores.iloc[idx]):
                continue

            score = float(z_scores.iloc[idx])
            abs_score = abs(score)
            severity = score_to_severity(abs_score)

            if severity is None:
                continue

            actual = float(row["Value"])
            expected = float(rolling_median.iloc[idx])
            dev = float(deviations.iloc[idx])
            dev_pct = (dev / expected * 100.0) if expected != 0 else 0.0
            direction = "spike" if dev > 0 else "drop"

            # Contextual explanation
            unit_label = (
                "units" if metric == "quantity" else "in revenue"
            )
            val_format = f"{actual:,.0f}" if metric == "quantity" else f"${actual:,.2f}"
            exp_format = (
                f"{expected:,.0f}" if metric == "quantity" else f"${expected:,.2f}"
            )
            sign = "+" if dev >= 0 else ""
            context_clause = []
            if row["Promotion"] > 0:
                context_clause.append("active promotional campaign")
            if row["Holiday"] > 0:
                context_clause.append("holiday trading")

            context_str = (
                f" (coincided with {', '.join(context_clause)})"
                if context_clause
                else " (during standard trading)"
            )

            explanation = (
                f"{entity_name or entity_type.capitalize()} {metric.replace('_', ' ')} of "
                f"{val_format} {unit_label} was {sign}{dev_pct:.1f}% vs "
                f"{window}-day baseline ({exp_format} {unit_label}){context_str}, "
                f"yielding a robust z-score of {score:.2f} ({severity} {direction})."
            )

            anom_id = f"anom-{row['Date'].strftime('%Y%m%d')}-{metric[:3]}-{entity_type[:3]}"
            if entity_id:
                anom_id += f"-{entity_id}"

            anomalies.append(
                AnomalyItem(
                    id=anom_id,
                    date=row["Date"].date(),
                    metric=metric,
                    entity_type=entity_type,
                    entity_id=str(entity_id) if entity_id else None,
                    entity_name=entity_name,
                    actual_value=round(actual, 2),
                    expected_value=round(expected, 2),
                    deviation=round(dev, 2),
                    deviation_percent=round(dev_pct, 2),
                    anomaly_score=round(abs_score, 2),
                    severity=severity,
                    direction=direction,
                    explanation=explanation,
                )
            )

        return anomalies

    def detect_aggregate_anomalies(
        self,
        metrics: list[Literal["quantity", "sales_amount"]] | None = None,
        window: int = 28,
        min_periods: int = 7,
        user_id: int | None = None,
        db: Any = None,
    ) -> list[AnomalyItem]:
        """Detect anomalies across the daily aggregate sales dataset."""
        df = self._load_daily_data(user_id=user_id, db=db)
        if df.empty:
            return []
        all_metrics = metrics or ["quantity", "sales_amount"]
        results: list[AnomalyItem] = []

        for m in all_metrics:
            col_name = "Quantity" if m == "quantity" else "Sales_Amount"
            promo_col = df["Promotions"] if "Promotions" in df.columns else None
            hol_col = (
                df["Holiday_Flag"] if "Holiday_Flag" in df.columns else None
            )

            anoms = self._compute_series_anomalies(
                dates=df["Date"],
                values=df[col_name],
                metric=m,
                entity_type="aggregate",
                entity_name="Total System Sales",
                promotions=promo_col,
                holidays=hol_col,
                window=window,
                min_periods=min_periods,
            )
            results.extend(anoms)

        # Sort chronologically descending
        results.sort(key=lambda x: (x.date, x.anomaly_score), reverse=True)
        return results

    def detect_category_anomalies(
        self,
        category_id: str | None = None,
        metrics: list[Literal["quantity", "sales_amount"]] | None = None,
        window: int = 28,
        min_periods: int = 7,
        user_id: int | None = None,
        db: Any = None,
    ) -> list[AnomalyItem]:
        """Detect anomalies grouped by product category."""
        df = self._load_category_data(user_id=user_id, db=db)
        if df.empty:
            return []
        all_metrics = metrics or ["quantity", "sales_amount"]

        if category_id:
            df = df[df["Category_ID"].astype(str) == str(category_id)]

        results: list[AnomalyItem] = []

        for cat_id, group in df.groupby("Category_ID"):
            cat_name = str(group["Category_Name"].iloc[0])
            g_sorted = group.sort_values("Date").reset_index(drop=True)

            for m in all_metrics:
                col_name = "Quantity" if m == "quantity" else "Sales_Amount"
                anoms = self._compute_series_anomalies(
                    dates=g_sorted["Date"],
                    values=g_sorted[col_name],
                    metric=m,
                    entity_type="category",
                    entity_id=str(cat_id),
                    entity_name=cat_name,
                    promotions=g_sorted["Promotion"],
                    holidays=g_sorted["Is_Holiday"],
                    window=window,
                    min_periods=min_periods,
                )
                results.extend(anoms)

        results.sort(key=lambda x: (x.date, x.anomaly_score), reverse=True)
        return results

    def detect_product_anomalies(
        self,
        product_id: int | str,
        metrics: list[Literal["quantity", "sales_amount"]] | None = None,
        window: int = 28,
        min_periods: int = 7,
        user_id: int | None = None,
        db: Any = None,
    ) -> list[AnomalyItem]:
        """Detect anomalies for a specific product ID."""
        df = dataset_runtime_service.get_product_daily(user_id=user_id, db=db)
        if df.empty:
            return []

        df_prod = df[df["Product_ID"].astype(str) == str(product_id)].sort_values("Date")

        if df_prod.empty:
            return []

        prod_name = str(df_prod["Product_Name"].iloc[0])
        all_metrics = metrics or ["quantity", "sales_amount"]
        results: list[AnomalyItem] = []

        for m in all_metrics:
            col_name = "Quantity" if m == "quantity" else "Sales_Amount"
            anoms = self._compute_series_anomalies(
                dates=df_prod["Date"],
                values=df_prod[col_name],
                metric=m,
                entity_type="product",
                entity_id=str(product_id),
                entity_name=prod_name,
                promotions=df_prod["Promotion"],
                holidays=df_prod["Is_Holiday"],
                window=window,
                min_periods=min_periods,
            )
            results.extend(anoms)

        results.sort(key=lambda x: (x.date, x.anomaly_score), reverse=True)
        return results

    def get_anomalies(
        self,
        metric: str | None = None,
        severity: str | None = None,
        min_severity: str | None = None,
        direction: str | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        skip: int = 0,
        limit: int = 100,
        window: int = 28,
        user_id: int | None = None,
        db: Any = None,
    ) -> AnomalyListResponse:
        """
        Query detected anomalies with comprehensive filtering and summary KPIs.
        """
        # Select data source based on entity_type
        if entity_type == "category":
            anomalies = self.detect_category_anomalies(
                category_id=entity_id,
                metrics=[metric] if metric in ("quantity", "sales_amount") else None,
                window=window,
                user_id=user_id,
                db=db,
            )
            eval_count = len(self._load_category_data(user_id=user_id, db=db))
        elif entity_type == "product" and entity_id:
            anomalies = self.detect_product_anomalies(
                product_id=entity_id,
                metrics=[metric] if metric in ("quantity", "sales_amount") else None,
                window=window,
                user_id=user_id,
                db=db,
            )
            eval_count = len(anomalies)  # approximate for single product
        else:
            anomalies = self.detect_aggregate_anomalies(
                metrics=[metric] if metric in ("quantity", "sales_amount") else None,
                window=window,
                user_id=user_id,
                db=db,
            )
            eval_count = len(self._load_daily_data(user_id=user_id, db=db)) * (
                1 if metric in ("quantity", "sales_amount") else 2
            )

        # Apply in-memory filters
        filtered = anomalies

        if metric:
            filtered = [a for a in filtered if a.metric == metric]

        if severity:
            filtered = [a for a in filtered if a.severity == severity]
        elif min_severity and min_severity in SEVERITY_RANKS:
            min_rank = SEVERITY_RANKS[min_severity]
            filtered = [
                a for a in filtered if SEVERITY_RANKS.get(a.severity, 0) >= min_rank
            ]

        if direction:
            filtered = [a for a in filtered if a.direction == direction]

        if entity_type:
            filtered = [a for a in filtered if a.entity_type == entity_type]

        if entity_id:
            filtered = [a for a in filtered if a.entity_id == str(entity_id)]

        if start_date:
            filtered = [a for a in filtered if a.date >= start_date]

        if end_date:
            filtered = [a for a in filtered if a.date <= end_date]

        # Compute summary metrics on filtered collection
        total_matching = len(filtered)
        severity_counts = {
            "critical": sum(1 for a in filtered if a.severity == "critical"),
            "high": sum(1 for a in filtered if a.severity == "high"),
            "medium": sum(1 for a in filtered if a.severity == "medium"),
            "low": sum(1 for a in filtered if a.severity == "low"),
        }
        direction_counts = {
            "spike": sum(1 for a in filtered if a.direction == "spike"),
            "drop": sum(1 for a in filtered if a.direction == "drop"),
        }
        metric_counts = {
            "quantity": sum(1 for a in filtered if a.metric == "quantity"),
            "sales_amount": sum(1 for a in filtered if a.metric == "sales_amount"),
        }
        rate_pct = (
            round((total_matching / eval_count) * 100.0, 2)
            if eval_count > 0
            else 0.0
        )

        min_d = min((a.date for a in filtered), default=None)
        max_d = max((a.date for a in filtered), default=None)

        summary = AnomalySummary(
            total_anomalies=total_matching,
            total_records_evaluated=eval_count,
            anomaly_rate_percent=rate_pct,
            severity_breakdown=severity_counts,
            direction_breakdown=direction_counts,
            metric_breakdown=metric_counts,
            start_date=start_date or min_d,
            end_date=end_date or max_d,
        )

        # Apply pagination
        paginated_items = filtered[skip : skip + limit]

        return AnomalyListResponse(
            items=paginated_items,
            total=total_matching,
            skip=skip,
            limit=limit,
            summary=summary,
        )
