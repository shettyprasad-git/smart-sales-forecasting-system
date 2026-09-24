from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.app.schemas.anomalies import AnomalyItem
from backend.app.schemas.investigations import (
    ConfidenceLevel,
    DriverDirection,
    EstimatedImpact,
    InvestigationDriver,
    InvestigationResponse,
)
from backend.app.services.anomaly_service import AnomalyDetectionService

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRODUCT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "synthetic_product_daily_forecasting.csv"
)


from backend.app.services.dataset_runtime_service import (
    dataset_runtime_service,
)


class InvestigationService:
    """
    Production Root-Cause Attribution service for investigating sales anomalies.
    Decomposes observed deviations into contributing dimensions (promotions, holidays,
    categories, products, pricing, discounts, and pre-existing trend/drift) using
    strict causal historical baselines (zero lookahead leakage).
    Integrates with DatasetRuntimeService for user-isolated datasets.
    """

    def __init__(
        self, anomaly_service: AnomalyDetectionService | None = None
    ) -> None:
        self.anomaly_service = anomaly_service or AnomalyDetectionService()
        self._product_df: pd.DataFrame | None = None

    def _load_product_data(self, user_id: int | None = None, db: Any = None) -> pd.DataFrame:
        """Load the product-level daily sales dataset for user_id."""
        return dataset_runtime_service.get_product_daily(user_id=user_id, db=db)

    def find_anomaly_by_id(
        self,
        anomaly_id: str,
        user_id: int | None = None,
        db: Any = None,
    ) -> AnomalyItem | None:
        """
        Locate an existing anomaly record by ID.
        Uses ID structure: anom-YYYYMMDD-met-ent[-id]
        """
        parts = anomaly_id.split("-")
        if len(parts) < 4 or parts[0] != "anom":
            return None

        date_str = parts[1]
        met_str = parts[2]
        ent_str = parts[3]
        ent_id = parts[4] if len(parts) > 4 else None

        try:
            target_date = dt.datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            return None

        metric_map = {"qua": "quantity", "sal": "sales_amount"}
        entity_map = {"agg": "aggregate", "cat": "category", "pro": "product"}

        metric = metric_map.get(met_str)
        entity_type = entity_map.get(ent_str)

        if not metric or not entity_type:
            return None

        anomalies_resp = self.anomaly_service.get_anomalies(
            metric=metric,
            entity_type=entity_type,
            entity_id=ent_id,
            start_date=target_date,
            end_date=target_date,
            limit=50,
            user_id=user_id,
            db=db,
        )

        for item in anomalies_resp.items:
            if item.id == anomaly_id:
                return item

        return None

    def investigate_anomaly(
        self,
        anomaly_id: str,
        top_n: int = 5,
        include_products: bool = True,
        include_categories: bool = True,
        window: int = 28,
        user_id: int | None = None,
        db: Any = None,
    ) -> InvestigationResponse:
        """
        Execute root-cause attribution and impact quantification for an anomaly.
        Guarantees strict causal invariance: all historical reference stats use data strictly prior to date T.
        """
        anomaly = self.find_anomaly_by_id(anomaly_id, user_id=user_id, db=db)
        if anomaly is None:
            raise KeyError(f"Anomaly not found with ID: {anomaly_id}")

        target_date = pd.Timestamp(anomaly.date)
        window_start = target_date - pd.Timedelta(days=window)
        history_end = target_date - pd.Timedelta(days=1)

        daily_df = self.anomaly_service._load_daily_data(user_id=user_id, db=db)
        product_df = self._load_product_data(user_id=user_id, db=db)

        # Prior historical slice (STRICTLY CAUSAL: Date < target_date)
        daily_prior = daily_df[
            (daily_df["Date"] >= window_start) & (daily_df["Date"] <= history_end)
        ]
        daily_all_prior = daily_df[daily_df["Date"] < target_date]
        daily_target = daily_df[daily_df["Date"] == target_date]

        prod_prior = product_df[
            (product_df["Date"] >= window_start) & (product_df["Date"] <= history_end)
        ]
        prod_target = product_df[product_df["Date"] == target_date]

        # Filter by entity if category or product
        if anomaly.entity_type == "category" and anomaly.entity_id:
            prod_prior = prod_prior[
                prod_prior["Category_ID"].astype(str) == str(anomaly.entity_id)
            ]
            prod_target = prod_target[
                prod_target["Category_ID"].astype(str) == str(anomaly.entity_id)
            ]
        elif anomaly.entity_type == "product" and anomaly.entity_id:
            prod_prior = prod_prior[
                prod_prior["Product_ID"].astype(str) == str(anomaly.entity_id)
            ]
            prod_target = prod_target[
                prod_target["Product_ID"].astype(str) == str(anomaly.entity_id)
            ]

        # -------------------------------------------------------------
        # 1. Impact Estimation & Price Basis
        # -------------------------------------------------------------
        actual = anomaly.actual_value
        expected = anomaly.expected_value
        deviation = anomaly.deviation
        dev_pct = anomaly.deviation_percent
        is_spike = anomaly.direction == "spike"

        # Baseline pricing
        if not daily_prior.empty and daily_prior["Quantity"].sum() > 0:
            baseline_avg_price = float(
                daily_prior["Sales_Amount"].sum()
                / daily_prior["Quantity"].sum()
            )
        else:
            baseline_avg_price = 3400.0

        if anomaly.metric == "quantity":
            impact_val = abs(deviation)
            est_rev_impact = impact_val * baseline_avg_price
            price_basis_expl = (
                f"Valued at 28-day baseline average realized price of "
                f"${baseline_avg_price:,.2f} per unit."
            )
            if is_spike:
                interpretation = (
                    f"Estimated excess demand of {impact_val:,.0f} units (+{dev_pct:.1f}% vs baseline), "
                    f"representing an estimated ${est_rev_impact:,.2f} in excess sales potential."
                )
            else:
                interpretation = (
                    f"Estimated demand deficit of {impact_val:,.0f} units ({dev_pct:.1f}% vs baseline), "
                    f"representing a potential revenue gap of ${est_rev_impact:,.2f}."
                )
        else:  # sales_amount
            impact_val = abs(deviation)
            est_rev_impact = impact_val
            price_basis_expl = (
                "Direct sales revenue metric calculated against 28-day rolling median revenue baseline."
            )
            if is_spike:
                interpretation = (
                    f"Estimated excess revenue of ${impact_val:,.2f} (+{dev_pct:.1f}% vs baseline)."
                )
            else:
                interpretation = (
                    f"Estimated revenue gap of ${impact_val:,.2f} ({dev_pct:.1f}% vs baseline)."
                )

        estimated_impact = EstimatedImpact(
            impact_metric=anomaly.metric,
            impact_value=round(impact_val, 2),
            signed_deviation=round(deviation, 2),
            deviation_percent=round(dev_pct, 2),
            estimated_revenue_impact=round(est_rev_impact, 2),
            price_basis_explanation=price_basis_expl,
            interpretation=interpretation,
        )

        drivers: list[InvestigationDriver] = []
        evidence_notes: list[str] = []

        # -------------------------------------------------------------
        # 2. Promotion Driver
        # -------------------------------------------------------------
        promo_active = False
        promo_count_on_day = 0
        if not prod_target.empty:
            promo_count_on_day = int((prod_target["Promotion"] > 0).sum())
            promo_active = promo_count_on_day > 0
        elif not daily_target.empty:
            promo_val = int(daily_target["Promotions"].iloc[0])
            promo_active = promo_val > 0
            promo_count_on_day = promo_val

        metric_col = "Quantity" if anomaly.metric == "quantity" else "Sales_Amount"
        promo_hist = daily_all_prior[daily_all_prior["Promotions"] > 0]
        non_promo_hist = daily_all_prior[daily_all_prior["Promotions"] == 0]

        if promo_active:
            hist_promo_avg = (
                float(promo_hist[metric_col].mean())
                if not promo_hist.empty
                else expected
            )
            hist_non_promo_avg = (
                float(non_promo_hist[metric_col].mean())
                if not non_promo_hist.empty
                else expected
            )
            promo_lift_pct = (
                ((hist_promo_avg - hist_non_promo_avg) / hist_non_promo_avg * 100)
                if hist_non_promo_avg > 0
                else 0.0
            )

            p_conf: ConfidenceLevel = (
                "high" if len(promo_hist) >= 5 else "medium" if len(promo_hist) >= 1 else "low"
            )
            p_dir: DriverDirection = "positive" if promo_lift_pct > 0 else "neutral"

            drivers.append(
                InvestigationDriver(
                    driver_type="promotion",
                    driver_name="Active Promotional Campaign",
                    direction=p_dir,
                    observed_value=float(promo_count_on_day),
                    reference_value=0.0,
                    difference=float(promo_count_on_day),
                    difference_percent=None,
                    contribution_score=min(100.0, max(0.0, round(abs(promo_lift_pct), 1))),
                    evidence=(
                        f"Active promotion was running on anomaly date ({promo_count_on_day} products discounted). "
                        f"Historically, promotion dates prior to this day averaged {hist_promo_avg:,.0f} vs "
                        f"{hist_non_promo_avg:,.0f} on standard dates (+{promo_lift_pct:.1f}% historical lift across "
                        f"{len(promo_hist)} prior events), indicating promotion is a strong associated contributor."
                    ),
                    confidence=p_conf,
                    is_event_related=True,
                )
            )
            evidence_notes.append(
                f"Promotion: Active ({promo_count_on_day} products on sale; historical promo lift: +{promo_lift_pct:.1f}%)."
            )
        else:
            drivers.append(
                InvestigationDriver(
                    driver_type="promotion",
                    driver_name="Promotion Activity",
                    direction="neutral",
                    observed_value=0.0,
                    reference_value=0.0,
                    difference=0.0,
                    difference_percent=0.0,
                    contribution_score=0.0,
                    evidence="No promotional campaigns were active on the anomaly date.",
                    confidence="high",
                    is_event_related=False,
                )
            )
            evidence_notes.append("Promotion: No campaign active on anomaly date.")

        # -------------------------------------------------------------
        # 3. Holiday Driver
        # -------------------------------------------------------------
        is_holiday = False
        if not prod_target.empty:
            is_holiday = bool(prod_target["Is_Holiday"].iloc[0] > 0)
        elif not daily_target.empty:
            is_holiday = bool(daily_target["Holiday_Flag"].iloc[0] > 0)

        hol_hist = daily_all_prior[daily_all_prior["Holiday_Flag"] > 0]
        non_hol_hist = daily_all_prior[daily_all_prior["Holiday_Flag"] == 0]
        hol_sample = len(hol_hist)

        if is_holiday:
            hol_avg = (
                float(hol_hist[metric_col].mean())
                if hol_sample > 0
                else expected
            )
            non_hol_avg = (
                float(non_hol_hist[metric_col].mean())
                if not non_hol_hist.empty
                else expected
            )
            hol_diff_pct = (
                ((hol_avg - non_hol_avg) / non_hol_avg * 100)
                if non_hol_avg > 0
                else 0.0
            )

            h_conf: ConfidenceLevel = (
                "high" if hol_sample >= 10 else "medium" if hol_sample >= 3 else "low"
            )
            h_dir: DriverDirection = (
                "positive" if hol_diff_pct > 2 else "negative" if hol_diff_pct < -2 else "neutral"
            )

            drivers.append(
                InvestigationDriver(
                    driver_type="holiday",
                    driver_name="Holiday Trading Event",
                    direction=h_dir,
                    observed_value=1.0,
                    reference_value=0.0,
                    difference=1.0,
                    difference_percent=None,
                    contribution_score=round(abs(hol_diff_pct), 1),
                    evidence=(
                        f"The anomaly date coincided with a calendar holiday. Prior historical holidays averaged "
                        f"{hol_avg:,.0f} vs {non_hol_avg:,.0f} on normal trading days ({hol_diff_pct:+.1f}% shift, "
                        f"sample: {hol_sample} prior holidays)."
                    ),
                    confidence=h_conf,
                    is_event_related=True,
                )
            )
            evidence_notes.append(
                f"Holiday: Calendar holiday active (Historical holiday shift: {hol_diff_pct:+.1f}%, N={hol_sample})."
            )
        else:
            drivers.append(
                InvestigationDriver(
                    driver_type="holiday",
                    driver_name="Holiday Trading Event",
                    direction="neutral",
                    observed_value=0.0,
                    reference_value=0.0,
                    difference=0.0,
                    difference_percent=0.0,
                    contribution_score=0.0,
                    evidence="The anomaly date was a standard non-holiday trading day.",
                    confidence="high",
                    is_event_related=False,
                )
            )

        # -------------------------------------------------------------
        # 4. Category Contribution (for Aggregate anomalies)
        # -------------------------------------------------------------
        if include_categories and anomaly.entity_type == "aggregate" and not prod_target.empty and not prod_prior.empty:
            cat_curr = prod_target.groupby(
                ["Category_ID", "Category_Name"], as_index=False
            )[["Quantity", "Sales_Amount"]].sum()

            cat_daily_prior = prod_prior.groupby(
                ["Date", "Category_ID", "Category_Name"], as_index=False
            )[["Quantity", "Sales_Amount"]].sum()

            cat_baseline = cat_daily_prior.groupby(
                ["Category_ID", "Category_Name"], as_index=False
            )[["Quantity", "Sales_Amount"]].median()

            merged_cat = pd.merge(
                cat_curr,
                cat_baseline,
                on=["Category_ID", "Category_Name"],
                suffixes=("_actual", "_baseline"),
            )

            val_col_act = f"{metric_col}_actual"
            val_col_base = f"{metric_col}_baseline"
            merged_cat["deviation"] = merged_cat[val_col_act] - merged_cat[val_col_base]
            merged_cat["abs_deviation"] = merged_cat["deviation"].abs()

            sum_abs_cat_dev = merged_cat["abs_deviation"].sum()
            merged_cat = merged_cat.sort_values("abs_deviation", ascending=False)

            top_cats = merged_cat.head(min(top_n, len(merged_cat)))
            for _, c_row in top_cats.iterrows():
                c_act = float(c_row[val_col_act])
                c_base = float(c_row[val_col_base])
                c_dev = float(c_row["deviation"])
                c_pct = (c_dev / c_base * 100) if c_base > 0 else 0.0
                c_share = (
                    (abs(c_dev) / sum_abs_cat_dev * 100)
                    if sum_abs_cat_dev > 0
                    else 0.0
                )
                c_dir: DriverDirection = "positive" if c_dev > 0 else "negative"

                unit_lbl = "units" if anomaly.metric == "quantity" else "$"
                drivers.append(
                    InvestigationDriver(
                        driver_type="category",
                        driver_name=f"Category: {c_row['Category_Name']}",
                        direction=c_dir,
                        observed_value=round(c_act, 2),
                        reference_value=round(c_base, 2),
                        difference=round(c_dev, 2),
                        difference_percent=round(c_pct, 2),
                        contribution_score=round(c_share, 1),
                        evidence=(
                            f"{c_row['Category_Name']} actual was {c_act:,.0f} {unit_lbl} vs "
                            f"28-day baseline of {c_base:,.0f} {unit_lbl} ({c_pct:+.1f}%), "
                            f"accounting for {c_share:.1f}% of total category variance."
                        ),
                        confidence="high",
                        is_event_related=False,
                    )
                )

        # -------------------------------------------------------------
        # 5. Product Contribution (Top N Products)
        # -------------------------------------------------------------
        if include_products and not prod_target.empty and not prod_prior.empty:
            prod_base = prod_prior.groupby(
                ["Product_ID", "Product_Name", "Category_Name"], as_index=False
            )[["Quantity", "Sales_Amount"]].median()

            merged_prod = pd.merge(
                prod_target,
                prod_base,
                on=["Product_ID", "Product_Name", "Category_Name"],
                suffixes=("_actual", "_baseline"),
            )

            p_act_col = f"{metric_col}_actual"
            p_base_col = f"{metric_col}_baseline"
            merged_prod["deviation"] = merged_prod[p_act_col] - merged_prod[p_base_col]
            merged_prod["abs_deviation"] = merged_prod["deviation"].abs()

            sum_abs_prod_dev = merged_prod["abs_deviation"].sum()
            merged_prod = merged_prod.sort_values("abs_deviation", ascending=False)

            top_products = merged_prod.head(min(top_n, len(merged_prod)))
            for _, p_row in top_products.iterrows():
                p_act = float(p_row[p_act_col])
                p_base = float(p_row[p_base_col])
                p_dev = float(p_row["deviation"])
                p_pct = (p_dev / p_base * 100) if p_base > 0 else 0.0
                p_share = (
                    (abs(p_dev) / sum_abs_prod_dev * 100)
                    if sum_abs_prod_dev > 0
                    else 0.0
                )
                p_dir: DriverDirection = "positive" if p_dev > 0 else "negative"

                unit_lbl = "units" if anomaly.metric == "quantity" else "$"
                drivers.append(
                    InvestigationDriver(
                        driver_type="product",
                        driver_name=f"Product: {p_row['Product_Name']} ({p_row['Category_Name']})",
                        direction=p_dir,
                        observed_value=round(p_act, 2),
                        reference_value=round(p_base, 2),
                        difference=round(p_dev, 2),
                        difference_percent=round(p_pct, 2),
                        contribution_score=round(p_share, 1),
                        evidence=(
                            f"{p_row['Product_Name']} had actual of {p_act:,.0f} {unit_lbl} vs "
                            f"28-day baseline {p_base:,.0f} {unit_lbl} ({p_pct:+.1f}%), "
                            f"representing {p_share:.1f}% of catalog deviation."
                        ),
                        confidence="high",
                        is_event_related=False,
                    )
                )

        # -------------------------------------------------------------
        # 6. Price Shift Driver
        # -------------------------------------------------------------
        if not prod_target.empty and not prod_prior.empty:
            obs_unit_price = float(prod_target["Unit_Price"].mean())
            ref_unit_price = float(prod_prior["Unit_Price"].median())
            price_diff = obs_unit_price - ref_unit_price
            price_diff_pct = (
                (price_diff / ref_unit_price * 100)
                if ref_unit_price > 0
                else 0.0
            )

            p_conf: ConfidenceLevel = (
                "high" if abs(price_diff_pct) >= 5 else "medium" if abs(price_diff_pct) >= 2 else "low"
            )
            p_dir: DriverDirection = (
                "positive" if price_diff > 0 else "negative" if price_diff < 0 else "neutral"
            )

            drivers.append(
                InvestigationDriver(
                    driver_type="price",
                    driver_name="Average Unit Price Realization",
                    direction=p_dir,
                    observed_value=round(obs_unit_price, 2),
                    reference_value=round(ref_unit_price, 2),
                    difference=round(price_diff, 2),
                    difference_percent=round(price_diff_pct, 2),
                    contribution_score=round(min(100.0, abs(price_diff_pct)), 1),
                    evidence=(
                        f"Average unit price on anomaly date was ${obs_unit_price:,.2f} vs "
                        f"28-day historical reference of ${ref_unit_price:,.2f} ({price_diff_pct:+.1f}% shift)."
                    ),
                    confidence=p_conf,
                    is_event_related=False,
                )
            )

        # -------------------------------------------------------------
        # 7. Discount Rate Driver
        # -------------------------------------------------------------
        if not prod_target.empty and not prod_prior.empty:
            obs_discount = float(prod_target["Discount_Percent"].mean())
            ref_discount = float(prod_prior["Discount_Percent"].median())
            disc_diff = obs_discount - ref_discount

            d_conf: ConfidenceLevel = "high" if abs(disc_diff) >= 3 else "medium"
            d_dir: DriverDirection = (
                "positive" if disc_diff > 0 else "negative" if disc_diff < 0 else "neutral"
            )

            drivers.append(
                InvestigationDriver(
                    driver_type="discount",
                    driver_name="Discount Rate Shift",
                    direction=d_dir,
                    observed_value=round(obs_discount, 2),
                    reference_value=round(ref_discount, 2),
                    difference=round(disc_diff, 2),
                    difference_percent=None,
                    contribution_score=round(min(100.0, abs(disc_diff) * 2.5), 1),
                    evidence=(
                        f"Average discount rate on anomaly date was {obs_discount:.1f}% vs "
                        f"28-day reference of {ref_discount:.1f}% ({disc_diff:+.1f} percentage points)."
                    ),
                    confidence=d_conf,
                    is_event_related=False,
                )
            )

        # -------------------------------------------------------------
        # 8. Recent Trend / Baseline Drift
        # -------------------------------------------------------------
        recent_7 = daily_df[
            (daily_df["Date"] >= target_date - pd.Timedelta(days=7))
            & (daily_df["Date"] < target_date)
        ]
        if not recent_7.empty:
            prior_7_mean = float(recent_7[metric_col].mean())
            drift_val = prior_7_mean - expected
            drift_pct = (drift_val / expected * 100) if expected > 0 else 0.0

            t_conf: ConfidenceLevel = "high" if abs(drift_pct) >= 4 else "medium"
            t_dir: DriverDirection = (
                "positive" if drift_pct > 2 else "negative" if drift_pct < -2 else "neutral"
            )

            trend_phrase = (
                "preceded by strong upward momentum"
                if drift_pct > 3
                else "preceded by a downward softening trend"
                if drift_pct < -3
                else "preceded by relatively stable demand"
            )

            drivers.append(
                InvestigationDriver(
                    driver_type="recent_trend",
                    driver_name="Pre-Anomaly Trend Drift",
                    direction=t_dir,
                    observed_value=round(prior_7_mean, 2),
                    reference_value=round(expected, 2),
                    difference=round(drift_val, 2),
                    difference_percent=round(drift_pct, 2),
                    contribution_score=round(min(100.0, abs(drift_pct)), 1),
                    evidence=(
                        f"Prior 7-day average was {prior_7_mean:,.0f} vs 28-day baseline of "
                        f"{expected:,.0f} ({drift_pct:+.1f}% drift). The anomaly was {trend_phrase}."
                    ),
                    confidence=t_conf,
                    is_event_related=False,
                )
            )
            evidence_notes.append(
                f"Trend Drift: 7-day prior mean was {drift_pct:+.1f}% vs 28-day baseline ({trend_phrase})."
            )

        # Sort drivers by contribution score descending
        drivers.sort(key=lambda d: d.contribution_score, reverse=True)

        # -------------------------------------------------------------
        # 9. Deterministic Investigation Summary
        # -------------------------------------------------------------
        top_driver_names = [d.driver_name for d in drivers if d.contribution_score > 5][:3]
        top_str = f"Primary associated factors include {', '.join(top_driver_names)}." if top_driver_names else "No single dominant driver emerged."

        event_str = (
            "Coincided with promotional activity."
            if promo_active
            else "Coincided with a recognized holiday."
            if is_holiday
            else "Occurred during standard regular trading without active events."
        )

        val_label = f"{actual:,.0f} units" if anomaly.metric == "quantity" else f"${actual:,.2f}"
        base_label = f"{expected:,.0f} units" if anomaly.metric == "quantity" else f"${expected:,.2f}"

        summary = (
            f"Daily {anomaly.metric.replace('_', ' ')} of {val_label} was {dev_pct:+.1f}% vs its 28-day "
            f"baseline of {base_label} ({anomaly.severity} {anomaly.direction}). {event_str} "
            f"{top_str} All contributing dimensions were evaluated against pre-anomaly historical baselines."
        )

        limitations = [
            "Observational Attribution: Drivers reflect empirical decompositions and statistical associations; observational data cannot establish counterfactual causality.",
            "Historical Baseline Window: All reference baselines use a 28-day lookback strictly preceding the anomaly date (t < T) to prevent lookahead leakage.",
            "Monetary Translation: Revenue estimates for quantity anomalies utilize baseline realized prices without assuming price elasticity.",
        ]

        return InvestigationResponse(
            anomaly_id=anomaly.id,
            anomaly_date=anomaly.date,
            metric=anomaly.metric,
            entity_type=anomaly.entity_type,
            entity_id=anomaly.entity_id,
            entity_name=anomaly.entity_name,
            actual_value=anomaly.actual_value,
            expected_baseline=anomaly.expected_value,
            deviation=anomaly.deviation,
            deviation_percent=anomaly.deviation_percent,
            anomaly_score=anomaly.anomaly_score,
            severity=anomaly.severity,
            direction=anomaly.direction,
            estimated_impact=estimated_impact,
            impact_interpretation=interpretation,
            drivers=drivers,
            investigation_summary=summary,
            evidence_notes=evidence_notes,
            limitations=limitations,
        )
