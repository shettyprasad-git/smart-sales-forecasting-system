from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSeriesQualityReport:
    """
    Diagnostic assessment of a daily time series' continuity and data completeness.
    """
    frequency: str = "daily"
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    expected_days: int = 0
    observed_days: int = 0
    missing_days: int = 0
    coverage_ratio: float = 0.0
    duplicate_dates: int = 0
    zero_demand_days: int = 0
    missing_date_samples: list[str] = field(default_factory=list)
    longest_missing_gap: int = 0
    is_continuous: bool = False
    quality_status: str = "insufficient"  # "good", "warning", "insufficient"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary representation."""
        return asdict(self)


def assess_time_series_quality(
    df: pd.DataFrame,
    date_column: str = "Date",
    quantity_column: str = "Quantity",
) -> TimeSeriesQualityReport:
    """
    Inspect a time series dataframe and evaluate its calendar continuity,
    identifying missing dates, zero-demand days, and calendar gaps.
    """
    if df is None or df.empty:
        return TimeSeriesQualityReport(
            frequency="daily",
            min_date=None,
            max_date=None,
            expected_days=0,
            observed_days=0,
            missing_days=0,
            coverage_ratio=0.0,
            duplicate_dates=0,
            zero_demand_days=0,
            missing_date_samples=[],
            longest_missing_gap=0,
            is_continuous=False,
            quality_status="insufficient",
            warnings=["Dataset contains no observations."],
        )

    # 1. Normalize and extract dates
    if date_column in df.columns:
        dates_raw = pd.to_datetime(df[date_column], errors="coerce").dt.normalize()
    elif isinstance(df.index, pd.DatetimeIndex):
        dates_raw = pd.Series(df.index, index=df.index).dt.normalize()
    else:
        return TimeSeriesQualityReport(
            frequency="daily",
            min_date=None,
            max_date=None,
            expected_days=0,
            observed_days=0,
            missing_days=0,
            coverage_ratio=0.0,
            duplicate_dates=0,
            zero_demand_days=0,
            missing_date_samples=[],
            longest_missing_gap=0,
            is_continuous=False,
            quality_status="insufficient",
            warnings=["No valid Date column or DatetimeIndex found in time series."],
        )

    valid_mask = dates_raw.notna()
    if "is_missing" in df.columns:
        valid_mask = valid_mask & (~df["is_missing"].astype(bool))
    elif quantity_column in df.columns:
        valid_mask = valid_mask & pd.to_numeric(df[quantity_column], errors="coerce").notna()

    if not valid_mask.any():
        return TimeSeriesQualityReport(
            frequency="daily",
            min_date=None,
            max_date=None,
            expected_days=0,
            observed_days=0,
            missing_days=0,
            coverage_ratio=0.0,
            duplicate_dates=0,
            zero_demand_days=0,
            missing_date_samples=[],
            longest_missing_gap=0,
            is_continuous=False,
            quality_status="insufficient",
            warnings=["All Date values are invalid or null."],
        )

    clean_dates = dates_raw[valid_mask]
    duplicate_dates = int(clean_dates.duplicated().sum())

    # Count zero-demand observations where date is valid
    zero_demand_days = 0
    if quantity_column in df.columns:
        qty_series = pd.to_numeric(df.loc[valid_mask, quantity_column], errors="coerce")
        # Explicit observed zero demand (not NaN)
        zero_demand_days = int((qty_series == 0.0).sum())

    unique_dates = clean_dates.drop_duplicates().sort_values().reset_index(drop=True)
    observed_days = len(unique_dates)

    if observed_days < 2:
        single_date_str = unique_dates.iloc[0].strftime("%Y-%m-%d")
        return TimeSeriesQualityReport(
            frequency="daily",
            min_date=single_date_str,
            max_date=single_date_str,
            expected_days=1,
            observed_days=observed_days,
            missing_days=0,
            coverage_ratio=1.0,
            duplicate_dates=duplicate_dates,
            zero_demand_days=zero_demand_days,
            missing_date_samples=[],
            longest_missing_gap=0,
            is_continuous=False,
            quality_status="insufficient",
            warnings=["Dataset has fewer than 2 daily observations."],
        )

    min_date = unique_dates.min()
    max_date = unique_dates.max()
    min_str = min_date.strftime("%Y-%m-%d")
    max_str = max_date.strftime("%Y-%m-%d")

    full_calendar = pd.date_range(start=min_date, end=max_date, freq="D")
    expected_days = len(full_calendar)

    observed_set = set(unique_dates)
    missing_dates = [d for d in full_calendar if d not in observed_set]
    missing_days = len(missing_dates)

    coverage_ratio = round(observed_days / expected_days, 4) if expected_days > 0 else 0.0
    missing_date_samples = [d.strftime("%Y-%m-%d") for d in missing_dates[:10]]

    # Compute longest missing consecutive gap
    longest_gap = 0
    current_gap = 0
    for d in full_calendar:
        if d not in observed_set:
            current_gap += 1
            if current_gap > longest_gap:
                longest_gap = current_gap
        else:
            current_gap = 0

    warnings: list[str] = []
    if duplicate_dates > 0:
        warnings.append(f"{duplicate_dates} duplicate daily timestamps detected in series.")
    if missing_days > 0:
        warnings.append(
            f"{missing_days} calendar days missing between {min_str} and {max_str} "
            f"(longest gap: {longest_gap} days, coverage: {coverage_ratio:.1%})."
        )

    is_continuous = (missing_days == 0 and observed_days >= 2 and duplicate_dates == 0)

    if observed_days < 2:
        quality_status = "insufficient"
    elif missing_days == 0 and duplicate_dates == 0:
        quality_status = "good"
    elif coverage_ratio < 0.5 or longest_gap > 30:
        quality_status = "insufficient"
    else:
        quality_status = "warning"

    return TimeSeriesQualityReport(
        frequency="daily",
        min_date=min_str,
        max_date=max_str,
        expected_days=expected_days,
        observed_days=observed_days,
        missing_days=missing_days,
        coverage_ratio=coverage_ratio,
        duplicate_dates=duplicate_dates,
        zero_demand_days=zero_demand_days,
        missing_date_samples=missing_date_samples,
        longest_missing_gap=longest_gap,
        is_continuous=is_continuous,
        quality_status=quality_status,
        warnings=warnings,
    )


def build_canonical_daily_series(
    df: pd.DataFrame,
    date_column: str = "Date",
) -> tuple[pd.DataFrame, TimeSeriesQualityReport]:
    """
    Convert an observed daily or transaction dataset into a canonical daily series
    covering every single calendar day from min(Date) to max(Date).

    Key properties:
      1. Explicit daily frequency: exactly one row for every calendar day.
      2. Missing calendar dates are preserved with Quantity = NaN (and is_missing = True).
         They are NOT silently assumed to be zero demand.
      3. Explicit zero demand (dates observed with Quantity == 0) is preserved as 0.0
         (and is_missing = False).
      4. Dates are normalized to midnight and sorted ascending.
      5. Generates a comprehensive TimeSeriesQualityReport.
    """
    if df is None or df.empty:
        report = assess_time_series_quality(df, date_column=date_column)
        empty_df = pd.DataFrame(
            columns=["Date", "Quantity", "Sales_Amount", "Profit", "Promotions", "Holiday_Flag", "is_missing"]
        )
        return empty_df, report

    work_df = df.copy()

    # Normalize Date column
    if date_column in work_df.columns:
        work_df["Date"] = pd.to_datetime(work_df[date_column], errors="coerce").dt.normalize()
    elif isinstance(work_df.index, pd.DatetimeIndex):
        work_df["Date"] = pd.Series(work_df.index, index=work_df.index).dt.normalize()
    else:
        raise ValueError("Input data must contain a Date column or use a DatetimeIndex.")

    work_df = work_df.dropna(subset=["Date"])
    if work_df.empty:
        report = assess_time_series_quality(work_df)
        empty_df = pd.DataFrame(
            columns=["Date", "Quantity", "Sales_Amount", "Profit", "Promotions", "Holiday_Flag", "is_missing"]
        )
        return empty_df, report

    # Normalize column names if alternative names are used
    if "Promotions" not in work_df.columns and "Promotion" in work_df.columns:
        work_df["Promotions"] = work_df["Promotion"]
    if "Holiday_Flag" not in work_df.columns and "Is_Holiday" in work_df.columns:
        work_df["Holiday_Flag"] = work_df["Is_Holiday"]

    for col in ["Quantity", "Sales_Amount", "Profit"]:
        if col in work_df.columns:
            work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
        else:
            work_df[col] = np.nan

    for col in ["Promotions", "Holiday_Flag"]:
        if col in work_df.columns:
            work_df[col] = pd.to_numeric(work_df[col], errors="coerce").fillna(0).astype(int)
        else:
            work_df[col] = 0

    # Retain only genuine observations for daily aggregation
    if "is_missing" in work_df.columns:
        work_df = work_df[~work_df["is_missing"].astype(bool)].copy()
    else:
        work_df = work_df[work_df["Quantity"].notna()].copy()


    # Aggregate by Date (merging multiple intra-day transactions into single daily observation)
    agg_funcs: dict[str, Any] = {
        "Quantity": "sum",
        "Sales_Amount": "sum",
        "Profit": "sum",
        "Promotions": "max",
        "Holiday_Flag": "max",
    }
    daily_obs = (
        work_df.groupby("Date", as_index=False)
        .agg(agg_funcs)
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Assess quality on the aggregated observed days
    report = assess_time_series_quality(daily_obs, date_column="Date", quantity_column="Quantity")

    if report.observed_days < 1:
        empty_df = pd.DataFrame(
            columns=["Date", "Quantity", "Sales_Amount", "Profit", "Promotions", "Holiday_Flag", "is_missing"]
        )
        return empty_df, report

    min_date = pd.Timestamp(report.min_date).normalize()
    max_date = pd.Timestamp(report.max_date).normalize()

    # Construct the continuous calendar spine
    full_calendar = pd.date_range(start=min_date, end=max_date, freq="D", name="Date")
    calendar_df = pd.DataFrame({"Date": full_calendar})

    # Mark observed records
    daily_obs["is_missing"] = False

    # Outer/Left merge onto calendar spine
    canonical = pd.merge(calendar_df, daily_obs, on="Date", how="left")

    # For calendar gaps (missing observations), flag as missing without silently zero-filling Quantity
    missing_mask = canonical["is_missing"].isna()
    canonical.loc[missing_mask, "is_missing"] = True
    canonical.loc[missing_mask, "Promotions"] = 0
    canonical.loc[missing_mask, "Holiday_Flag"] = 0

    # Guarantee strict sorting, explicit types, and column layout
    canonical = canonical.sort_values("Date").reset_index(drop=True)
    canonical["Promotions"] = canonical["Promotions"].fillna(0).astype(int)
    canonical["Holiday_Flag"] = canonical["Holiday_Flag"].fillna(0).astype(int)
    canonical["is_missing"] = canonical["is_missing"].astype(bool)

    ordered_columns = [
        "Date",
        "Quantity",
        "Sales_Amount",
        "Profit",
        "Promotions",
        "Holiday_Flag",
        "is_missing",
    ]
    return canonical[ordered_columns], report
