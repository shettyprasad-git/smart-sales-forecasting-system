from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from backend.app.database.models import DatasetUpload, SalesRecord, User
from backend.app.main import app
from backend.app.services.dataset_runtime_service import dataset_runtime_service
from backend.app.services.forecast_service import BackendForecastService
from ml.data.canonical_series import (
    TimeSeriesQualityReport,
    assess_time_series_quality,
    build_canonical_daily_series,
)
from ml.training.company_trainer import (
    benchmark_and_train_horizon,
    calculate_min_required_days,
    check_data_sufficiency,
)


# ==============================================================================
# SECTION 11: TIME SERIES CONTINUITY & QUALITY EDGE CASES (A THROUGH I)
# ==============================================================================

def test_edge_case_a_continuous_daily_dataset():
    """Case A: Continuous daily dataset without any gaps."""
    dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Quantity": np.random.uniform(50, 150, size=len(dates)),
        "Sales_Amount": np.random.uniform(500, 1500, size=len(dates)),
        "Profit": np.random.uniform(50, 150, size=len(dates)),
        "Promotions": np.zeros(len(dates), dtype=int),
        "Holiday_Flag": np.zeros(len(dates), dtype=int),
    })

    canonical_df, report = build_canonical_daily_series(df)

    assert len(canonical_df) == 31
    assert report.expected_days == 31
    assert report.observed_days == 31
    assert report.missing_days == 0
    assert report.coverage_ratio == 1.0
    assert report.is_continuous is True
    assert report.quality_status == "good"
    assert report.longest_missing_gap == 0
    assert len(report.missing_date_samples) == 0
    assert not canonical_df["is_missing"].any()
    assert canonical_df["Quantity"].notna().all()


def test_edge_case_b_one_missing_date():
    """Case B: Exactly one missing date between observations."""
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-03"],
        "Quantity": [100.0, 200.0],
        "Sales_Amount": [1000.0, 2000.0],
        "Profit": [100.0, 200.0],
    })

    canonical_df, report = build_canonical_daily_series(df)

    assert len(canonical_df) == 3
    assert report.expected_days == 3
    assert report.observed_days == 2
    assert report.missing_days == 1
    assert report.coverage_ratio == pytest.approx(2 / 3, rel=1e-3)
    assert report.is_continuous is False
    assert report.longest_missing_gap == 1
    assert report.missing_date_samples == ["2024-01-02"]

    # Canonical row for missing day must have Quantity=NaN, is_missing=True
    missing_row = canonical_df.iloc[1]
    assert missing_row["Date"] == pd.Timestamp("2024-01-02")
    assert pd.isna(missing_row["Quantity"])
    assert pd.isna(missing_row["Sales_Amount"])
    assert bool(missing_row["is_missing"]) is True

    # Preserved rows
    assert bool(canonical_df.iloc[0]["is_missing"]) is False
    assert canonical_df.iloc[0]["Quantity"] == 100.0
    assert bool(canonical_df.iloc[2]["is_missing"]) is False
    assert canonical_df.iloc[2]["Quantity"] == 200.0


def test_edge_case_c_multiple_missing_dates_and_longest_gap():
    """Case C: Multiple missing dates across multiple calendar gaps."""
    # Days: Jan 1 (obs), Jan 5 (obs: gap of 3), Jan 7 (obs: gap of 1)
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-05", "2024-01-07"],
        "Quantity": [10.0, 50.0, 70.0],
    })

    canonical_df, report = build_canonical_daily_series(df)

    assert len(canonical_df) == 7
    assert report.expected_days == 7
    assert report.observed_days == 3
    assert report.missing_days == 4
    assert report.longest_missing_gap == 3
    assert report.is_continuous is False
    assert report.missing_date_samples == ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-06"]
    assert canonical_df["is_missing"].sum() == 4


def test_edge_case_d_multiple_transactions_on_same_date():
    """Case D: Multiple raw transactions on the same date are aggregated, not treated as duplicate dates."""
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02"],
        "Quantity": [10.0, 20.0, 30.0, 40.0],
        "Sales_Amount": [100.0, 200.0, 300.0, 400.0],
        "Profit": [10.0, 20.0, 30.0, 40.0],
        "Promotions": [0, 1, 0, 0],
        "Holiday_Flag": [0, 0, 0, 1],
    })

    canonical_df, report = build_canonical_daily_series(df)

    assert len(canonical_df) == 2
    assert report.duplicate_dates == 0  # Aggregated, so daily series has 0 duplicate dates
    assert report.expected_days == 2
    assert report.observed_days == 2
    assert report.missing_days == 0
    assert report.is_continuous is True

    # 2024-01-01 should be sum of 10+20+30=60
    assert canonical_df.iloc[0]["Quantity"] == 60.0
    assert canonical_df.iloc[0]["Sales_Amount"] == 600.0
    assert canonical_df.iloc[0]["Profit"] == 60.0
    assert canonical_df.iloc[0]["Promotions"] == 1
    assert canonical_df.iloc[0]["Holiday_Flag"] == 0

    # 2024-01-02
    assert canonical_df.iloc[1]["Quantity"] == 40.0
    assert canonical_df.iloc[1]["Promotions"] == 0
    assert canonical_df.iloc[1]["Holiday_Flag"] == 1


def test_edge_case_e_explicit_zero_demand_day():
    """Case E: Explicit zero-demand day is preserved as 0.0 and never treated as missing."""
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Quantity": [100.0, 0.0, 200.0],
        "Sales_Amount": [1000.0, 0.0, 2000.0],
        "Profit": [100.0, 0.0, 200.0],
    })

    canonical_df, report = build_canonical_daily_series(df)

    assert len(canonical_df) == 3
    assert report.expected_days == 3
    assert report.observed_days == 3
    assert report.missing_days == 0
    assert report.zero_demand_days == 1
    assert report.is_continuous is True
    assert report.quality_status == "good"

    # Row 1 (2024-01-02) must have explicit 0.0, NOT NaN, and is_missing=False
    zero_row = canonical_df.iloc[1]
    assert zero_row["Date"] == pd.Timestamp("2024-01-02")
    assert zero_row["Quantity"] == 0.0
    assert bool(zero_row["is_missing"]) is False


def test_edge_case_f_empty_dataset():
    """Case F: Completely empty dataset assessment."""
    df = pd.DataFrame()
    canonical_df, report = build_canonical_daily_series(df)

    assert canonical_df.empty
    assert report.expected_days == 0
    assert report.observed_days == 0
    assert report.missing_days == 0
    assert report.is_continuous is False
    assert report.quality_status == "insufficient"


def test_edge_case_g_one_day_dataset():
    """Case G: Dataset with only 1 daily observation is insufficient."""
    df = pd.DataFrame({
        "Date": ["2024-01-01"],
        "Quantity": [50.0],
    })

    canonical_df, report = build_canonical_daily_series(df)

    assert len(canonical_df) == 1
    assert report.expected_days == 1
    assert report.observed_days == 1
    assert report.missing_days == 0
    assert report.is_continuous is False
    assert report.quality_status == "insufficient"
    assert "fewer than 2" in report.warnings[0]


def test_edge_case_h_unsorted_raw_transactions():
    """Case H: Unsorted raw transactions are sorted strictly ascending."""
    df = pd.DataFrame({
        "Date": ["2024-01-03", "2024-01-01", "2024-01-02"],
        "Quantity": [30.0, 10.0, 20.0],
    })

    canonical_df, report = build_canonical_daily_series(df)

    assert len(canonical_df) == 3
    assert report.is_continuous is True
    assert canonical_df["Date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert canonical_df["Quantity"].tolist() == [10.0, 20.0, 30.0]


def test_edge_case_i_duplicate_canonical_daily_dates():
    """Case I: Duplicate dates in daily series are flagged in assessment."""
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-01", "2024-01-02"],
        "Quantity": [10.0, 20.0, 30.0],
    })

    report = assess_time_series_quality(df)

    assert report.duplicate_dates == 1
    assert report.is_continuous is False
    assert any("duplicate" in w.lower() for w in report.warnings)


# ==============================================================================
# SECTION 12: TRAINING & FORECASTING SAFETY TESTS
# ==============================================================================

def test_missing_dates_prevent_company_model_training():
    """Discontinuous dataset prevents company model training and returns clear diagnostics."""
    # Create 100 days of data, but delete day 40 and 41 to introduce a calendar gap
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Quantity": np.random.uniform(50, 100, size=100),
        "Sales_Amount": np.random.uniform(500, 1000, size=100),
        "Profit": np.random.uniform(50, 100, size=100),
        "Promotions": np.zeros(100, dtype=int),
        "Holiday_Flag": np.zeros(100, dtype=int),
    })

    # Drop 2 days
    gapped_df = df[~df["Date"].isin([pd.Timestamp("2024-02-09"), pd.Timestamp("2024-02-10")])].copy()

    result = benchmark_and_train_horizon(gapped_df, horizon=7)

    assert result["status"] == "insufficient_data"
    assert "discontinuous" in result["status_message"].lower()
    assert "2 calendar days are missing" in result["status_message"]
    assert "2024-02-09" in result["status_message"]
    assert result["artifact_bytes"] is None


def test_continuous_dataset_trains_successfully():
    """A continuous dataset with sufficient rows trains successfully."""
    # 70 continuous days (28 lag + 28 train + 7 val + 7 test = 70 needed for H=7)
    dates = pd.date_range("2024-01-01", periods=75, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Quantity": np.random.uniform(50, 100, size=75),
        "Sales_Amount": np.random.uniform(500, 1000, size=75),
        "Profit": np.random.uniform(50, 100, size=75),
        "Promotions": np.zeros(75, dtype=int),
        "Holiday_Flag": np.zeros(75, dtype=int),
    })

    result = benchmark_and_train_horizon(df, horizon=7)

    assert result["status"] == "ready"
    assert result["artifact_bytes"] is not None
    assert result["model_type"] in ["Ridge Regression", "Random Forest", "HistGradientBoosting", "Seasonal Naive"]



def test_horizon_sufficiency_checks():
    """Verify that calculate_min_required_days accurately governs sufficiency."""
    assert calculate_min_required_days(7) == 28 + 28 + 14  # 70
    assert calculate_min_required_days(30) == 28 + 28 + 60  # 116
    assert calculate_min_required_days(90) == 28 + 28 + 180  # 236

    # 80 continuous days is enough for 7D, but not for 30D or 90D
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Quantity": np.random.uniform(50, 100, size=80),
        "Sales_Amount": np.random.uniform(500, 1000, size=80),
        "Profit": np.random.uniform(50, 100, size=80),
        "Promotions": np.zeros(80, dtype=int),
        "Holiday_Flag": np.zeros(80, dtype=int),
    })

    is_suf_7, _ = check_data_sufficiency(df, 7)
    is_suf_30, msg_30 = check_data_sufficiency(df, 30)
    is_suf_90, msg_90 = check_data_sufficiency(df, 90)

    assert is_suf_7 is True
    assert is_suf_30 is False
    assert "requires at least 116 days" in msg_30
    assert is_suf_90 is False
    assert "requires at least 236 days" in msg_90


def test_forecast_service_discontinuity_raises_error_without_global_fallback():
    """
    Forecasting on a discontinuous dataset must fail fast with a descriptive ValueError
    and MUST NOT silently fall back to the global model.
    """
    service = BackendForecastService()

    # Create 50 days of data with a gap on day 25
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "Quantity": np.random.uniform(50, 100, size=50),
        "Sales_Amount": np.random.uniform(500, 1000, size=50),
        "Profit": np.random.uniform(50, 100, size=50),
        "Promotions": np.zeros(50, dtype=int),
        "Holiday_Flag": np.zeros(50, dtype=int),
    })
    # Remove day 25 (2024-01-25)
    gapped_df = df[df["Date"] != pd.Timestamp("2024-01-25")].copy()

    with patch.object(dataset_runtime_service, "get_daily_aggregate", return_value=gapped_df):
        with pytest.raises(ValueError) as excinfo:
            service.generate_forecast(horizon=7)

        err_msg = str(excinfo.value)
        assert "discontinuous" in err_msg.lower()
        assert "1 calendar days are missing" in err_msg
        assert "2024-01-25" in err_msg


# ==============================================================================
# SECTION 13: QUALITY DIAGNOSTICS REST API ENDPOINTS
# ==============================================================================

def register_and_login(client: TestClient, email: str, password: str = "TestPassword123!"):
    client.post("/api/auth/register", json={"email": email, "password": password})
    res = client.post("/api/auth/login", data={"username": email, "password": password})
    assert res.status_code == 200
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_quality_endpoints_end_to_end(db_session, client: TestClient):
    """
    Test GET /api/datasets/current/quality and GET /api/datasets/{dataset_id}/quality.
    """
    auth_headers = register_and_login(client, "quality_test_user@example.com")
    user = db_session.scalars(select(User).where(User.email == "quality_test_user@example.com")).first()


    # 1. Create a continuous dataset
    ds = DatasetUpload(
        id="ds-quality-test-1",
        user_id=user.id,
        original_filename="continuous.csv",
        dataset_key="key-qual-1",
        status="active",
        row_count=10,
        product_count=1,
        category_count=1,
        min_date=date(2024, 1, 1),
        max_date=date(2024, 1, 10),
    )
    db_session.add(ds)
    db_session.flush()

    for i in range(10):
        sr = SalesRecord(
            user_id=user.id,
            dataset_id=ds.id,
            product_id=1,
            sale_date=date(2024, 1, 1) + timedelta(days=i),
            quantity=100.0,
            unit_price=10.0,
            sales_amount=1000.0,
            profit=100.0,
            promotion=False,
            holiday_flag=False,
        )
        db_session.add(sr)
    db_session.commit()

    # Clear caches
    dataset_runtime_service.invalidate_cache(user.id)

    # 2. Test GET /api/datasets/current/quality
    resp = client.get("/api/datasets/current/quality", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["dataset_id"] == ds.id
    assert data["frequency"] == "daily"
    assert data["expected_days"] == 10
    assert data["observed_days"] == 10
    assert data["missing_days"] == 0
    assert data["coverage_ratio"] == 1.0
    assert data["is_continuous"] is True
    assert data["quality_status"] == "good"

    # 3. Test GET /api/datasets/{dataset_id}/quality
    resp2 = client.get(f"/api/datasets/{ds.id}/quality", headers=auth_headers)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["dataset_id"] == ds.id
    assert data2["is_continuous"] is True

    # 4. Test non-existent dataset returns 404
    resp_404 = client.get("/api/datasets/ds-nonexistent/quality", headers=auth_headers)
    assert resp_404.status_code == 404
