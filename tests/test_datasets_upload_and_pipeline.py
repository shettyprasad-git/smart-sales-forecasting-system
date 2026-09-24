import io
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.database.models import DatasetUpload, Product, SalesRecord, User
from backend.app.services.dataset_runtime_service import dataset_runtime_service, NoActiveDatasetError


def register_and_login(client: TestClient, email: str, password: str = "TestPassword123!"):
    client.post("/api/auth/register", json={"email": email, "password": password})
    res = client.post("/api/auth/login", data={"username": email, "password": password})
    assert res.status_code == 200
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


SAMPLE_CSV_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_sales_upload.csv"


# --------------------------------------------------------------------------
# 1. Unauthenticated access tests
# --------------------------------------------------------------------------

def test_unauthenticated_dataset_endpoints_rejected(client: TestClient):
    assert client.get("/api/datasets/current").status_code == 401
    assert client.get("/api/datasets/history").status_code == 401
    assert client.post("/api/datasets/upload", files={"file": ("test.csv", b"a,b\n1,2")}).status_code == 401
    assert client.post("/api/datasets/1/activate").status_code == 401
    assert client.delete("/api/datasets/1").status_code == 401


# --------------------------------------------------------------------------
# 2. File and Column Validation Tests
# --------------------------------------------------------------------------

def test_upload_non_csv_rejected(client: TestClient):
    headers = register_and_login(client, "test_file_fmt@example.com")
    files = {"file": ("data.txt", b"some,text,data", "text/plain")}
    res = client.post("/api/datasets/upload", files=files, headers=headers)
    assert res.status_code == 400
    assert "Only standard CSV files (.csv) are accepted" in res.json()["detail"]


def test_upload_size_limit_exceeded(client: TestClient, monkeypatch):
    headers = register_and_login(client, "test_file_size@example.com")
    monkeypatch.setattr(settings, "max_dataset_upload_bytes", 50)
    large_content = b"date,product_id,quantity,unit_price\n" + b"2020-01-01,P1,10,10.0\n" * 10
    files = {"file": ("large.csv", large_content, "text/csv")}
    res = client.post("/api/datasets/upload", files=files, headers=headers)
    assert res.status_code == 413
    assert "exceeds maximum allowed size" in res.json()["detail"]


def test_upload_missing_required_columns(client: TestClient):
    headers = register_and_login(client, "test_missing_cols@example.com")
    # Missing unit_price
    content = b"date,product_id,quantity\n2020-01-01,P1,10\n"
    files = {"file": ("invalid_cols.csv", content, "text/csv")}
    res = client.post("/api/datasets/upload", files=files, headers=headers)
    assert res.status_code == 400
    assert "Missing required logical column" in str(res.json()["detail"])
    assert "unit_price" in str(res.json()["detail"])


def test_upload_invalid_row_values(client: TestClient):
    headers = register_and_login(client, "test_invalid_rows@example.com")

    # Invalid date
    bad_date = b"date,product_id,product_name,category,quantity,unit_price\nnot-a-date,P1,Widget,Electronics,10,5.0\n"
    res = client.post("/api/datasets/upload", files={"file": ("bad_date.csv", bad_date, "text/csv")}, headers=headers)
    assert res.status_code == 400
    assert "invalid date" in str(res.json()["detail"]).lower()

    # Non-positive quantity
    bad_qty = b"date,product_id,product_name,category,quantity,unit_price\n2020-01-01,P1,Widget,Electronics,-2,5.0\n"
    res = client.post("/api/datasets/upload", files={"file": ("bad_qty.csv", bad_qty, "text/csv")}, headers=headers)
    assert res.status_code == 400
    assert "quantity must be positive" in str(res.json()["detail"]).lower()

    # Negative price
    bad_price = b"date,product_id,product_name,category,quantity,unit_price\n2020-01-01,P1,Widget,Electronics,5,-10.0\n"
    res = client.post("/api/datasets/upload", files={"file": ("bad_price.csv", bad_price, "text/csv")}, headers=headers)
    assert res.status_code == 400
    assert "unit_price must be non-negative" in str(res.json()["detail"]).lower()

    # Invalid discount (>100)
    bad_disc = b"date,product_id,product_name,category,quantity,unit_price,discount_percent\n2020-01-01,P1,Widget,Electronics,5,10.0,150.0\n"
    res = client.post("/api/datasets/upload", files={"file": ("bad_disc.csv", bad_disc, "text/csv")}, headers=headers)
    assert res.status_code == 400
    assert "discount_percent must be between 0 and 100" in str(res.json()["detail"]).lower()


# --------------------------------------------------------------------------
# 3. Atomic Rollback on Failure
# --------------------------------------------------------------------------

def test_atomic_rollback_on_row_failure(client: TestClient, db_session):
    headers = register_and_login(client, "test_rollback@example.com")

    # 10 good rows followed by 1 bad row
    lines = ["date,product_id,product_name,category,quantity,unit_price"]
    for i in range(10):
        lines.append(f"2020-01-{i+1:02d},P1,Widget,Electronics,5,10.0")
    lines.append("2020-01-11,P1,Widget,Electronics,-99,10.0")  # Bad row

    content = "\n".join(lines).encode("utf-8")
    res = client.post("/api/datasets/upload", files={"file": ("rollback.csv", content, "text/csv")}, headers=headers)
    assert res.status_code == 400

    # Verify zero records committed
    user = db_session.query(User).filter(User.email == "test_rollback@example.com").first()
    sales = db_session.query(SalesRecord).filter(SalesRecord.user_id == user.id).all()
    datasets = db_session.query(DatasetUpload).filter(DatasetUpload.user_id == user.id).all()

    assert len(sales) == 0
    assert len(datasets) == 0


# --------------------------------------------------------------------------
# 4. Successful Upload, History, and Lifecycle Management
# --------------------------------------------------------------------------

def test_upload_success_and_lifecycle(client: TestClient, db_session):
    headers = register_and_login(client, "test_lifecycle@example.com")

    with open(SAMPLE_CSV_PATH, "rb") as f:
        file_content = f.read()

    # Upload Dataset 1
    res1 = client.post("/api/datasets/upload", files={"file": ("dataset_1.csv", file_content, "text/csv")}, headers=headers)
    assert res1.status_code == 201
    d1 = res1.json()
    assert d1["filename"] == "dataset_1.csv"
    assert d1["status"] == "active"
    assert d1["rows_imported"] == 180
    assert d1["products"] == 3
    assert d1["categories"] == 2
    assert d1["start_date"] == "2018-02-01"
    assert d1["end_date"] == "2018-04-01"
    d1_id = d1["dataset_id"]

    # Check /api/datasets/current
    cur_res = client.get("/api/datasets/current", headers=headers)
    assert cur_res.status_code == 200
    assert cur_res.json()["dataset_id"] == d1_id
    assert cur_res.json()["status"] == "active"

    # Upload Dataset 2
    res2 = client.post("/api/datasets/upload", files={"file": ("dataset_2.csv", file_content, "text/csv")}, headers=headers)
    assert res2.status_code == 201
    d2 = res2.json()
    assert d2["status"] == "active"
    d2_id = d2["dataset_id"]

    # History should contain both; d2 is active, d1 is archived
    hist_res = client.get("/api/datasets/history", headers=headers)
    assert hist_res.status_code == 200
    items = hist_res.json()["datasets"]
    assert len(items) == 2

    d2_item = next(x for x in items if x["id"] == d2_id)
    d1_item = next(x for x in items if x["id"] == d1_id)
    assert d2_item["status"] == "active"
    assert d1_item["status"] == "archived"

    # Activate Dataset 1 again
    act_res = client.post(f"/api/datasets/{d1_id}/activate", headers=headers)
    assert act_res.status_code == 200

    cur_res2 = client.get("/api/datasets/current", headers=headers)
    assert cur_res2.status_code == 200
    assert cur_res2.json()["dataset_id"] == d1_id

    # Delete Dataset 2
    del_res = client.delete(f"/api/datasets/{d2_id}", headers=headers)
    assert del_res.status_code == 200

    hist_res2 = client.get("/api/datasets/history", headers=headers)
    assert len(hist_res2.json()["datasets"]) == 1
    assert hist_res2.json()["datasets"][0]["id"] == d1_id


# --------------------------------------------------------------------------
# 5. Multi-Tenant User Isolation & Conflict Resolution
# --------------------------------------------------------------------------

def test_multi_tenant_isolation_and_product_scoping(client: TestClient, db_session):
    headers_a = register_and_login(client, "user_a@example.com")
    headers_b = register_and_login(client, "user_b@example.com")

    # User A uploads product 'PROD-CONFLICT' with name 'Alpha Camera'
    csv_a = (
        "date,product_id,product_name,category,quantity,unit_price\n"
        "2020-01-01,PROD-CONFLICT,Alpha Camera,Electronics,5,100.0\n"
        "2020-01-02,PROD-CONFLICT,Alpha Camera,Electronics,5,100.0\n"
    ).encode("utf-8")
    res_a = client.post("/api/datasets/upload", files={"file": ("data_a.csv", csv_a, "text/csv")}, headers=headers_a)
    assert res_a.status_code == 201
    dataset_a_id = res_a.json()["dataset_id"]

    # User B uploads product 'PROD-CONFLICT' with conflicting name 'Beta Shoes' & category 'Footwear'
    csv_b = (
        "date,product_id,product_name,category,quantity,unit_price\n"
        "2020-01-01,PROD-CONFLICT,Beta Shoes,Footwear,2,80.0\n"
        "2020-01-02,PROD-CONFLICT,Beta Shoes,Footwear,2,80.0\n"
    ).encode("utf-8")
    res_b = client.post("/api/datasets/upload", files={"file": ("data_b.csv", csv_b, "text/csv")}, headers=headers_b)
    assert res_b.status_code == 201
    dataset_b_id = res_b.json()["dataset_id"]

    # User B cannot activate or delete User A's dataset
    assert client.post(f"/api/datasets/{dataset_a_id}/activate", headers=headers_b).status_code == 404
    assert client.delete(f"/api/datasets/{dataset_a_id}", headers=headers_b).status_code == 404

    # Verify query isolation at runtime level
    user_a = db_session.query(User).filter(User.email == "user_a@example.com").first()
    user_b = db_session.query(User).filter(User.email == "user_b@example.com").first()

    prod_df_a = dataset_runtime_service.get_product_daily(user_a.id, db_session)
    prod_df_b = dataset_runtime_service.get_product_daily(user_b.id, db_session)

    # Product ID is stripped clean of any internal prefix
    assert (prod_df_a["Product_ID"] == "PROD-CONFLICT").all()
    assert (prod_df_b["Product_ID"] == "PROD-CONFLICT").all()

    # Product names remain isolated
    assert (prod_df_a["Product_Name"] == "Alpha Camera").all()
    assert (prod_df_b["Product_Name"] == "Beta Shoes").all()


# --------------------------------------------------------------------------
# 6. Legacy Sales Preservation
# --------------------------------------------------------------------------

def test_legacy_sales_records_preserved(client: TestClient, db_session):
    headers = register_and_login(client, "test_legacy@example.com")
    user = db_session.query(User).filter(User.email == "test_legacy@example.com").first()

    # Create a product and legacy sale with dataset_id = NULL
    prod = Product(product_id="LEGACY-P01", product_name="Legacy Product", category_id="C1", category_name="Cat1", unit_price=50.0)
    db_session.add(prod)
    db_session.flush()

    from datetime import date
    legacy_sale = SalesRecord(
        user_id=user.id,
        product_id=prod.id,
        sale_date=date(2019, 1, 1),
        quantity=3,
        unit_price=50.0,
        discount_percent=0.0,
        sales_amount=150.0,
        profit=50.0,
        promotion=False,
        holiday_flag=False,
        dataset_id=None,  # Legacy row
    )
    db_session.add(legacy_sale)
    db_session.commit()
    legacy_id = legacy_sale.id

    # User uploads a dataset
    csv_data = (
        "date,product_id,product_name,category,quantity,unit_price\n"
        "2020-01-01,P9,Product 9,Category 9,1,10.0\n"
    ).encode("utf-8")
    res = client.post("/api/datasets/upload", files={"file": ("upload.csv", csv_data, "text/csv")}, headers=headers)
    assert res.status_code == 201
    dataset_id = res.json()["dataset_id"]

    # Verify legacy sale is intact
    db_session.expire_all()
    check_legacy = db_session.query(SalesRecord).filter(SalesRecord.id == legacy_id).first()
    assert check_legacy is not None
    assert check_legacy.dataset_id is None

    # Delete the uploaded dataset
    client.delete(f"/api/datasets/{dataset_id}", headers=headers)

    # Legacy sale is still intact!
    check_legacy2 = db_session.query(SalesRecord).filter(SalesRecord.id == legacy_id).first()
    assert check_legacy2 is not None


# --------------------------------------------------------------------------
# 7. Downstream Forecast Integration & Production Mode Fallback
# --------------------------------------------------------------------------

def test_forecast_integration_with_uploaded_dataset(client: TestClient):
    headers = register_and_login(client, "test_forecast_user@example.com")

    with open(SAMPLE_CSV_PATH, "rb") as f:
        file_content = f.read()

    res = client.post("/api/datasets/upload", files={"file": ("forecast_data.csv", file_content, "text/csv")}, headers=headers)
    assert res.status_code == 201

    # Request forecast 7, 30, 90 days
    for h in (7, 30, 90):
        fc_res = client.post("/api/forecast", json={"horizon": h}, headers=headers)
        assert fc_res.status_code == 200
        data = fc_res.json()
        assert data["horizon"] == h
        assert len(data["forecast"]) == h

    # Dashboard endpoint
    dash_res = client.get("/api/forecast/dashboard?horizon=7", headers=headers)
    assert dash_res.status_code == 200
    dash_data = dash_res.json()
    assert len(dash_data["forecast"]) == 7
    assert len(dash_data["historical"]) > 0


def test_production_mode_404_when_no_active_dataset(client: TestClient, monkeypatch):
    headers = register_and_login(client, "test_prod_no_ds@example.com")
    monkeypatch.setattr(settings, "environment", "production")

    # In production without an active dataset, forecast should return 404
    fc_res = client.post("/api/forecast", json={"horizon": 7}, headers=headers)
    assert fc_res.status_code == 404
    assert "No active dataset found" in fc_res.json()["detail"]

    # History should also return 404
    hist_res = client.get("/api/forecast/history", headers=headers)
    assert hist_res.status_code == 404

    # Dashboard should also return 404
    dash_res = client.get("/api/forecast/dashboard", headers=headers)
    assert dash_res.status_code == 404


# --------------------------------------------------------------------------
# 8. Hardening: Single Active Dataset DB Constraint & Concurrency Invariant
# --------------------------------------------------------------------------

def test_database_enforces_single_active_dataset_invariant(client: TestClient, db_session):
    from sqlalchemy.exc import IntegrityError
    from datetime import datetime, timezone

    headers = register_and_login(client, "test_idx_invariant@example.com")
    user = db_session.query(User).filter(User.email == "test_idx_invariant@example.com").first()

    now = datetime.now(timezone.utc)
    ds1 = DatasetUpload(
        id="ds-test-inv-1",
        user_id=user.id,
        original_filename="inv1.csv",
        dataset_key=f"inv1_{user.id}",
        status="active",
        created_at=now,
        activated_at=now,
    )
    db_session.add(ds1)
    db_session.commit()

    # Attempting to insert another active dataset directly for the same user must violate the partial unique index
    ds2 = DatasetUpload(
        id="ds-test-inv-2",
        user_id=user.id,
        original_filename="inv2.csv",
        dataset_key=f"inv2_{user.id}",
        status="active",
        created_at=now,
        activated_at=now,
    )
    db_session.add(ds2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Archiving ds1 and setting ds2 to active succeeds
    ds1_record = db_session.query(DatasetUpload).filter(DatasetUpload.id == "ds-test-inv-1").first()
    ds1_record.status = "archived"
    db_session.commit()

    db_session.add(ds2)
    db_session.commit()

    # Exactly one active dataset exists
    active_count = db_session.query(DatasetUpload).filter(
        DatasetUpload.user_id == user.id,
        DatasetUpload.status == "active",
    ).count()
    assert active_count == 1


# --------------------------------------------------------------------------
# 9. Hardening: /api/products Tenant Scoping and Cross-Tenant Metadata Safety
# --------------------------------------------------------------------------

def test_products_api_tenant_safety(client: TestClient):
    # Register User A and User B
    headers_a = register_and_login(client, "user_a_product_safety@example.com")
    headers_b = register_and_login(client, "user_b_product_safety@example.com")

    # User A uploads dataset containing proprietary product "PROD-SECRET-A"
    csv_a = (
        "date,product_id,product_name,category,quantity,unit_price\n"
        "2023-01-01,PROD-SECRET-A,Confidential AI Widget,SecretCategoryA,5,999.0\n"
    ).encode("utf-8")
    res_a = client.post("/api/datasets/upload", files={"file": ("sales_a.csv", csv_a, "text/csv")}, headers=headers_a)
    assert res_a.status_code == 201

    # User B uploads dataset containing product "PROD-BETA-B"
    csv_b = (
        "date,product_id,product_name,category,quantity,unit_price\n"
        "2023-01-01,PROD-BETA-B,Standard Beta Product,PublicCategoryB,2,50.0\n"
    ).encode("utf-8")
    res_b = client.post("/api/datasets/upload", files={"file": ("sales_b.csv", csv_b, "text/csv")}, headers=headers_b)
    assert res_b.status_code == 201

    # 1. Unauthenticated request to /api/products must NEVER include User A's or User B's product
    unauth_list = client.get("/api/products")
    assert unauth_list.status_code == 200
    unauth_ids = [p["product_id"] for p in unauth_list.json()]
    assert "PROD-SECRET-A" not in unauth_ids
    assert "PROD-BETA-B" not in unauth_ids

    # Unauthenticated lookup for PROD-SECRET-A returns 404
    assert client.get("/api/products/PROD-SECRET-A").status_code == 404

    # 2. User B querying /api/products must NOT see User A's confidential product
    user_b_list = client.get("/api/products", headers=headers_b)
    assert user_b_list.status_code == 200
    user_b_ids = [p["product_id"] for p in user_b_list.json()]
    assert "PROD-SECRET-A" not in user_b_ids
    assert "PROD-BETA-B" in user_b_ids

    # 3. User B direct lookup/update/delete on User A's product returns 404
    assert client.get("/api/products/PROD-SECRET-A", headers=headers_b).status_code == 404
    assert client.put("/api/products/PROD-SECRET-A", json={"unit_price": 1.0}, headers=headers_b).status_code == 404
    assert client.delete("/api/products/PROD-SECRET-A", headers=headers_b).status_code == 404

    # 4. User A querying /api/products sees their own clean raw product_id
    user_a_list = client.get("/api/products", headers=headers_a)
    assert user_a_list.status_code == 200
    user_a_prods = {p["product_id"]: p for p in user_a_list.json()}
    assert "PROD-SECRET-A" in user_a_prods
    assert user_a_prods["PROD-SECRET-A"]["product_name"] == "Confidential AI Widget"
    assert user_a_prods["PROD-SECRET-A"]["unit_price"] == 999.0

    # User A direct lookup on PROD-SECRET-A succeeds
    user_a_get = client.get("/api/products/PROD-SECRET-A", headers=headers_a)
    assert user_a_get.status_code == 200
    assert user_a_get.json()["product_id"] == "PROD-SECRET-A"


# --------------------------------------------------------------------------
# 10. Hardening: Long Product ID Collision Resistance and Raw Preservation
# --------------------------------------------------------------------------

def test_long_product_id_collision_resistance_and_raw_preservation(client: TestClient, db_session):
    headers = register_and_login(client, "test_collision_hardening@example.com")
    user = db_session.query(User).filter(User.email == "test_collision_hardening@example.com").first()

    # Two distinct product IDs sharing the first 80 characters
    prefix = "SKU_GLOBAL_MANUFACTURING_SERIAL_IDENTIFIER_PART_BATCH_2026_TEST_VERY_LONG_STRING_"
    raw_pid_1 = prefix + "ALPHA_EXT_001"
    raw_pid_2 = prefix + "BETA_EXT_002"
    assert len(raw_pid_1) > 80
    assert len(raw_pid_2) > 80
    assert raw_pid_1 != raw_pid_2

    csv_content = (
        f"date,product_id,product_name,category,quantity,unit_price\n"
        f"2023-01-01,{raw_pid_1},Alpha Long Product,Hardware,10,120.0\n"
        f"2023-01-02,{raw_pid_2},Beta Long Product,Hardware,15,150.0\n"
    ).encode("utf-8")

    res = client.post("/api/datasets/upload", files={"file": ("long_skus.csv", csv_content, "text/csv")}, headers=headers)
    assert res.status_code == 201
    assert res.json()["products"] == 2

    # Query Product table directly to verify distinct, bounded product_ids and exact raw_product_ids
    products = db_session.query(Product).filter(Product.tenant_id == user.id).all()
    prod_map = {p.raw_product_id: p for p in products if p.raw_product_id in (raw_pid_1, raw_pid_2)}

    assert len(prod_map) == 2
    assert prod_map[raw_pid_1].product_id != prod_map[raw_pid_2].product_id
    assert len(prod_map[raw_pid_1].product_id) <= 100
    assert len(prod_map[raw_pid_2].product_id) <= 100

    # Verify runtime DataFrame contains exact untruncated raw_product_id strings
    df_product = dataset_runtime_service.get_product_daily(user_id=user.id, db=db_session)
    unique_df_pids = set(df_product["Product_ID"].unique())
    assert raw_pid_1 in unique_df_pids
    assert raw_pid_2 in unique_df_pids


# --------------------------------------------------------------------------
# 11. Hardening: Model Inference Transparency
# --------------------------------------------------------------------------

def test_model_inference_transparency(client: TestClient):
    headers = register_and_login(client, "test_transparency@example.com")
    csv_data = (
        "date,product_id,product_name,category,quantity,unit_price\n"
        "2023-01-01,P_TRANS,Transparency Product,General,5,10.0\n"
    ).encode("utf-8")

    upload_res = client.post("/api/datasets/upload", files={"file": ("trans.csv", csv_data, "text/csv")}, headers=headers)
    assert upload_res.status_code == 201
    data = upload_res.json()
    assert data["model_inference_mode"] == "runtime_inference_pretrained_models"

    curr_res = client.get("/api/datasets/current", headers=headers)
    assert curr_res.status_code == 200
    assert curr_res.json()["model_inference_mode"] == "runtime_inference_pretrained_models"

    hist_res = client.get("/api/datasets/history", headers=headers)
    assert hist_res.status_code == 200
    assert hist_res.json()["datasets"][0]["model_inference_mode"] == "runtime_inference_pretrained_models"


# --------------------------------------------------------------------------
# 12. Regression: Legacy Product Rows Migration & Global Compatibility
# --------------------------------------------------------------------------

def test_legacy_product_rows_without_tenant_id(client: TestClient, db_session):
    from datetime import datetime, timezone

    # Create a pre-migration product where tenant_id and raw_product_id are NULL
    legacy_prod = Product(
        product_id="LEGACY-GLOBAL-PROD-01",
        raw_product_id=None,
        tenant_id=None,
        product_name="Legacy Global Widget",
        category_id="LEGACY-CAT",
        category_name="Legacy Category",
        unit_price=29.99,
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(legacy_prod)
    db_session.commit()

    # 1. Unauthenticated client can list and find legacy global product
    unauth_list = client.get("/api/products")
    assert unauth_list.status_code == 200
    prod_ids = [p["product_id"] for p in unauth_list.json()]
    assert "LEGACY-GLOBAL-PROD-01" in prod_ids

    # Unauthenticated client can get legacy product by ID
    unauth_get = client.get("/api/products/LEGACY-GLOBAL-PROD-01")
    assert unauth_get.status_code == 200
    data = unauth_get.json()
    assert data["product_id"] == "LEGACY-GLOBAL-PROD-01"
    assert data["product_name"] == "Legacy Global Widget"
    assert data["unit_price"] == 29.99

    # 2. Authenticated user can also access legacy global product as shared catalog
    headers = register_and_login(client, "test_legacy_user@example.com")
    auth_list = client.get("/api/products", headers=headers)
    assert auth_list.status_code == 200
    auth_prod_ids = [p["product_id"] for p in auth_list.json()]
    assert "LEGACY-GLOBAL-PROD-01" in auth_prod_ids

    auth_get = client.get("/api/products/LEGACY-GLOBAL-PROD-01", headers=headers)
    assert auth_get.status_code == 200
    assert auth_get.json()["product_id"] == "LEGACY-GLOBAL-PROD-01"


