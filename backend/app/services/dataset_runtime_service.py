from __future__ import annotations

import csv
from datetime import date, datetime, timezone
import hashlib
import io
import logging
from pathlib import Path
import re
import threading
import time
from typing import Any
import uuid

import numpy as np
import pandas as pd
from sqlalchemy import case, delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.models import DatasetUpload, Product, SalesRecord, User
from backend.app.schemas.datasets import DatasetSummaryResponse

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FALLBACK_DAILY_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed_daily_forecasting_features.csv"
)
FALLBACK_DAILY_SYNTHETIC_PATH = (
    PROJECT_ROOT / "data" / "synthetic_daily_forecasting.csv"
)
FALLBACK_PRODUCT_PATH = (
    PROJECT_ROOT / "data" / "synthetic_product_daily_forecasting.csv"
)

COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["date", "sale_date", "saledate", "order_date", "orderdate", "transaction_date"],
    "product_id": ["product_id", "productid", "item_id", "itemid", "sku"],
    "product_name": ["product_name", "productname", "item_name", "itemname", "product"],
    "category": ["category", "category_name", "categoryname", "product_category"],
    "quantity": ["quantity", "qty", "units_sold", "units", "volume"],
    "unit_price": ["unit_price", "unitprice", "price", "item_price"],
    "discount_percent": ["discount_percent", "discount", "discount_pct", "discount_rate"],
    "promotion": ["promotion", "promotions", "promo", "is_promotion", "promo_flag"],
    "holiday_flag": ["holiday_flag", "holiday", "is_holiday", "holidayflag"],
    "sales_amount": ["sales_amount", "salesamount", "sales", "revenue", "total_amount", "total_sales"],
    "profit": ["profit", "net_profit", "margin", "earnings"],
    "category_id": ["category_id", "categoryid"],
}


class NoActiveDatasetError(Exception):
    """Raised in production when an operation requires an active dataset but none is found."""
    pass


class DatasetConflictError(Exception):
    """Raised when a concurrent conflict or invariant prevents activating a dataset."""
    pass


def generate_tenant_product_id(user_id: int, raw_pid: str) -> str:
    """
    Generate a collision-resistant deterministic identifier for tenant-scoped products.
    Uses a 16-hex-character SHA-256 digest suffix of (user_id:raw_pid) to guarantee
    collision resistance even when long raw identifiers share identical 100+ character prefixes.
    Total length is strictly <= 100 characters.
    """
    digest = hashlib.sha256(f"{user_id}:{raw_pid}".encode("utf-8")).hexdigest()[:16]
    clean_slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_pid)[:50]
    return f"u{user_id}_{clean_slug}_{digest}"[:100]


class DatasetValidationError(Exception):
    """Raised when uploaded dataset content violates schema or data validation rules."""
    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.errors = errors or []


def normalize_column_name(raw_name: str) -> str | None:
    """Deterministically map a raw CSV column header to a canonical column name."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "", raw_name.strip().lower().replace(" ", "_").replace("-", "_"))
    for canonical, aliases in COLUMN_ALIASES.items():
        if clean in aliases or clean == canonical:
            return canonical
    return None


def parse_boolean_value(val: Any) -> bool:
    """Deterministically normalize truthy/falsy boolean representations."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return False
    if isinstance(val, bool):
        return val
    str_val = str(val).strip().lower()
    if str_val in ("1", "1.0", "true", "t", "yes", "y"):
        return True
    if str_val in ("0", "0.0", "false", "f", "no", "n", ""):
        return False
    raise ValueError(f"Value '{val}' cannot be normalized to boolean.")


def parse_date_value(val: Any) -> date:
    """Parse string/timestamp into standard Python date."""
    if isinstance(val, (datetime, pd.Timestamp)):
        return val.date()
    if isinstance(val, date):
        return val
    str_val = str(val).strip()
    # Try ISO YYYY-MM-DD or common formats
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(str_val, fmt).date()
        except ValueError:
            continue
    # Fallback to pd.to_datetime
    ts = pd.to_datetime(str_val)
    if pd.isna(ts):
        raise ValueError(f"Could not parse date '{val}'.")
    return ts.date()


class DatasetRuntimeService:
    """
    Canonical service for dataset ingestion, transactional validation,
    deterministic product management, and isolated multi-tenant runtime DataFrames.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[int | None, str | None, str], pd.DataFrame] = {}
        self._lock = threading.Lock()

    def invalidate_cache(self, user_id: int | None = None) -> None:
        """Purge cached DataFrames for a specific user, or all caches if user_id is None."""
        with self._lock:
            if user_id is None:
                self._cache.clear()
            else:
                keys_to_del = [k for k in self._cache if k[0] == user_id]
                for k in keys_to_del:
                    del self._cache[k]
        logger.info("Dataset runtime cache invalidated for user_id=%s", user_id)

    def get_active_dataset(self, db: Session, user_id: int) -> DatasetUpload | None:
        """Query currently active dataset for user_id."""
        stmt = (
            select(DatasetUpload)
            .where(
                DatasetUpload.user_id == user_id,
                DatasetUpload.status == "active",
            )
            .order_by(DatasetUpload.activated_at.desc())
        )
        return db.scalars(stmt).first()

    def process_and_import_csv(
        self,
        db: Session,
        user_id: int,
        original_filename: str,
        file_content: bytes,
    ) -> DatasetSummaryResponse:
        """
        Validate and transactionally import CSV data into PostgreSQL.
        Guarantees single active dataset invariant and atomic rollback on any error.
        """
        # 1. Parse CSV safely
        try:
            text_stream = io.StringIO(file_content.decode("utf-8-sig", errors="replace"))
            reader = csv.reader(text_stream)
            headers = next(reader, None)
        except Exception as exc:
            raise DatasetValidationError(f"Malformed CSV file: {exc}")

        if not headers:
            raise DatasetValidationError("Uploaded CSV file is completely empty.")

        # 2. Map columns deterministically
        col_map: dict[int, str] = {}
        for idx, h in enumerate(headers):
            canon = normalize_column_name(h)
            if canon:
                col_map[idx] = canon

        mapped_canonicals = set(col_map.values())
        required = {"date", "product_id", "product_name", "category", "quantity", "unit_price"}
        missing = required - mapped_canonicals
        if missing:
            raise DatasetValidationError(
                f"Missing required logical column(s): {', '.join(sorted(missing))}. "
                f"Supplied headers: {headers}"
            )

        # 3. Validate rows and collect data
        validated_rows: list[dict[str, Any]] = []
        validation_errors: list[str] = []
        validation_warnings: list[str] = []

        distinct_products: dict[str, dict[str, Any]] = {}
        distinct_categories: set[str] = set()
        min_sale_date: date | None = None
        max_sale_date: date | None = None

        row_num = 1  # 1-based header is row 1
        for row in reader:
            row_num += 1
            if not row or all(not cell.strip() for cell in row):
                continue  # skip completely blank rows

            row_data: dict[str, Any] = {}
            for idx, cell in enumerate(row):
                if idx in col_map:
                    row_data[col_map[idx]] = cell.strip()

            # Row validation
            # Date
            try:
                sale_date = parse_date_value(row_data.get("date", ""))
            except Exception as exc:
                validation_errors.append(f"Row {row_num}: Invalid date '{row_data.get('date')}': {exc}")
                if len(validation_errors) >= 25:
                    break
                continue

            # Product ID, Product Name, Category
            pid = str(row_data.get("product_id", "")).strip()
            pname = str(row_data.get("product_name", "")).strip()
            cat = str(row_data.get("category", "")).strip()
            cat_id = str(row_data.get("category_id", "")).strip() or None

            if not pid:
                validation_errors.append(f"Row {row_num}: product_id cannot be empty.")
            if not pname:
                validation_errors.append(f"Row {row_num}: product_name cannot be empty.")
            if not cat:
                validation_errors.append(f"Row {row_num}: category cannot be empty.")

            # Quantity
            try:
                qty = float(row_data.get("quantity", 0))
                if qty <= 0:
                    validation_errors.append(f"Row {row_num}: quantity must be positive (> 0), got {qty}.")
            except (ValueError, TypeError):
                validation_errors.append(f"Row {row_num}: quantity must be numeric, got '{row_data.get('quantity')}'.")
                qty = 0.0

            # Unit Price
            try:
                uprice = float(row_data.get("unit_price", 0))
                if uprice < 0:
                    validation_errors.append(f"Row {row_num}: unit_price must be non-negative (>= 0), got {uprice}.")
            except (ValueError, TypeError):
                validation_errors.append(f"Row {row_num}: unit_price must be numeric, got '{row_data.get('unit_price')}'.")
                uprice = 0.0

            # Discount Percent
            discount_val = row_data.get("discount_percent")
            try:
                discount = float(discount_val) if discount_val not in (None, "") else 0.0
                if not (0.0 <= discount <= 100.0):
                    validation_errors.append(f"Row {row_num}: discount_percent must be between 0 and 100, got {discount}.")
            except (ValueError, TypeError):
                validation_errors.append(f"Row {row_num}: discount_percent must be numeric, got '{discount_val}'.")
                discount = 0.0

            # Promotion
            try:
                promo = parse_boolean_value(row_data.get("promotion", False))
            except ValueError as exc:
                validation_errors.append(f"Row {row_num}: {exc}")
                promo = False

            # Holiday Flag
            try:
                holiday = parse_boolean_value(row_data.get("holiday_flag", False))
            except ValueError as exc:
                validation_errors.append(f"Row {row_num}: {exc}")
                holiday = False

            # Sales Amount
            sales_val = row_data.get("sales_amount")
            if sales_val not in (None, ""):
                try:
                    s_amt = float(sales_val)
                    if s_amt < 0:
                        validation_errors.append(f"Row {row_num}: sales_amount cannot be negative, got {s_amt}.")
                except (ValueError, TypeError):
                    validation_errors.append(f"Row {row_num}: sales_amount must be numeric, got '{sales_val}'.")
                    s_amt = 0.0
            else:
                # Deterministic formula: quantity * unit_price * (1 - discount_percent / 100)
                s_amt = round(qty * uprice * (1.0 - discount / 100.0), 2)

            # Profit
            profit_val = row_data.get("profit")
            if profit_val not in (None, ""):
                try:
                    profit = float(profit_val)
                except (ValueError, TypeError):
                    validation_errors.append(f"Row {row_num}: profit must be numeric, got '{profit_val}'.")
                    profit = 0.0
            else:
                # Deterministic default matching SalesRecordCreate schema default
                profit = 0.0

            if len(validation_errors) >= 25:
                break

            if pid and pname and cat:
                distinct_products[pid] = {
                    "raw_pid": pid,
                    "product_name": pname,
                    "category_name": cat,
                    "category_id": cat_id,
                    "unit_price": uprice,
                }
                distinct_categories.add(cat)

            if min_sale_date is None or sale_date < min_sale_date:
                min_sale_date = sale_date
            if max_sale_date is None or sale_date > max_sale_date:
                max_sale_date = sale_date

            validated_rows.append(
                {
                    "raw_pid": pid,
                    "sale_date": sale_date,
                    "quantity": qty,
                    "unit_price": uprice,
                    "discount_percent": discount,
                    "promotion": promo,
                    "holiday_flag": holiday,
                    "sales_amount": s_amt,
                    "profit": profit,
                }
            )

        if validation_errors:
            raise DatasetValidationError(
                f"Dataset validation failed with {len(validation_errors)} error(s).",
                errors=validation_errors,
            )

        if not validated_rows:
            raise DatasetValidationError("No valid data rows found in uploaded CSV.")

        # 4. Deterministic Product Resolution and Scoping
        # Map raw_pid -> Product.id
        product_pk_map: dict[str, int] = {}
        now = datetime.now(timezone.utc)

        for raw_pid, pinfo in distinct_products.items():
            tenant_pid = generate_tenant_product_id(user_id, raw_pid)

            # Check if this user already has an existing product for raw_pid
            tenant_p = db.scalars(
                select(Product).where(
                    (Product.product_id == tenant_pid)
                    | ((Product.tenant_id == user_id) & (Product.raw_product_id == raw_pid))
                )
            ).first()

            if tenant_p:
                tenant_p.product_name = pinfo["product_name"]
                tenant_p.category_id = pinfo["category_id"]
                tenant_p.category_name = pinfo["category_name"]
                tenant_p.unit_price = pinfo["unit_price"]
                product_pk_map[raw_pid] = tenant_p.id
            else:
                new_tp = Product(
                    product_id=tenant_pid,
                    tenant_id=user_id,
                    raw_product_id=raw_pid,
                    product_name=pinfo["product_name"],
                    category_id=pinfo["category_id"],
                    category_name=pinfo["category_name"],
                    unit_price=pinfo["unit_price"],
                    created_at=now,
                )
                db.add(new_tp)
                db.flush()
                product_pk_map[raw_pid] = new_tp.id

        # 5. Create DatasetUpload record & enforce single active dataset
        dataset_id = f"ds-{uuid.uuid4().hex[:12]}"
        safe_filename = re.sub(r"[^\w\.-]", "_", Path(original_filename).name)
        dataset_key = f"usr_{user_id}_{int(time.time())}_{safe_filename[:50]}"

        try:
            # Archive any currently active dataset for this user
            db.execute(
                update(DatasetUpload)
                .where(
                    DatasetUpload.user_id == user_id,
                    DatasetUpload.status == "active",
                )
                .values(status="archived")
            )

            dataset_record = DatasetUpload(
                id=dataset_id,
                user_id=user_id,
                original_filename=safe_filename,
                dataset_key=dataset_key,
                row_count=len(validated_rows),
                product_count=len(distinct_products),
                category_count=len(distinct_categories),
                min_date=min_sale_date,
                max_date=max_sale_date,
                status="active",
                validation_summary={
                    "imported_rows": len(validated_rows),
                    "warnings": validation_warnings,
                },
                created_at=now,
                activated_at=now,
            )
            db.add(dataset_record)
            db.flush()

            # 6. Bulk insert SalesRecords
            sales_records_to_insert = [
                {
                    "user_id": user_id,
                    "dataset_id": dataset_id,
                    "product_id": product_pk_map[r["raw_pid"]],
                    "sale_date": r["sale_date"],
                    "quantity": r["quantity"],
                    "unit_price": r["unit_price"],
                    "discount_percent": r["discount_percent"],
                    "promotion": r["promotion"],
                    "holiday_flag": r["holiday_flag"],
                    "sales_amount": r["sales_amount"],
                    "profit": r["profit"],
                    "created_at": now,
                }
                for r in validated_rows
            ]

            # Chunked bulk insertion
            chunk_size = 5000
            for i in range(0, len(sales_records_to_insert), chunk_size):
                chunk = sales_records_to_insert[i : i + chunk_size]
                db.bulk_insert_mappings(SalesRecord, chunk)

            db.commit()
        except IntegrityError as exc:
            db.rollback()
            logger.warning("Integrity conflict during dataset upload for user_id=%s: %s", user_id, exc)
            raise DatasetConflictError(
                "A concurrent dataset activation occurred. Please retry dataset activation."
            ) from exc

        self.invalidate_cache(user_id)

        return DatasetSummaryResponse(
            dataset_id=dataset_id,
            filename=safe_filename,
            rows_imported=len(validated_rows),
            products=len(distinct_products),
            categories=len(distinct_categories),
            start_date=min_sale_date.isoformat() if min_sale_date else None,
            end_date=max_sale_date.isoformat() if max_sale_date else None,
            status="active",
            validation_warnings=validation_warnings,
            validation_errors=[],
        )

    def activate_dataset(self, db: Session, user_id: int, dataset_id: str) -> DatasetSummaryResponse:
        """
        Transactionally activate dataset for user_id and archive previous active dataset.
        Enforces at most one active dataset per user.
        """
        target = db.scalars(
            select(DatasetUpload).where(
                DatasetUpload.id == dataset_id,
                DatasetUpload.user_id == user_id,
            )
        ).first()

        if not target:
            raise KeyError(f"Dataset '{dataset_id}' not found for current user.")

        if target.status == "failed":
            raise ValueError(f"Cannot activate dataset '{dataset_id}' in 'failed' state.")

        now = datetime.now(timezone.utc)

        try:
            # Archive previous active datasets
            db.execute(
                update(DatasetUpload)
                .where(
                    DatasetUpload.user_id == user_id,
                    DatasetUpload.status == "active",
                    DatasetUpload.id != dataset_id,
                )
                .values(status="archived")
            )

            target.status = "active"
            target.activated_at = now
            db.commit()
            db.refresh(target)
        except IntegrityError as exc:
            db.rollback()
            logger.warning(
                "Integrity conflict during dataset activation for user_id=%s: %s. Retrying...",
                user_id,
                exc,
            )
            try:
                target = db.scalars(
                    select(DatasetUpload).where(
                        DatasetUpload.id == dataset_id,
                        DatasetUpload.user_id == user_id,
                    )
                ).first()
                if not target:
                    raise KeyError(f"Dataset '{dataset_id}' not found for current user.")

                db.execute(
                    update(DatasetUpload)
                    .where(
                        DatasetUpload.user_id == user_id,
                        DatasetUpload.status == "active",
                        DatasetUpload.id != dataset_id,
                    )
                    .values(status="archived")
                )
                target.status = "active"
                target.activated_at = datetime.now(timezone.utc)
                db.commit()
                db.refresh(target)
            except IntegrityError as retry_exc:
                db.rollback()
                raise DatasetConflictError(
                    "A concurrent dataset activation conflict occurred. Exactly one dataset can be active per user."
                ) from retry_exc

        self.invalidate_cache(user_id)

        return DatasetSummaryResponse(
            dataset_id=target.id,
            filename=target.original_filename,
            rows_imported=target.row_count,
            products=target.product_count,
            categories=target.category_count,
            start_date=target.min_date.isoformat() if target.min_date else None,
            end_date=target.max_date.isoformat() if target.max_date else None,
            status=target.status,
            validation_warnings=[],
            validation_errors=[],
        )

    def delete_dataset(self, db: Session, user_id: int, dataset_id: str) -> None:
        """
        Delete dataset belonging to user_id.
        Cascades delete only to its linked sales records.
        Preserves shared products and legacy sales.
        """
        target = db.scalars(
            select(DatasetUpload).where(
                DatasetUpload.id == dataset_id,
                DatasetUpload.user_id == user_id,
            )
        ).first()

        if not target:
            raise KeyError(f"Dataset '{dataset_id}' not found for current user.")

        db.delete(target)
        db.commit()
        self.invalidate_cache(user_id)

    def get_daily_aggregate(
        self,
        user_id: int | None = None,
        db: Session | None = None,
        dataset_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Return daily aggregate DataFrame:
        Date, Quantity, Sales_Amount, Profit, Promotions, Holiday_Flag.
        Strictly isolated by user_id and dataset_id (or active dataset).
        """
        target_ds = None
        if db and user_id:
            if dataset_id:
                target_ds = db.scalars(
                    select(DatasetUpload).where(
                        DatasetUpload.id == dataset_id,
                        DatasetUpload.user_id == user_id,
                    )
                ).first()
            else:
                target_ds = self.get_active_dataset(db, user_id)

        cache_key = (user_id, target_ds.id if target_ds else None, "daily_aggregate")

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key].copy()

        if target_ds and db and user_id:
            # Load from database for target dataset
            stmt = (
                select(
                    SalesRecord.sale_date.label("Date"),
                    func.sum(SalesRecord.quantity).label("Quantity"),
                    func.sum(SalesRecord.sales_amount).label("Sales_Amount"),
                    func.sum(SalesRecord.profit).label("Profit"),
                    func.max(case((SalesRecord.promotion == True, 1), else_=0)).label("Promotions"),
                    func.max(case((SalesRecord.holiday_flag == True, 1), else_=0)).label("Holiday_Flag"),
                )
                .where(
                    SalesRecord.user_id == user_id,
                    SalesRecord.dataset_id == target_ds.id,
                )
                .group_by(SalesRecord.sale_date)
                .order_by(SalesRecord.sale_date.asc())
            )

            rows = db.execute(stmt).all()
            if not rows:
                df = pd.DataFrame(
                    columns=["Date", "Quantity", "Sales_Amount", "Profit", "Promotions", "Holiday_Flag"]
                )
            else:
                data = [
                    {
                        "Date": pd.to_datetime(r.Date),
                        "Quantity": float(r.Quantity or 0.0),
                        "Sales_Amount": float(r.Sales_Amount or 0.0),
                        "Profit": float(r.Profit or 0.0),
                        "Promotions": int(bool(r.Promotions)),
                        "Holiday_Flag": int(bool(r.Holiday_Flag)),
                    }
                    for r in rows
                ]
                df = pd.DataFrame(data).sort_values("Date").reset_index(drop=True)

            with self._lock:
                self._cache[cache_key] = df
            return df.copy()

        # Fallback handling
        if dataset_id:
            # Explicit dataset_id not found for this user: tenant isolated empty DataFrame
            return pd.DataFrame(
                columns=["Date", "Quantity", "Sales_Amount", "Profit", "Promotions", "Holiday_Flag"]
            )

        if settings.is_production:
            raise NoActiveDatasetError(
                "No active dataset found for this account. Please upload and activate a sales dataset via /datasets."
            )

        # Development / test fallback
        df = self._load_fallback_daily()
        with self._lock:
            self._cache[cache_key] = df
        return df.copy()

    def get_product_daily(
        self,
        user_id: int | None = None,
        db: Session | None = None,
    ) -> pd.DataFrame:
        """
        Return product-level daily DataFrame:
        Date, Product_ID, Product_Name, Category_ID, Category_Name, Quantity,
        Sales_Amount, Unit_Price, Discount_Percent, Promotion, Is_Holiday, Profit.
        """
        active_ds = self.get_active_dataset(db, user_id) if (db and user_id) else None
        cache_key = (user_id, active_ds.id if active_ds else None, "product_daily")

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key].copy()

        if active_ds and db and user_id:
            # Query joined SalesRecord and Product
            stmt = (
                select(
                    SalesRecord.sale_date.label("Date"),
                    Product.product_id.label("Product_ID"),
                    Product.raw_product_id.label("Raw_Product_ID"),
                    Product.product_name.label("Product_Name"),
                    Product.category_id.label("Category_ID"),
                    Product.category_name.label("Category_Name"),
                    SalesRecord.quantity.label("Quantity"),
                    SalesRecord.sales_amount.label("Sales_Amount"),
                    SalesRecord.unit_price.label("Unit_Price"),
                    SalesRecord.discount_percent.label("Discount_Percent"),
                    SalesRecord.promotion.label("Promotion"),
                    SalesRecord.holiday_flag.label("Is_Holiday"),
                    SalesRecord.profit.label("Profit"),
                )
                .join(Product, SalesRecord.product_id == Product.id)
                .where(
                    SalesRecord.user_id == user_id,
                    SalesRecord.dataset_id == active_ds.id,
                )
                .order_by(SalesRecord.sale_date.asc())
            )

            rows = db.execute(stmt).all()
            if not rows:
                df = pd.DataFrame(
                    columns=[
                        "Date",
                        "Product_ID",
                        "Product_Name",
                        "Category_ID",
                        "Category_Name",
                        "Quantity",
                        "Sales_Amount",
                        "Unit_Price",
                        "Discount_Percent",
                        "Promotion",
                        "Is_Holiday",
                        "Profit",
                    ]
                )
            else:
                prefix_to_strip = f"u{user_id}_"
                data = [
                    {
                        "Date": pd.to_datetime(r.Date),
                        "Product_ID": r.Raw_Product_ID or (
                            r.Product_ID[len(prefix_to_strip):]
                            if r.Product_ID.startswith(prefix_to_strip)
                            else r.Product_ID
                        ),
                        "Product_Name": r.Product_Name,
                        "Category_ID": r.Category_ID or r.Category_Name or "1",
                        "Category_Name": r.Category_Name or "General",
                        "Quantity": float(r.Quantity or 0.0),
                        "Sales_Amount": float(r.Sales_Amount or 0.0),
                        "Unit_Price": float(r.Unit_Price or 0.0),
                        "Discount_Percent": float(r.Discount_Percent or 0.0),
                        "Promotion": int(bool(r.Promotion)),
                        "Is_Holiday": int(bool(r.Is_Holiday)),
                        "Profit": float(r.Profit or 0.0),
                    }
                    for r in rows
                ]
                df = pd.DataFrame(data).sort_values("Date").reset_index(drop=True)

            with self._lock:
                self._cache[cache_key] = df
            return df.copy()

        # Fallback handling
        if settings.is_production:
            raise NoActiveDatasetError(
                "No active dataset found for this account. Please upload and activate a sales dataset via /datasets."
            )

        df = self._load_fallback_product()
        with self._lock:
            self._cache[cache_key] = df
        return df.copy()

    def _load_fallback_daily(self) -> pd.DataFrame:
        """Load static daily dataset for development and testing."""
        if FALLBACK_DAILY_SYNTHETIC_PATH.exists():
            df = pd.read_csv(FALLBACK_DAILY_SYNTHETIC_PATH, parse_dates=["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            if "Promotions" not in df.columns and "Promotion" in df.columns:
                df["Promotions"] = df["Promotion"]
            return df
        elif FALLBACK_DAILY_FEATURES_PATH.exists():
            df = pd.read_csv(FALLBACK_DAILY_FEATURES_PATH, parse_dates=["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            return df
        else:
            raise FileNotFoundError("Fallback daily forecasting dataset not found.")

    def _load_fallback_product(self) -> pd.DataFrame:
        """Load static product-level dataset for development and testing."""
        if not FALLBACK_PRODUCT_PATH.exists():
            raise FileNotFoundError(f"Fallback product dataset not found: {FALLBACK_PRODUCT_PATH}")
        df = pd.read_csv(
            FALLBACK_PRODUCT_PATH,
            usecols=[
                "Date",
                "Product_ID",
                "Product_Name",
                "Category_ID",
                "Category_Name",
                "Quantity",
                "Sales_Amount",
                "Unit_Price",
                "Discount_Percent",
                "Promotion",
                "Is_Holiday",
                "Profit",
            ],
            parse_dates=["Date"],
        )
        return df.sort_values("Date").reset_index(drop=True)


# Global singleton instance
dataset_runtime_service = DatasetRuntimeService()
