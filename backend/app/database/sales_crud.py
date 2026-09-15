from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import SalesRecord


def create_sales_record(
    db: Session,
    user_id: int,
    data: dict,
) -> SalesRecord:

    sales_record = SalesRecord(
        user_id=user_id,
        **data,
    )

    db.add(sales_record)
    db.commit()
    db.refresh(sales_record)

    return sales_record


def get_sales_records(
    db: Session,
    user_id: int,
    skip: int = 0,
    limit: int = 100,
) -> list[SalesRecord]:

    statement = (
        select(SalesRecord)
        .where(SalesRecord.user_id == user_id)
        .order_by(SalesRecord.sale_date.desc())
        .offset(skip)
        .limit(limit)
    )

    return list(db.scalars(statement).all())


def get_sales_record(
    db: Session,
    user_id: int,
    sales_id: int,
) -> SalesRecord | None:

    statement = select(SalesRecord).where(
        SalesRecord.id == sales_id,
        SalesRecord.user_id == user_id,
    )

    return db.scalars(statement).first()


def update_sales_record(
    db: Session,
    sales_record: SalesRecord,
    data: dict,
) -> SalesRecord:

    for field, value in data.items():
        setattr(
            sales_record,
            field,
            value,
        )

    db.commit()
    db.refresh(sales_record)

    return sales_record


def delete_sales_record(
    db: Session,
    sales_record: SalesRecord,
) -> None:

    db.delete(sales_record)
    db.commit()
