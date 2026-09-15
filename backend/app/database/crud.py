from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import Product
from backend.app.schemas.products import (
    ProductCreate,
    ProductUpdate,
)


def get_products(
    db: Session,
    skip: int = 0,
    limit: int = 100,
) -> list[Product]:

    statement = (
        select(Product)
        .offset(skip)
        .limit(limit)
    )

    return list(
        db.scalars(statement).all()
    )


def get_product(
    db: Session,
    product_id: int,
) -> Product | None:

    return db.get(
        Product,
        product_id,
    )


def get_product_by_external_id(
    db: Session,
    product_id: str,
) -> Product | None:

    statement = select(Product).where(
        Product.product_id == product_id
    )

    return db.scalars(
        statement
    ).first()


def create_product(
    db: Session,
    product_data: ProductCreate,
) -> Product:

    product = Product(
        product_id=product_data.product_id,
        product_name=product_data.product_name,
        category_id=product_data.category_id,
        category_name=product_data.category_name,
        unit_price=product_data.unit_price,
    )

    db.add(product)
    db.commit()
    db.refresh(product)

    return product


def update_product(
    db: Session,
    product: Product,
    product_data: ProductUpdate,
) -> Product:

    update_data = product_data.model_dump(
        exclude_unset=True
    )

    for field, value in update_data.items():
        setattr(product, field, value)

    db.commit()
    db.refresh(product)

    return product


def delete_product(
    db: Session,
    product: Product,
) -> None:

    db.delete(product)
    db.commit()
