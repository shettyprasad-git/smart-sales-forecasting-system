from sqlalchemy import and_, or_, select
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
    user_id: int | None = None,
) -> list[Product]:

    if user_id is not None:
        statement = (
            select(Product)
            .where(
                or_(
                    Product.tenant_id.is_(None),
                    Product.tenant_id == user_id,
                )
            )
            .offset(skip)
            .limit(limit)
        )
    else:
        statement = (
            select(Product)
            .where(Product.tenant_id.is_(None))
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
    user_id: int | None = None,
) -> Product | None:

    if user_id is not None:
        statement = select(Product).where(
            or_(
                Product.product_id == product_id,
                and_(
                    Product.raw_product_id == product_id,
                    Product.tenant_id == user_id,
                ),
            )
        )
    else:
        statement = select(Product).where(
            or_(
                Product.product_id == product_id,
                Product.raw_product_id == product_id,
            )
        )

    product = db.scalars(
        statement
    ).first()

    if product is not None and product.tenant_id is not None:
        if user_id is None or product.tenant_id != user_id:
            return None

    return product


def create_product(
    db: Session,
    product_data: ProductCreate,
    tenant_id: int | None = None,
) -> Product:

    product = Product(
        product_id=product_data.product_id,
        raw_product_id=product_data.product_id,
        tenant_id=tenant_id,
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
