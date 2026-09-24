from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.database.crud import (
    create_product,
    delete_product,
    get_product_by_external_id,
    get_products,
    update_product,
)
from backend.app.database.database import get_db
from backend.app.database.models import Product, User
from backend.app.dependencies import get_current_user_optional
from backend.app.schemas.products import (
    ProductCreate,
    ProductResponse,
    ProductUpdate,
)


router = APIRouter(
    prefix="/api/products",
    tags=["Products"],
)


def to_product_response(product: Product) -> ProductResponse:
    """Map internal Product model to ProductResponse, preserving raw_product_id."""
    return ProductResponse(
        id=product.id,
        product_id=product.raw_product_id or product.product_id,
        product_name=product.product_name,
        category_id=product.category_id,
        category_name=product.category_name,
        unit_price=product.unit_price,
        created_at=product.created_at,
    )


@router.post(
    "",
    response_model=ProductResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_product_endpoint(
    product_data: ProductCreate,
    current_user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Create a new product.
    """
    user_id = current_user.id if current_user else None
    existing_product = get_product_by_external_id(
        db,
        product_data.product_id,
        user_id=user_id,
    )

    if existing_product:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Product with product_id "
                f"'{product_data.product_id}' "
                f"already exists."
            ),
        )

    product = create_product(
        db,
        product_data,
    )
    return to_product_response(product)


@router.get(
    "",
    response_model=list[ProductResponse],
)
def list_products(
    skip: int = Query(
        default=0,
        ge=0,
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
    ),
    current_user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Return a paginated list of products.
    Shared products (tenant_id is None) and products belonging to the
    authenticated user are returned. Cross-tenant products are isolated.
    """
    user_id = current_user.id if current_user else None
    products = get_products(
        db,
        skip=skip,
        limit=limit,
        user_id=user_id,
    )
    return [to_product_response(p) for p in products]


@router.get(
    "/{product_id}",
    response_model=ProductResponse,
)
def get_product_endpoint(
    product_id: str,
    current_user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Return a single product by external product ID.
    Enforces tenant scoping: tenant-specific products are not accessible
    to other tenants or unauthenticated callers.
    """
    user_id = current_user.id if current_user else None
    product = get_product_by_external_id(
        db,
        product_id,
        user_id=user_id,
    )

    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found.",
        )

    return to_product_response(product)


@router.put(
    "/{product_id}",
    response_model=ProductResponse,
)
def update_product_endpoint(
    product_id: str,
    product_data: ProductUpdate,
    current_user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Update an existing product by external product ID.
    Enforces tenant scoping.
    """
    user_id = current_user.id if current_user else None
    product = get_product_by_external_id(
        db,
        product_id,
        user_id=user_id,
    )

    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found.",
        )

    updated = update_product(
        db,
        product,
        product_data,
    )
    return to_product_response(updated)


@router.delete(
    "/{product_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_product_endpoint(
    product_id: str,
    current_user: User | None = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """
    Delete an existing product by external product ID.
    Enforces tenant scoping.
    """
    user_id = current_user.id if current_user else None
    product = get_product_by_external_id(
        db,
        product_id,
        user_id=user_id,
    )

    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found.",
        )

    delete_product(
        db,
        product,
    )

    return None
