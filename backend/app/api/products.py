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
from backend.app.schemas.products import (
    ProductCreate,
    ProductResponse,
    ProductUpdate,
)


router = APIRouter(
    prefix="/api/products",
    tags=["Products"],
)


@router.post(
    "",
    response_model=ProductResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_product_endpoint(
    product_data: ProductCreate,
    db: Session = Depends(get_db),
):
    """
    Create a new product.
    """

    existing_product = get_product_by_external_id(
        db,
        product_data.product_id,
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

    return create_product(
        db,
        product_data,
    )


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
    db: Session = Depends(get_db),
):
    """
    Return a paginated list of products.
    """

    return get_products(
        db,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{product_id}",
    response_model=ProductResponse,
)
def get_product_endpoint(
    product_id: str,
    db: Session = Depends(get_db),
):
    """
    Return a single product by external product ID.
    """

    product = get_product_by_external_id(
        db,
        product_id,
    )

    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found.",
        )

    return product


@router.put(
    "/{product_id}",
    response_model=ProductResponse,
)
def update_product_endpoint(
    product_id: str,
    product_data: ProductUpdate,
    db: Session = Depends(get_db),
):
    """
    Update an existing product by external product ID.
    """

    product = get_product_by_external_id(
        db,
        product_id,
    )

    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found.",
        )

    return update_product(
        db,
        product,
        product_data,
    )


@router.delete(
    "/{product_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_product_endpoint(
    product_id: str,
    db: Session = Depends(get_db),
):
    """
    Delete an existing product by external product ID.
    """

    product = get_product_by_external_id(
        db,
        product_id,
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
