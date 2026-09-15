from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.database.database import get_db
from backend.app.database.models import Product
from backend.app.database.sales_crud import (
    create_sales_record,
    delete_sales_record,
    get_sales_record,
    get_sales_records,
    update_sales_record,
)
from backend.app.dependencies import get_current_user
from backend.app.schemas.sales import (
    SalesRecordCreate,
    SalesRecordResponse,
    SalesRecordUpdate,
)


router = APIRouter(
    prefix="/api/sales",
    tags=["Sales"],
)


@router.post(
    "",
    response_model=SalesRecordResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_sale(
    sale_data: SalesRecordCreate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):

    product = db.query(Product).filter(
        Product.id == sale_data.product_id
    ).first()

    if product is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Product not found.",
        )

    sale = create_sales_record(
        db=db,
        user_id=current_user.id,
        data=sale_data.model_dump(),
    )

    return sale


@router.get(
    "",
    response_model=list[SalesRecordResponse],
)
def list_sales(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):

    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="skip cannot be negative.",
        )

    if limit < 1 or limit > 500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="limit must be between 1 and 500.",
        )

    return get_sales_records(
        db=db,
        user_id=current_user.id,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/{sales_id}",
    response_model=SalesRecordResponse,
)
def get_sale(
    sales_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):

    sale = get_sales_record(
        db=db,
        user_id=current_user.id,
        sales_id=sales_id,
    )

    if sale is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sales record not found.",
        )

    return sale


@router.put(
    "/{sales_id}",
    response_model=SalesRecordResponse,
)
def update_sale(
    sales_id: int,
    sale_data: SalesRecordUpdate,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):

    sale = get_sales_record(
        db=db,
        user_id=current_user.id,
        sales_id=sales_id,
    )

    if sale is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sales record not found.",
        )

    if sale_data.product_id is not None:
        product = db.query(Product).filter(
            Product.id == sale_data.product_id
        ).first()

        if product is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Product not found.",
            )

    update_data = sale_data.model_dump(
        exclude_unset=True
    )

    return update_sales_record(
        db=db,
        sales_record=sale,
        data=update_data,
    )


@router.delete(
    "/{sales_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_sale(
    sales_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):

    sale = get_sales_record(
        db=db,
        user_id=current_user.id,
        sales_id=sales_id,
    )

    if sale is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sales record not found.",
        )

    delete_sales_record(
        db=db,
        sales_record=sale,
    )

    return None
