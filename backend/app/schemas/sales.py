from datetime import date, datetime

from pydantic import BaseModel, Field


class SalesRecordCreate(BaseModel):
    product_id: int = Field(..., gt=0)
    sale_date: date
    quantity: float = Field(..., gt=0)
    unit_price: float = Field(..., ge=0)
    discount_percent: float = Field(
        default=0,
        ge=0,
        le=100,
    )
    promotion: bool = False
    holiday_flag: bool = False
    sales_amount: float = Field(..., ge=0)
    profit: float = 0


class SalesRecordUpdate(BaseModel):
    product_id: int | None = Field(default=None, gt=0)
    sale_date: date | None = None
    quantity: float | None = Field(default=None, gt=0)
    unit_price: float | None = Field(default=None, ge=0)
    discount_percent: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )
    promotion: bool | None = None
    holiday_flag: bool | None = None
    sales_amount: float | None = Field(
        default=None,
        ge=0,
    )
    profit: float | None = None


class SalesRecordResponse(BaseModel):
    id: int
    user_id: int
    product_id: int
    sale_date: date
    quantity: float
    unit_price: float
    discount_percent: float
    promotion: bool
    holiday_flag: bool
    sales_amount: float
    profit: float
    created_at: datetime

    model_config = {
        "from_attributes": True,
    }
