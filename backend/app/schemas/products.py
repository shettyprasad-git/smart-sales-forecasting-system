from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ProductBase(BaseModel):
    product_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    product_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
    )

    category_id: str | None = Field(
        default=None,
        max_length=100,
    )

    category_name: str | None = Field(
        default=None,
        max_length=255,
    )

    unit_price: float = Field(
        ...,
        ge=0,
    )


class ProductCreate(ProductBase):
    pass


class ProductUpdate(BaseModel):
    product_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
    )

    category_id: str | None = Field(
        default=None,
        max_length=100,
    )

    category_name: str | None = Field(
        default=None,
        max_length=255,
    )

    unit_price: float | None = Field(
        default=None,
        ge=0,
    )


class ProductResponse(ProductBase):
    id: int
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
    )

