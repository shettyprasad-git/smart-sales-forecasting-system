from __future__ import annotations

from datetime import date, datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class DatasetSummaryResponse(BaseModel):
    dataset_id: str = Field(..., description="Unique dataset identifier")
    filename: str = Field(..., description="Original filename of the uploaded CSV")
    rows_imported: int = Field(..., description="Total validated sales rows imported")
    products: int = Field(..., description="Distinct products identified and linked")
    categories: int = Field(..., description="Distinct categories identified")
    start_date: str | None = Field(default=None, description="Earliest transaction date (YYYY-MM-DD)")
    end_date: str | None = Field(default=None, description="Latest transaction date (YYYY-MM-DD)")
    status: str = Field(default="active", description="Dataset lifecycle status (active, archived, etc.)")
    model_inference_mode: str = Field(
        default="runtime_inference_pretrained_models",
        description="Dataset is used for runtime inference using the existing pre-trained 7/30/90-day models without retraining.",
    )
    validation_warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings encountered during validation")
    validation_errors: list[str] = Field(default_factory=list, description="Validation failure details if any")


class DatasetHistoryItem(BaseModel):
    id: str
    original_filename: str
    row_count: int
    product_count: int
    category_count: int
    min_date: str | None = None
    max_date: str | None = None
    status: str
    model_inference_mode: str = "runtime_inference_pretrained_models"
    created_at: datetime
    activated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class DatasetHistoryResponse(BaseModel):
    datasets: list[DatasetHistoryItem]
    total: int


class DatasetActionResponse(BaseModel):
    message: str
    dataset_id: str
    status: str
