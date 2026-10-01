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


class TimeSeriesQualityResponse(BaseModel):
    dataset_id: str | None = Field(default=None, description="Dataset identifier")
    frequency: str = Field(default="daily", description="Detected calendar frequency")
    min_date: str | None = Field(default=None, description="Earliest calendar date in series (YYYY-MM-DD)")
    max_date: str | None = Field(default=None, description="Latest calendar date in series (YYYY-MM-DD)")
    expected_days: int = Field(default=0, description="Total calendar days between min_date and max_date")
    observed_days: int = Field(default=0, description="Days with recorded transactions")
    missing_days: int = Field(default=0, description="Missing calendar days with no observations")
    coverage_ratio: float = Field(default=0.0, description="Ratio of observed days to expected days")
    duplicate_dates: int = Field(default=0, description="Count of duplicate date timestamps")
    zero_demand_days: int = Field(default=0, description="Days explicitly observed with 0 demand")
    missing_date_samples: list[str] = Field(default_factory=list, description="Sample of missing calendar dates (up to 10)")
    longest_missing_gap: int = Field(default=0, description="Longest consecutive sequence of missing days")
    is_continuous: bool = Field(default=False, description="Whether dataset has a complete daily calendar without gaps or duplicates")
    quality_status: str = Field(default="insufficient", description="Quality tier: 'good', 'warning', or 'insufficient'")
    warnings: list[str] = Field(default_factory=list, description="Diagnostic warnings regarding continuity and coverage")
