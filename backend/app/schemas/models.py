from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class CompanyModelItem(BaseModel):
    horizon: int = Field(..., description="Forecasting horizon (7, 30, or 90 days)")
    status: str = Field(..., description="Model status: queued, processing, training, evaluating, ready, insufficient_data, failed, none")
    model_type: str | None = Field(None, description="Winning candidate algorithm family, e.g. HistGradientBoosting")
    model_version: int | None = Field(None, description="Model iteration version number")
    validation_wape: float | None = Field(None, description="Validation Weighted Absolute Percentage Error used for selection")
    test_wape: float | None = Field(None, description="Unbiased test Weighted Absolute Percentage Error")
    validation_mae: float | None = Field(None, description="Validation Mean Absolute Error")
    test_mae: float | None = Field(None, description="Test Mean Absolute Error")
    validation_rmse: float | None = Field(None, description="Validation Root Mean Squared Error")
    test_rmse: float | None = Field(None, description="Test Root Mean Squared Error")
    training_rows: int | None = Field(None, description="Number of daily rows used in final training fit")
    trained_at: datetime | None = Field(None, description="Timestamp when model benchmarking and training completed")
    is_active: bool = Field(False, description="Whether this model is currently active for runtime inference")
    source: str = Field("company", description="Model source: 'company' or 'global_fallback'")
    status_message: str | None = Field(None, description="Status detail or insufficiency explanation")


class TrainingJobStatus(BaseModel):
    job_id: str = Field(..., description="Unique training job ID")
    status: str = Field(..., description="Job status: queued, processing, training, evaluating, ready, failed")
    progress_stage: str | None = Field(None, description="Current progress milestone description")
    error_message: str | None = Field(None, description="Failure reason if job failed")
    started_at: datetime | None = Field(None, description="When job processing started")
    completed_at: datetime | None = Field(None, description="When job reached terminal state")


class CurrentModelsResponse(BaseModel):
    active_dataset_id: str | None = Field(None, description="Identifier of the currently active dataset")
    active_model_version: int | None = Field(None, description="Active model iteration version")
    training_job: TrainingJobStatus | None = Field(None, description="Latest training job status for active dataset")
    models: list[CompanyModelItem] = Field(default_factory=list, description="Forecasting models across 7D, 30D, and 90D")


class TrainTriggerResponse(BaseModel):
    dataset_id: str = Field(..., description="Dataset ID for which training was queued")
    job_id: str = Field(..., description="Unique training job ID")
    status: str = Field(..., description="Training state: queued, processing, etc.")
    message: str = Field(..., description="User-facing status confirmation")
