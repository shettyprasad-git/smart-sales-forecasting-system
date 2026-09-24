from datetime import date

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    horizon: int = Field(
        ...,
        description="Forecast horizon in days. Supported values: 7, 30, 90.",
    )


class ForecastPoint(BaseModel):
    date: date
    predicted_quantity: float


class ForecastResponse(BaseModel):
    horizon: int
    model: str
    source: str = "global"
    model_type: str | None = None
    model_version: int | None = None
    validation_wape: float | None = None
    test_wape: float | None = None
    fallback_reason: str | None = None
    forecast: list[ForecastPoint]


class HistoricalPoint(BaseModel):
    date: date
    quantity: float
    sales_amount: float
    profit: float


class HistoricalResponse(BaseModel):
    records: list[HistoricalPoint]


class DashboardKPIs(BaseModel):
    total_historical_quantity: float
    total_historical_sales: float
    total_historical_profit: float
    average_daily_quantity: float


class DashboardResponse(BaseModel):
    kpis: DashboardKPIs
    historical: list[HistoricalPoint]
    forecast: list[ForecastPoint]
    forecast_horizon: int
    forecast_model: str
