from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.forecast import (
    DashboardKPIs,
    DashboardResponse,
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HistoricalPoint,
    HistoricalResponse,
)
from backend.app.services.forecast_service import (
    BackendForecastService,
)
from backend.app.services.history_service import (
    HistoryService,
)


router = APIRouter(
    prefix="/api/forecast",
    tags=["Forecast"],
)


forecast_service = BackendForecastService()
history_service = HistoryService()


@router.post(
    "",
    response_model=ForecastResponse,
)
def generate_forecast(
    request: ForecastRequest,
):
    """
    Generate a demand forecast using the production ML model.
    """

    if request.horizon not in (7, 30, 90):
        raise HTTPException(
            status_code=400,
            detail=(
                "Supported forecast horizons "
                "are 7, 30, and 90 days."
            ),
        )

    try:
        model_name, forecast_df = (
            forecast_service.generate_forecast(
                horizon=request.horizon
            )
        )

        forecast = [
            ForecastPoint(
                date=row["Date"],
                predicted_quantity=float(
                    row["Predicted_Quantity"]
                ),
            )
            for _, row in forecast_df.iterrows()
        ]

        return ForecastResponse(
            horizon=request.horizon,
            model=model_name,
            forecast=forecast,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast generation failed: {exc}",
        ) from exc


@router.get(
    "/history",
    response_model=HistoricalResponse,
)
def get_forecast_history(
    limit: int = Query(
        default=365,
        ge=1,
        le=2922,
        description="Number of historical daily records to return.",
    ),
):
    """
    Return historical demand, sales, and profit data
    for dashboard visualization.
    """

    try:
        history_df = history_service.get_history(
            limit=limit
        )

        records = [
            HistoricalPoint(
                date=row["Date"],
                quantity=float(row["Quantity"]),
                sales_amount=float(row["Sales_Amount"]),
                profit=float(row["Profit"]),
            )
            for _, row in history_df.iterrows()
        ]

        return HistoricalResponse(
            records=records
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Historical data retrieval failed: {exc}",
        ) from exc


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
)
def get_dashboard(
    horizon: int = Query(
        default=7,
        description="Forecast horizon: 7, 30, or 90 days.",
    ),
    history_limit: int = Query(
        default=365,
        ge=1,
        le=2922,
        description="Number of historical records.",
    ),
):
    """
    Return dashboard KPIs, historical data, and forecast
    in a single response.
    """

    if horizon not in (7, 30, 90):
        raise HTTPException(
            status_code=400,
            detail=(
                "Supported forecast horizons "
                "are 7, 30, and 90 days."
            ),
        )

    try:
        history_df = history_service.get_history(
            limit=history_limit
        )

        if history_df.empty:
            raise ValueError(
                "No historical data available."
            )

        model_name, forecast_df = (
            forecast_service.generate_forecast(
                horizon=horizon
            )
        )

        historical = [
            HistoricalPoint(
                date=row["Date"],
                quantity=float(row["Quantity"]),
                sales_amount=float(row["Sales_Amount"]),
                profit=float(row["Profit"]),
            )
            for _, row in history_df.iterrows()
        ]

        forecast = [
            ForecastPoint(
                date=row["Date"],
                predicted_quantity=float(
                    row["Predicted_Quantity"]
                ),
            )
            for _, row in forecast_df.iterrows()
        ]

        total_quantity = float(
            history_df["Quantity"].sum()
        )

        total_sales = float(
            history_df["Sales_Amount"].sum()
        )

        total_profit = float(
            history_df["Profit"].sum()
        )

        average_quantity = float(
            history_df["Quantity"].mean()
        )

        kpis = DashboardKPIs(
            total_historical_quantity=total_quantity,
            total_historical_sales=total_sales,
            total_historical_profit=total_profit,
            average_daily_quantity=average_quantity,
        )

        return DashboardResponse(
            kpis=kpis,
            historical=historical,
            forecast=forecast,
            forecast_horizon=horizon,
            forecast_model=model_name,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Dashboard data generation failed: {exc}",
        ) from exc
