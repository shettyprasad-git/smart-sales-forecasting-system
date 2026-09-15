from unittest.mock import patch

import pandas as pd


# ---------------------------------------------------------
# Helper data
# ---------------------------------------------------------

def mock_forecast_dataframe(horizon: int):
    dates = pd.date_range(
        start="2026-01-01",
        periods=horizon,
        freq="D",
    )

    return pd.DataFrame(
        {
            "Date": dates,
            "Predicted_Quantity": [5000.0 + i for i in range(horizon)],
        }
    )


def mock_history_dataframe(limit: int):
    dates = pd.date_range(
        start="2025-01-01",
        periods=limit,
        freq="D",
    )

    return pd.DataFrame(
        {
            "Date": dates,
            "Quantity": [5000.0 + i for i in range(limit)],
            "Sales_Amount": [100000.0 + i * 1000 for i in range(limit)],
            "Profit": [20000.0 + i * 500 for i in range(limit)],
        }
    )


# ---------------------------------------------------------
# POST /api/forecast
# ---------------------------------------------------------

def test_generate_forecast_7_days(client):
    mock_df = mock_forecast_dataframe(7)

    with patch(
        "backend.app.api.forecast.forecast_service.generate_forecast",
        return_value=("Random Forest", mock_df),
    ):
        response = client.post(
            "/api/forecast",
            json={"horizon": 7},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["horizon"] == 7
    assert data["model"] == "Random Forest"
    assert len(data["forecast"]) == 7

    assert "date" in data["forecast"][0]
    assert "predicted_quantity" in data["forecast"][0]

    assert data["forecast"][0]["predicted_quantity"] == 5000.0


def test_generate_forecast_30_days(client):
    mock_df = mock_forecast_dataframe(30)

    with patch(
        "backend.app.api.forecast.forecast_service.generate_forecast",
        return_value=("Gradient Boosting", mock_df),
    ):
        response = client.post(
            "/api/forecast",
            json={"horizon": 30},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["horizon"] == 30
    assert data["model"] == "Gradient Boosting"
    assert len(data["forecast"]) == 30


def test_generate_forecast_90_days(client):
    mock_df = mock_forecast_dataframe(90)

    with patch(
        "backend.app.api.forecast.forecast_service.generate_forecast",
        return_value=("Linear Regression", mock_df),
    ):
        response = client.post(
            "/api/forecast",
            json={"horizon": 90},
        )

    assert response.status_code == 200

    data = response.json()

    assert data["horizon"] == 90
    assert data["model"] == "Linear Regression"
    assert len(data["forecast"]) == 90


def test_generate_forecast_invalid_horizon(client):
    response = client.post(
        "/api/forecast",
        json={"horizon": 14},
    )

    assert response.status_code == 400

    data = response.json()

    assert (
        data["detail"]
        == "Supported forecast horizons are 7, 30, and 90 days."
    )


def test_generate_forecast_missing_horizon(client):
    response = client.post(
        "/api/forecast",
        json={},
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# GET /api/forecast/history
# ---------------------------------------------------------

def test_get_forecast_history_default(client):
    mock_df = mock_history_dataframe(365)

    with patch(
        "backend.app.api.forecast.history_service.get_history",
        return_value=mock_df,
    ):
        response = client.get(
            "/api/forecast/history"
        )

    assert response.status_code == 200

    data = response.json()

    assert "records" in data
    assert len(data["records"]) == 365

    record = data["records"][0]

    assert "date" in record
    assert "quantity" in record
    assert "sales_amount" in record
    assert "profit" in record


def test_get_forecast_history_custom_limit(client):
    mock_df = mock_history_dataframe(30)

    with patch(
        "backend.app.api.forecast.history_service.get_history",
        return_value=mock_df,
    ):
        response = client.get(
            "/api/forecast/history?limit=30"
        )

    assert response.status_code == 200

    data = response.json()

    assert len(data["records"]) == 30


def test_get_forecast_history_invalid_limit(client):
    response = client.get(
        "/api/forecast/history?limit=0"
    )

    assert response.status_code == 422


def test_get_forecast_history_limit_too_large(client):
    response = client.get(
        "/api/forecast/history?limit=2923"
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# GET /api/forecast/dashboard
# ---------------------------------------------------------

def test_get_dashboard_default(client):
    mock_history = mock_history_dataframe(365)
    mock_forecast = mock_forecast_dataframe(7)

    with patch(
        "backend.app.api.forecast.history_service.get_history",
        return_value=mock_history,
    ), patch(
        "backend.app.api.forecast.forecast_service.generate_forecast",
        return_value=("Random Forest", mock_forecast),
    ):
        response = client.get(
            "/api/forecast/dashboard"
        )

    assert response.status_code == 200

    data = response.json()

    assert "kpis" in data
    assert "historical" in data
    assert "forecast" in data

    assert data["forecast_horizon"] == 7
    assert data["forecast_model"] == "Random Forest"

    assert len(data["historical"]) == 365
    assert len(data["forecast"]) == 7


def test_get_dashboard_30_days(client):
    mock_history = mock_history_dataframe(30)
    mock_forecast = mock_forecast_dataframe(30)

    with patch(
        "backend.app.api.forecast.history_service.get_history",
        return_value=mock_history,
    ), patch(
        "backend.app.api.forecast.forecast_service.generate_forecast",
        return_value=("Gradient Boosting", mock_forecast),
    ):
        response = client.get(
            "/api/forecast/dashboard?horizon=30&history_limit=30"
        )

    assert response.status_code == 200

    data = response.json()

    assert data["forecast_horizon"] == 30
    assert data["forecast_model"] == "Gradient Boosting"

    assert len(data["historical"]) == 30
    assert len(data["forecast"]) == 30


def test_get_dashboard_90_days(client):
    mock_history = mock_history_dataframe(90)
    mock_forecast = mock_forecast_dataframe(90)

    with patch(
        "backend.app.api.forecast.history_service.get_history",
        return_value=mock_history,
    ), patch(
        "backend.app.api.forecast.forecast_service.generate_forecast",
        return_value=("Linear Regression", mock_forecast),
    ):
        response = client.get(
            "/api/forecast/dashboard?horizon=90&history_limit=90"
        )

    assert response.status_code == 200

    data = response.json()

    assert data["forecast_horizon"] == 90
    assert data["forecast_model"] == "Linear Regression"

    assert len(data["historical"]) == 90
    assert len(data["forecast"]) == 90


def test_get_dashboard_invalid_horizon(client):
    response = client.get(
        "/api/forecast/dashboard?horizon=14"
    )

    assert response.status_code == 400

    data = response.json()

    assert (
        data["detail"]
        == "Supported forecast horizons are 7, 30, and 90 days."
    )


def test_get_dashboard_invalid_history_limit(client):
    response = client.get(
        "/api/forecast/dashboard?history_limit=0"
    )

    assert response.status_code == 422


# ---------------------------------------------------------
# KPI validation
# ---------------------------------------------------------

def test_get_dashboard_kpis(client):
    mock_history = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                ]
            ),
            "Quantity": [100.0, 200.0, 300.0],
            "Sales_Amount": [1000.0, 2000.0, 3000.0],
            "Profit": [100.0, 200.0, 300.0],
        }
    )

    mock_forecast = mock_forecast_dataframe(7)

    with patch(
        "backend.app.api.forecast.history_service.get_history",
        return_value=mock_history,
    ), patch(
        "backend.app.api.forecast.forecast_service.generate_forecast",
        return_value=("Random Forest", mock_forecast),
    ):
        response = client.get(
            "/api/forecast/dashboard?history_limit=3"
        )

    assert response.status_code == 200

    data = response.json()

    kpis = data["kpis"]

    assert kpis["total_historical_quantity"] == 600.0
    assert kpis["total_historical_sales"] == 6000.0
    assert kpis["total_historical_profit"] == 600.0
    assert kpis["average_daily_quantity"] == 200.0


# ---------------------------------------------------------
# Error handling
# ---------------------------------------------------------

def test_generate_forecast_file_not_found(client):
    with patch(
        "backend.app.api.forecast.forecast_service.generate_forecast",
        side_effect=FileNotFoundError("Model file not found."),
    ):
        response = client.post(
            "/api/forecast",
            json={"horizon": 7},
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Model file not found."


def test_generate_forecast_value_error(client):
    with patch(
        "backend.app.api.forecast.forecast_service.generate_forecast",
        side_effect=ValueError("Invalid forecasting data."),
    ):
        response = client.post(
            "/api/forecast",
            json={"horizon": 7},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid forecasting data."


def test_get_history_file_not_found(client):
    with patch(
        "backend.app.api.forecast.history_service.get_history",
        side_effect=FileNotFoundError("Dataset not found."),
    ):
        response = client.get(
            "/api/forecast/history?limit=30"
        )

    assert response.status_code == 500
    assert response.json()["detail"] == "Dataset not found."