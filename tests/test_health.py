from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Smart Sales Forecasting API is running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()
    assert data.get("status") in ["ok", "healthy"]
