from fastapi.testclient import TestClient


def test_create_product(client: TestClient):
    response = client.post(
        "/api/products",
        json={
            "product_id": "TEST-P001",
            "product_name": "Test Laptop",
            "category_id": "CAT-001",
            "category_name": "Electronics",
            "unit_price": 55000.0,
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["product_id"] == "TEST-P001"
    assert data["product_name"] == "Test Laptop"
    assert data["category_id"] == "CAT-001"
    assert data["category_name"] == "Electronics"
    assert data["unit_price"] == 55000.0
    assert "id" in data
    assert "created_at" in data


def test_create_duplicate_product(client: TestClient):
    payload = {
        "product_id": "TEST-DUP-001",
        "product_name": "Duplicate Test Product",
        "category_id": "CAT-001",
        "category_name": "Electronics",
        "unit_price": 1000.0,
    }

    first_response = client.post(
        "/api/products",
        json=payload,
    )

    assert first_response.status_code == 201

    second_response = client.post(
        "/api/products",
        json=payload,
    )

    assert second_response.status_code == 409
    assert "already exists" in second_response.json()["detail"].lower()


def test_get_products(client: TestClient):
    first_create = client.post(
        "/api/products",
        json={
            "product_id": "TEST-GET-001",
            "product_name": "Product One",
            "category_id": "CAT-001",
            "category_name": "Electronics",
            "unit_price": 1000.0,
        },
    )

    assert first_create.status_code == 201

    second_create = client.post(
        "/api/products",
        json={
            "product_id": "TEST-GET-002",
            "product_name": "Product Two",
            "category_id": "CAT-002",
            "category_name": "Office",
            "unit_price": 2000.0,
        },
    )

    assert second_create.status_code == 201

    response = client.get("/api/products")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)
    assert len(data) >= 2

    product_ids = [
        product["product_id"]
        for product in data
    ]

    assert "TEST-GET-001" in product_ids
    assert "TEST-GET-002" in product_ids


def test_get_product_by_id(client: TestClient):
    create_response = client.post(
        "/api/products",
        json={
            "product_id": "TEST-BY-ID-001",
            "product_name": "Find Me Product",
            "category_id": "CAT-001",
            "category_name": "Electronics",
            "unit_price": 3000.0,
        },
    )

    assert create_response.status_code == 201

    response = client.get(
        "/api/products/TEST-BY-ID-001"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["product_id"] == "TEST-BY-ID-001"
    assert data["product_name"] == "Find Me Product"
    assert data["unit_price"] == 3000.0


def test_get_nonexistent_product(client: TestClient):
    response = client.get(
        "/api/products/DOES-NOT-EXIST"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Product not found."


def test_update_product(client: TestClient):
    create_response = client.post(
        "/api/products",
        json={
            "product_id": "TEST-UPDATE-001",
            "product_name": "Original Product",
            "category_id": "CAT-001",
            "category_name": "Electronics",
            "unit_price": 1000.0,
        },
    )

    assert create_response.status_code == 201

    update_response = client.put(
        "/api/products/TEST-UPDATE-001",
        json={
            "product_name": "Updated Product",
            "category_id": "CAT-002",
            "category_name": "Updated Category",
            "unit_price": 1500.0,
        },
    )

    assert update_response.status_code == 200

    data = update_response.json()

    assert data["product_id"] == "TEST-UPDATE-001"
    assert data["product_name"] == "Updated Product"
    assert data["category_id"] == "CAT-002"
    assert data["category_name"] == "Updated Category"
    assert data["unit_price"] == 1500.0


def test_delete_product(client: TestClient):
    create_response = client.post(
        "/api/products",
        json={
            "product_id": "TEST-DELETE-001",
            "product_name": "Delete Me",
            "category_id": "CAT-001",
            "category_name": "Electronics",
            "unit_price": 500.0,
        },
    )

    assert create_response.status_code == 201

    delete_response = client.delete(
        "/api/products/TEST-DELETE-001"
    )

    assert delete_response.status_code == 204
    assert delete_response.content == b""

    get_response = client.get(
        "/api/products/TEST-DELETE-001"
    )

    assert get_response.status_code == 404


def test_invalid_product_data(client: TestClient):
    response = client.post(
        "/api/products",
        json={
            "product_id": "",
            "product_name": "",
            "unit_price": -100,
        },
    )

    assert response.status_code == 422

