from datetime import date

from fastapi.testclient import TestClient


def register_and_login(
    client: TestClient,
    email: str,
    password: str = "TestPassword123!",
):
    response = client.post(
        "/api/auth/register",
        json={
            "email": email,
            "password": password,
        },
    )

    assert response.status_code == 201

    login_response = client.post(
        "/api/auth/login",
        data={
            "username": email,
            "password": password,
        },
    )

    assert login_response.status_code == 200

    token = login_response.json()["access_token"]

    return {
        "Authorization": f"Bearer {token}",
    }


def create_product(
    client: TestClient,
    product_id: str = "TEST-SALES-P001",
):
    response = client.post(
        "/api/products",
        json={
            "product_id": product_id,
            "product_name": "Sales Test Product",
            "category_id": "CAT-SALES",
            "category_name": "Sales Testing",
            "unit_price": 1000.0,
        },
    )

    assert response.status_code == 201

    return response.json()


def create_sale(
    client: TestClient,
    headers: dict,
    product_database_id: int,
):
    response = client.post(
        "/api/sales",
        headers=headers,
        json={
            "product_id": product_database_id,
            "sale_date": "2025-01-15",
            "quantity": 5,
            "unit_price": 1000.0,
            "discount_percent": 10.0,
            "promotion": True,
            "holiday_flag": False,
            "sales_amount": 4500.0,
            "profit": 1000.0,
        },
    )

    assert response.status_code == 201

    return response.json()


def test_create_sale(client: TestClient):
    headers = register_and_login(
        client,
        "sales-create@example.com",
    )

    product = create_product(
        client,
        "TEST-SALES-CREATE-001",
    )

    response = client.post(
        "/api/sales",
        headers=headers,
        json={
            "product_id": product["id"],
            "sale_date": "2025-01-15",
            "quantity": 5,
            "unit_price": 1000.0,
            "discount_percent": 10.0,
            "promotion": True,
            "holiday_flag": False,
            "sales_amount": 4500.0,
            "profit": 1000.0,
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["user_id"] > 0
    assert data["product_id"] == product["id"]
    assert data["sale_date"] == "2025-01-15"
    assert data["quantity"] == 5
    assert data["unit_price"] == 1000.0
    assert data["discount_percent"] == 10.0
    assert data["promotion"] is True
    assert data["holiday_flag"] is False
    assert data["sales_amount"] == 4500.0
    assert data["profit"] == 1000.0
    assert "id" in data
    assert "created_at" in data


def test_create_sale_without_authentication(client: TestClient):
    product = create_product(
        client,
        "TEST-SALES-NOAUTH-001",
    )

    response = client.post(
        "/api/sales",
        json={
            "product_id": product["id"],
            "sale_date": "2025-01-15",
            "quantity": 5,
            "unit_price": 1000.0,
            "discount_percent": 0,
            "promotion": False,
            "holiday_flag": False,
            "sales_amount": 5000.0,
            "profit": 1000.0,
        },
    )

    assert response.status_code == 401


def test_get_sales(client: TestClient):
    headers = register_and_login(
        client,
        "sales-list@example.com",
    )

    product = create_product(
        client,
        "TEST-SALES-LIST-001",
    )

    create_sale(
        client,
        headers,
        product["id"],
    )

    response = client.get(
        "/api/sales",
        headers=headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)
    assert len(data) >= 1

    sales_ids = [
        sale["product_id"]
        for sale in data
    ]

    assert product["id"] in sales_ids


def test_get_sale_by_id(client: TestClient):
    headers = register_and_login(
        client,
        "sales-get@example.com",
    )

    product = create_product(
        client,
        "TEST-SALES-GET-001",
    )

    sale = create_sale(
        client,
        headers,
        product["id"],
    )

    response = client.get(
        f"/api/sales/{sale['id']}",
        headers=headers,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == sale["id"]
    assert data["product_id"] == product["id"]
    assert data["quantity"] == 5


def test_get_nonexistent_sale(client: TestClient):
    headers = register_and_login(
        client,
        "sales-missing@example.com",
    )

    response = client.get(
        "/api/sales/999999",
        headers=headers,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Sales record not found."


def test_update_sale(client: TestClient):
    headers = register_and_login(
        client,
        "sales-update@example.com",
    )

    product = create_product(
        client,
        "TEST-SALES-UPDATE-001",
    )

    sale = create_sale(
        client,
        headers,
        product["id"],
    )

    response = client.put(
        f"/api/sales/{sale['id']}",
        headers=headers,
        json={
            "quantity": 10,
            "unit_price": 1200.0,
            "discount_percent": 5.0,
            "promotion": False,
            "sales_amount": 11400.0,
            "profit": 2500.0,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == sale["id"]
    assert data["quantity"] == 10
    assert data["unit_price"] == 1200.0
    assert data["discount_percent"] == 5.0
    assert data["promotion"] is False
    assert data["sales_amount"] == 11400.0
    assert data["profit"] == 2500.0


def test_update_sale_with_invalid_product(client: TestClient):
    headers = register_and_login(
        client,
        "sales-invalid-product@example.com",
    )

    product = create_product(
        client,
        "TEST-SALES-INVALID-PRODUCT-001",
    )

    sale = create_sale(
        client,
        headers,
        product["id"],
    )

    response = client.put(
        f"/api/sales/{sale['id']}",
        headers=headers,
        json={
            "product_id": 999999,
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Product not found."


def test_delete_sale(client: TestClient):
    headers = register_and_login(
        client,
        "sales-delete@example.com",
    )

    product = create_product(
        client,
        "TEST-SALES-DELETE-001",
    )

    sale = create_sale(
        client,
        headers,
        product["id"],
    )

    delete_response = client.delete(
        f"/api/sales/{sale['id']}",
        headers=headers,
    )

    assert delete_response.status_code == 204
    assert delete_response.content == b""

    get_response = client.get(
        f"/api/sales/{sale['id']}",
        headers=headers,
    )

    assert get_response.status_code == 404


def test_sales_user_isolation(client: TestClient):
    user_a_headers = register_and_login(
        client,
        "sales-user-a@example.com",
    )

    user_b_headers = register_and_login(
        client,
        "sales-user-b@example.com",
    )

    product = create_product(
        client,
        "TEST-SALES-ISOLATION-001",
    )

    sale = create_sale(
        client,
        user_a_headers,
        product["id"],
    )

    response = client.get(
        f"/api/sales/{sale['id']}",
        headers=user_b_headers,
    )

    assert response.status_code == 404

    list_response = client.get(
        "/api/sales",
        headers=user_b_headers,
    )

    assert list_response.status_code == 200

    data = list_response.json()

    assert all(
        item["id"] != sale["id"]
        for item in data
    )


def test_invalid_sale_data(client: TestClient):
    headers = register_and_login(
        client,
        "sales-invalid-data@example.com",
    )

    response = client.post(
        "/api/sales",
        headers=headers,
        json={
            "product_id": -1,
            "sale_date": "invalid-date",
            "quantity": -5,
            "unit_price": -100,
            "discount_percent": 150,
            "sales_amount": -500,
        },
    )

    assert response.status_code == 422
