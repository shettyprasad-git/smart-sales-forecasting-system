def test_register_user(client):

    response = client.post(
        "/api/auth/register",
        json={
            "email": "pytest_user@example.com",
            "password": "TestPassword123!",
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["email"] == "pytest_user@example.com"
    assert "id" in data
    assert "created_at" in data
    assert "password_hash" not in data


def test_duplicate_registration(client):

    client.post(
        "/api/auth/register",
        json={
            "email": "pytest_duplicate@example.com",
            "password": "TestPassword123!",
        },
    )

    response = client.post(
        "/api/auth/register",
        json={
            "email": "pytest_duplicate@example.com",
            "password": "TestPassword123!",
        },
    )

    assert response.status_code == 409

    assert response.json()["detail"] == (
        "An account with this email already exists."
    )


def test_login(client):

    client.post(
        "/api/auth/register",
        json={
            "email": "pytest_login@example.com",
            "password": "TestPassword123!",
        },
    )

    response = client.post(
        "/api/auth/login",
        data={
            "username": "pytest_login@example.com",
            "password": "TestPassword123!",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["email"] == (
        "pytest_login@example.com"
    )


def test_invalid_login(client):

    client.post(
        "/api/auth/register",
        json={
            "email": "pytest_invalid@example.com",
            "password": "TestPassword123!",
        },
    )

    response = client.post(
        "/api/auth/login",
        data={
            "username": "pytest_invalid@example.com",
            "password": "WrongPassword123!",
        },
    )

    assert response.status_code == 401

    assert response.json()["detail"] == (
        "Incorrect email or password."
    )


def test_me_without_authentication(client):

    response = client.get(
        "/api/auth/me"
    )

    assert response.status_code == 401


def test_me_with_authentication(client):

    client.post(
        "/api/auth/register",
        json={
            "email": "pytest_me@example.com",
            "password": "TestPassword123!",
        },
    )

    login_response = client.post(
        "/api/auth/login",
        data={
            "username": "pytest_me@example.com",
            "password": "TestPassword123!",
        },
    )

    assert login_response.status_code == 200

    token = login_response.json()["access_token"]

    response = client.get(
        "/api/auth/me",
        headers={
            "Authorization": f"Bearer {token}"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["email"] == "pytest_me@example.com"
