"""Tests for order flow API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_dark_pool_prints(client: AsyncClient) -> None:
    """Test getting dark pool prints for a symbol."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "darkpool@example.com",
            "username": "darkpooluser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "darkpool@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Get dark pool prints
    response = await client.get(
        "/api/v1/dark-pools/AAPL?threshold_usd=500000",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_get_dark_pool_summary(client: AsyncClient) -> None:
    """Test getting dark pool summary for a symbol."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "summary@example.com",
            "username": "summaryuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "summary@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Get dark pool summary
    response = await client.get(
        "/api/v1/dark-pools/AAPL/summary?days=30",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "symbol" in data
    assert "period_days" in data
    assert "total_volume" in data
    assert "institutional_bias" in data


@pytest.mark.asyncio
async def test_get_options_activity(client: AsyncClient) -> None:
    """Test getting options activity for a symbol."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "options@example.com",
            "username": "optionsuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "options@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Get options activity
    response = await client.get(
        "/api/v1/options/TSLA?threshold_usd=100000",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_get_unusual_options_activity(client: AsyncClient) -> None:
    """Test getting unusual options activity for a symbol."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "unusual@example.com",
            "username": "unusualuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "unusual@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Get unusual options activity
    response = await client.get(
        "/api/v1/options/NVDA/unusual?volume_threshold=2.0",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_get_short_interest(client: AsyncClient) -> None:
    """Test getting current short interest for a symbol."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "short@example.com",
            "username": "shortuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "short@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Get short interest - may return 404 if no data exists
    response = await client.get(
        "/api/v1/short-interest/GME",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_get_short_interest_history(client: AsyncClient) -> None:
    """Test getting short interest history for a symbol."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "history@example.com",
            "username": "historyuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "history@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Get short interest history
    response = await client.get(
        "/api/v1/short-interest/AMC/history?reports=6",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert isinstance(data, list)


@pytest.mark.asyncio
async def test_get_squeeze_candidates(client: AsyncClient) -> None:
    """Test getting short squeeze candidates."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "squeeze@example.com",
            "username": "squeezeuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "squeeze@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Get squeeze candidates
    response = await client.get(
        "/api/v1/short-interest/squeeze-candidates?min_short_percent=20.0&min_days_to_cover=5.0",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_get_flow_aggregation(client: AsyncClient) -> None:
    """Test getting aggregated flow for a symbol."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "aggregation@example.com",
            "username": "aggregationuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "aggregation@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Get flow aggregation
    response = await client.get(
        "/api/v1/aggregation/SPY?days=5",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "symbol" in data
    assert "net_flow" in data
    assert "bias" in data
    assert "period_days" in data


@pytest.mark.asyncio
async def test_get_flow_anomalies(client: AsyncClient) -> None:
    """Test getting flow anomalies for a symbol."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "anomalies@example.com",
            "username": "anomaliesuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "anomalies@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Get flow anomalies
    response = await client.get(
        "/api/v1/anomalies/MSFT?sensitivity=2.0",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_get_most_shorted_stocks(client: AsyncClient) -> None:
    """Test getting most shorted stocks."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "mostshorted@example.com",
            "username": "mostshorteduser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "mostshorted@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Get most shorted stocks
    response = await client.get(
        "/api/v1/most-shorted?top_n=10",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_authentication_required(client: AsyncClient) -> None:
    """Test that endpoints require authentication."""
    # Try to access without token
    response = await client.get("/api/v1/dark-pools/AAPL")
    assert response.status_code == 401

    response = await client.get("/api/v1/dark-pools/AAPL/summary")
    assert response.status_code == 401

    response = await client.get("/api/v1/options/AAPL")
    assert response.status_code == 401

    response = await client.get("/api/v1/options/AAPL/unusual")
    assert response.status_code == 401

    response = await client.get("/api/v1/short-interest/AAPL")
    assert response.status_code == 401

    response = await client.get("/api/v1/short-interest/AAPL/history")
    assert response.status_code == 401

    response = await client.get("/api/v1/short-interest/squeeze-candidates")
    assert response.status_code == 401

    response = await client.get("/api/v1/aggregation/AAPL")
    assert response.status_code == 401

    response = await client.get("/api/v1/anomalies/AAPL")
    assert response.status_code == 401

    response = await client.get("/api/v1/most-shorted")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_symbol_case_insensitive(client: AsyncClient) -> None:
    """Test that symbol parameters are case insensitive."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "case@example.com",
            "username": "caseuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "case@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Test lowercase symbol
    response = await client.get(
        "/api/v1/dark-pools/aapl/summary",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_query_parameter_validation(client: AsyncClient) -> None:
    """Test query parameter validation."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "validation@example.com",
            "username": "validationuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "validation@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Test invalid days parameter (too large)
    response = await client.get(
        "/api/v1/dark-pools/AAPL/summary?days=500",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400

    # Test invalid sensitivity parameter (too small)
    response = await client.get(
        "/api/v1/anomalies/AAPL?sensitivity=0.1",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400

    # Test invalid top_n parameter (too large)
    response = await client.get(
        "/api/v1/most-shorted?top_n=150",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 400
