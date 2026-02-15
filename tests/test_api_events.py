"""Tests for event calendar API endpoints."""

from datetime import UTC, datetime, timedelta

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_events(client: AsyncClient) -> None:
    """Test listing events."""
    # Create a test user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "events@example.com",
            "username": "eventsuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "events@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # List events (should be empty initially)
    response = await client.get(
        "/api/v1/events",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["total"] == 0
    assert data["events"] == []


@pytest.mark.asyncio
async def test_create_event(client: AsyncClient) -> None:
    """Test creating a new event."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "create@example.com",
            "username": "createuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "create@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Create an event
    event_data = {
        "symbol": "AAPL",
        "event_type": "earnings",
        "event_date": (datetime.now(UTC) + timedelta(days=7)).isoformat(),
        "importance": "high",
        "title": "Apple Q4 Earnings",
        "description": "Quarterly earnings release for Apple Inc.",
        "expected_value": 1.25,
        "source": "test",
    }

    response = await client.post(
        "/api/v1/events",
        headers={"Authorization": f"Bearer {token}"},
        json=event_data,
    )
    assert response.status_code == 201
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert data["event_type"] == "earnings"
    assert data["importance"] == "high"
    assert data["title"] == "Apple Q4 Earnings"
    assert data["expected_value"] == 1.25
    assert "id" in data
    assert "created_at" in data


@pytest.mark.asyncio
async def test_get_event_by_id(client: AsyncClient) -> None:
    """Test retrieving a specific event by ID."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "get@example.com",
            "username": "getuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "get@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Create an event
    event_data = {
        "symbol": "TSLA",
        "event_type": "earnings",
        "event_date": (datetime.now(UTC) + timedelta(days=5)).isoformat(),
        "importance": "high",
        "title": "Tesla Q1 Earnings",
        "source": "test",
    }

    create_response = await client.post(
        "/api/v1/events",
        headers={"Authorization": f"Bearer {token}"},
        json=event_data,
    )
    event_id = create_response.json()["id"]

    # Get the event by ID
    response = await client.get(
        f"/api/v1/events/{event_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == event_id
    assert data["symbol"] == "TSLA"
    assert data["title"] == "Tesla Q1 Earnings"


@pytest.mark.asyncio
async def test_get_nonexistent_event(client: AsyncClient) -> None:
    """Test getting a nonexistent event returns 404."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "notfound@example.com",
            "username": "notfounduser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "notfound@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Try to get a nonexistent event
    response = await client.get(
        "/api/v1/events/99999",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_update_event(client: AsyncClient) -> None:
    """Test updating an event."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "update@example.com",
            "username": "updateuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "update@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Create an event
    event_data = {
        "symbol": "GOOGL",
        "event_type": "earnings",
        "event_date": (datetime.now(UTC) + timedelta(days=10)).isoformat(),
        "importance": "medium",
        "title": "Google Q2 Earnings",
        "source": "test",
    }

    create_response = await client.post(
        "/api/v1/events",
        headers={"Authorization": f"Bearer {token}"},
        json=event_data,
    )
    event_id = create_response.json()["id"]

    # Update the event
    update_data = {
        "importance": "high",
        "title": "Google Q2 Earnings - Updated",
        "actual_value": 1.35,
    }

    response = await client.put(
        f"/api/v1/events/{event_id}",
        headers={"Authorization": f"Bearer {token}"},
        json=update_data,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == event_id
    assert data["importance"] == "high"
    assert data["title"] == "Google Q2 Earnings - Updated"
    assert data["actual_value"] == 1.35


@pytest.mark.asyncio
async def test_delete_event(client: AsyncClient) -> None:
    """Test deleting an event."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "delete@example.com",
            "username": "deleteuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "delete@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Create an event
    event_data = {
        "symbol": "MSFT",
        "event_type": "earnings",
        "event_date": (datetime.now(UTC) + timedelta(days=3)).isoformat(),
        "importance": "high",
        "title": "Microsoft Q3 Earnings",
        "source": "test",
    }

    create_response = await client.post(
        "/api/v1/events",
        headers={"Authorization": f"Bearer {token}"},
        json=event_data,
    )
    event_id = create_response.json()["id"]

    # Delete the event
    response = await client.delete(
        f"/api/v1/events/{event_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 204

    # Verify deletion
    get_response = await client.get(
        f"/api/v1/events/{event_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_list_events_with_filters(client: AsyncClient) -> None:
    """Test listing events with various filters."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "filter@example.com",
            "username": "filteruser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "filter@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Create multiple events
    events = [
        {
            "symbol": "AAPL",
            "event_type": "earnings",
            "event_date": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            "importance": "high",
            "title": "Apple Earnings",
            "source": "test",
        },
        {
            "symbol": "AAPL",
            "event_type": "dividend",
            "event_date": (datetime.now(UTC) + timedelta(days=2)).isoformat(),
            "importance": "medium",
            "title": "Apple Dividend",
            "source": "test",
        },
        {
            "symbol": "TSLA",
            "event_type": "earnings",
            "event_date": (datetime.now(UTC) + timedelta(days=3)).isoformat(),
            "importance": "high",
            "title": "Tesla Earnings",
            "source": "test",
        },
    ]

    for event_data in events:
        await client.post(
            "/api/v1/events",
            headers={"Authorization": f"Bearer {token}"},
            json=event_data,
        )

    # Filter by symbol
    response = await client.get(
        "/api/v1/events?symbol=AAPL",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert all(e["symbol"] == "AAPL" for e in data["events"])

    # Filter by event type
    response = await client.get(
        "/api/v1/events?event_type=earnings",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert all(e["event_type"] == "earnings" for e in data["events"])

    # Filter by importance
    response = await client.get(
        "/api/v1/events?importance=high",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert all(e["importance"] == "high" for e in data["events"])


@pytest.mark.asyncio
async def test_get_earnings_for_symbol(client: AsyncClient) -> None:
    """Test getting earnings calendar for a specific symbol."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "earnings@example.com",
            "username": "earningsuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "earnings@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Create earnings events
    event_data = {
        "symbol": "NVDA",
        "event_type": "earnings",
        "event_date": (datetime.now(UTC) + timedelta(days=15)).isoformat(),
        "importance": "high",
        "title": "NVIDIA Q1 Earnings",
        "source": "test",
        "metadata_json": {
            "eps_estimate": 2.50,
            "revenue_estimate": 10000000000,
        },
    }

    await client.post(
        "/api/v1/events",
        headers={"Authorization": f"Bearer {token}"},
        json=event_data,
    )

    # Get earnings for symbol
    response = await client.get(
        "/api/v1/events/earnings/NVDA",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert data[0]["symbol"] == "NVDA"


@pytest.mark.asyncio
async def test_get_upcoming_earnings(client: AsyncClient) -> None:
    """Test getting upcoming earnings events."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "upcoming@example.com",
            "username": "upcominguser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "upcoming@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Get upcoming earnings
    response = await client.get(
        "/api/v1/events/earnings/upcoming?days=7",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_get_fomc_schedule(client: AsyncClient) -> None:
    """Test getting FOMC meeting schedule."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "fomc@example.com",
            "username": "fomcuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "fomc@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Get FOMC schedule
    response = await client.get(
        "/api/v1/events/fed/schedule?year=2025",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert all("event_date" in event for event in data)


@pytest.mark.asyncio
async def test_get_economic_releases(client: AsyncClient) -> None:
    """Test getting economic releases calendar."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "economic@example.com",
            "username": "economicuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "economic@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Create economic events
    event_data = {
        "symbol": None,
        "event_type": "cpi",
        "event_date": (datetime.now(UTC) + timedelta(days=5)).isoformat(),
        "importance": "critical",
        "title": "CPI Release",
        "source": "test",
        "metadata_json": {
            "indicator_name": "CPI",
            "forecast": 3.5,
        },
    }

    await client.post(
        "/api/v1/events",
        headers={"Authorization": f"Bearer {token}"},
        json=event_data,
    )

    # Get economic releases
    response = await client.get(
        "/api/v1/events/economic?days=14",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_authentication_required(client: AsyncClient) -> None:
    """Test that endpoints require authentication."""
    # Try to access without token
    response = await client.get("/api/v1/events")
    assert response.status_code == 401

    response = await client.get("/api/v1/events/1")
    assert response.status_code == 401

    response = await client.post(
        "/api/v1/events",
        json={
            "symbol": "TEST",
            "event_type": "earnings",
            "event_date": datetime.now(UTC).isoformat(),
            "importance": "high",
            "title": "Test",
            "source": "test",
        },
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_pagination(client: AsyncClient) -> None:
    """Test event pagination."""
    # Create user and get token
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "pagination@example.com",
            "username": "paginationuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "pagination@example.com",
            "password": "testpassword123",
        },
    )
    token = login_response.json()["access_token"]

    # Create multiple events
    for i in range(15):
        event_data = {
            "symbol": "TEST",
            "event_type": "earnings",
            "event_date": (datetime.now(UTC) + timedelta(days=i)).isoformat(),
            "importance": "medium",
            "title": f"Test Event {i}",
            "source": "test",
        }
        await client.post(
            "/api/v1/events",
            headers={"Authorization": f"Bearer {token}"},
            json=event_data,
        )

    # Test first page
    response = await client.get(
        "/api/v1/events?limit=10&offset=0",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 10
    assert data["total"] == 15
    assert data["limit"] == 10
    assert data["offset"] == 0

    # Test second page
    response = await client.get(
        "/api/v1/events?limit=10&offset=10",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 5
    assert data["total"] == 15
