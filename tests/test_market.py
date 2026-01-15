"""Tests for market data endpoints."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.price import Price


async def create_test_user_and_get_token(client: AsyncClient) -> str:
    """Helper to create a test user and return access token."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "market_test@example.com",
            "username": "marketuser",
            "password": "testpassword123",
        },
    )
    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "market_test@example.com",
            "password": "testpassword123",
        },
    )
    return login_response.json()["access_token"]


async def insert_test_prices(db_session: AsyncSession, symbol: str = "AAPL") -> None:
    """Insert test price data into the database."""
    base_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)

    for i in range(5):
        price = Price(
            symbol=symbol,
            timestamp=datetime(
                base_time.year,
                base_time.month,
                base_time.day + i,
                base_time.hour,
                base_time.minute,
                tzinfo=timezone.utc,
            ),
            open=Decimal("150.00") + Decimal(str(i)),
            high=Decimal("155.00") + Decimal(str(i)),
            low=Decimal("148.00") + Decimal(str(i)),
            close=Decimal("153.00") + Decimal(str(i)),
            volume=1000000 + (i * 100000),
            adj_close=Decimal("152.50") + Decimal(str(i)),
        )
        db_session.add(price)

    await db_session.commit()


@pytest.mark.asyncio
async def test_get_price_history_requires_auth(client: AsyncClient) -> None:
    """Test that price history endpoint requires authentication."""
    response = await client.get("/api/v1/market/prices/AAPL")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_price_history_no_data(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test getting price history for a symbol with no data."""
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/market/prices/NONEXISTENT",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 404
    assert "No price data found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_price_history_success(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test successful retrieval of price history."""
    await insert_test_prices(db_session, "AAPL")
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/market/prices/AAPL",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    assert data["count"] == 5
    assert len(data["data"]) == 5

    first_price = data["data"][0]
    assert "timestamp" in first_price
    assert "open" in first_price
    assert "high" in first_price
    assert "low" in first_price
    assert "close" in first_price
    assert "volume" in first_price


@pytest.mark.asyncio
async def test_get_price_history_case_insensitive(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test that symbol lookup is case insensitive."""
    await insert_test_prices(db_session, "MSFT")
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/market/prices/msft",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["symbol"] == "MSFT"


@pytest.mark.asyncio
async def test_get_price_history_with_limit(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test price history with limit parameter."""
    await insert_test_prices(db_session, "GOOGL")
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/market/prices/GOOGL?limit=2",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["data"]) == 2


@pytest.mark.asyncio
async def test_get_price_history_limit_validation(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test that limit parameter is validated."""
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/market/prices/AAPL?limit=0",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422

    response = await client.get(
        "/api/v1/market/prices/AAPL?limit=1001",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_available_symbols_requires_auth(client: AsyncClient) -> None:
    """Test that symbols endpoint requires authentication."""
    response = await client.get("/api/v1/market/symbols")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_available_symbols_empty(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test getting symbols when no data exists."""
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/market/symbols",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["symbols"] == []
    assert data["count"] == 0


@pytest.mark.asyncio
async def test_get_available_symbols_success(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test successful retrieval of available symbols."""
    await insert_test_prices(db_session, "AAPL")
    await insert_test_prices(db_session, "MSFT")
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/market/symbols",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2

    symbols = [s["symbol"] for s in data["symbols"]]
    assert "AAPL" in symbols
    assert "MSFT" in symbols

    for symbol_info in data["symbols"]:
        assert symbol_info["data_available"] is True
        assert symbol_info["first_date"] is not None
        assert symbol_info["last_date"] is not None
