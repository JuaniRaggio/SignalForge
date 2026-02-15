"""Tests for user preferences and activity endpoints."""

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.user import (
    ExperienceLevel,
    InvestmentHorizon,
    RiskTolerance,
    User,
)
from signalforge.models.user_activity import ActivityType, UserActivity


async def create_test_user_and_login(client: AsyncClient) -> dict[str, str]:
    """Helper function to create a user and return auth headers."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "testuser@example.com",
            "username": "testuser",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "testuser@example.com",
            "password": "testpassword123",
        },
    )

    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_get_user_preferences_default_values(client: AsyncClient) -> None:
    """Test getting user preferences returns default values."""
    headers = await create_test_user_and_login(client)

    response = await client.get("/api/v1/users/me/preferences", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["risk_tolerance"] == "medium"
    assert data["investment_horizon"] == "medium"
    assert data["experience_level"] == "informed"
    assert data["preferred_sectors"] == []
    assert data["watchlist"] == []
    assert data["notification_preferences"] == {}


@pytest.mark.asyncio
async def test_get_user_preferences_unauthorized(client: AsyncClient) -> None:
    """Test getting preferences without authentication fails."""
    response = await client.get("/api/v1/users/me/preferences")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_update_risk_tolerance(client: AsyncClient) -> None:
    """Test updating user risk tolerance."""
    headers = await create_test_user_and_login(client)

    response = await client.put(
        "/api/v1/users/me/preferences",
        headers=headers,
        json={"risk_tolerance": "high"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["risk_tolerance"] == "high"


@pytest.mark.asyncio
async def test_update_investment_horizon(client: AsyncClient) -> None:
    """Test updating user investment horizon."""
    headers = await create_test_user_and_login(client)

    response = await client.put(
        "/api/v1/users/me/preferences",
        headers=headers,
        json={"investment_horizon": "long"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["investment_horizon"] == "long"


@pytest.mark.asyncio
async def test_update_experience_level(client: AsyncClient) -> None:
    """Test updating user experience level."""
    headers = await create_test_user_and_login(client)

    response = await client.put(
        "/api/v1/users/me/preferences",
        headers=headers,
        json={"experience_level": "quant"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["experience_level"] == "quant"


@pytest.mark.asyncio
async def test_update_preferred_sectors(client: AsyncClient) -> None:
    """Test updating preferred sectors."""
    headers = await create_test_user_and_login(client)

    sectors = ["Technology", "Healthcare", "Finance"]
    response = await client.put(
        "/api/v1/users/me/preferences",
        headers=headers,
        json={"preferred_sectors": sectors},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["preferred_sectors"] == sectors


@pytest.mark.asyncio
async def test_update_notification_preferences(client: AsyncClient) -> None:
    """Test updating notification preferences."""
    headers = await create_test_user_and_login(client)

    prefs = {
        "email": True,
        "push": False,
        "frequency": "daily",
        "signals": ["momentum", "volatility"],
    }
    response = await client.put(
        "/api/v1/users/me/preferences",
        headers=headers,
        json={"notification_preferences": prefs},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["notification_preferences"] == prefs


@pytest.mark.asyncio
async def test_update_multiple_preferences(client: AsyncClient) -> None:
    """Test updating multiple preferences at once."""
    headers = await create_test_user_and_login(client)

    response = await client.put(
        "/api/v1/users/me/preferences",
        headers=headers,
        json={
            "risk_tolerance": "low",
            "investment_horizon": "short",
            "experience_level": "casual",
            "preferred_sectors": ["Energy", "Utilities"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["risk_tolerance"] == "low"
    assert data["investment_horizon"] == "short"
    assert data["experience_level"] == "casual"
    assert data["preferred_sectors"] == ["Energy", "Utilities"]


@pytest.mark.asyncio
async def test_update_preferences_partial(client: AsyncClient) -> None:
    """Test partial update does not affect unspecified fields."""
    headers = await create_test_user_and_login(client)

    await client.put(
        "/api/v1/users/me/preferences",
        headers=headers,
        json={"risk_tolerance": "high"},
    )

    response = await client.put(
        "/api/v1/users/me/preferences",
        headers=headers,
        json={"preferred_sectors": ["Technology"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["risk_tolerance"] == "high"
    assert data["preferred_sectors"] == ["Technology"]


@pytest.mark.asyncio
async def test_add_to_watchlist(client: AsyncClient) -> None:
    """Test adding symbols to watchlist."""
    headers = await create_test_user_and_login(client)

    response = await client.post(
        "/api/v1/users/me/watchlist",
        headers=headers,
        json={"symbols": ["AAPL", "MSFT", "GOOGL"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert set(data["watchlist"]) == {"AAPL", "GOOGL", "MSFT"}


@pytest.mark.asyncio
async def test_add_to_watchlist_normalizes_case(client: AsyncClient) -> None:
    """Test that watchlist symbols are normalized to uppercase."""
    headers = await create_test_user_and_login(client)

    response = await client.post(
        "/api/v1/users/me/watchlist",
        headers=headers,
        json={"symbols": ["aapl", "msft", "GoOgL"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert set(data["watchlist"]) == {"AAPL", "GOOGL", "MSFT"}


@pytest.mark.asyncio
async def test_add_to_watchlist_handles_duplicates(client: AsyncClient) -> None:
    """Test that adding duplicate symbols does not duplicate watchlist entries."""
    headers = await create_test_user_and_login(client)

    await client.post(
        "/api/v1/users/me/watchlist",
        headers=headers,
        json={"symbols": ["AAPL", "MSFT"]},
    )

    response = await client.post(
        "/api/v1/users/me/watchlist",
        headers=headers,
        json={"symbols": ["AAPL", "GOOGL"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert set(data["watchlist"]) == {"AAPL", "GOOGL", "MSFT"}


@pytest.mark.asyncio
async def test_add_to_watchlist_sorted(client: AsyncClient) -> None:
    """Test that watchlist is sorted alphabetically."""
    headers = await create_test_user_and_login(client)

    response = await client.post(
        "/api/v1/users/me/watchlist",
        headers=headers,
        json={"symbols": ["TSLA", "AAPL", "MSFT"]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["watchlist"] == ["AAPL", "MSFT", "TSLA"]


@pytest.mark.asyncio
async def test_remove_from_watchlist(client: AsyncClient) -> None:
    """Test removing a symbol from watchlist."""
    headers = await create_test_user_and_login(client)

    await client.post(
        "/api/v1/users/me/watchlist",
        headers=headers,
        json={"symbols": ["AAPL", "MSFT", "GOOGL"]},
    )

    response = await client.delete("/api/v1/users/me/watchlist/MSFT", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert set(data["watchlist"]) == {"AAPL", "GOOGL"}


@pytest.mark.asyncio
async def test_remove_from_watchlist_case_insensitive(client: AsyncClient) -> None:
    """Test removing symbol is case insensitive."""
    headers = await create_test_user_and_login(client)

    await client.post(
        "/api/v1/users/me/watchlist",
        headers=headers,
        json={"symbols": ["AAPL", "MSFT"]},
    )

    response = await client.delete("/api/v1/users/me/watchlist/msft", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["watchlist"] == ["AAPL"]


@pytest.mark.asyncio
async def test_remove_from_watchlist_not_found(client: AsyncClient) -> None:
    """Test removing non-existent symbol returns 404."""
    headers = await create_test_user_and_login(client)

    response = await client.delete("/api/v1/users/me/watchlist/AAPL", headers=headers)

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_create_user_activity(client: AsyncClient, db_session: AsyncSession) -> None:
    """Test creating a user activity record."""
    headers = await create_test_user_and_login(client)

    response = await client.post(
        "/api/v1/users/me/activity",
        headers=headers,
        json={
            "activity_type": "view",
            "symbol": "AAPL",
            "sector": "Technology",
            "metadata": {"page": "dashboard", "duration_seconds": 45},
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["activity_type"] == "view"
    assert data["symbol"] == "AAPL"
    assert data["sector"] == "Technology"
    assert data["metadata"]["page"] == "dashboard"

    result = await db_session.execute(select(UserActivity))
    activities = result.scalars().all()
    assert len(activities) == 1


@pytest.mark.asyncio
async def test_create_user_activity_normalizes_symbol(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test that activity symbols are normalized to uppercase."""
    headers = await create_test_user_and_login(client)

    response = await client.post(
        "/api/v1/users/me/activity",
        headers=headers,
        json={
            "activity_type": "click",
            "symbol": "aapl",
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_create_user_activity_all_types(client: AsyncClient) -> None:
    """Test creating activities of all types."""
    headers = await create_test_user_and_login(client)

    for activity_type in ["view", "click", "follow", "dismiss"]:
        response = await client.post(
            "/api/v1/users/me/activity",
            headers=headers,
            json={
                "activity_type": activity_type,
                "symbol": "AAPL",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["activity_type"] == activity_type


@pytest.mark.asyncio
async def test_create_user_activity_minimal(client: AsyncClient) -> None:
    """Test creating activity with minimal required fields."""
    headers = await create_test_user_and_login(client)

    response = await client.post(
        "/api/v1/users/me/activity",
        headers=headers,
        json={"activity_type": "view"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["activity_type"] == "view"
    assert data["symbol"] is None
    assert data["sector"] is None
    assert data["signal_id"] is None


@pytest.mark.asyncio
async def test_get_user_activity(client: AsyncClient) -> None:
    """Test retrieving user activity history."""
    headers = await create_test_user_and_login(client)

    for i in range(5):
        await client.post(
            "/api/v1/users/me/activity",
            headers=headers,
            json={
                "activity_type": "view",
                "symbol": f"TEST{i}",
            },
        )

    response = await client.get("/api/v1/users/me/activity", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5


@pytest.mark.asyncio
async def test_get_user_activity_pagination(client: AsyncClient) -> None:
    """Test user activity pagination."""
    headers = await create_test_user_and_login(client)

    for i in range(15):
        await client.post(
            "/api/v1/users/me/activity",
            headers=headers,
            json={
                "activity_type": "view",
                "symbol": f"TEST{i}",
            },
        )

    response = await client.get(
        "/api/v1/users/me/activity?limit=10&offset=0",
        headers=headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 10

    response = await client.get(
        "/api/v1/users/me/activity?limit=10&offset=10",
        headers=headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5


@pytest.mark.asyncio
async def test_get_user_activity_ordered_by_most_recent(client: AsyncClient) -> None:
    """Test that activities are returned in reverse chronological order."""
    headers = await create_test_user_and_login(client)

    symbols = ["AAPL", "MSFT", "GOOGL"]
    for symbol in symbols:
        await client.post(
            "/api/v1/users/me/activity",
            headers=headers,
            json={
                "activity_type": "view",
                "symbol": symbol,
            },
        )

    response = await client.get("/api/v1/users/me/activity", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data[0]["symbol"] == "GOOGL"
    assert data[1]["symbol"] == "MSFT"
    assert data[2]["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_user_activity_isolated_per_user(client: AsyncClient) -> None:
    """Test that each user only sees their own activities."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "user1@example.com",
            "username": "user1",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "user1@example.com",
            "password": "testpassword123",
        },
    )

    headers1 = {"Authorization": f"Bearer {login_response.json()['access_token']}"}

    await client.post(
        "/api/v1/users/me/activity",
        headers=headers1,
        json={"activity_type": "view", "symbol": "USER1STOCK"},
    )

    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "user2@example.com",
            "username": "user2",
            "password": "testpassword123",
        },
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "user2@example.com",
            "password": "testpassword123",
        },
    )

    headers2 = {"Authorization": f"Bearer {login_response.json()['access_token']}"}

    response = await client.get("/api/v1/users/me/activity", headers=headers2)

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 0


@pytest.mark.asyncio
async def test_user_model_has_new_fields(db_session: AsyncSession) -> None:
    """Test that User model includes all new preference fields."""
    from signalforge.core.security import get_password_hash

    user = User(
        email="model_test@example.com",
        username="modeltest",
        hashed_password=get_password_hash("password"),
        risk_tolerance=RiskTolerance.HIGH,
        investment_horizon=InvestmentHorizon.LONG,
        experience_level=ExperienceLevel.QUANT,
        preferred_sectors=["Technology", "Healthcare"],
        watchlist=["AAPL", "MSFT"],
        notification_preferences={"email": True},
    )

    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    assert user.risk_tolerance == RiskTolerance.HIGH
    assert user.investment_horizon == InvestmentHorizon.LONG
    assert user.experience_level == ExperienceLevel.QUANT
    assert user.preferred_sectors == ["Technology", "Healthcare"]
    assert user.watchlist == ["AAPL", "MSFT"]
    assert user.notification_preferences == {"email": True}


@pytest.mark.asyncio
async def test_user_activity_model_relationships(db_session: AsyncSession) -> None:
    """Test UserActivity model can be created with all fields."""
    from uuid import uuid4

    from signalforge.core.security import get_password_hash

    user = User(
        email="activity_test@example.com",
        username="activitytest",
        hashed_password=get_password_hash("password"),
    )

    db_session.add(user)
    await db_session.flush()

    activity = UserActivity(
        user_id=user.id,
        activity_type=ActivityType.VIEW,
        symbol="AAPL",
        sector="Technology",
        signal_id=uuid4(),
        metadata_={"source": "dashboard"},
    )

    db_session.add(activity)
    await db_session.flush()
    await db_session.refresh(activity)

    assert activity.user_id == user.id
    assert activity.activity_type == ActivityType.VIEW
    assert activity.symbol == "AAPL"
    assert activity.sector == "Technology"
    assert activity.metadata_["source"] == "dashboard"
