"""Tests for news endpoints."""

from datetime import UTC, datetime

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.news import NewsArticle


async def create_test_user_and_get_token(client: AsyncClient) -> str:
    """Helper to create a test user and return access token."""
    await client.post(
        "/api/v1/auth/register",
        json={
            "email": "news_test@example.com",
            "username": "newsuser",
            "password": "testpassword123",
        },
    )
    login_response = await client.post(
        "/api/v1/auth/login",
        data={
            "username": "news_test@example.com",
            "password": "testpassword123",
        },
    )
    return login_response.json()["access_token"]


async def insert_test_articles(
    db_session: AsyncSession,
    count: int = 5,
    source: str = "test_source",
) -> None:
    """Insert test news articles into the database."""
    base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)

    for i in range(count):
        article = NewsArticle(
            url=f"https://example.com/article-{source}-{i}",
            title=f"Test Article {i} from {source}",
            source=source,
            published_at=datetime(
                base_time.year,
                base_time.month,
                base_time.day,
                base_time.hour + i,
                base_time.minute,
                tzinfo=UTC,
            ),
            summary=f"This is a summary for test article {i}",
            metadata_={"author": "Test Author", "tags": ["finance", "market"]},
            symbols=["AAPL", "MSFT"] if i % 2 == 0 else ["GOOGL"],
            categories=["technology"] if i % 2 == 0 else ["business"],
        )
        db_session.add(article)

    await db_session.commit()


@pytest.mark.asyncio
async def test_get_news_requires_auth(client: AsyncClient) -> None:
    """Test that news endpoint requires authentication."""
    response = await client.get("/api/v1/news/")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_news_empty(
    client: AsyncClient,
    _db_session: AsyncSession,
) -> None:
    """Test getting news when no articles exist."""
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/news/",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["articles"] == []
    assert data["count"] == 0
    assert data["total"] == 0


@pytest.mark.asyncio
async def test_get_news_success(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test successful retrieval of news articles."""
    await insert_test_articles(db_session, count=5)
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/news/",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 5
    assert data["total"] == 5
    assert len(data["articles"]) == 5

    first_article = data["articles"][0]
    assert "id" in first_article
    assert "url" in first_article
    assert "title" in first_article
    assert "source" in first_article
    assert "published_at" in first_article
    assert "summary" in first_article
    assert "categories" in first_article
    assert "symbols" in first_article
    assert "created_at" in first_article


@pytest.mark.asyncio
async def test_get_news_ordered_by_published_at(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test that news articles are ordered by published_at descending."""
    await insert_test_articles(db_session, count=5)
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/news/",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()

    dates = [
        datetime.fromisoformat(a["published_at"].replace("Z", "+00:00"))
        for a in data["articles"]
        if a["published_at"]
    ]
    assert dates == sorted(dates, reverse=True)


@pytest.mark.asyncio
async def test_get_news_filter_by_source(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test filtering news by source."""
    await insert_test_articles(db_session, count=3, source="bloomberg")
    await insert_test_articles(db_session, count=2, source="reuters")
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/news/?source=bloomberg",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    assert data["total"] == 3
    for article in data["articles"]:
        assert article["source"] == "bloomberg"


@pytest.mark.asyncio
async def test_get_news_with_limit(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test news pagination with limit parameter."""
    await insert_test_articles(db_session, count=10)
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/news/?limit=5",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 5
    assert data["total"] == 10


@pytest.mark.asyncio
async def test_get_news_with_offset(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test news pagination with offset parameter."""
    await insert_test_articles(db_session, count=10)
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/news/?limit=5&offset=5",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 5
    assert data["total"] == 10


@pytest.mark.asyncio
async def test_get_news_limit_validation(
    client: AsyncClient,
    _db_session: AsyncSession,
) -> None:
    """Test that limit parameter is validated."""
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/news/?limit=0",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422

    response = await client.get(
        "/api/v1/news/?limit=101",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_news_offset_validation(
    client: AsyncClient,
    _db_session: AsyncSession,
) -> None:
    """Test that offset parameter is validated."""
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/news/?offset=-1",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_news_nonexistent_source(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test filtering by non-existent source returns empty results."""
    await insert_test_articles(db_session, count=5)
    token = await create_test_user_and_get_token(client)

    response = await client.get(
        "/api/v1/news/?source=nonexistent_source",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["articles"] == []
    assert data["count"] == 0
    assert data["total"] == 0
