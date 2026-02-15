"""Tests for NLP API endpoints."""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.news import NewsArticle
from signalforge.models.user import User


@pytest.fixture
async def test_article(db_session: AsyncSession) -> NewsArticle:
    """Create test news article."""
    article = NewsArticle(
        id=uuid4(),
        url="https://test.com/article-1",
        title="Apple Reports Strong Q4 Earnings",
        source="test",
        published_at=datetime.now(UTC),
        content="Apple Inc. reported strong Q4 earnings that beat analyst expectations.",
        summary="Apple beats earnings",
        metadata_={"sector": "Information Technology"},
        symbols=["AAPL"],
        categories=["earnings"],
    )
    db_session.add(article)
    await db_session.commit()
    await db_session.refresh(article)
    return article


@pytest.fixture
async def auth_headers(authenticated_user_token: str) -> dict[str, str]:
    """Create authorization headers for authenticated requests."""
    return {"Authorization": f"Bearer {authenticated_user_token}"}


@pytest.mark.asyncio
async def test_analyze_document_with_document_id(
    client: AsyncClient,
    auth_headers: dict[str, str],
    test_article: NewsArticle,
):
    """Test analyzing document by ID."""
    response = await client.post(
        "/api/v1/nlp/analyze-document",
        json={
            "document_id": str(test_article.id),
            "include_price_targets": True,
            "include_sector_signals": True,
        },
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert "ticker" in data
    assert "sentiment" in data
    assert "price_targets" in data
    assert "analyst_consensus" in data
    assert "sector_signals" in data
    assert "urgency" in data
    assert data["urgency"] in ["low", "medium", "high", "critical"]


@pytest.mark.asyncio
async def test_analyze_document_with_text(
    client: AsyncClient,
    auth_headers: dict[str, str],
):
    """Test analyzing document from raw text."""
    response = await client.post(
        "/api/v1/nlp/analyze-document",
        json={
            "text": "Microsoft announces strong Q3 results with 15% revenue growth.",
            "title": "Microsoft Q3 Results",
            "symbols": ["MSFT"],
            "include_price_targets": False,
            "include_sector_signals": True,
        },
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["ticker"] in ["MSFT", "UNKNOWN"]
    assert "sentiment" in data
    assert data["sentiment"]["label"] in ["bullish", "bearish", "neutral"]


@pytest.mark.asyncio
async def test_analyze_document_missing_input(
    client: AsyncClient,
    auth_headers: dict[str, str],
):
    """Test analyzing document without document_id or text."""
    response = await client.post(
        "/api/v1/nlp/analyze-document",
        json={
            "include_price_targets": True,
        },
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.asyncio
async def test_analyze_document_not_found(
    client: AsyncClient,
    auth_headers: dict[str, str],
):
    """Test analyzing non-existent document."""
    response = await client.post(
        "/api/v1/nlp/analyze-document",
        json={
            "document_id": str(uuid4()),
        },
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_analyze_document_unauthorized(
    client: AsyncClient,
    test_article: NewsArticle,
):
    """Test analyzing document without authentication."""
    response = await client.post(
        "/api/v1/nlp/analyze-document",
        json={
            "document_id": str(test_article.id),
        },
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.asyncio
async def test_get_nlp_signals(
    client: AsyncClient,
    auth_headers: dict[str, str],
    test_article: NewsArticle,
):
    """Test getting NLP signals for a symbol."""
    response = await client.get(
        "/api/v1/nlp/signals/AAPL",
        params={"days": 7},
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["symbol"] == "AAPL"
    assert "signals" in data
    assert "count" in data
    assert "period_days" in data
    assert data["period_days"] == 7
    assert "summary" in data


@pytest.mark.asyncio
async def test_get_nlp_signals_no_data(
    client: AsyncClient,
    auth_headers: dict[str, str],
):
    """Test getting NLP signals for symbol with no data."""
    response = await client.get(
        "/api/v1/nlp/signals/UNKNOWN",
        params={"days": 7},
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["symbol"] == "UNKNOWN"
    assert data["count"] == 0
    assert len(data["signals"]) == 0


@pytest.mark.asyncio
async def test_get_nlp_signals_custom_period(
    client: AsyncClient,
    auth_headers: dict[str, str],
):
    """Test getting NLP signals with custom time period."""
    response = await client.get(
        "/api/v1/nlp/signals/AAPL",
        params={"days": 30},
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["period_days"] == 30


@pytest.mark.asyncio
async def test_get_sector_report(
    client: AsyncClient,
    auth_headers: dict[str, str],
):
    """Test getting sector report."""
    response = await client.get(
        "/api/v1/nlp/sector-report/Information Technology",
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["sector"] == "Information Technology"
    assert "overall_sentiment" in data
    assert "analyst_consensus" in data
    assert "top_symbols" in data


@pytest.mark.asyncio
async def test_get_sector_report_invalid_sector(
    client: AsyncClient,
    auth_headers: dict[str, str],
):
    """Test getting report for invalid sector."""
    response = await client.get(
        "/api/v1/nlp/sector-report/InvalidSector",
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.asyncio
async def test_get_analyst_consensus(
    client: AsyncClient,
    auth_headers: dict[str, str],
):
    """Test getting analyst consensus."""
    response = await client.get(
        "/api/v1/nlp/analyst-consensus/AAPL",
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["symbol"] == "AAPL"
    assert "consensus" in data
    assert "price_targets" in data
    assert data["consensus"]["rating"] in [
        "strong_buy",
        "buy",
        "hold",
        "sell",
        "strong_sell",
    ]


@pytest.mark.asyncio
async def test_aggregate_signals(
    client: AsyncClient,
    auth_headers: dict[str, str],
    test_article: NewsArticle,
):
    """Test aggregating ML and NLP signals."""
    response = await client.post(
        "/api/v1/nlp/aggregate-signals",
        json={
            "symbol": "AAPL",
            "include_ml_prediction": False,
            "include_nlp_signals": True,
            "include_execution_quality": False,
            "ml_weight": 0.6,
            "nlp_weight": 0.4,
        },
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert "signal" in data
    assert data["signal"]["symbol"] == "AAPL"
    assert data["signal"]["direction"] in ["long", "short", "neutral"]
    assert -1.0 <= data["signal"]["strength"] <= 1.0
    assert 0.0 <= data["signal"]["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_aggregate_signals_invalid_weights(
    client: AsyncClient,
    auth_headers: dict[str, str],
):
    """Test aggregating signals with invalid weights."""
    response = await client.post(
        "/api/v1/nlp/aggregate-signals",
        json={
            "symbol": "AAPL",
            "ml_weight": 0.7,
            "nlp_weight": 0.5,  # Sum > 1.0
        },
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.asyncio
async def test_aggregate_signals_with_regime(
    client: AsyncClient,
    auth_headers: dict[str, str],
    test_article: NewsArticle,
):
    """Test aggregating signals with market regime conditioning."""
    response = await client.post(
        "/api/v1/nlp/aggregate-signals",
        json={
            "symbol": "AAPL",
            "include_nlp_signals": True,
            "ml_weight": 0.6,
            "nlp_weight": 0.4,
            "market_regime": "bear",
        },
        headers=auth_headers,
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert data["market_regime"] == "bear"
    assert "signal" in data
