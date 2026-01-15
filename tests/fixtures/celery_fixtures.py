"""Celery fixtures for testing task execution."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest

from signalforge.ingestion.scrapers.base import ScrapedArticle

if TYPE_CHECKING:
    from celery import Celery


@pytest.fixture
def celery_config() -> dict[str, Any]:
    """Celery configuration for testing.

    Uses eager execution to run tasks synchronously in tests.
    """
    return {
        "broker_url": "memory://",
        "result_backend": "cache+memory://",
        "task_always_eager": True,
        "task_eager_propagates": True,
        "task_store_eager_result": True,
    }


@pytest.fixture
def celery_app(celery_config: dict[str, Any]) -> Celery:
    """Create a test Celery application."""
    from celery import Celery

    app = Celery("signalforge_test")
    app.config_from_object(celery_config)

    # Register tasks
    app.autodiscover_tasks(["signalforge.ingestion"])

    return app


@pytest.fixture
def mock_yahoo_client() -> MagicMock:
    """Create a mock Yahoo Finance client."""
    base_date = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)

    # Create sample DataFrame response
    sample_data = {
        "symbol": ["AAPL"] * 5,
        "timestamp": [base_date + timedelta(days=i) for i in range(5)],
        "open": [150.0, 152.0, 154.0, 156.0, 158.0],
        "high": [155.0, 157.0, 159.0, 161.0, 163.0],
        "low": [148.0, 150.0, 152.0, 154.0, 156.0],
        "close": [153.0, 155.0, 157.0, 159.0, 161.0],
        "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
        "adj_close": [152.9, 154.9, 156.9, 158.9, 160.9],
    }
    sample_df = pl.DataFrame(sample_data)

    mock = MagicMock()
    mock.fetch_data = AsyncMock(return_value=sample_df)
    mock.fetch_multiple = AsyncMock(return_value={"AAPL": sample_df})
    mock.close = MagicMock()

    return mock


@pytest.fixture
def mock_rss_scraper() -> MagicMock:
    """Create a mock RSS scraper."""
    base_date = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)

    sample_articles = [
        ScrapedArticle(
            url=f"https://news.example.com/article-{i}",
            title=f"Test News Article {i}",
            source="Test News Source",
            published_at=base_date - timedelta(hours=i),
            summary=f"Summary for article {i}",
            categories=["finance", "markets"],
            symbols=["AAPL", "MSFT"] if i % 2 == 0 else [],
            metadata={"author": f"Author {i}"},
        )
        for i in range(5)
    ]

    mock = MagicMock()
    mock.scrape_all = AsyncMock(return_value=sample_articles)
    mock.reset_dedup_cache = MagicMock()

    return mock


@pytest.fixture
def mock_database_session() -> MagicMock:
    """Create a mock database session for testing."""
    mock_session = MagicMock()
    mock_session.merge = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.commit = AsyncMock()
    mock_session.execute = AsyncMock()

    # Mock context manager
    async_context = MagicMock()
    async_context.__aenter__ = AsyncMock(return_value=mock_session)
    async_context.__aexit__ = AsyncMock()

    return async_context


@pytest.fixture
def celery_worker(celery_app: Celery) -> None:
    """Configure Celery for synchronous task execution in tests.

    This fixture doesn't actually start a worker since we use
    task_always_eager=True for synchronous execution.
    """
    # In eager mode, tasks run synchronously without a worker
    pass


@pytest.fixture
def mock_price_model() -> MagicMock:
    """Create a mock Price model for testing."""
    mock = MagicMock()
    mock.symbol = "AAPL"
    mock.timestamp = datetime.now(UTC)
    mock.open = 150.0
    mock.high = 155.0
    mock.low = 148.0
    mock.close = 153.0
    mock.volume = 1000000
    return mock


@pytest.fixture
def mock_news_article_model() -> MagicMock:
    """Create a mock NewsArticle model for testing."""
    mock = MagicMock()
    mock.id = "test-uuid"
    mock.url = "https://example.com/news"
    mock.title = "Test Article"
    mock.source = "Test Source"
    mock.published_at = datetime.now(UTC)
    mock.summary = "Test summary"
    mock.categories = ["finance"]
    mock.symbols = ["AAPL"]
    mock.created_at = datetime.now(UTC)
    return mock
