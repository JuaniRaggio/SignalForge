"""Data fixtures for testing ingestion and processing pipelines."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl
import pytest


@pytest.fixture
def sample_price_data() -> list[dict[str, Any]]:
    """Generate sample OHLCV price data for testing."""
    base_date = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    data = []

    for i in range(5):
        date = base_date + timedelta(days=i)
        base_price = 150.0 + i * 2

        data.append(
            {
                "symbol": "AAPL",
                "timestamp": date,
                "open": base_price,
                "high": base_price + 5.0,
                "low": base_price - 2.0,
                "close": base_price + 3.0,
                "volume": 1000000 + i * 100000,
                "adj_close": base_price + 2.9,
            }
        )

    return data


@pytest.fixture
def sample_price_dataframe() -> pl.DataFrame:
    """Generate sample price data as a Polars DataFrame."""
    base_date = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)

    data = {
        "symbol": ["AAPL"] * 5,
        "timestamp": [base_date + timedelta(days=i) for i in range(5)],
        "open": [150.0, 152.0, 154.0, 156.0, 158.0],
        "high": [155.0, 157.0, 159.0, 161.0, 163.0],
        "low": [148.0, 150.0, 152.0, 154.0, 156.0],
        "close": [153.0, 155.0, 157.0, 159.0, 161.0],
        "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
    }

    return pl.DataFrame(data)


@pytest.fixture
def sample_news_articles() -> list[dict[str, Any]]:
    """Generate sample news articles for testing."""
    base_date = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)

    return [
        {
            "url": "https://finance.example.com/news/article-1",
            "title": "Apple Reports Record Quarterly Earnings",
            "source": "Finance News",
            "published_at": base_date,
            "summary": "Apple Inc. reported record quarterly earnings...",
            "categories": ["earnings", "technology"],
            "symbols": ["AAPL"],
            "metadata": {"author": "John Doe"},
        },
        {
            "url": "https://finance.example.com/news/article-2",
            "title": "Microsoft Azure Growth Accelerates",
            "source": "Tech Daily",
            "published_at": base_date - timedelta(hours=2),
            "summary": "Microsoft's cloud business continues to grow...",
            "categories": ["cloud", "technology"],
            "symbols": ["MSFT"],
            "metadata": {"author": "Jane Smith"},
        },
        {
            "url": "https://finance.example.com/news/article-3",
            "title": "Fed Signals Rate Cuts May Come Soon",
            "source": "Market Watch",
            "published_at": base_date - timedelta(hours=4),
            "summary": "Federal Reserve officials indicated...",
            "categories": ["economy", "markets"],
            "symbols": [],
            "metadata": {},
        },
    ]


@pytest.fixture
def sample_yahoo_response() -> dict[str, Any]:
    """Generate a mock Yahoo Finance API response."""
    base_date = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)

    return {
        "symbol": "AAPL",
        "period": "5d",
        "interval": "1d",
        "data": [
            {
                "Date": base_date + timedelta(days=i),
                "Open": 150.0 + i * 2,
                "High": 155.0 + i * 2,
                "Low": 148.0 + i * 2,
                "Close": 153.0 + i * 2,
                "Volume": 1000000 + i * 100000,
                "Adj Close": 152.9 + i * 2,
            }
            for i in range(5)
        ],
    }


@pytest.fixture
def sample_rss_feed() -> dict[str, Any]:
    """Generate a mock RSS feed response."""
    base_date = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)

    return {
        "feed": {
            "title": "Test Financial News",
            "link": "https://news.example.com/rss",
        },
        "entries": [
            {
                "link": f"https://news.example.com/article-{i}",
                "title": f"Test Article {i}",
                "published": (base_date - timedelta(hours=i)).strftime("%a, %d %b %Y %H:%M:%S %z"),
                "summary": f"This is the summary for article {i}...",
                "author": f"Author {i}",
                "tags": [{"term": "finance"}, {"term": "markets"}],
            }
            for i in range(5)
        ],
    }


@pytest.fixture
def sample_malformed_price_data() -> list[dict[str, Any]]:
    """Generate malformed price data for validation testing."""
    return [
        # Missing required fields
        {
            "symbol": "AAPL",
            "timestamp": datetime.now(UTC),
            # Missing OHLCV
        },
        # Invalid price relationship (high < low)
        {
            "symbol": "MSFT",
            "timestamp": datetime.now(UTC),
            "open": 150.0,
            "high": 145.0,  # Invalid: less than low
            "low": 148.0,
            "close": 147.0,
            "volume": 1000000,
        },
        # Negative volume
        {
            "symbol": "GOOGL",
            "timestamp": datetime.now(UTC),
            "open": 150.0,
            "high": 155.0,
            "low": 148.0,
            "close": 153.0,
            "volume": -1000,  # Invalid
        },
        # Missing timestamp
        {
            "symbol": "NVDA",
            "open": 150.0,
            "high": 155.0,
            "low": 148.0,
            "close": 153.0,
            "volume": 1000000,
        },
    ]


@pytest.fixture
def sample_malformed_articles() -> list[dict[str, Any]]:
    """Generate malformed article data for validation testing."""
    return [
        # Missing URL
        {
            "title": "Article Without URL",
            "source": "Test Source",
        },
        # Invalid URL
        {
            "url": "not-a-valid-url",
            "title": "Article with Invalid URL",
            "source": "Test Source",
        },
        # Missing title
        {
            "url": "https://example.com/article",
            "source": "Test Source",
        },
        # Empty title
        {
            "url": "https://example.com/article-2",
            "title": "",
            "source": "Test Source",
        },
    ]


@pytest.fixture
def multi_symbol_price_data() -> dict[str, pl.DataFrame]:
    """Generate price data for multiple symbols."""
    base_date = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
    result = {}

    for idx, symbol in enumerate(symbols):
        base_price = 100.0 + idx * 50

        data = {
            "symbol": [symbol] * 5,
            "timestamp": [base_date + timedelta(days=i) for i in range(5)],
            "open": [base_price + i * 2 for i in range(5)],
            "high": [base_price + 5 + i * 2 for i in range(5)],
            "low": [base_price - 2 + i * 2 for i in range(5)],
            "close": [base_price + 3 + i * 2 for i in range(5)],
            "volume": [1000000 + i * 100000 for i in range(5)],
        }

        result[symbol] = pl.DataFrame(data)

    return result
