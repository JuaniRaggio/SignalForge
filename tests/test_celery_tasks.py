"""Integration tests for Celery tasks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from signalforge.ingestion.scrapers.base import ScrapedArticle


class TestIngestDailyPricesTask:
    """Tests for the ingest_daily_prices Celery task."""

    @pytest.fixture
    def sample_yahoo_data(self) -> pl.DataFrame:
        """Create sample Yahoo Finance data."""
        base_date = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        return pl.DataFrame(
            {
                "symbol": ["AAPL"] * 5,
                "timestamp": [base_date + timedelta(days=i) for i in range(5)],
                "open": [150.0, 152.0, 154.0, 156.0, 158.0],
                "high": [155.0, 157.0, 159.0, 161.0, 163.0],
                "low": [148.0, 150.0, 152.0, 154.0, 156.0],
                "close": [153.0, 155.0, 157.0, 159.0, 161.0],
                "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            }
        )

    @pytest.mark.asyncio
    async def test_ingest_daily_prices_success(
        self,
        sample_yahoo_data: pl.DataFrame,
    ) -> None:
        """Test successful price ingestion."""
        mock_client = MagicMock()
        mock_client.fetch_multiple = AsyncMock(return_value={"AAPL": sample_yahoo_data})
        mock_client.close = MagicMock()

        mock_session = MagicMock()
        mock_session.merge = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with (
            patch(
                "signalforge.ingestion.tasks.YahooFinanceClient",
                return_value=mock_client,
            ),
            patch(
                "signalforge.ingestion.tasks.get_session_context",
                return_value=mock_session,
            ),
        ):
            from signalforge.ingestion.tasks import _ingest_daily_prices_async

            result = await _ingest_daily_prices_async(["AAPL"])

            assert "success" in result
            assert "AAPL" in result["success"]
            assert result["records_inserted"] == 5

    @pytest.mark.asyncio
    async def test_ingest_daily_prices_partial_failure(
        self,
        sample_yahoo_data: pl.DataFrame,
    ) -> None:
        """Test price ingestion with partial failures."""
        mock_client = MagicMock()
        mock_client.fetch_multiple = AsyncMock(
            return_value={"AAPL": sample_yahoo_data}  # Only AAPL succeeds
        )
        mock_client.close = MagicMock()

        mock_session = MagicMock()
        mock_session.merge = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with (
            patch(
                "signalforge.ingestion.tasks.YahooFinanceClient",
                return_value=mock_client,
            ),
            patch(
                "signalforge.ingestion.tasks.get_session_context",
                return_value=mock_session,
            ),
        ):
            from signalforge.ingestion.tasks import _ingest_daily_prices_async

            # Request multiple symbols but only AAPL returns
            result = await _ingest_daily_prices_async(["AAPL", "INVALID"])

            assert "AAPL" in result["success"]
            # INVALID should not be in success (not returned by mock)

    @pytest.mark.asyncio
    async def test_ingest_daily_prices_empty_symbols(self) -> None:
        """Test price ingestion with empty symbol list."""
        mock_client = MagicMock()
        mock_client.fetch_multiple = AsyncMock(return_value={})
        mock_client.close = MagicMock()

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with (
            patch(
                "signalforge.ingestion.tasks.YahooFinanceClient",
                return_value=mock_client,
            ),
            patch(
                "signalforge.ingestion.tasks.get_session_context",
                return_value=mock_session,
            ),
        ):
            from signalforge.ingestion.tasks import _ingest_daily_prices_async

            result = await _ingest_daily_prices_async([])

            assert result["success"] == []
            assert result["records_inserted"] == 0


class TestScrapeNewsRSSTask:
    """Tests for the scrape_news_rss Celery task."""

    @pytest.fixture
    def sample_articles(self) -> list[ScrapedArticle]:
        """Create sample scraped articles."""
        base_date = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        return [
            ScrapedArticle(
                url=f"https://news.example.com/article-{i}",
                title=f"Test Article {i}",
                source="Test Source",
                published_at=base_date - timedelta(hours=i),
                summary=f"Summary {i}",
                categories=["finance"],
                symbols=["AAPL"] if i % 2 == 0 else [],
                metadata={},
            )
            for i in range(3)
        ]

    @pytest.mark.asyncio
    async def test_scrape_news_success(
        self,
        sample_articles: list[ScrapedArticle],
    ) -> None:
        """Test successful news scraping."""
        mock_scraper = MagicMock()
        mock_scraper.scrape_all = AsyncMock(return_value=sample_articles)

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(
            return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None))
        )
        mock_session.add = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with (
            patch(
                "signalforge.ingestion.tasks.MultiSourceRSSScraper",
                return_value=mock_scraper,
            ),
            patch(
                "signalforge.ingestion.tasks.get_session_context",
                return_value=mock_session,
            ),
        ):
            from signalforge.ingestion.tasks import _scrape_news_rss_async

            result = await _scrape_news_rss_async()

            assert result["articles_scraped"] == 3
            assert result["articles_saved"] == 3
            assert result["duplicates_skipped"] == 0

    @pytest.mark.asyncio
    async def test_scrape_news_with_duplicates(
        self,
        sample_articles: list[ScrapedArticle],
    ) -> None:
        """Test news scraping handles duplicates correctly."""
        mock_scraper = MagicMock()
        mock_scraper.scrape_all = AsyncMock(return_value=sample_articles)

        # Simulate first article already exists
        call_count = 0

        def mock_scalar_one_or_none() -> MagicMock | None:
            nonlocal call_count
            call_count += 1
            return MagicMock() if call_count == 1 else None

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none = mock_scalar_one_or_none

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_execute_result)
        mock_session.add = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with (
            patch(
                "signalforge.ingestion.tasks.MultiSourceRSSScraper",
                return_value=mock_scraper,
            ),
            patch(
                "signalforge.ingestion.tasks.get_session_context",
                return_value=mock_session,
            ),
        ):
            from signalforge.ingestion.tasks import _scrape_news_rss_async

            result = await _scrape_news_rss_async()

            assert result["articles_scraped"] == 3
            assert result["duplicates_skipped"] == 1
            assert result["articles_saved"] == 2

    @pytest.mark.asyncio
    async def test_scrape_news_empty_feed(self) -> None:
        """Test news scraping with empty feed."""
        mock_scraper = MagicMock()
        mock_scraper.scrape_all = AsyncMock(return_value=[])

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with (
            patch(
                "signalforge.ingestion.tasks.MultiSourceRSSScraper",
                return_value=mock_scraper,
            ),
            patch(
                "signalforge.ingestion.tasks.get_session_context",
                return_value=mock_session,
            ),
        ):
            from signalforge.ingestion.tasks import _scrape_news_rss_async

            result = await _scrape_news_rss_async()

            assert result["articles_scraped"] == 0
            assert result["articles_saved"] == 0


class TestHistoricalBackfillTask:
    """Tests for the ingest_historical_backfill Celery task."""

    @pytest.fixture
    def sample_historical_data(self) -> pl.DataFrame:
        """Create sample historical data."""
        base_date = datetime(2023, 1, 15, 10, 0, 0, tzinfo=UTC)
        return pl.DataFrame(
            {
                "symbol": ["AAPL"] * 252,  # One year of trading days
                "timestamp": [base_date + timedelta(days=i) for i in range(252)],
                "open": [150.0 + i * 0.1 for i in range(252)],
                "high": [155.0 + i * 0.1 for i in range(252)],
                "low": [148.0 + i * 0.1 for i in range(252)],
                "close": [153.0 + i * 0.1 for i in range(252)],
                "volume": [1000000 + i * 1000 for i in range(252)],
            }
        )

    @pytest.mark.asyncio
    async def test_historical_backfill_success(
        self,
        sample_historical_data: pl.DataFrame,
    ) -> None:
        """Test successful historical backfill."""
        mock_client = MagicMock()
        mock_client.fetch_data = AsyncMock(return_value=sample_historical_data)
        mock_client.close = MagicMock()

        mock_session = MagicMock()
        mock_session.merge = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with (
            patch(
                "signalforge.ingestion.tasks.YahooFinanceClient",
                return_value=mock_client,
            ),
            patch(
                "signalforge.ingestion.tasks.get_session_context",
                return_value=mock_session,
            ),
        ):
            from signalforge.ingestion.tasks import _ingest_historical_backfill_async

            result = await _ingest_historical_backfill_async("AAPL", "1y")

            assert result["success"] is True
            assert result["symbol"] == "AAPL"
            assert result["records_inserted"] == 252

    @pytest.mark.asyncio
    async def test_historical_backfill_failure(self) -> None:
        """Test historical backfill failure handling."""
        mock_client = MagicMock()
        mock_client.fetch_data = AsyncMock(side_effect=Exception("API error"))
        mock_client.close = MagicMock()

        with patch(
            "signalforge.ingestion.tasks.YahooFinanceClient",
            return_value=mock_client,
        ):
            from signalforge.ingestion.tasks import _ingest_historical_backfill_async

            result = await _ingest_historical_backfill_async("INVALID", "1y")

            assert result["success"] is False
            assert "error" in result


class TestTaskPipeline:
    """Integration tests for complete task pipelines."""

    @pytest.mark.asyncio
    async def test_full_price_ingestion_pipeline(self) -> None:
        """Test complete price ingestion pipeline from API to database."""
        base_date = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        sample_data = pl.DataFrame(
            {
                "symbol": ["AAPL"] * 3,
                "timestamp": [base_date + timedelta(days=i) for i in range(3)],
                "open": [150.0, 152.0, 154.0],
                "high": [155.0, 157.0, 159.0],
                "low": [148.0, 150.0, 152.0],
                "close": [153.0, 155.0, 157.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        mock_client = MagicMock()
        mock_client.fetch_multiple = AsyncMock(return_value={"AAPL": sample_data})
        mock_client.close = MagicMock()

        merged_records: list[Any] = []

        async def capture_merge(record: Any) -> None:
            merged_records.append(record)

        mock_session = MagicMock()
        mock_session.merge = AsyncMock(side_effect=capture_merge)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with (
            patch(
                "signalforge.ingestion.tasks.YahooFinanceClient",
                return_value=mock_client,
            ),
            patch(
                "signalforge.ingestion.tasks.get_session_context",
                return_value=mock_session,
            ),
        ):
            from signalforge.ingestion.tasks import _ingest_daily_prices_async

            result = await _ingest_daily_prices_async(["AAPL"])

            # Verify pipeline completed
            assert result["success"] == ["AAPL"]
            assert result["records_inserted"] == 3

            # Verify records were passed to database
            assert len(merged_records) == 3

    @pytest.mark.asyncio
    async def test_full_news_scraping_pipeline(self) -> None:
        """Test complete news scraping pipeline from RSS to database."""
        base_date = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
        sample_articles = [
            ScrapedArticle(
                url="https://example.com/news/1",
                title="Breaking News",
                source="Test Source",
                published_at=base_date,
                summary="Important news summary",
                categories=["markets"],
                symbols=["AAPL"],
                metadata={},
            )
        ]

        mock_scraper = MagicMock()
        mock_scraper.scrape_all = AsyncMock(return_value=sample_articles)

        added_records: list[Any] = []

        def capture_add(record: Any) -> None:
            added_records.append(record)

        mock_execute_result = MagicMock()
        mock_execute_result.scalar_one_or_none = MagicMock(return_value=None)

        mock_session = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_execute_result)
        mock_session.add = MagicMock(side_effect=capture_add)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock()

        with (
            patch(
                "signalforge.ingestion.tasks.MultiSourceRSSScraper",
                return_value=mock_scraper,
            ),
            patch(
                "signalforge.ingestion.tasks.get_session_context",
                return_value=mock_session,
            ),
        ):
            from signalforge.ingestion.tasks import _scrape_news_rss_async

            result = await _scrape_news_rss_async()

            # Verify pipeline completed
            assert result["articles_scraped"] == 1
            assert result["articles_saved"] == 1

            # Verify article was passed to database
            assert len(added_records) == 1
