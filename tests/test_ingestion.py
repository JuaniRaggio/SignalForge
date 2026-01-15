"""Tests for data ingestion components."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from signalforge.core.exceptions import ExternalAPIError
from signalforge.ingestion.api_clients.yahoo import YahooFinanceClient
from signalforge.ingestion.scrapers.base import BaseScraper, ScrapedArticle
from signalforge.ingestion.scrapers.rss import (
    MultiSourceRSSScraper,
    RSSFeed,
    RSSFeedScraper,
)


class TestBaseScraper:
    """Tests for BaseScraper class."""

    def test_normalize_url_removes_query_params(self) -> None:
        """Test that URL normalization removes query parameters."""

        class ConcreteScraper(BaseScraper):
            async def scrape(self) -> list[ScrapedArticle]:
                return []

        scraper = ConcreteScraper("test")
        url = "https://example.com/article?utm_source=test&utm_medium=email"
        normalized = scraper._normalize_url(url)
        assert normalized == "https://example.com/article"

    def test_normalize_url_removes_trailing_slash(self) -> None:
        """Test that URL normalization removes trailing slashes."""

        class ConcreteScraper(BaseScraper):
            async def scrape(self) -> list[ScrapedArticle]:
                return []

        scraper = ConcreteScraper("test")
        url = "https://example.com/article/"
        normalized = scraper._normalize_url(url)
        assert normalized == "https://example.com/article"


class TestScrapedArticle:
    """Tests for ScrapedArticle dataclass."""

    def test_scraped_article_creation(self) -> None:
        """Test creating a ScrapedArticle instance."""
        article = ScrapedArticle(
            url="https://example.com/article",
            title="Test Article",
            source="test_source",
            published_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            content="Full article content",
            summary="Article summary",
            metadata={"author": "Test Author"},
            symbols=["AAPL", "MSFT"],
            categories=["technology"],
        )

        assert article.url == "https://example.com/article"
        assert article.title == "Test Article"
        assert article.source == "test_source"
        assert article.symbols == ["AAPL", "MSFT"]

    def test_scraped_article_optional_fields(self) -> None:
        """Test ScrapedArticle with only required fields."""
        article = ScrapedArticle(
            url="https://example.com/article",
            title="Test Article",
            source="test_source",
        )

        assert article.published_at is None
        assert article.content is None
        assert article.summary is None
        assert article.metadata is None
        assert article.symbols is None
        assert article.categories is None


class TestRSSFeedScraper:
    """Tests for RSSFeedScraper class."""

    def test_parse_date_rfc2822(self) -> None:
        """Test parsing RFC 2822 date format."""
        feed = RSSFeed(name="Test", url="https://example.com/feed", category="test")
        scraper = RSSFeedScraper(feed)

        date_str = "Mon, 15 Jan 2024 10:00:00 +0000"
        result = scraper._parse_date(date_str)

        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_date_iso_format(self) -> None:
        """Test parsing ISO date format."""
        feed = RSSFeed(name="Test", url="https://example.com/feed", category="test")
        scraper = RSSFeedScraper(feed)

        date_str = "2024-01-15T10:00:00Z"
        result = scraper._parse_date(date_str)

        assert result is not None
        assert result.year == 2024

    def test_parse_date_none(self) -> None:
        """Test parsing None date."""
        feed = RSSFeed(name="Test", url="https://example.com/feed", category="test")
        scraper = RSSFeedScraper(feed)

        result = scraper._parse_date(None)
        assert result is None

    def test_parse_date_invalid(self) -> None:
        """Test parsing invalid date returns None."""
        feed = RSSFeed(name="Test", url="https://example.com/feed", category="test")
        scraper = RSSFeedScraper(feed)

        result = scraper._parse_date("not a date")
        assert result is None

    @pytest.mark.asyncio
    async def test_scrape_http_error(self) -> None:
        """Test scrape handles HTTP errors gracefully."""
        import httpx

        feed = RSSFeed(name="Test", url="https://example.com/feed", category="test")
        scraper = RSSFeedScraper(feed, timeout=5.0)

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.side_effect = httpx.HTTPError("Connection failed")
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            result = await scraper.scrape()
            assert result == []

    @pytest.mark.asyncio
    async def test_scrape_success(self) -> None:
        """Test successful RSS feed scraping."""
        feed = RSSFeed(name="Test", url="https://example.com/feed", category="test")
        scraper = RSSFeedScraper(feed)

        mock_rss_content = """<?xml version="1.0"?>
        <rss version="2.0">
            <channel>
                <title>Test Feed</title>
                <item>
                    <title>Test Article</title>
                    <link>https://example.com/article</link>
                    <pubDate>Mon, 15 Jan 2024 10:00:00 +0000</pubDate>
                    <description>Test summary</description>
                </item>
            </channel>
        </rss>
        """

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.text = mock_rss_content
            mock_response.raise_for_status = MagicMock()

            mock_instance = AsyncMock()
            mock_instance.get.return_value = mock_response
            mock_instance.__aenter__.return_value = mock_instance
            mock_instance.__aexit__.return_value = None
            mock_client.return_value = mock_instance

            result = await scraper.scrape()

            assert len(result) == 1
            assert result[0].title == "Test Article"
            assert result[0].source == "Test"


class TestMultiSourceRSSScraper:
    """Tests for MultiSourceRSSScraper class."""

    def test_default_feeds(self) -> None:
        """Test that default feeds are configured."""
        scraper = MultiSourceRSSScraper()
        assert len(scraper.feeds) > 0

    def test_custom_feeds(self) -> None:
        """Test using custom feeds."""
        custom_feeds = [
            RSSFeed(name="Custom1", url="https://custom1.com/feed", category="test"),
            RSSFeed(name="Custom2", url="https://custom2.com/feed", category="test"),
        ]
        scraper = MultiSourceRSSScraper(feeds=custom_feeds)
        assert len(scraper.feeds) == 2

    def test_reset_dedup_cache(self) -> None:
        """Test deduplication cache reset."""
        scraper = MultiSourceRSSScraper()
        scraper._seen_urls.add("https://example.com/article")
        assert len(scraper._seen_urls) == 1

        scraper.reset_dedup_cache()
        assert len(scraper._seen_urls) == 0

    @pytest.mark.asyncio
    async def test_scrape_all_deduplicates(self) -> None:
        """Test that scrape_all deduplicates articles by URL."""
        custom_feeds = [
            RSSFeed(name="Feed1", url="https://feed1.com/feed", category="test"),
        ]
        scraper = MultiSourceRSSScraper(feeds=custom_feeds)

        mock_articles = [
            ScrapedArticle(
                url="https://example.com/article1",
                title="Article 1",
                source="Feed1",
            ),
            ScrapedArticle(
                url="https://example.com/article1",
                title="Article 1 Duplicate",
                source="Feed1",
            ),
        ]

        with patch.object(RSSFeedScraper, "scrape", return_value=mock_articles):
            result = await scraper.scrape_all()
            assert len(result) == 1


class TestYahooFinanceClient:
    """Tests for YahooFinanceClient class."""

    def test_client_initialization(self) -> None:
        """Test client initialization with default parameters."""
        client = YahooFinanceClient()
        assert client.max_retries == 3
        assert client.base_delay == 1.0
        client.close()

    def test_client_custom_parameters(self) -> None:
        """Test client initialization with custom parameters."""
        client = YahooFinanceClient(
            max_retries=5,
            base_delay=2.0,
            max_delay=120.0,
        )
        assert client.max_retries == 5
        assert client.base_delay == 2.0
        client.close()

    @pytest.mark.asyncio
    async def test_fetch_data_empty_result(self) -> None:
        """Test fetch_data raises error for empty result."""
        client = YahooFinanceClient()

        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.history.return_value = MagicMock()
            mock_instance.history.return_value.empty = True
            mock_ticker.return_value = mock_instance

            with pytest.raises(ExternalAPIError) as exc_info:
                client._fetch_sync("INVALID")

            assert "No data returned" in str(exc_info.value)

        client.close()

    @pytest.mark.asyncio
    async def test_fetch_multiple_partial_failure(self) -> None:
        """Test fetch_multiple handles partial failures."""
        client = YahooFinanceClient()

        mock_df = pl.DataFrame(
            {
                "symbol": ["AAPL"],
                "timestamp": [datetime(2024, 1, 15, tzinfo=timezone.utc)],
                "open": [150.0],
                "high": [155.0],
                "low": [148.0],
                "close": [153.0],
                "volume": [1000000],
            }
        )

        async def mock_fetch_data(symbol: str, **kwargs) -> pl.DataFrame:
            if symbol == "INVALID":
                raise ExternalAPIError("Failed", source="yahoo")
            return mock_df

        with patch.object(client, "fetch_data", side_effect=mock_fetch_data):
            result = await client.fetch_multiple(["AAPL", "INVALID"])
            assert "AAPL" in result
            assert "INVALID" not in result

        client.close()

    def test_close_shuts_down_executor(self) -> None:
        """Test that close properly shuts down the thread pool."""
        client = YahooFinanceClient()
        assert client._executor is not None
        client.close()


class TestIngestionTaskHelpers:
    """Tests for ingestion task helper functions."""

    def test_run_async_executes_coroutine(self) -> None:
        """Test run_async helper executes coroutines in sync context."""
        from signalforge.ingestion.tasks import run_async

        async def sample_coro() -> str:
            return "success"

        result = run_async(sample_coro())
        assert result == "success"

    def test_default_symbols_configured(self) -> None:
        """Test that default symbols are configured."""
        from signalforge.ingestion.tasks import DEFAULT_SYMBOLS

        assert len(DEFAULT_SYMBOLS) > 0
        assert "AAPL" in DEFAULT_SYMBOLS
        assert "SPY" in DEFAULT_SYMBOLS
