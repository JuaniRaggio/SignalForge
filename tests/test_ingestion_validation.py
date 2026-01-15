"""Tests for ingestion validation schemas."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from signalforge.schemas.ingestion import (
    DataQualityStatus,
    RSSArticleRaw,
    RSSArticleValidated,
    YahooPriceDataRaw,
    YahooPriceDataValidated,
    validate_rss_articles,
    validate_yahoo_data,
)


class TestYahooPriceDataRaw:
    """Tests for raw Yahoo Finance data parsing."""

    def test_valid_raw_data(self) -> None:
        """Test parsing valid raw price data."""
        data = YahooPriceDataRaw(
            symbol="aapl",  # Should be normalized to uppercase
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
        )

        assert data.symbol == "AAPL"
        assert data.open == 150.0
        assert data.timestamp.tzinfo == UTC

    def test_symbol_normalization(self) -> None:
        """Test that symbols are normalized to uppercase."""
        data = YahooPriceDataRaw(
            symbol="  msft  ",
            timestamp=datetime.now(UTC),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=500000,
        )
        assert data.symbol == "MSFT"

    def test_timezone_added_if_missing(self) -> None:
        """Test that UTC timezone is added to naive datetimes."""
        data = YahooPriceDataRaw(
            symbol="NVDA",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),  # Naive datetime
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=500000,
        )
        assert data.timestamp.tzinfo == UTC

    def test_negative_volume_converted_to_zero(self) -> None:
        """Test that negative volumes are converted to zero."""
        data = YahooPriceDataRaw(
            symbol="TEST",
            timestamp=datetime.now(UTC),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=-100,
        )
        assert data.volume == 0

    def test_missing_optional_fields(self) -> None:
        """Test that optional fields can be None."""
        data = YahooPriceDataRaw(
            symbol="TEST",
            timestamp=datetime.now(UTC),
        )
        assert data.open is None
        assert data.adj_close is None


class TestYahooPriceDataValidated:
    """Tests for validated Yahoo Finance data."""

    def test_valid_data(self) -> None:
        """Test creating valid price data."""
        data = YahooPriceDataValidated(
            symbol="AAPL",
            timestamp=datetime.now(UTC),
            open=Decimal("150.00"),
            high=Decimal("155.00"),
            low=Decimal("149.00"),
            close=Decimal("154.00"),
            volume=1000000,
        )
        assert data.symbol == "AAPL"

    def test_high_less_than_low_fails(self) -> None:
        """Test that high < low fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            YahooPriceDataValidated(
                symbol="AAPL",
                timestamp=datetime.now(UTC),
                open=Decimal("150.00"),
                high=Decimal("145.00"),  # Less than low
                low=Decimal("149.00"),
                close=Decimal("154.00"),
                volume=1000000,
            )
        assert "cannot be less than low" in str(exc_info.value)

    def test_open_outside_range_fails(self) -> None:
        """Test that open outside high-low range fails."""
        with pytest.raises(ValidationError) as exc_info:
            YahooPriceDataValidated(
                symbol="AAPL",
                timestamp=datetime.now(UTC),
                open=Decimal("160.00"),  # Above high
                high=Decimal("155.00"),
                low=Decimal("149.00"),
                close=Decimal("154.00"),
                volume=1000000,
            )
        assert "must be between low and high" in str(exc_info.value)

    def test_negative_price_fails(self) -> None:
        """Test that negative prices fail validation."""
        with pytest.raises(ValidationError):
            YahooPriceDataValidated(
                symbol="AAPL",
                timestamp=datetime.now(UTC),
                open=Decimal("-150.00"),
                high=Decimal("155.00"),
                low=Decimal("149.00"),
                close=Decimal("154.00"),
                volume=1000000,
            )

    def test_from_raw_conversion(self) -> None:
        """Test conversion from raw to validated data."""
        raw = YahooPriceDataRaw(
            symbol="aapl",
            timestamp=datetime.now(UTC),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            adj_close=153.5,
        )

        validated = YahooPriceDataValidated.from_raw(raw)
        assert validated.symbol == "AAPL"
        assert validated.open == Decimal("150.0")
        assert validated.adj_close == Decimal("153.5")

    def test_from_raw_missing_required_fields_fails(self) -> None:
        """Test that conversion fails when required fields are missing."""
        raw = YahooPriceDataRaw(
            symbol="AAPL",
            timestamp=datetime.now(UTC),
            open=150.0,
            # Missing high, low, close, volume
        )

        with pytest.raises(ValueError) as exc_info:
            YahooPriceDataValidated.from_raw(raw)
        assert "Missing required" in str(exc_info.value)


class TestRSSArticleValidation:
    """Tests for RSS article validation."""

    def test_valid_article(self) -> None:
        """Test parsing valid RSS article."""
        article = RSSArticleValidated(
            url="https://example.com/news/article",
            title="Breaking News",
            source="Example News",
        )
        assert article.title == "Breaking News"

    def test_url_validation(self) -> None:
        """Test URL validation."""
        with pytest.raises(ValidationError):
            RSSArticleValidated(
                url="not-a-url",
                title="Test",
                source="Test Source",
            )

    def test_url_without_protocol_fails(self) -> None:
        """Test that URLs without protocol fail."""
        with pytest.raises(ValidationError):
            RSSArticleValidated(
                url="example.com/news",
                title="Test",
                source="Test Source",
            )

    def test_symbols_normalized_to_uppercase(self) -> None:
        """Test that stock symbols are normalized."""
        article = RSSArticleValidated(
            url="https://example.com/news",
            title="Test Article",
            source="Test",
            symbols=["aapl", "msft", "googl"],
        )
        assert article.symbols == ["AAPL", "MSFT", "GOOGL"]

    def test_categories_normalized_to_lowercase(self) -> None:
        """Test that categories are normalized."""
        article = RSSArticleValidated(
            url="https://example.com/news",
            title="Test Article",
            source="Test",
            categories=["Business", "FINANCE", "Markets"],
        )
        assert article.categories == ["business", "finance", "markets"]

    def test_title_whitespace_normalized(self) -> None:
        """Test that title whitespace is normalized."""
        article = RSSArticleValidated(
            url="https://example.com/news",
            title="  Multiple   Spaces   Here  ",
            source="Test",
        )
        assert article.title == "Multiple Spaces Here"

    def test_from_raw_with_alternative_fields(self) -> None:
        """Test conversion from raw with alternative field names."""
        raw = RSSArticleRaw(
            link="https://example.com/news",  # Alternative to url
            title="Test Article",
            description="This is a description",  # Alternative to summary
            published="Mon, 15 Jan 2024 10:00:00 GMT",  # Alternative to published_at
        )

        validated = RSSArticleValidated.from_raw(raw, default_source="Test Source")
        assert validated.url == "https://example.com/news"
        assert validated.summary == "This is a description"
        assert validated.source == "Test Source"


class TestBatchValidation:
    """Tests for batch validation utilities."""

    def test_validate_yahoo_data_all_valid(self) -> None:
        """Test batch validation with all valid records."""
        records = [
            {
                "symbol": "AAPL",
                "timestamp": datetime.now(UTC),
                "open": 150.0,
                "high": 155.0,
                "low": 149.0,
                "close": 154.0,
                "volume": 1000000,
            },
            {
                "symbol": "MSFT",
                "timestamp": datetime.now(UTC),
                "open": 350.0,
                "high": 355.0,
                "low": 349.0,
                "close": 354.0,
                "volume": 500000,
            },
        ]

        result = validate_yahoo_data(records)

        assert result.status == DataQualityStatus.VALID
        assert result.valid_count == 2
        assert result.invalid_count == 0
        assert len(result.valid_records) == 2

    def test_validate_yahoo_data_partial(self) -> None:
        """Test batch validation with mixed valid/invalid records."""
        records = [
            {
                "symbol": "AAPL",
                "timestamp": datetime.now(UTC),
                "open": 150.0,
                "high": 155.0,
                "low": 149.0,
                "close": 154.0,
                "volume": 1000000,
            },
            {
                "symbol": "INVALID",
                "timestamp": datetime.now(UTC),
                # Missing required fields
            },
        ]

        result = validate_yahoo_data(records)

        assert result.status == DataQualityStatus.PARTIAL
        assert result.valid_count == 1
        assert result.invalid_count == 1
        assert len(result.errors) > 0

    def test_validate_yahoo_data_all_invalid(self) -> None:
        """Test batch validation with all invalid records."""
        records = [
            {"symbol": "TEST"},  # Missing required fields
            {"symbol": "TEST2"},  # Missing required fields
        ]

        result = validate_yahoo_data(records)

        assert result.status == DataQualityStatus.INVALID
        assert result.valid_count == 0
        assert result.invalid_count == 2

    def test_validate_rss_articles_all_valid(self) -> None:
        """Test RSS batch validation with all valid articles."""
        articles = [
            {
                "url": "https://example.com/news/1",
                "title": "Article 1",
                "source": "News Site",
            },
            {
                "url": "https://example.com/news/2",
                "title": "Article 2",
                "source": "News Site",
            },
        ]

        result = validate_rss_articles(articles, source="Test Source")

        assert result.status == DataQualityStatus.VALID
        assert result.valid_count == 2
        assert result.source == "Test Source"

    def test_validate_rss_articles_partial(self) -> None:
        """Test RSS batch validation with mixed results."""
        articles = [
            {
                "url": "https://example.com/news/1",
                "title": "Valid Article",
            },
            {
                "url": "invalid-url",  # Invalid URL
                "title": "Invalid Article",
            },
        ]

        result = validate_rss_articles(articles)

        assert result.status == DataQualityStatus.PARTIAL
        assert result.valid_count == 1
        assert result.invalid_count == 1
