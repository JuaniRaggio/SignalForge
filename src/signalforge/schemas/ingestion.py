"""Pydantic schemas for data ingestion validation.

This module provides comprehensive validation schemas for:
- Yahoo Finance price data
- RSS feed articles
- Data quality validation
- Malformed data handling
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


class DataQualityStatus(str, Enum):
    """Status of data quality validation."""

    VALID = "valid"
    PARTIAL = "partial"  # Some fields missing but usable
    INVALID = "invalid"  # Critical data missing or malformed


# ============================================================================
# Yahoo Finance Schemas
# ============================================================================


class YahooPriceDataRaw(BaseModel):
    """Schema for raw Yahoo Finance price data before validation.

    This schema accepts the raw data format from yfinance and normalizes it.
    """

    model_config = ConfigDict(extra="ignore")

    symbol: str = Field(..., min_length=1, max_length=20)
    timestamp: datetime
    open: float | Decimal | None = None
    high: float | Decimal | None = None
    low: float | Decimal | None = None
    close: float | Decimal | None = None
    volume: int | None = None
    adj_close: float | Decimal | None = Field(default=None, alias="adj_close")

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, v: Any) -> str:
        """Normalize symbol to uppercase."""
        if isinstance(v, str):
            return v.strip().upper()
        return str(v).upper()

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_timezone(cls, v: Any) -> datetime:
        """Ensure timestamp has timezone info."""
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=UTC)
            return v
        raise ValueError(f"Invalid timestamp format: {v}")

    @field_validator("volume", mode="before")
    @classmethod
    def validate_volume(cls, v: Any) -> int | None:
        """Validate and convert volume to integer."""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return max(0, int(v))
        return None


class YahooPriceDataValidated(BaseModel):
    """Schema for validated Yahoo Finance price data.

    All required fields must be present and valid.
    """

    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(..., min_length=1, max_length=20)
    timestamp: datetime
    open: Decimal = Field(..., ge=0)
    high: Decimal = Field(..., ge=0)
    low: Decimal = Field(..., ge=0)
    close: Decimal = Field(..., ge=0)
    volume: int = Field(..., ge=0)
    adj_close: Decimal | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_price_relationships(self) -> Self:
        """Validate OHLC price relationships."""
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) cannot be less than low ({self.low})")
        if self.open < self.low or self.open > self.high:
            raise ValueError(f"Open ({self.open}) must be between low and high")
        if self.close < self.low or self.close > self.high:
            raise ValueError(f"Close ({self.close}) must be between low and high")
        return self

    @classmethod
    def from_raw(cls, raw: YahooPriceDataRaw) -> YahooPriceDataValidated:
        """Create validated data from raw data.

        Raises:
            ValidationError: If required fields are missing or invalid.
        """
        if raw.open is None or raw.high is None or raw.low is None or raw.close is None:
            raise ValueError("Missing required OHLC price fields")
        if raw.volume is None:
            raise ValueError("Missing required volume field")

        return cls(
            symbol=raw.symbol,
            timestamp=raw.timestamp,
            open=Decimal(str(raw.open)),
            high=Decimal(str(raw.high)),
            low=Decimal(str(raw.low)),
            close=Decimal(str(raw.close)),
            volume=raw.volume,
            adj_close=Decimal(str(raw.adj_close)) if raw.adj_close is not None else None,
        )


class YahooDataValidationResult(BaseModel):
    """Result of Yahoo Finance data validation."""

    status: DataQualityStatus
    valid_records: list[YahooPriceDataValidated]
    invalid_records: list[dict[str, Any]]
    errors: list[str]
    total_processed: int
    valid_count: int
    invalid_count: int


# ============================================================================
# RSS Feed Schemas
# ============================================================================


class RSSArticleRaw(BaseModel):
    """Schema for raw RSS article data before validation."""

    model_config = ConfigDict(extra="ignore")

    url: str | None = None
    link: str | None = None  # Alternative field name
    title: str | None = None
    source: str | None = None
    published_at: datetime | str | None = None
    published: str | None = None  # Alternative field name
    updated: str | None = None  # Alternative field name
    summary: str | None = None
    description: str | None = None  # Alternative field name
    content: str | None = None  # Alternative field name
    author: str | None = None
    categories: list[str] | None = None
    tags: list[str] | None = None
    symbols: list[str] | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def resolve_alternative_fields(self) -> Self:
        """Resolve alternative field names to canonical names."""
        # URL resolution
        if self.url is None and self.link is not None:
            self.url = self.link

        # Summary resolution
        if self.summary is None:
            self.summary = self.description or self.content

        # Published date resolution
        if self.published_at is None:
            self.published_at = self.published or self.updated

        return self


class RSSArticleValidated(BaseModel):
    """Schema for validated RSS article data."""

    model_config = ConfigDict(extra="forbid")

    url: Annotated[str, HttpUrl]
    title: str = Field(..., min_length=1, max_length=1000)
    source: str = Field(..., min_length=1, max_length=200)
    published_at: datetime | None = None
    summary: str | None = Field(default=None, max_length=10000)
    author: str | None = Field(default=None, max_length=200)
    categories: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("url", mode="before")
    @classmethod
    def validate_url(cls, v: Any) -> str:
        """Validate and normalize URL."""
        if not isinstance(v, str):
            raise ValueError("URL must be a string")
        url: str = v.strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return url

    @field_validator("title", mode="before")
    @classmethod
    def clean_title(cls, v: Any) -> str:
        """Clean and validate title."""
        if not isinstance(v, str):
            raise ValueError("Title must be a string")
        title: str = v.strip()
        # Remove HTML entities and normalize whitespace
        title = " ".join(title.split())
        return title

    @field_validator("symbols", mode="before")
    @classmethod
    def normalize_symbols(cls, v: Any) -> list[str]:
        """Normalize stock symbols to uppercase."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip().upper()]
        if isinstance(v, list):
            return [s.strip().upper() for s in v if isinstance(s, str) and s.strip()]
        return []

    @field_validator("categories", mode="before")
    @classmethod
    def normalize_categories(cls, v: Any) -> list[str]:
        """Normalize categories."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v.strip().lower()]
        if isinstance(v, list):
            return [c.strip().lower() for c in v if isinstance(c, str) and c.strip()]
        return []

    @classmethod
    def from_raw(
        cls,
        raw: RSSArticleRaw,
        default_source: str = "unknown",
    ) -> RSSArticleValidated:
        """Create validated article from raw data.

        Args:
            raw: The raw article data to validate.
            default_source: Default source name if not provided.

        Raises:
            ValidationError: If required fields are missing or invalid.
        """
        if not raw.url:
            raise ValueError("URL is required")
        if not raw.title:
            raise ValueError("Title is required")

        # Parse published_at if it's a string
        published_at = None
        if raw.published_at:
            if isinstance(raw.published_at, datetime):
                published_at = raw.published_at
            elif isinstance(raw.published_at, str):
                published_at = cls._parse_date(raw.published_at)

        return cls(
            url=raw.url,
            title=raw.title,
            source=raw.source or default_source,
            published_at=published_at,
            summary=raw.summary,
            author=raw.author,
            categories=raw.categories or raw.tags or [],
            symbols=raw.symbols or [],
            metadata=raw.metadata or {},
        )

    @staticmethod
    def _parse_date(date_str: str) -> datetime | None:
        """Parse various date formats."""
        from email.utils import parsedate_to_datetime

        # Try RFC 2822 format (common in RSS)
        try:
            return parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            pass

        # Try ISO format
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

        return None


class RSSValidationResult(BaseModel):
    """Result of RSS feed validation."""

    status: DataQualityStatus
    valid_articles: list[RSSArticleValidated]
    invalid_articles: list[dict[str, Any]]
    errors: list[str]
    total_processed: int
    valid_count: int
    invalid_count: int
    source: str


# ============================================================================
# Validation Utilities
# ============================================================================


def validate_yahoo_data(
    raw_records: list[dict[str, Any]],
) -> YahooDataValidationResult:
    """Validate a batch of Yahoo Finance records.

    Args:
        raw_records: List of raw price data dictionaries.

    Returns:
        Validation result with valid and invalid records separated.
    """
    valid_records: list[YahooPriceDataValidated] = []
    invalid_records: list[dict[str, Any]] = []
    errors: list[str] = []

    for i, record in enumerate(raw_records):
        try:
            raw = YahooPriceDataRaw(**record)
            validated = YahooPriceDataValidated.from_raw(raw)
            valid_records.append(validated)
        except Exception as e:
            invalid_records.append(record)
            errors.append(f"Record {i}: {e!s}")

    valid_count = len(valid_records)
    invalid_count = len(invalid_records)
    total = valid_count + invalid_count

    # Determine overall status
    if invalid_count == 0:
        status = DataQualityStatus.VALID
    elif valid_count == 0:
        status = DataQualityStatus.INVALID
    else:
        status = DataQualityStatus.PARTIAL

    return YahooDataValidationResult(
        status=status,
        valid_records=valid_records,
        invalid_records=invalid_records,
        errors=errors,
        total_processed=total,
        valid_count=valid_count,
        invalid_count=invalid_count,
    )


def validate_rss_articles(
    raw_articles: list[dict[str, Any]],
    source: str = "unknown",
) -> RSSValidationResult:
    """Validate a batch of RSS articles.

    Args:
        raw_articles: List of raw article dictionaries.
        source: Default source name for articles without one.

    Returns:
        Validation result with valid and invalid articles separated.
    """
    valid_articles: list[RSSArticleValidated] = []
    invalid_articles: list[dict[str, Any]] = []
    errors: list[str] = []

    for i, article in enumerate(raw_articles):
        try:
            raw = RSSArticleRaw(**article)
            validated = RSSArticleValidated.from_raw(raw, default_source=source)
            valid_articles.append(validated)
        except Exception as e:
            invalid_articles.append(article)
            errors.append(f"Article {i}: {e!s}")

    valid_count = len(valid_articles)
    invalid_count = len(invalid_articles)
    total = valid_count + invalid_count

    # Determine overall status
    if invalid_count == 0:
        status = DataQualityStatus.VALID
    elif valid_count == 0:
        status = DataQualityStatus.INVALID
    else:
        status = DataQualityStatus.PARTIAL

    return RSSValidationResult(
        status=status,
        valid_articles=valid_articles,
        invalid_articles=invalid_articles,
        errors=errors,
        total_processed=total,
        valid_count=valid_count,
        invalid_count=invalid_count,
        source=source,
    )
