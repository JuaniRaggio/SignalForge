"""Schema definitions for document processing."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Document type classification."""

    EARNINGS_REPORT = "earnings_report"
    MARKET_REPORT = "market_report"
    SECTOR_ANALYSIS = "sector_analysis"
    SEC_FILING = "sec_filing"
    ANALYST_REPORT = "analyst_report"
    NEWS_ARTICLE = "news_article"
    PRESS_RELEASE = "press_release"
    RESEARCH_NOTE = "research_note"
    UNKNOWN = "unknown"


class DocumentSource(str, Enum):
    """Document source classification."""

    SEC_EDGAR = "sec_edgar"
    ANALYST_FEED = "analyst_feed"
    NEWS_RSS = "news_rss"
    MANUAL_UPLOAD = "manual_upload"


class FinancialDocument(BaseModel):
    """Financial document schema."""

    document_id: str
    document_type: DocumentType
    source: DocumentSource
    title: str
    content: str
    symbols: list[str]
    sectors: list[str]
    published_at: datetime
    ingested_at: datetime
    version: int = 1
    content_hash: str
    metadata: dict[str, str] = Field(default_factory=dict)


class ClassificationResult(BaseModel):
    """Document classification result."""

    document_type: DocumentType
    confidence: float
    secondary_types: list[tuple[DocumentType, float]] = Field(default_factory=list)
    detected_symbols: list[str]
    detected_sectors: list[str]


class IngestionResult(BaseModel):
    """Document ingestion result."""

    document_id: str
    status: str  # "ingested", "duplicate", "error"
    is_new: bool
    version: int
    message: str | None = None
