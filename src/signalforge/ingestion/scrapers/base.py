"""Base class for web scrapers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ScrapedArticle:
    """Represents a scraped news article."""

    url: str
    title: str
    source: str
    published_at: datetime | None = None
    content: str | None = None
    summary: str | None = None
    metadata: dict[str, Any] | None = None
    symbols: list[str] | None = None
    categories: list[str] | None = None


class BaseScraper(ABC):
    """Abstract base class for scrapers."""

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name

    @abstractmethod
    async def scrape(self) -> list[ScrapedArticle]:
        """Scrape and return articles. Must be implemented by subclasses."""
        pass

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        url = url.split("?")[0]
        url = url.rstrip("/")
        return url
