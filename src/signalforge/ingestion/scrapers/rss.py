"""RSS feed scraper for financial news."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx

from signalforge.ingestion.scrapers.base import BaseScraper, ScrapedArticle

logger = logging.getLogger(__name__)


@dataclass
class RSSFeed:
    """Configuration for an RSS feed."""

    name: str
    url: str
    category: str


DEFAULT_FEEDS: list[RSSFeed] = [
    RSSFeed(
        name="Yahoo Finance",
        url="https://finance.yahoo.com/news/rssindex",
        category="general",
    ),
    RSSFeed(
        name="MarketWatch",
        url="https://feeds.marketwatch.com/marketwatch/topstories/",
        category="general",
    ),
    RSSFeed(
        name="Seeking Alpha",
        url="https://seekingalpha.com/market_currents.xml",
        category="analysis",
    ),
    RSSFeed(
        name="Reuters Business",
        url="https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        category="general",
    ),
]


class RSSScraperError(Exception):
    """RSS scraper specific error."""

    pass


class RSSFeedScraper(BaseScraper):
    """Scraper for a single RSS feed."""

    def __init__(
        self,
        feed: RSSFeed,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(feed.name)
        self.feed = feed
        self.timeout = timeout

    def _parse_date(self, date_str: str | None) -> datetime | None:
        """Parse various date formats from RSS feeds."""
        if not date_str:
            return None

        try:
            return parsedate_to_datetime(date_str)
        except (ValueError, TypeError):
            pass

        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

        return None

    def _extract_metadata(self, entry: Any) -> dict[str, Any]:
        """Extract additional metadata from feed entry."""
        metadata: dict[str, Any] = {}

        if hasattr(entry, "author"):
            metadata["author"] = entry.author

        if hasattr(entry, "tags"):
            metadata["tags"] = [tag.term for tag in entry.tags if hasattr(tag, "term")]

        if hasattr(entry, "media_content"):
            metadata["media"] = [
                {"url": m.get("url"), "type": m.get("type")}
                for m in entry.media_content
            ]

        return metadata

    async def scrape(self) -> list[ScrapedArticle]:
        """Scrape articles from the RSS feed."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.feed.url)
                response.raise_for_status()
                content = response.text

        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch {self.feed.name}: {e}")
            return []

        try:
            parsed = feedparser.parse(content)
        except Exception as e:
            logger.error(f"Failed to parse {self.feed.name}: {e}")
            return []

        articles: list[ScrapedArticle] = []

        for entry in parsed.entries:
            try:
                url = self._normalize_url(entry.get("link", ""))
                if not url:
                    continue

                title = entry.get("title", "").strip()
                if not title:
                    continue

                published_str = entry.get("published") or entry.get("updated")
                published_at = self._parse_date(published_str)

                summary = entry.get("summary") or entry.get("description")
                if summary:
                    summary = summary.strip()

                metadata = self._extract_metadata(entry)

                article = ScrapedArticle(
                    url=url,
                    title=title,
                    source=self.feed.name,
                    published_at=published_at,
                    summary=summary,
                    metadata=metadata,
                    categories=[self.feed.category],
                )

                articles.append(article)

            except Exception as e:
                logger.warning(f"Failed to parse entry in {self.feed.name}: {e}")
                continue

        logger.info(f"Scraped {len(articles)} articles from {self.feed.name}")
        return articles


class MultiSourceRSSScraper:
    """Aggregates articles from multiple RSS feeds."""

    def __init__(
        self,
        feeds: list[RSSFeed] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.feeds = feeds or DEFAULT_FEEDS
        self.timeout = timeout
        self._seen_urls: set[str] = set()

    async def scrape_all(self) -> list[ScrapedArticle]:
        """Scrape all configured feeds and deduplicate results."""
        scrapers = [
            RSSFeedScraper(feed, timeout=self.timeout)
            for feed in self.feeds
        ]

        tasks = [scraper.scrape() for scraper in scrapers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_articles: list[ScrapedArticle] = []

        for feed, result in zip(self.feeds, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(f"Scraper failed for {feed.name}: {result}")
                continue

            # result is now guaranteed to be list[ScrapedArticle]
            for article in result:
                normalized_url = article.url.lower()
                if normalized_url not in self._seen_urls:
                    self._seen_urls.add(normalized_url)
                    all_articles.append(article)

        all_articles.sort(
            key=lambda a: a.published_at or datetime.min.replace(tzinfo=UTC),
            reverse=True,
        )

        logger.info(
            f"Total unique articles scraped: {len(all_articles)} from {len(self.feeds)} feeds"
        )

        return all_articles

    def reset_dedup_cache(self) -> None:
        """Reset the deduplication cache."""
        self._seen_urls.clear()
