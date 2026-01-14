"""Celery tasks for data ingestion."""

import asyncio
import logging
from decimal import Decimal

from celery import shared_task

from signalforge.core.database import get_session_context
from signalforge.ingestion.api_clients.yahoo import YahooFinanceClient
from signalforge.ingestion.scrapers.rss import MultiSourceRSSScraper
from signalforge.models.news import NewsArticle
from signalforge.models.price import Price

logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "SPY", "QQQ"]


def run_async(coro):
    """Run async function in sync context for Celery."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@shared_task(bind=True, name="signalforge.ingestion.tasks.ingest_daily_prices")
def ingest_daily_prices(_self, symbols: list[str] | None = None) -> dict:  # noqa: ARG001
    """Ingest daily prices for configured symbols."""
    symbols = symbols or DEFAULT_SYMBOLS
    return run_async(_ingest_daily_prices_async(symbols))


async def _ingest_daily_prices_async(symbols: list[str]) -> dict:
    """Async implementation of daily price ingestion."""
    client = YahooFinanceClient()
    results = {"success": [], "failed": [], "records_inserted": 0}

    try:
        data = await client.fetch_multiple(symbols, period="5d", interval="1d")

        async with get_session_context() as session:
            for symbol, df in data.items():
                try:
                    for row in df.iter_rows(named=True):
                        price = Price(
                            symbol=row["symbol"],
                            timestamp=row["timestamp"],
                            open=Decimal(str(row["open"])),
                            high=Decimal(str(row["high"])),
                            low=Decimal(str(row["low"])),
                            close=Decimal(str(row["close"])),
                            volume=row["volume"],
                            adj_close=Decimal(str(row["adj_close"])) if row.get("adj_close") else None,
                        )
                        await session.merge(price)
                        results["records_inserted"] += 1

                    results["success"].append(symbol)
                    logger.info(f"Ingested prices for {symbol}")

                except Exception as e:
                    results["failed"].append({"symbol": symbol, "error": str(e)})
                    logger.error(f"Failed to ingest {symbol}: {e}")

    except Exception as e:
        logger.error(f"Price ingestion failed: {e}")
        results["error"] = str(e)
    finally:
        client.close()

    return results


@shared_task(bind=True, name="signalforge.ingestion.tasks.ingest_historical_backfill")
def ingest_historical_backfill(
    _self,  # noqa: ARG001
    symbol: str,
    period: str = "1y",
) -> dict:
    """Backfill historical data for a new symbol."""
    return run_async(_ingest_historical_backfill_async(symbol, period))


async def _ingest_historical_backfill_async(symbol: str, period: str) -> dict:
    """Async implementation of historical backfill."""
    client = YahooFinanceClient()
    results = {"symbol": symbol, "records_inserted": 0, "success": False}

    try:
        df = await client.fetch_data(symbol, period=period, interval="1d")

        async with get_session_context() as session:
            for row in df.iter_rows(named=True):
                price = Price(
                    symbol=row["symbol"],
                    timestamp=row["timestamp"],
                    open=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    volume=row["volume"],
                    adj_close=Decimal(str(row["adj_close"])) if row.get("adj_close") else None,
                )
                await session.merge(price)
                results["records_inserted"] += 1

        results["success"] = True
        logger.info(f"Backfilled {results['records_inserted']} records for {symbol}")

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"Backfill failed for {symbol}: {e}")
    finally:
        client.close()

    return results


@shared_task(bind=True, name="signalforge.ingestion.tasks.scrape_news_rss")
def scrape_news_rss(_self) -> dict:  # noqa: ARG001
    """Scrape news from configured RSS feeds."""
    return run_async(_scrape_news_rss_async())


async def _scrape_news_rss_async() -> dict:
    """Async implementation of RSS news scraping."""
    scraper = MultiSourceRSSScraper()
    results = {"articles_scraped": 0, "articles_saved": 0, "duplicates_skipped": 0}

    try:
        articles = await scraper.scrape_all()
        results["articles_scraped"] = len(articles)

        async with get_session_context() as session:
            for article in articles:
                try:
                    from sqlalchemy import select
                    existing = await session.execute(
                        select(NewsArticle).where(NewsArticle.url == article.url)
                    )
                    if existing.scalar_one_or_none():
                        results["duplicates_skipped"] += 1
                        continue

                    news_article = NewsArticle(
                        url=article.url,
                        title=article.title,
                        source=article.source,
                        published_at=article.published_at,
                        summary=article.summary,
                        metadata_=article.metadata or {},
                        symbols=article.symbols or [],
                        categories=article.categories or [],
                    )
                    session.add(news_article)
                    results["articles_saved"] += 1

                except Exception as e:
                    logger.warning(f"Failed to save article {article.url}: {e}")

        logger.info(
            f"RSS scrape complete: {results['articles_saved']} new, "
            f"{results['duplicates_skipped']} duplicates"
        )

    except Exception as e:
        results["error"] = str(e)
        logger.error(f"RSS scrape failed: {e}")

    return results
