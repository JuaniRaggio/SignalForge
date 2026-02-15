"""Celery tasks for Events module."""

import asyncio
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any, TypeVar

import structlog
from celery import shared_task

from signalforge.core.database import get_session_context
from signalforge.events.calendar import EventCalendar
from signalforge.events.earnings import EarningsTracker
from signalforge.events.economic_releases import EconomicCalendar
from signalforge.events.fed import FedTracker

logger = structlog.get_logger(__name__)

T = TypeVar("T")

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "SPY", "QQQ"]


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run async function in sync context for Celery."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@shared_task(bind=True, name="signalforge.events.tasks.sync_earnings_calendar")  # type: ignore[untyped-decorator]
def sync_earnings_calendar(
    _self: Any,  # noqa: ARG001
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    """Sync earnings calendar for symbols using EarningsTracker."""
    symbols = symbols or DEFAULT_SYMBOLS
    return run_async(_sync_earnings_calendar_async(symbols))


async def _sync_earnings_calendar_async(symbols: list[str]) -> dict[str, Any]:
    """Async implementation of earnings calendar sync."""
    results: dict[str, Any] = {
        "success": [],
        "failed": [],
        "events_stored": 0,
    }

    async with get_session_context() as session:
        tracker = EarningsTracker(session)

        for symbol in symbols:
            try:
                earnings_events = await tracker.fetch_earnings_from_yfinance(symbol)

                for earnings_event in earnings_events:
                    await tracker.store_earnings_event(earnings_event)
                    results["events_stored"] += 1

                results["success"].append(symbol)
                logger.info(
                    "synced_earnings_calendar",
                    symbol=symbol,
                    events_count=len(earnings_events),
                )

            except Exception as e:
                results["failed"].append({"symbol": symbol, "error": str(e)})
                logger.error("failed_to_sync_earnings", symbol=symbol, error=str(e))

    return results


@shared_task(bind=True, name="signalforge.events.tasks.sync_fed_schedule")  # type: ignore[untyped-decorator]
def sync_fed_schedule(_self: Any) -> dict[str, Any]:  # noqa: ARG001
    """Sync FOMC schedule using FedTracker."""
    return run_async(_sync_fed_schedule_async())


async def _sync_fed_schedule_async() -> dict[str, Any]:
    """Async implementation of Fed schedule sync."""
    results: dict[str, Any] = {
        "events_stored": 0,
        "year": datetime.now(UTC).year,
    }

    async with get_session_context() as session:
        tracker = FedTracker(session)

        try:
            current_year = datetime.now(UTC).year
            fomc_schedule = await tracker.get_fomc_schedule(year=current_year)

            for fed_event in fomc_schedule:
                if fed_event.event_date > datetime.now(UTC):
                    await tracker.store_fed_event(fed_event)
                    results["events_stored"] += 1

            logger.info(
                "synced_fed_schedule",
                year=current_year,
                events_count=results["events_stored"],
            )
            results["success"] = True

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            logger.error("failed_to_sync_fed_schedule", error=str(e))

    return results


@shared_task(bind=True, name="signalforge.events.tasks.sync_economic_calendar")  # type: ignore[untyped-decorator]
def sync_economic_calendar(_self: Any) -> dict[str, Any]:  # noqa: ARG001
    """Sync economic releases using EconomicCalendar."""
    return run_async(_sync_economic_calendar_async())


async def _sync_economic_calendar_async() -> dict[str, Any]:
    """Async implementation of economic calendar sync."""
    results: dict[str, Any] = {
        "events_stored": 0,
        "success": False,
    }

    async with get_session_context() as session:
        calendar = EconomicCalendar(session)

        try:
            upcoming_releases = await calendar.get_upcoming_releases(days=30)

            for economic_event in upcoming_releases:
                await calendar.store_economic_event(economic_event)
                results["events_stored"] += 1

            logger.info(
                "synced_economic_calendar",
                events_count=results["events_stored"],
            )
            results["success"] = True

        except Exception as e:
            results["error"] = str(e)
            logger.error("failed_to_sync_economic_calendar", error=str(e))

        finally:
            await calendar.close()

    return results


@shared_task(bind=True, name="signalforge.events.tasks.check_upcoming_events")  # type: ignore[untyped-decorator]
def check_upcoming_events(_self: Any) -> dict[str, Any]:  # noqa: ARG001
    """Check for upcoming events in next 24h and trigger alerts."""
    return run_async(_check_upcoming_events_async())


async def _check_upcoming_events_async() -> dict[str, Any]:
    """Async implementation of upcoming events check."""
    results: dict[str, Any] = {
        "upcoming_count": 0,
        "alerts_sent": 0,
        "events": [],
    }

    async with get_session_context() as session:
        calendar = EventCalendar(session)

        try:
            upcoming_events = await calendar.get_upcoming_events(days=1)
            results["upcoming_count"] = len(upcoming_events)

            for event in upcoming_events:
                time_until = event.event_date - datetime.now(UTC)
                hours_until = time_until.total_seconds() / 3600

                event_info = {
                    "symbol": event.symbol,
                    "event_type": event.event_type.value,
                    "event_date": event.event_date.isoformat(),
                    "hours_until": round(hours_until, 2),
                    "importance": event.importance.value,
                    "title": event.title,
                }
                results["events"].append(event_info)

                logger.info(
                    "upcoming_event_alert",
                    symbol=event.symbol,
                    event_type=event.event_type.value,
                    hours_until=round(hours_until, 2),
                    importance=event.importance.value,
                )

                results["alerts_sent"] += 1

            results["success"] = True

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)
            logger.error("failed_to_check_upcoming_events", error=str(e))

    return results
