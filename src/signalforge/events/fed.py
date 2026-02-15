"""Federal Reserve events tracking and analysis module."""

from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.events.schemas import FedEvent
from signalforge.models.event import Event, EventImportance, EventType

logger = structlog.get_logger(__name__)


class FedTracker:
    """Service for tracking Federal Reserve events and monetary policy."""

    # FOMC meeting schedule (approximate dates - should be updated annually)
    FOMC_SCHEDULE_2024 = [
        datetime(2024, 1, 31, tzinfo=UTC),
        datetime(2024, 3, 20, tzinfo=UTC),
        datetime(2024, 5, 1, tzinfo=UTC),
        datetime(2024, 6, 12, tzinfo=UTC),
        datetime(2024, 7, 31, tzinfo=UTC),
        datetime(2024, 9, 18, tzinfo=UTC),
        datetime(2024, 11, 7, tzinfo=UTC),
        datetime(2024, 12, 18, tzinfo=UTC),
    ]

    FOMC_SCHEDULE_2025 = [
        datetime(2025, 1, 29, tzinfo=UTC),
        datetime(2025, 3, 19, tzinfo=UTC),
        datetime(2025, 5, 7, tzinfo=UTC),
        datetime(2025, 6, 18, tzinfo=UTC),
        datetime(2025, 7, 30, tzinfo=UTC),
        datetime(2025, 9, 17, tzinfo=UTC),
        datetime(2025, 10, 29, tzinfo=UTC),
        datetime(2025, 12, 10, tzinfo=UTC),
    ]

    FOMC_SCHEDULE_2026 = [
        datetime(2026, 1, 28, tzinfo=UTC),
        datetime(2026, 3, 18, tzinfo=UTC),
        datetime(2026, 4, 29, tzinfo=UTC),
        datetime(2026, 6, 17, tzinfo=UTC),
        datetime(2026, 7, 29, tzinfo=UTC),
        datetime(2026, 9, 16, tzinfo=UTC),
        datetime(2026, 11, 4, tzinfo=UTC),
        datetime(2026, 12, 16, tzinfo=UTC),
    ]

    def __init__(self, db_session: AsyncSession) -> None:
        """Initialize the Fed tracker.

        Args:
            db_session: Async database session for queries.
        """
        self.db = db_session
        self.logger = logger.bind(service="fed_tracker")

    async def get_fomc_schedule(self, year: int | None = None) -> list[FedEvent]:
        """Get FOMC meeting schedule for a given year.

        Args:
            year: Year to get schedule for. If None, returns current year.

        Returns:
            List of FOMC meeting events.
        """
        if year is None:
            year = datetime.now(UTC).year

        # Get schedule from predefined dates
        schedule_map = {
            2024: self.FOMC_SCHEDULE_2024,
            2025: self.FOMC_SCHEDULE_2025,
            2026: self.FOMC_SCHEDULE_2026,
        }

        fomc_dates = schedule_map.get(year, [])

        if not fomc_dates:
            self.logger.warning("No FOMC schedule available for year", year=year)
            return []

        # Check database for stored FOMC events
        query = (
            select(Event)
            .where(
                Event.event_type == EventType.FOMC,
                Event.event_date >= datetime(year, 1, 1, tzinfo=UTC),
                Event.event_date < datetime(year + 1, 1, 1, tzinfo=UTC),
            )
            .order_by(Event.event_date.asc())
        )

        result = await self.db.execute(query)
        stored_events = list(result.scalars().all())

        fed_events = []
        for event in stored_events:
            metadata = event.metadata_json or {}
            fed_event = FedEvent(
                event_date=event.event_date,
                importance=event.importance,
                rate_decision=metadata.get("rate_decision"),
                statement_summary=metadata.get("statement_summary"),
                title=event.title,
                description=event.description,
                source=event.source,
            )
            fed_events.append(fed_event)

        # If no stored events, create placeholder events from schedule
        if not fed_events:
            for date in fomc_dates:
                fed_event = FedEvent(
                    event_date=date,
                    importance=EventImportance.CRITICAL,
                    title=f"FOMC Meeting - {date.strftime('%B %Y')}",
                    description="Federal Open Market Committee meeting",
                    source="federal_reserve",
                )
                fed_events.append(fed_event)

        self.logger.info("Retrieved FOMC schedule", year=year, count=len(fed_events))
        return fed_events

    async def get_next_fomc_meeting(self) -> FedEvent | None:
        """Get the next upcoming FOMC meeting.

        Returns:
            Next FOMC meeting event or None if not found.
        """
        now = datetime.now(UTC)

        # Combine all schedules and filter for future dates
        all_dates = (
            self.FOMC_SCHEDULE_2024 + self.FOMC_SCHEDULE_2025 + self.FOMC_SCHEDULE_2026
        )
        future_dates = [date for date in all_dates if date > now]

        if not future_dates:
            self.logger.warning("No upcoming FOMC meetings found")
            return None

        next_date = min(future_dates)

        # Check if we have stored event data
        query = select(Event).where(
            Event.event_type == EventType.FOMC,
            Event.event_date == next_date,
        )

        result = await self.db.execute(query)
        event = result.scalar_one_or_none()

        if event:
            metadata = event.metadata_json or {}
            fed_event = FedEvent(
                event_date=event.event_date,
                importance=event.importance,
                rate_decision=metadata.get("rate_decision"),
                statement_summary=metadata.get("statement_summary"),
                title=event.title,
                description=event.description,
                source=event.source,
            )
        else:
            # Create placeholder event
            fed_event = FedEvent(
                event_date=next_date,
                importance=EventImportance.CRITICAL,
                title=f"FOMC Meeting - {next_date.strftime('%B %Y')}",
                description="Federal Open Market Committee meeting",
                source="federal_reserve",
            )

        self.logger.info("Retrieved next FOMC meeting", event_date=next_date.isoformat())
        return fed_event

    def parse_fed_statement(self, text: str) -> dict[str, str | list[str] | int | None]:
        """Parse a Federal Reserve statement for key information.

        This is a simplified implementation. In production, you might use
        NLP techniques for more sophisticated analysis.

        Args:
            text: Full text of the Fed statement.

        Returns:
            Dictionary containing parsed information including sentiment,
            key phrases, and rate action.
        """
        text_lower = text.lower()

        # Determine sentiment based on keywords
        hawkish_keywords = [
            "inflation",
            "tighten",
            "raise",
            "increase",
            "elevated",
            "persistent",
        ]
        dovish_keywords = [
            "support",
            "accommodative",
            "lower",
            "decrease",
            "patient",
            "gradual",
        ]

        hawkish_count = sum(1 for keyword in hawkish_keywords if keyword in text_lower)
        dovish_count = sum(1 for keyword in dovish_keywords if keyword in text_lower)

        if hawkish_count > dovish_count:
            sentiment = "hawkish"
        elif dovish_count > hawkish_count:
            sentiment = "dovish"
        else:
            sentiment = "neutral"

        # Extract key phrases (simplified - just looking for common phrases)
        key_phrases = []
        if "inflation" in text_lower:
            key_phrases.append("inflation concerns")
        if "employment" in text_lower or "labor market" in text_lower:
            key_phrases.append("employment/labor market")
        if "economic growth" in text_lower:
            key_phrases.append("economic growth")
        if "interest rate" in text_lower:
            key_phrases.append("interest rate policy")

        # Detect rate action
        rate_action = None
        if "raise" in text_lower or "increase" in text_lower:
            if "interest rate" in text_lower or "federal funds rate" in text_lower:
                rate_action = "increase"
        elif "lower" in text_lower or "decrease" in text_lower:
            if "interest rate" in text_lower or "federal funds rate" in text_lower:
                rate_action = "decrease"
        elif ("maintain" in text_lower or "unchanged" in text_lower) and (
            "interest rate" in text_lower or "federal funds rate" in text_lower
        ):
            rate_action = "hold"

        self.logger.info(
            "Parsed Fed statement",
            sentiment=sentiment,
            rate_action=rate_action,
            key_phrases_count=len(key_phrases),
        )

        return {
            "sentiment": sentiment,
            "key_phrases": key_phrases,
            "rate_action": rate_action,
            "hawkish_count": hawkish_count,
            "dovish_count": dovish_count,
        }

    async def get_fed_rate_history(self) -> list[dict[str, object]]:
        """Get historical Federal Reserve rate decisions.

        Returns:
            List of historical rate decisions with dates and values.
        """
        query = (
            select(Event)
            .where(Event.event_type == EventType.FOMC)
            .where(Event.event_date < datetime.now(UTC))
            .order_by(Event.event_date.desc())
            .limit(20)
        )

        result = await self.db.execute(query)
        events = list(result.scalars().all())

        rate_history = []
        for event in events:
            metadata = event.metadata_json or {}
            rate_decision = metadata.get("rate_decision")

            if rate_decision is not None:
                rate_history.append({
                    "date": event.event_date.isoformat(),
                    "rate": float(rate_decision),
                    "title": event.title,
                    "description": event.description or "",
                })

        self.logger.info("Retrieved Fed rate history", count=len(rate_history))
        return rate_history

    def calculate_fed_sentiment_delta(self, current: str, previous: str) -> float:
        """Calculate change in Fed sentiment between statements.

        Args:
            current: Current statement text.
            previous: Previous statement text.

        Returns:
            Sentiment delta score. Positive means more hawkish, negative means more dovish.
        """
        current_parsed = self.parse_fed_statement(current)
        previous_parsed = self.parse_fed_statement(previous)

        # Calculate sentiment scores
        sentiment_map = {"hawkish": 1.0, "neutral": 0.0, "dovish": -1.0}

        current_score = sentiment_map.get(str(current_parsed["sentiment"]), 0.0)
        previous_score = sentiment_map.get(str(previous_parsed["sentiment"]), 0.0)

        delta = current_score - previous_score

        self.logger.info(
            "Calculated Fed sentiment delta",
            current_sentiment=current_parsed["sentiment"],
            previous_sentiment=previous_parsed["sentiment"],
            delta=delta,
        )

        return delta

    async def store_fed_event(self, fed_event: FedEvent) -> Event:
        """Store a Federal Reserve event in the database.

        Args:
            fed_event: Fed event to store.

        Returns:
            The stored Event.
        """
        metadata = {
            "rate_decision": fed_event.rate_decision,
            "statement_summary": fed_event.statement_summary,
        }

        event = Event(
            symbol=None,
            event_type=EventType.FOMC,
            event_date=fed_event.event_date,
            importance=fed_event.importance,
            title=fed_event.title,
            description=fed_event.description,
            expected_value=None,
            actual_value=fed_event.rate_decision,
            metadata_json=metadata,
            source=fed_event.source,
        )

        self.db.add(event)
        await self.db.commit()
        await self.db.refresh(event)

        self.logger.info("Stored Fed event", event_id=event.id, event_date=event.event_date)
        return event

    async def fetch_fed_data_from_fred(
        self,
        series_id: str = "FEDFUNDS",
    ) -> list[dict[str, str | float]]:
        """Fetch Federal Funds Rate data from FRED API.

        Note: This is a placeholder. In production, you would need a FRED API key
        and proper error handling.

        Args:
            series_id: FRED series ID (default: FEDFUNDS for Federal Funds Rate).

        Returns:
            List of data points with dates and values.
        """
        self.logger.info("Fetching Fed data from FRED", series_id=series_id)

        # This would require FRED API key in production
        # For now, return empty list as placeholder
        self.logger.warning("FRED API integration not implemented - returning empty data")

        return []
