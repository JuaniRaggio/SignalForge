"""Event calendar service for managing financial events."""

from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import func, select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.events.schemas import EventCalendarResponse, EventCreate, EventQuery, EventUpdate
from signalforge.models.event import Event, EventType
from signalforge.models.price import Price

logger = structlog.get_logger(__name__)


class EventCalendar:
    """Service for managing and querying financial events."""

    def __init__(self, db_session: AsyncSession) -> None:
        """Initialize the event calendar service.

        Args:
            db_session: Async database session for queries.
        """
        self.db = db_session
        self.logger = logger.bind(service="event_calendar")

    async def get_upcoming_events(
        self,
        days: int = 7,
        symbols: list[str] | None = None,
    ) -> list[Event]:
        """Get upcoming events for the next N days.

        Args:
            days: Number of days to look ahead.
            symbols: Optional list of symbols to filter by.

        Returns:
            List of upcoming events ordered by date.
        """
        now = datetime.now(UTC)
        end_date = now + timedelta(days=days)

        query = select(Event).where(
            Event.event_date >= now,
            Event.event_date <= end_date,
        )

        if symbols:
            query = query.where(Event.symbol.in_(symbols))

        query = query.order_by(Event.event_date.asc())

        result = await self.db.execute(query)
        events = list(result.scalars().all())

        self.logger.info(
            "Retrieved upcoming events",
            days=days,
            symbols=symbols,
            count=len(events),
        )
        return events

    async def get_events_for_symbol(
        self,
        symbol: str,
        days_back: int = 30,
        days_forward: int = 30,
    ) -> list[Event]:
        """Get events for a specific symbol in a date range.

        Args:
            symbol: Stock symbol to query.
            days_back: Number of days to look back.
            days_forward: Number of days to look forward.

        Returns:
            List of events for the symbol ordered by date descending.
        """
        now = datetime.now(UTC)
        start_date = now - timedelta(days=days_back)
        end_date = now + timedelta(days=days_forward)

        query = (
            select(Event)
            .where(
                Event.symbol == symbol,
                Event.event_date >= start_date,
                Event.event_date <= end_date,
            )
            .order_by(Event.event_date.desc())
        )

        result = await self.db.execute(query)
        events = list(result.scalars().all())

        self.logger.info(
            "Retrieved events for symbol",
            symbol=symbol,
            days_back=days_back,
            days_forward=days_forward,
            count=len(events),
        )
        return events

    async def get_events_by_type(
        self,
        event_type: EventType,
        date_from: datetime,
        date_to: datetime,
    ) -> list[Event]:
        """Get events by type within a date range.

        Args:
            event_type: Type of event to query.
            date_from: Start date for the query.
            date_to: End date for the query.

        Returns:
            List of events of the specified type ordered by date descending.
        """
        query = (
            select(Event)
            .where(
                Event.event_type == event_type,
                Event.event_date >= date_from,
                Event.event_date <= date_to,
            )
            .order_by(Event.event_date.desc())
        )

        result = await self.db.execute(query)
        events = list(result.scalars().all())

        self.logger.info(
            "Retrieved events by type",
            event_type=event_type.value,
            date_from=date_from.isoformat(),
            date_to=date_to.isoformat(),
            count=len(events),
        )
        return events

    async def add_event(self, event_data: EventCreate) -> Event:
        """Add a new event to the calendar.

        Args:
            event_data: Event creation data.

        Returns:
            The created event.

        Raises:
            IntegrityError: If event creation violates database constraints.
        """
        event = Event(
            symbol=event_data.symbol,
            event_type=event_data.event_type,
            event_date=event_data.event_date,
            importance=event_data.importance,
            title=event_data.title,
            description=event_data.description,
            expected_value=event_data.expected_value,
            actual_value=event_data.actual_value,
            previous_value=event_data.previous_value,
            metadata_json=event_data.metadata_json,
            source=event_data.source,
        )

        self.db.add(event)
        await self.db.commit()
        await self.db.refresh(event)

        self.logger.info(
            "Added new event",
            event_id=event.id,
            event_type=event.event_type.value,
            symbol=event.symbol,
        )
        return event

    async def update_event(self, event_id: int, update_data: EventUpdate) -> Event:
        """Update an existing event.

        Args:
            event_id: ID of the event to update.
            update_data: Event update data.

        Returns:
            The updated event.

        Raises:
            NoResultFound: If event does not exist.
        """
        query = select(Event).where(Event.id == event_id)
        result = await self.db.execute(query)
        event = result.scalar_one_or_none()

        if event is None:
            self.logger.error("Event not found for update", event_id=event_id)
            raise NoResultFound(f"Event with ID {event_id} not found")

        # Update only provided fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            setattr(event, field, value)

        await self.db.commit()
        await self.db.refresh(event)

        self.logger.info("Updated event", event_id=event_id, fields=list(update_dict.keys()))
        return event

    async def get_event_impact_history(
        self,
        symbol: str,
        event_type: EventType,
    ) -> list[dict[str, object]]:
        """Get historical impact of events on price.

        Analyzes how past events affected the symbol's price by calculating
        price changes before and after the event.

        Args:
            symbol: Stock symbol to analyze.
            event_type: Type of event to analyze.

        Returns:
            List of dictionaries containing event date, price changes, and event details.
        """
        # Get historical events of this type for the symbol
        query = (
            select(Event)
            .where(
                Event.symbol == symbol,
                Event.event_type == event_type,
                Event.event_date < datetime.now(UTC),
            )
            .order_by(Event.event_date.desc())
            .limit(20)
        )

        result = await self.db.execute(query)
        events = list(result.scalars().all())

        impact_history = []
        for event in events:
            # Get prices around the event date
            before_date = event.event_date - timedelta(days=1)
            after_date = event.event_date + timedelta(days=1)

            price_query = select(Price).where(
                Price.symbol == symbol,
                Price.timestamp >= before_date,
                Price.timestamp <= after_date,
            )

            price_result = await self.db.execute(price_query)
            prices = list(price_result.scalars().all())

            if len(prices) >= 2:
                # Calculate price change
                prices_sorted = sorted(prices, key=lambda p: p.timestamp)
                price_before = prices_sorted[0].close
                price_after = prices_sorted[-1].close
                price_change_pct = ((price_after - price_before) / price_before) * 100

                impact_history.append({
                    "event_date": event.event_date.isoformat(),
                    "event_title": event.title,
                    "price_before": float(price_before),
                    "price_after": float(price_after),
                    "price_change_pct": float(price_change_pct),
                    "expected_value": event.expected_value,
                    "actual_value": event.actual_value,
                    "surprise": (
                        float((event.actual_value - event.expected_value) / event.expected_value * 100)
                        if event.expected_value and event.actual_value and event.expected_value != 0
                        else None
                    ),
                })

        self.logger.info(
            "Retrieved event impact history",
            symbol=symbol,
            event_type=event_type.value,
            count=len(impact_history),
        )
        return impact_history

    async def search_events(self, query: EventQuery) -> EventCalendarResponse:
        """Search events with filters and pagination.

        Args:
            query: Query parameters for filtering and pagination.

        Returns:
            Paginated event calendar response.
        """
        # Build base query
        stmt = select(Event)

        # Apply filters
        if query.symbol:
            stmt = stmt.where(Event.symbol == query.symbol)
        if query.event_type:
            stmt = stmt.where(Event.event_type == query.event_type)
        if query.date_from:
            stmt = stmt.where(Event.event_date >= query.date_from)
        if query.date_to:
            stmt = stmt.where(Event.event_date <= query.date_to)
        if query.importance:
            stmt = stmt.where(Event.importance == query.importance)

        # Get total count
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total_result = await self.db.execute(count_stmt)
        total = total_result.scalar_one()

        # Apply ordering and pagination
        stmt = stmt.order_by(Event.event_date.desc())
        stmt = stmt.limit(query.limit).offset(query.offset)

        result = await self.db.execute(stmt)
        events = list(result.scalars().all())

        self.logger.info(
            "Searched events",
            filters={
                "symbol": query.symbol,
                "event_type": query.event_type.value if query.event_type else None,
                "importance": query.importance.value if query.importance else None,
            },
            total=total,
            count=len(events),
        )

        return EventCalendarResponse(
            events=events,
            count=len(events),
            total=total,
            limit=query.limit,
            offset=query.offset,
        )

    async def delete_event(self, event_id: int) -> None:
        """Delete an event from the calendar.

        Args:
            event_id: ID of the event to delete.

        Raises:
            NoResultFound: If event does not exist.
        """
        query = select(Event).where(Event.id == event_id)
        result = await self.db.execute(query)
        event = result.scalar_one_or_none()

        if event is None:
            self.logger.error("Event not found for deletion", event_id=event_id)
            raise NoResultFound(f"Event with ID {event_id} not found")

        await self.db.delete(event)
        await self.db.commit()

        self.logger.info("Deleted event", event_id=event_id)

    async def get_event_by_id(self, event_id: int) -> Event:
        """Get an event by its ID.

        Args:
            event_id: ID of the event to retrieve.

        Returns:
            The event.

        Raises:
            NoResultFound: If event does not exist.
        """
        query = select(Event).where(Event.id == event_id)
        result = await self.db.execute(query)
        event = result.scalar_one_or_none()

        if event is None:
            self.logger.error("Event not found", event_id=event_id)
            raise NoResultFound(f"Event with ID {event_id} not found")

        return event
