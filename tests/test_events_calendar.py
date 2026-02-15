"""Tests for event calendar service."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.exc import NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.events.calendar import EventCalendar
from signalforge.events.schemas import EventCreate, EventQuery, EventUpdate
from signalforge.models.event import Event, EventImportance, EventType
from signalforge.models.price import Price


async def create_test_event(
    db_session: AsyncSession,
    symbol: str = "AAPL",
    event_type: EventType = EventType.EARNINGS,
    days_offset: int = 0,
    importance: EventImportance = EventImportance.HIGH,
) -> Event:
    """Helper to create a test event."""
    event_date = datetime.now(UTC) + timedelta(days=days_offset)

    event = Event(
        symbol=symbol,
        event_type=event_type,
        event_date=event_date,
        importance=importance,
        title=f"Test {event_type.value} for {symbol}",
        description=f"Test event description for {symbol}",
        expected_value=100.0,
        actual_value=None,
        previous_value=95.0,
        metadata_json={"test": "data"},
        source="test",
    )
    db_session.add(event)
    await db_session.commit()
    await db_session.refresh(event)
    return event


@pytest.mark.asyncio
async def test_get_upcoming_events(db_session: AsyncSession) -> None:
    """Test retrieving upcoming events."""
    calendar = EventCalendar(db_session)

    # Create events at different time offsets
    await create_test_event(db_session, "AAPL", EventType.EARNINGS, days_offset=1)
    await create_test_event(db_session, "MSFT", EventType.EARNINGS, days_offset=3)
    await create_test_event(db_session, "GOOGL", EventType.DIVIDEND, days_offset=5)
    await create_test_event(db_session, "TSLA", EventType.EARNINGS, days_offset=-1)  # Past event

    events = await calendar.get_upcoming_events(days=7)

    assert len(events) == 3
    assert all(event.event_date > datetime.now(UTC) for event in events)
    assert events[0].event_date < events[-1].event_date  # Ascending order


@pytest.mark.asyncio
async def test_get_upcoming_events_filtered_by_symbols(db_session: AsyncSession) -> None:
    """Test retrieving upcoming events filtered by symbols."""
    calendar = EventCalendar(db_session)

    await create_test_event(db_session, "AAPL", EventType.EARNINGS, days_offset=1)
    await create_test_event(db_session, "MSFT", EventType.EARNINGS, days_offset=2)
    await create_test_event(db_session, "GOOGL", EventType.EARNINGS, days_offset=3)

    events = await calendar.get_upcoming_events(days=7, symbols=["AAPL", "MSFT"])

    assert len(events) == 2
    assert all(event.symbol in ["AAPL", "MSFT"] for event in events)


@pytest.mark.asyncio
async def test_get_upcoming_events_empty(db_session: AsyncSession) -> None:
    """Test retrieving upcoming events when none exist."""
    calendar = EventCalendar(db_session)

    events = await calendar.get_upcoming_events(days=7)

    assert len(events) == 0


@pytest.mark.asyncio
async def test_get_events_for_symbol(db_session: AsyncSession) -> None:
    """Test retrieving events for a specific symbol."""
    calendar = EventCalendar(db_session)

    await create_test_event(db_session, "AAPL", EventType.EARNINGS, days_offset=-10)
    await create_test_event(db_session, "AAPL", EventType.DIVIDEND, days_offset=5)
    await create_test_event(db_session, "MSFT", EventType.EARNINGS, days_offset=3)

    events = await calendar.get_events_for_symbol("AAPL", days_back=30, days_forward=30)

    assert len(events) == 2
    assert all(event.symbol == "AAPL" for event in events)
    # Descending order
    assert events[0].event_date > events[1].event_date


@pytest.mark.asyncio
async def test_get_events_for_symbol_no_events(db_session: AsyncSession) -> None:
    """Test retrieving events for symbol with no events."""
    calendar = EventCalendar(db_session)

    events = await calendar.get_events_for_symbol("NONEXISTENT", days_back=30, days_forward=30)

    assert len(events) == 0


@pytest.mark.asyncio
async def test_get_events_by_type(db_session: AsyncSession) -> None:
    """Test retrieving events by type."""
    calendar = EventCalendar(db_session)

    date_from = datetime.now(UTC) - timedelta(days=30)
    date_to = datetime.now(UTC) + timedelta(days=30)

    await create_test_event(db_session, "AAPL", EventType.EARNINGS, days_offset=1)
    await create_test_event(db_session, "MSFT", EventType.EARNINGS, days_offset=5)
    await create_test_event(db_session, "GOOGL", EventType.DIVIDEND, days_offset=3)

    events = await calendar.get_events_by_type(EventType.EARNINGS, date_from, date_to)

    assert len(events) == 2
    assert all(event.event_type == EventType.EARNINGS for event in events)


@pytest.mark.asyncio
async def test_add_event(db_session: AsyncSession) -> None:
    """Test adding a new event."""
    calendar = EventCalendar(db_session)

    event_data = EventCreate(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_date=datetime.now(UTC) + timedelta(days=7),
        importance=EventImportance.HIGH,
        title="Q4 Earnings",
        description="Apple Q4 earnings release",
        expected_value=1.50,
        source="test",
    )

    event = await calendar.add_event(event_data)

    assert event.id is not None
    assert event.symbol == "AAPL"
    assert event.event_type == EventType.EARNINGS
    assert event.title == "Q4 Earnings"


@pytest.mark.asyncio
async def test_update_event(db_session: AsyncSession) -> None:
    """Test updating an existing event."""
    calendar = EventCalendar(db_session)

    # Create initial event
    event = await create_test_event(db_session, "AAPL", EventType.EARNINGS, days_offset=7)

    # Update event
    update_data = EventUpdate(
        title="Updated Title",
        actual_value=105.0,
        description="Updated description",
    )

    updated_event = await calendar.update_event(event.id, update_data)

    assert updated_event.id == event.id
    assert updated_event.title == "Updated Title"
    assert updated_event.actual_value == 105.0
    assert updated_event.description == "Updated description"
    # Unchanged fields should remain the same
    assert updated_event.symbol == "AAPL"
    assert updated_event.expected_value == 100.0


@pytest.mark.asyncio
async def test_update_event_not_found(db_session: AsyncSession) -> None:
    """Test updating a non-existent event."""
    calendar = EventCalendar(db_session)

    update_data = EventUpdate(title="Updated Title")

    with pytest.raises(NoResultFound):
        await calendar.update_event(99999, update_data)


@pytest.mark.asyncio
async def test_delete_event(db_session: AsyncSession) -> None:
    """Test deleting an event."""
    calendar = EventCalendar(db_session)

    event = await create_test_event(db_session, "AAPL", EventType.EARNINGS, days_offset=7)
    event_id = event.id

    await calendar.delete_event(event_id)

    # Verify deletion
    query = select(Event).where(Event.id == event_id)
    result = await db_session.execute(query)
    deleted_event = result.scalar_one_or_none()

    assert deleted_event is None


@pytest.mark.asyncio
async def test_delete_event_not_found(db_session: AsyncSession) -> None:
    """Test deleting a non-existent event."""
    calendar = EventCalendar(db_session)

    with pytest.raises(NoResultFound):
        await calendar.delete_event(99999)


@pytest.mark.asyncio
async def test_get_event_by_id(db_session: AsyncSession) -> None:
    """Test retrieving an event by ID."""
    calendar = EventCalendar(db_session)

    created_event = await create_test_event(db_session, "AAPL", EventType.EARNINGS, days_offset=7)

    event = await calendar.get_event_by_id(created_event.id)

    assert event.id == created_event.id
    assert event.symbol == "AAPL"
    assert event.event_type == EventType.EARNINGS


@pytest.mark.asyncio
async def test_get_event_by_id_not_found(db_session: AsyncSession) -> None:
    """Test retrieving a non-existent event by ID."""
    calendar = EventCalendar(db_session)

    with pytest.raises(NoResultFound):
        await calendar.get_event_by_id(99999)


@pytest.mark.asyncio
async def test_search_events_no_filters(db_session: AsyncSession) -> None:
    """Test searching events without filters."""
    calendar = EventCalendar(db_session)

    await create_test_event(db_session, "AAPL", EventType.EARNINGS, days_offset=1)
    await create_test_event(db_session, "MSFT", EventType.DIVIDEND, days_offset=2)
    await create_test_event(db_session, "GOOGL", EventType.EARNINGS, days_offset=3)

    query = EventQuery()
    result = await calendar.search_events(query)

    assert result.total == 3
    assert result.count == 3
    assert len(result.events) == 3


@pytest.mark.asyncio
async def test_search_events_filter_by_symbol(db_session: AsyncSession) -> None:
    """Test searching events filtered by symbol."""
    calendar = EventCalendar(db_session)

    await create_test_event(db_session, "AAPL", EventType.EARNINGS, days_offset=1)
    await create_test_event(db_session, "AAPL", EventType.DIVIDEND, days_offset=2)
    await create_test_event(db_session, "MSFT", EventType.EARNINGS, days_offset=3)

    query = EventQuery(symbol="AAPL")
    result = await calendar.search_events(query)

    assert result.total == 2
    assert result.count == 2
    assert all(event.symbol == "AAPL" for event in result.events)


@pytest.mark.asyncio
async def test_search_events_filter_by_type(db_session: AsyncSession) -> None:
    """Test searching events filtered by type."""
    calendar = EventCalendar(db_session)

    await create_test_event(db_session, "AAPL", EventType.EARNINGS, days_offset=1)
    await create_test_event(db_session, "MSFT", EventType.EARNINGS, days_offset=2)
    await create_test_event(db_session, "GOOGL", EventType.DIVIDEND, days_offset=3)

    query = EventQuery(event_type=EventType.EARNINGS)
    result = await calendar.search_events(query)

    assert result.total == 2
    assert result.count == 2
    assert all(event.event_type == EventType.EARNINGS for event in result.events)


@pytest.mark.asyncio
async def test_search_events_filter_by_importance(db_session: AsyncSession) -> None:
    """Test searching events filtered by importance."""
    calendar = EventCalendar(db_session)

    await create_test_event(
        db_session, "AAPL", EventType.EARNINGS, days_offset=1, importance=EventImportance.HIGH
    )
    await create_test_event(
        db_session, "MSFT", EventType.EARNINGS, days_offset=2, importance=EventImportance.CRITICAL
    )
    await create_test_event(
        db_session, "GOOGL", EventType.DIVIDEND, days_offset=3, importance=EventImportance.LOW
    )

    query = EventQuery(importance=EventImportance.HIGH)
    result = await calendar.search_events(query)

    assert result.total == 1
    assert result.count == 1
    assert result.events[0].importance == EventImportance.HIGH


@pytest.mark.asyncio
async def test_search_events_filter_by_date_range(db_session: AsyncSession) -> None:
    """Test searching events filtered by date range."""
    calendar = EventCalendar(db_session)

    await create_test_event(db_session, "AAPL", EventType.EARNINGS, days_offset=1)
    await create_test_event(db_session, "MSFT", EventType.EARNINGS, days_offset=5)
    await create_test_event(db_session, "GOOGL", EventType.EARNINGS, days_offset=10)

    date_from = datetime.now(UTC)
    date_to = datetime.now(UTC) + timedelta(days=7)

    query = EventQuery(date_from=date_from, date_to=date_to)
    result = await calendar.search_events(query)

    assert result.total == 2
    assert result.count == 2
    assert all(date_from <= event.event_date <= date_to for event in result.events)


@pytest.mark.asyncio
async def test_search_events_pagination(db_session: AsyncSession) -> None:
    """Test event search pagination."""
    calendar = EventCalendar(db_session)

    # Create 10 events
    for i in range(10):
        await create_test_event(db_session, f"SYM{i}", EventType.EARNINGS, days_offset=i + 1)

    # First page
    query = EventQuery(limit=5, offset=0)
    result = await calendar.search_events(query)

    assert result.total == 10
    assert result.count == 5
    assert result.limit == 5
    assert result.offset == 0

    # Second page
    query = EventQuery(limit=5, offset=5)
    result = await calendar.search_events(query)

    assert result.total == 10
    assert result.count == 5
    assert result.limit == 5
    assert result.offset == 5


@pytest.mark.asyncio
async def test_search_events_multiple_filters(db_session: AsyncSession) -> None:
    """Test searching events with multiple filters."""
    calendar = EventCalendar(db_session)

    await create_test_event(
        db_session, "AAPL", EventType.EARNINGS, days_offset=1, importance=EventImportance.HIGH
    )
    await create_test_event(
        db_session, "AAPL", EventType.DIVIDEND, days_offset=2, importance=EventImportance.LOW
    )
    await create_test_event(
        db_session, "MSFT", EventType.EARNINGS, days_offset=3, importance=EventImportance.HIGH
    )

    query = EventQuery(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        importance=EventImportance.HIGH,
    )
    result = await calendar.search_events(query)

    assert result.total == 1
    assert result.count == 1
    assert result.events[0].symbol == "AAPL"
    assert result.events[0].event_type == EventType.EARNINGS
    assert result.events[0].importance == EventImportance.HIGH


@pytest.mark.asyncio
async def test_get_event_impact_history(db_session: AsyncSession) -> None:
    """Test retrieving event impact history."""
    calendar = EventCalendar(db_session)

    # Create past event
    event_date = datetime.now(UTC) - timedelta(days=30)
    event = Event(
        symbol="AAPL",
        event_type=EventType.EARNINGS,
        event_date=event_date,
        importance=EventImportance.HIGH,
        title="Q3 Earnings",
        description="Apple Q3 earnings",
        expected_value=1.40,
        actual_value=1.50,
        previous_value=1.30,
        metadata_json={},
        source="test",
    )
    db_session.add(event)

    # Create price data around the event
    price_before = Price(
        symbol="AAPL",
        timestamp=event_date - timedelta(days=1),
        open=150.0,
        high=152.0,
        low=149.0,
        close=151.0,
        volume=1000000,
    )
    price_after = Price(
        symbol="AAPL",
        timestamp=event_date + timedelta(days=1),
        open=152.0,
        high=155.0,
        low=151.0,
        close=154.0,
        volume=1500000,
    )
    db_session.add(price_before)
    db_session.add(price_after)
    await db_session.commit()

    # Get impact history
    history = await calendar.get_event_impact_history("AAPL", EventType.EARNINGS)

    assert len(history) == 1
    assert history[0]["price_before"] == 151.0
    assert history[0]["price_after"] == 154.0
    assert history[0]["price_change_pct"] > 0
    assert history[0]["surprise"] is not None


@pytest.mark.asyncio
async def test_get_event_impact_history_no_prices(db_session: AsyncSession) -> None:
    """Test impact history when no price data exists."""
    calendar = EventCalendar(db_session)

    await create_test_event(db_session, "AAPL", EventType.EARNINGS, days_offset=-30)

    history = await calendar.get_event_impact_history("AAPL", EventType.EARNINGS)

    # Should return empty list since no price data exists
    assert len(history) == 0


@pytest.mark.asyncio
async def test_get_event_impact_history_no_events(db_session: AsyncSession) -> None:
    """Test impact history when no events exist."""
    calendar = EventCalendar(db_session)

    history = await calendar.get_event_impact_history("AAPL", EventType.EARNINGS)

    assert len(history) == 0
