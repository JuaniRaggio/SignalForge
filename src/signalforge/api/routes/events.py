"""Event calendar routes."""

from datetime import datetime
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.api.dependencies.auth import get_current_active_user
from signalforge.api.dependencies.database import get_db
from signalforge.events.calendar import EventCalendar
from signalforge.events.earnings import EarningsTracker
from signalforge.events.economic_releases import EconomicCalendar
from signalforge.events.fed import FedTracker
from signalforge.events.schemas import (
    EarningsEvent,
    EconomicEvent,
    EventCalendarResponse,
    EventCreate,
    EventQuery,
    EventResponse,
    EventUpdate,
    FedEvent,
)
from signalforge.models.event import EventImportance, EventType
from signalforge.models.user import User

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/events", response_model=EventCalendarResponse)
async def list_events(
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    symbol: str | None = Query(None, description="Filter by stock symbol"),
    event_type: EventType | None = Query(None, description="Filter by event type"),
    date_from: datetime | None = Query(None, description="Start date for filtering"),
    date_to: datetime | None = Query(None, description="End date for filtering"),
    importance: EventImportance | None = Query(None, description="Filter by importance level"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of records"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
) -> EventCalendarResponse:
    """List events with optional filters and pagination."""
    calendar = EventCalendar(db)

    query = EventQuery(
        symbol=symbol,
        event_type=event_type,
        date_from=date_from,
        date_to=date_to,
        importance=importance,
        limit=limit,
        offset=offset,
    )

    result = await calendar.search_events(query)

    logger.info(
        "Listed events",
        filters={
            "symbol": symbol,
            "event_type": event_type.value if event_type else None,
            "importance": importance.value if importance else None,
        },
        total=result.total,
        count=result.count,
    )

    return result


@router.post("/events", response_model=EventResponse, status_code=status.HTTP_201_CREATED)
async def create_event(
    event_data: EventCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
) -> EventResponse:
    """Create a new event."""
    calendar = EventCalendar(db)

    try:
        event = await calendar.add_event(event_data)
        logger.info(
            "Created event",
            event_id=event.id,
            event_type=event.event_type.value,
            symbol=event.symbol,
        )
        return EventResponse.model_validate(event)
    except IntegrityError as e:
        logger.error("Failed to create event", error=str(e))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create event. Check data integrity.",
        )


# Specific routes MUST come before generic /{event_id} routes
@router.get("/events/economic", response_model=list[EconomicEvent])
async def get_economic_releases(
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    days: int = Query(14, ge=1, le=90, description="Number of days to look ahead"),
) -> list[EconomicEvent]:
    """Get upcoming economic releases calendar."""
    calendar = EconomicCalendar(db)

    try:
        releases = await calendar.get_upcoming_releases(days=days)

        logger.info(
            "Retrieved economic releases",
            days=days,
            count=len(releases),
        )

        return releases
    finally:
        await calendar.close()


@router.get("/events/earnings/upcoming", response_model=list[EarningsEvent])
async def get_upcoming_earnings(
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    days: int = Query(7, ge=1, le=30, description="Number of days to look ahead"),
    symbols: list[str] | None = Query(None, description="Optional list of symbols to filter"),
) -> list[EarningsEvent]:
    """Get upcoming earnings events."""
    tracker = EarningsTracker(db)

    # Convert symbols to uppercase if provided
    if symbols:
        symbols = [s.upper() for s in symbols]

    # Get all upcoming earnings, then filter if symbols provided
    # Note: This could be optimized by modifying the service method
    earnings = await tracker.get_earnings_calendar(
        symbols if symbols else [],
        days=days,
    )

    logger.info(
        "Retrieved upcoming earnings",
        days=days,
        symbols=symbols,
        count=len(earnings),
    )

    return earnings


@router.get("/events/earnings/{symbol}", response_model=list[EarningsEvent])
async def get_earnings_for_symbol(
    symbol: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    days: int = Query(30, ge=1, le=365, description="Number of days to look ahead"),
) -> list[EarningsEvent]:
    """Get earnings calendar for a specific symbol."""
    symbol = symbol.upper()
    tracker = EarningsTracker(db)

    earnings = await tracker.get_earnings_calendar([symbol], days=days)

    logger.info(
        "Retrieved earnings for symbol",
        symbol=symbol,
        days=days,
        count=len(earnings),
    )

    return earnings


@router.get("/events/fed/schedule", response_model=list[FedEvent])
async def get_fomc_schedule(
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    year: int | None = Query(None, ge=2020, le=2030, description="Year for FOMC schedule"),
) -> list[FedEvent]:
    """Get Federal Open Market Committee (FOMC) meeting schedule."""
    tracker = FedTracker(db)

    schedule = await tracker.get_fomc_schedule(year=year)

    logger.info(
        "Retrieved FOMC schedule",
        year=year or datetime.now().year,
        count=len(schedule),
    )

    return schedule


# Generic /{event_id} routes MUST come after specific routes
@router.get("/events/{event_id}", response_model=EventResponse)
async def get_event(
    event_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
) -> EventResponse:
    """Get specific event by ID."""
    calendar = EventCalendar(db)

    try:
        event = await calendar.get_event_by_id(event_id)
        logger.info("Retrieved event", event_id=event_id)
        return EventResponse.model_validate(event)
    except NoResultFound:
        logger.warning("Event not found", event_id=event_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event with ID {event_id} not found",
        )


@router.put("/events/{event_id}", response_model=EventResponse)
async def update_event(
    event_id: int,
    update_data: EventUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
) -> EventResponse:
    """Update an existing event."""
    calendar = EventCalendar(db)

    try:
        event = await calendar.update_event(event_id, update_data)
        logger.info("Updated event", event_id=event_id)
        return EventResponse.model_validate(event)
    except NoResultFound:
        logger.warning("Event not found for update", event_id=event_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event with ID {event_id} not found",
        )
    except IntegrityError as e:
        logger.error("Failed to update event", event_id=event_id, error=str(e))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update event. Check data integrity.",
        )


@router.delete("/events/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_event(
    event_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
) -> None:
    """Delete an event."""
    calendar = EventCalendar(db)

    try:
        await calendar.delete_event(event_id)
        logger.info("Deleted event", event_id=event_id)
    except NoResultFound:
        logger.warning("Event not found for deletion", event_id=event_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event with ID {event_id} not found",
        )
