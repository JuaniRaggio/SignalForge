"""Event schemas for API requests and responses."""

from datetime import datetime

from pydantic import BaseModel, Field

from signalforge.models.event import EventImportance, EventType


class EventCreate(BaseModel):
    """Schema for creating a new event."""

    symbol: str | None = None
    event_type: EventType
    event_date: datetime
    importance: EventImportance
    title: str = Field(..., max_length=500)
    description: str | None = Field(None, max_length=2000)
    expected_value: float | None = None
    actual_value: float | None = None
    previous_value: float | None = None
    metadata_json: dict[str, str | int | float | bool] | None = None
    source: str = Field(..., max_length=100)


class EventUpdate(BaseModel):
    """Schema for updating an existing event."""

    symbol: str | None = None
    event_type: EventType | None = None
    event_date: datetime | None = None
    importance: EventImportance | None = None
    title: str | None = Field(None, max_length=500)
    description: str | None = Field(None, max_length=2000)
    expected_value: float | None = None
    actual_value: float | None = None
    previous_value: float | None = None
    metadata_json: dict[str, str | int | float | bool] | None = None
    source: str | None = Field(None, max_length=100)


class EventResponse(BaseModel):
    """Schema for event response."""

    id: int
    symbol: str | None
    event_type: EventType
    event_date: datetime
    importance: EventImportance
    title: str
    description: str | None
    expected_value: float | None
    actual_value: float | None
    previous_value: float | None
    metadata_json: dict[str, str | int | float | bool] | None
    source: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class EarningsEvent(BaseModel):
    """Schema for earnings-specific events."""

    symbol: str
    event_date: datetime
    importance: EventImportance = EventImportance.HIGH
    eps_estimate: float | None = None
    eps_actual: float | None = None
    revenue_estimate: float | None = None
    revenue_actual: float | None = None
    title: str
    description: str | None = None
    source: str = "yfinance"

    model_config = {"from_attributes": True}


class FedEvent(BaseModel):
    """Schema for Federal Reserve events."""

    event_date: datetime
    importance: EventImportance = EventImportance.CRITICAL
    rate_decision: float | None = None
    statement_summary: str | None = None
    title: str
    description: str | None = None
    source: str = "federal_reserve"

    model_config = {"from_attributes": True}


class EconomicEvent(BaseModel):
    """Schema for economic indicator events."""

    event_date: datetime
    importance: EventImportance
    indicator_name: str
    forecast: float | None = None
    actual: float | None = None
    prior: float | None = None
    title: str
    description: str | None = None
    source: str = "fred"

    model_config = {"from_attributes": True}


class EventQuery(BaseModel):
    """Schema for querying events with filters."""

    symbol: str | None = None
    event_type: EventType | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    importance: EventImportance | None = None
    limit: int = Field(default=50, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class EventCalendarResponse(BaseModel):
    """Schema for event calendar response with pagination."""

    events: list[EventResponse]
    count: int
    total: int
    limit: int
    offset: int
