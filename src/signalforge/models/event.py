"""Event models for financial calendar."""

from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import DateTime, Enum as SQLEnum
from sqlalchemy import Float, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from signalforge.models.base import Base, TimestampMixin


class EventType(str, Enum):
    """Type of financial event."""

    EARNINGS = "earnings"
    DIVIDEND = "dividend"
    SPLIT = "split"
    FOMC = "fomc"
    CPI = "cpi"
    NFP = "nfp"
    GDP = "gdp"
    OPTIONS_EXPIRY = "options_expiry"
    ECONOMIC = "economic"


class EventImportance(str, Enum):
    """Importance level of the event."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Event(Base, TimestampMixin):
    """Financial event model for storing calendar events."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        index=True,
    )
    event_type: Mapped[EventType] = mapped_column(
        SQLEnum(EventType, native_enum=False, length=50),
        index=True,
        nullable=False,
    )
    event_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    importance: Mapped[EventImportance] = mapped_column(
        SQLEnum(EventImportance, native_enum=False, length=20),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(
        String(2000),
        nullable=True,
    )
    expected_value: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    actual_value: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    previous_value: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    source: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_events_symbol_date", "symbol", "event_date"),
        Index("ix_events_type_date", "event_type", "event_date"),
    )
