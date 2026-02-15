"""Order flow models for institutional activity tracking."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column

from signalforge.models.base import Base


class FlowType(str, Enum):
    """Type of order flow activity."""

    DARK_POOL = "dark_pool"
    OPTIONS = "options"
    SHORT_INTEREST = "short_interest"
    INSIDER = "insider"
    BLOCK_TRADE = "block_trade"


class FlowDirection(str, Enum):
    """Direction of flow sentiment."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class OrderFlowRecord(Base):
    """Record of institutional order flow activity."""

    __tablename__ = "orderflow_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    flow_type: Mapped[FlowType] = mapped_column(SQLEnum(FlowType), index=True, nullable=False)
    direction: Mapped[FlowDirection] = mapped_column(
        SQLEnum(FlowDirection), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, nullable=False
    )
    value: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_unusual: Mapped[bool] = mapped_column(Boolean, default=False, index=True, nullable=False)
    z_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )


class OptionsActivity(Base):
    """Options trading activity record."""

    __tablename__ = "options_activity"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, nullable=False
    )
    option_type: Mapped[str] = mapped_column(String(10), nullable=False)
    strike: Mapped[float] = mapped_column(Float, nullable=False)
    expiry: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    open_interest: Mapped[int] = mapped_column(Integer, nullable=False)
    premium: Mapped[float] = mapped_column(Float, nullable=False)
    implied_volatility: Mapped[float | None] = mapped_column(Float, nullable=True)
    delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_unusual: Mapped[bool] = mapped_column(Boolean, default=False, index=True, nullable=False)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class ShortInterest(Base):
    """Short interest reporting data."""

    __tablename__ = "short_interest"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True, nullable=False)
    report_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, nullable=False
    )
    short_interest: Mapped[int] = mapped_column(Integer, nullable=False)
    shares_outstanding: Mapped[int] = mapped_column(Integer, nullable=False)
    short_percent: Mapped[float] = mapped_column(Float, nullable=False)
    days_to_cover: Mapped[float | None] = mapped_column(Float, nullable=True)
    change_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
