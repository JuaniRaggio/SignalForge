"""Competition models for paper trading tournaments."""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from signalforge.models.base import Base, TimestampMixin


class CompetitionStatus(str, enum.Enum):
    """Competition lifecycle status."""

    DRAFT = "draft"
    REGISTRATION_OPEN = "registration_open"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class CompetitionType(str, enum.Enum):
    """Competition type."""

    PUBLIC = "public"
    PRIVATE = "private"
    SPONSORED = "sponsored"


class Competition(Base, TimestampMixin):
    """Paper trading competition/tournament."""

    __tablename__ = "competitions"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    competition_type: Mapped[CompetitionType] = mapped_column(
        Enum(CompetitionType, values_callable=lambda x: [e.value for e in x]),
        default=CompetitionType.PUBLIC,
    )
    status: Mapped[CompetitionStatus] = mapped_column(
        Enum(CompetitionStatus, values_callable=lambda x: [e.value for e in x]),
        default=CompetitionStatus.DRAFT,
    )

    # Timing
    registration_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    registration_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    competition_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    competition_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # Configuration
    initial_capital: Mapped[Decimal] = mapped_column(Numeric(14, 2), default=Decimal("100000"))
    max_participants: Mapped[int | None] = mapped_column(Integer, nullable=True)
    min_participants: Mapped[int] = mapped_column(Integer, default=2)

    # Rules (JSON)
    rules: Mapped[dict[str, object] | None] = mapped_column(JSONB, nullable=True)
    # Example rules: {"max_position_pct": 20, "allowed_symbols": ["AAPL", "GOOGL"], "max_trades_per_day": 10}

    # Prizes (JSON)
    prizes: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)
    # Example: {"1": "Premium subscription 1 year", "2": "Premium 6 months", "3": "Premium 3 months"}

    # Creator
    created_by: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("users.id"))

    # Relationships
    participants: Mapped[list[CompetitionParticipant]] = relationship(
        back_populates="competition", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_competitions_status", "status"),
        Index("ix_competitions_dates", "competition_start", "competition_end"),
        Index("ix_competitions_type_status", "competition_type", "status"),
    )


class CompetitionParticipant(Base, TimestampMixin):
    """Participant in a competition."""

    __tablename__ = "competition_participants"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    competition_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("competitions.id"), index=True
    )
    user_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    portfolio_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), index=True)

    # Registration
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    disqualified: Mapped[bool] = mapped_column(Boolean, default=False)
    disqualification_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Final standing (set when competition ends)
    final_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    final_return_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 4), nullable=True)
    final_sharpe: Mapped[Decimal | None] = mapped_column(Numeric(8, 4), nullable=True)
    prize_awarded: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # Relationships
    competition: Mapped[Competition] = relationship(back_populates="participants")

    __table_args__ = (
        Index("ix_competition_participants_comp_user", "competition_id", "user_id", unique=True),
        Index("ix_competition_participants_portfolio", "portfolio_id"),
    )
