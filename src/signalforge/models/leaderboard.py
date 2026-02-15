"""Leaderboard models for paper trading rankings."""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, Numeric
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from signalforge.models.base import Base


class LeaderboardPeriod(str, enum.Enum):
    """Leaderboard time periods."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ALL_TIME = "all_time"


class LeaderboardType(str, enum.Enum):
    """Leaderboard ranking criteria."""

    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    RISK_ADJUSTED = "risk_adjusted"


class LeaderboardEntry(Base):
    """User ranking entry in leaderboard."""

    __tablename__ = "leaderboard_entries"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid4
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id"), index=True
    )
    portfolio_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), index=True)
    period: Mapped[LeaderboardPeriod] = mapped_column(
        Enum(LeaderboardPeriod, values_callable=lambda x: [e.value for e in x])
    )
    period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    leaderboard_type: Mapped[LeaderboardType] = mapped_column(
        Enum(LeaderboardType, values_callable=lambda x: [e.value for e in x])
    )
    rank: Mapped[int] = mapped_column(Integer)
    score: Mapped[Decimal] = mapped_column(Numeric(14, 6))
    total_return_pct: Mapped[Decimal] = mapped_column(Numeric(10, 4))
    sharpe_ratio: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 4), nullable=True
    )
    max_drawdown_pct: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 4), nullable=True
    )
    total_trades: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index(
            "ix_leaderboard_period_type_rank",
            "period",
            "leaderboard_type",
            "rank",
        ),
        Index("ix_leaderboard_user_period", "user_id", "period"),
    )
