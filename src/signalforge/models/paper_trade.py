"""SQLAlchemy models for paper trading positions and portfolio snapshots."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import DateTime, Index, Integer, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

from signalforge.models.base import Base


class LegacyPaperPosition(Base):
    """Paper trading position model.

    Tracks individual trades in the paper portfolio with entry/exit details,
    stop-loss/take-profit levels, and P&L tracking.
    """

    __tablename__ = "paper_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_id: Mapped[str] = mapped_column(
        String(36), index=True, comment="UUID of originating signal"
    )
    symbol: Mapped[str] = mapped_column(String(10), index=True, comment="Stock symbol")
    direction: Mapped[str] = mapped_column(String(10), comment="long or short")
    entry_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), comment="Position entry date"
    )
    entry_price: Mapped[Decimal] = mapped_column(Numeric(12, 4), comment="Entry price")
    shares: Mapped[int] = mapped_column(Integer, comment="Number of shares")
    stop_loss: Mapped[Decimal] = mapped_column(Numeric(12, 4), comment="Stop loss price")
    take_profit: Mapped[Decimal] = mapped_column(Numeric(12, 4), comment="Take profit price")
    status: Mapped[str] = mapped_column(
        String(20), default="open", index=True, comment="open or closed"
    )
    exit_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="stop_loss, take_profit, timeout, manual"
    )
    pnl: Mapped[Decimal | None] = mapped_column(
        Numeric(14, 2), nullable=True, comment="Profit/Loss in dollars"
    )
    pnl_pct: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 4), nullable=True, comment="Profit/Loss percentage"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_paper_positions_symbol_status", "symbol", "status"),
        Index("ix_paper_positions_entry_date", "entry_date"),
    )


class PortfolioSnapshot(Base):
    """Daily snapshot of paper portfolio state.

    Records the portfolio value, positions, and returns at end of each trading day.
    Used for equity curve and performance tracking.
    """

    __tablename__ = "portfolio_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), index=True, unique=True, comment="Date of snapshot (EOD)"
    )
    equity_value: Mapped[Decimal] = mapped_column(Numeric(14, 2), comment="Total portfolio value")
    cash_balance: Mapped[Decimal] = mapped_column(Numeric(14, 2), comment="Cash not in positions")
    positions_value: Mapped[Decimal] = mapped_column(
        Numeric(14, 2), comment="Value of open positions"
    )
    positions_count: Mapped[int] = mapped_column(Integer, comment="Number of open positions")
    daily_return_pct: Mapped[Decimal | None] = mapped_column(
        Numeric(8, 4), nullable=True, comment="Daily return %"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
    )
