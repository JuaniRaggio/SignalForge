"""Paper trading models for virtual portfolios, orders, and competitions."""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from signalforge.models.base import Base, TimestampMixin


class OrderType(str, enum.Enum):
    """Order type enum."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(str, enum.Enum):
    """Order side (buy/sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, enum.Enum):
    """Order status enum."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PortfolioStatus(str, enum.Enum):
    """Portfolio status enum."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    CLOSED = "closed"


class PaperPortfolio(Base, TimestampMixin):
    """User's paper trading portfolio."""

    __tablename__ = "paper_portfolios"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    initial_capital: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    current_cash: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    status: Mapped[PortfolioStatus] = mapped_column(
        Enum(PortfolioStatus, values_callable=lambda x: [e.value for e in x]),
        default=PortfolioStatus.ACTIVE,
    )
    is_competition_portfolio: Mapped[bool] = mapped_column(Boolean, default=False)
    competition_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)

    # Relationships
    orders: Mapped[list[PaperOrder]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")
    positions: Mapped[list[PaperPosition]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")
    snapshots: Mapped[list[PaperPortfolioSnapshot]] = relationship(back_populates="portfolio", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_paper_portfolios_user_status", "user_id", "status"),
    )


class PaperOrder(Base, TimestampMixin):
    """Paper trading order."""

    __tablename__ = "paper_orders"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("paper_portfolios.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    order_type: Mapped[OrderType] = mapped_column(
        Enum(OrderType, values_callable=lambda x: [e.value for e in x])
    )
    side: Mapped[OrderSide] = mapped_column(
        Enum(OrderSide, values_callable=lambda x: [e.value for e in x])
    )
    quantity: Mapped[int] = mapped_column(Integer)
    limit_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    stop_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    status: Mapped[OrderStatus] = mapped_column(
        Enum(OrderStatus, values_callable=lambda x: [e.value for e in x]),
        default=OrderStatus.PENDING,
    )
    filled_quantity: Mapped[int] = mapped_column(Integer, default=0)
    filled_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    filled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Relationships
    portfolio: Mapped[PaperPortfolio] = relationship(back_populates="orders")
    trade: Mapped[PaperTrade | None] = relationship(back_populates="order", uselist=False)

    __table_args__ = (
        Index("ix_paper_orders_portfolio_status", "portfolio_id", "status"),
        Index("ix_paper_orders_symbol_status", "symbol", "status"),
    )


class PaperPosition(Base, TimestampMixin):
    """Active paper trading position."""

    __tablename__ = "paper_positions_v2"  # New table, different from old paper_positions

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("paper_portfolios.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    quantity: Mapped[int] = mapped_column(Integer)
    avg_entry_price: Mapped[Decimal] = mapped_column(Numeric(12, 4))
    current_price: Mapped[Decimal] = mapped_column(Numeric(12, 4))
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(14, 2), default=Decimal("0"))
    unrealized_pnl_pct: Mapped[Decimal] = mapped_column(Numeric(8, 4), default=Decimal("0"))

    # Relationships
    portfolio: Mapped[PaperPortfolio] = relationship(back_populates="positions")

    __table_args__ = (
        UniqueConstraint("portfolio_id", "symbol", name="uq_paper_position_portfolio_symbol"),
        Index("ix_paper_positions_v2_portfolio", "portfolio_id"),
    )


class PaperTrade(Base, TimestampMixin):
    """Executed paper trade (historical record)."""

    __tablename__ = "paper_trades"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("paper_portfolios.id"), index=True)
    order_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("paper_orders.id"), index=True)
    symbol: Mapped[str] = mapped_column(String(10), index=True)
    side: Mapped[OrderSide] = mapped_column(
        Enum(OrderSide, values_callable=lambda x: [e.value for e in x])
    )
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[Decimal] = mapped_column(Numeric(12, 4))
    commission: Mapped[Decimal] = mapped_column(Numeric(10, 4), default=Decimal("0"))
    slippage: Mapped[Decimal] = mapped_column(Numeric(10, 4), default=Decimal("0"))
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    # Relationships
    order: Mapped[PaperOrder] = relationship(back_populates="trade")

    __table_args__ = (
        Index("ix_paper_trades_portfolio_executed", "portfolio_id", "executed_at"),
    )


class PaperPortfolioSnapshot(Base):
    """Daily snapshot of portfolio state."""

    __tablename__ = "paper_portfolio_snapshots"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    portfolio_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("paper_portfolios.id"), index=True)
    snapshot_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    equity_value: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    cash_balance: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    positions_value: Mapped[Decimal] = mapped_column(Numeric(14, 2))
    positions_count: Mapped[int] = mapped_column(Integer)
    daily_return_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    cumulative_return_pct: Mapped[Decimal | None] = mapped_column(Numeric(10, 6), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    # Relationships
    portfolio: Mapped[PaperPortfolio] = relationship(back_populates="snapshots")

    __table_args__ = (
        UniqueConstraint("portfolio_id", "snapshot_date", name="uq_portfolio_snapshot_date"),
        Index("ix_paper_portfolio_snapshots_date", "portfolio_id", "snapshot_date"),
    )
