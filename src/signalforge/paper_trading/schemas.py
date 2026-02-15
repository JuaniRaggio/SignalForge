"""Pydantic schemas for paper trading API."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from signalforge.models.paper_trading import (
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioStatus,
)


# Portfolio Schemas
class PortfolioCreate(BaseModel):
    """Schema for creating a new portfolio."""
    name: str = Field(..., min_length=1, max_length=100, description="Portfolio name")
    initial_capital: Decimal = Field(..., gt=0, description="Starting capital in dollars")
    is_competition_portfolio: bool = Field(default=False, description="Is this a competition portfolio")
    competition_id: UUID | None = Field(default=None, description="Competition ID if applicable")

    model_config = ConfigDict(from_attributes=True)


class PortfolioResponse(BaseModel):
    """Schema for portfolio response."""
    id: UUID
    user_id: UUID
    name: str
    initial_capital: Decimal
    current_cash: Decimal
    status: PortfolioStatus
    is_competition_portfolio: bool
    competition_id: UUID | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PortfolioSummary(BaseModel):
    """Schema for portfolio summary with calculated metrics."""
    id: UUID
    name: str
    equity_value: Decimal = Field(..., description="Total portfolio value (cash + positions)")
    cash_balance: Decimal
    positions_value: Decimal = Field(..., description="Total value of open positions")
    positions_count: int
    total_pnl: Decimal = Field(..., description="Total profit/loss (realized + unrealized)")
    total_pnl_pct: Decimal = Field(..., description="Total P&L as percentage of initial capital")
    status: PortfolioStatus

    model_config = ConfigDict(from_attributes=True)


# Order Schemas
class OrderCreate(BaseModel):
    """Schema for creating a new order."""
    symbol: str = Field(..., min_length=1, max_length=10, description="Stock symbol")
    order_type: OrderType = Field(..., description="Type of order")
    side: OrderSide = Field(..., description="Buy or sell")
    quantity: int = Field(..., gt=0, description="Number of shares")
    limit_price: Decimal | None = Field(default=None, gt=0, description="Limit price for limit orders")
    stop_price: Decimal | None = Field(default=None, gt=0, description="Stop price for stop orders")
    expires_at: datetime | None = Field(default=None, description="Order expiration time")

    model_config = ConfigDict(from_attributes=True)


class OrderResponse(BaseModel):
    """Schema for order response."""
    id: UUID
    portfolio_id: UUID
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: int
    limit_price: Decimal | None
    stop_price: Decimal | None
    status: OrderStatus
    filled_quantity: int
    filled_price: Decimal | None
    filled_at: datetime | None
    expires_at: datetime | None
    rejection_reason: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Position Schemas
class PositionResponse(BaseModel):
    """Schema for position response."""
    id: UUID
    portfolio_id: UUID
    symbol: str
    quantity: int
    avg_entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: Decimal
    market_value: Decimal = Field(..., description="Current market value of position")
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Trade Schemas
class TradeResponse(BaseModel):
    """Schema for trade response."""
    id: UUID
    portfolio_id: UUID
    order_id: UUID
    symbol: str
    side: OrderSide
    quantity: int
    price: Decimal
    commission: Decimal
    slippage: Decimal
    executed_at: datetime
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Snapshot Schemas
class SnapshotResponse(BaseModel):
    """Schema for portfolio snapshot response."""
    id: UUID
    portfolio_id: UUID
    snapshot_date: datetime
    equity_value: Decimal
    cash_balance: Decimal
    positions_value: Decimal
    positions_count: int
    daily_return_pct: Decimal | None
    cumulative_return_pct: Decimal | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Price Update Schema
class PriceUpdate(BaseModel):
    """Schema for updating position prices."""
    prices: dict[str, Decimal] = Field(..., description="Map of symbol to current price")

    model_config = ConfigDict(from_attributes=True)
