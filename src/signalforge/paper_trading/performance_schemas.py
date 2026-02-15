"""Pydantic schemas for performance metrics API responses."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field


class PerformanceMetricsResponse(BaseModel):
    """Performance metrics response schema."""

    portfolio_id: UUID
    total_return_pct: Decimal = Field(..., description="Total return percentage")
    annualized_return_pct: Decimal = Field(..., description="Annualized return percentage")
    sharpe_ratio: Decimal = Field(..., description="Sharpe ratio (risk-adjusted return)")
    sortino_ratio: Decimal = Field(..., description="Sortino ratio (downside risk-adjusted return)")
    max_drawdown_pct: Decimal = Field(..., description="Maximum drawdown percentage")
    max_drawdown_duration_days: int = Field(..., description="Maximum drawdown duration in days")
    win_rate: Decimal = Field(..., description="Percentage of winning trades")
    profit_factor: Decimal = Field(
        ..., description="Ratio of gross profit to gross loss"
    )
    total_trades: int = Field(..., description="Total number of closed trades")
    winning_trades: int = Field(..., description="Number of winning trades")
    losing_trades: int = Field(..., description="Number of losing trades")
    avg_win_pct: Decimal = Field(..., description="Average winning trade percentage")
    avg_loss_pct: Decimal = Field(..., description="Average losing trade percentage")
    best_trade_pct: Decimal = Field(..., description="Best trade return percentage")
    worst_trade_pct: Decimal = Field(..., description="Worst trade return percentage")
    avg_holding_period_days: Decimal = Field(
        ..., description="Average holding period in days"
    )
    volatility_annualized: Decimal = Field(..., description="Annualized volatility")
    calmar_ratio: Decimal = Field(
        ..., description="Calmar ratio (return / max drawdown)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "portfolio_id": "123e4567-e89b-12d3-a456-426614174000",
                "total_return_pct": "15.25",
                "annualized_return_pct": "18.50",
                "sharpe_ratio": "1.85",
                "sortino_ratio": "2.10",
                "max_drawdown_pct": "8.25",
                "max_drawdown_duration_days": 14,
                "win_rate": "62.50",
                "profit_factor": "1.75",
                "total_trades": 48,
                "winning_trades": 30,
                "losing_trades": 18,
                "avg_win_pct": "3.20",
                "avg_loss_pct": "-1.85",
                "best_trade_pct": "12.50",
                "worst_trade_pct": "-5.25",
                "avg_holding_period_days": "5.5",
                "volatility_annualized": "12.30",
                "calmar_ratio": "2.24",
            }
        }


class EquityCurvePoint(BaseModel):
    """Single point on the equity curve."""

    date: datetime = Field(..., description="Snapshot date")
    equity: Decimal = Field(..., description="Total portfolio equity value")
    cash_balance: Decimal = Field(..., description="Cash balance")
    positions_value: Decimal = Field(..., description="Value of open positions")
    positions_count: int = Field(..., description="Number of open positions")
    daily_return_pct: Decimal | None = Field(None, description="Daily return percentage")
    cumulative_return_pct: Decimal | None = Field(
        None, description="Cumulative return percentage from inception"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2024-01-15T16:00:00Z",
                "equity": "105250.00",
                "cash_balance": "45000.00",
                "positions_value": "60250.00",
                "positions_count": 8,
                "daily_return_pct": "0.75",
                "cumulative_return_pct": "5.25",
            }
        }


class EquityCurveResponse(BaseModel):
    """Equity curve data response."""

    portfolio_id: UUID
    start_date: datetime | None = Field(None, description="Start date of the curve")
    end_date: datetime | None = Field(None, description="End date of the curve")
    data: list[EquityCurvePoint] = Field(..., description="Equity curve data points")

    class Config:
        json_schema_extra = {
            "example": {
                "portfolio_id": "123e4567-e89b-12d3-a456-426614174000",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-31T00:00:00Z",
                "data": [
                    {
                        "date": "2024-01-15T16:00:00Z",
                        "equity": "105250.00",
                        "cash_balance": "45000.00",
                        "positions_value": "60250.00",
                        "positions_count": 8,
                        "daily_return_pct": "0.75",
                        "cumulative_return_pct": "5.25",
                    }
                ],
            }
        }


class ReturnsSeriesResponse(BaseModel):
    """Daily returns series response."""

    portfolio_id: UUID
    start_date: datetime | None = Field(None, description="Start date of the series")
    end_date: datetime | None = Field(None, description="End date of the series")
    returns: list[Decimal] = Field(..., description="Daily return percentages")
    mean_return: Decimal = Field(..., description="Mean daily return")
    std_return: Decimal = Field(..., description="Standard deviation of returns")

    class Config:
        json_schema_extra = {
            "example": {
                "portfolio_id": "123e4567-e89b-12d3-a456-426614174000",
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": "2024-01-31T00:00:00Z",
                "returns": ["0.25", "-0.15", "0.50", "1.25", "-0.35"],
                "mean_return": "0.30",
                "std_return": "0.55",
            }
        }
