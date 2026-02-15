"""Pydantic schemas for leaderboard API."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class LeaderboardEntryResponse(BaseModel):
    """Schema for a single leaderboard entry."""

    rank: int = Field(..., description="User's rank in the leaderboard")
    user_id: UUID = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    portfolio_name: str = Field(..., description="Portfolio name")
    score: Decimal = Field(..., description="Primary ranking score")
    total_return_pct: Decimal = Field(
        ..., description="Total return percentage"
    )
    sharpe_ratio: Decimal | None = Field(
        None, description="Sharpe ratio (risk-adjusted return)"
    )
    max_drawdown_pct: Decimal | None = Field(
        None, description="Maximum drawdown percentage"
    )
    total_trades: int = Field(..., description="Total number of trades executed")

    model_config = ConfigDict(from_attributes=True)


class LeaderboardResponse(BaseModel):
    """Schema for leaderboard response with metadata."""

    period: str = Field(..., description="Time period (daily, weekly, monthly, all_time)")
    leaderboard_type: str = Field(
        ..., description="Ranking criteria (total_return, sharpe_ratio, risk_adjusted)"
    )
    period_start: datetime = Field(..., description="Start of the period")
    period_end: datetime = Field(..., description="End of the period")
    total_entries: int = Field(..., description="Total number of entries")
    entries: list[LeaderboardEntryResponse] = Field(
        ..., description="Leaderboard entries"
    )

    model_config = ConfigDict(from_attributes=True)


class UserRankingResponse(BaseModel):
    """Schema for individual user's ranking information."""

    user_id: UUID = Field(..., description="User ID")
    period: str = Field(..., description="Time period")
    leaderboard_type: str = Field(..., description="Ranking criteria")
    rank: int = Field(..., description="User's rank")
    total_participants: int = Field(..., description="Total number of participants")
    percentile: Decimal = Field(
        ..., description="User's percentile (0-100)", ge=0, le=100
    )
    score: Decimal = Field(..., description="User's score")
    total_return_pct: Decimal = Field(..., description="Total return percentage")
    sharpe_ratio: Decimal | None = Field(None, description="Sharpe ratio")
    max_drawdown_pct: Decimal | None = Field(None, description="Maximum drawdown")
    total_trades: int = Field(..., description="Total trades")

    model_config = ConfigDict(from_attributes=True)


class RankingHistoryEntry(BaseModel):
    """Schema for a single historical ranking entry."""

    date: datetime = Field(..., description="Date of the ranking")
    period: str = Field(..., description="Time period")
    leaderboard_type: str = Field(..., description="Ranking criteria")
    rank: int = Field(..., description="Rank at that time")
    score: float = Field(..., description="Score at that time")
    total_return_pct: float = Field(..., description="Total return percentage")

    model_config = ConfigDict(from_attributes=True)


class RankingHistoryResponse(BaseModel):
    """Schema for user's ranking history."""

    user_id: UUID = Field(..., description="User ID")
    history: list[RankingHistoryEntry] = Field(
        ..., description="List of historical rankings"
    )

    model_config = ConfigDict(from_attributes=True)
