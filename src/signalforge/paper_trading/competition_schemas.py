"""Pydantic schemas for competition API."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CompetitionCreate(BaseModel):
    """Schema for creating a new competition."""

    name: str = Field(..., min_length=3, max_length=200, description="Competition name")
    description: str | None = Field(default=None, description="Competition description")
    registration_start: datetime = Field(..., description="When registration opens")
    registration_end: datetime = Field(..., description="When registration closes")
    competition_start: datetime = Field(..., description="When competition starts")
    competition_end: datetime = Field(..., description="When competition ends")
    initial_capital: Decimal = Field(
        default=Decimal("100000"),
        ge=1000,
        description="Starting capital for participants",
    )
    max_participants: int | None = Field(
        default=None,
        ge=2,
        description="Maximum number of participants",
    )
    min_participants: int = Field(
        default=2,
        ge=2,
        description="Minimum participants required",
    )
    rules: dict | None = Field(default=None, description="Competition rules as JSON")
    prizes: dict | None = Field(default=None, description="Prize structure as JSON")
    competition_type: str = Field(default="public", description="Competition type (public/private/sponsored)")

    model_config = ConfigDict(from_attributes=True)


class CompetitionResponse(BaseModel):
    """Schema for competition response."""

    id: UUID
    name: str
    description: str | None
    competition_type: str
    status: str
    registration_start: datetime
    registration_end: datetime
    competition_start: datetime
    competition_end: datetime
    initial_capital: Decimal
    max_participants: int | None
    min_participants: int
    current_participants: int = Field(..., description="Number of registered participants")
    rules: dict | None
    prizes: dict | None
    created_by: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CompetitionStandingResponse(BaseModel):
    """Schema for individual standing in competition."""

    rank: int
    user_id: UUID
    username: str
    portfolio_id: UUID
    total_return_pct: Decimal
    sharpe_ratio: Decimal | None
    total_trades: int
    is_disqualified: bool

    model_config = ConfigDict(from_attributes=True)


class CompetitionStandingsResponse(BaseModel):
    """Schema for competition standings list."""

    competition_id: UUID
    competition_name: str
    status: str
    standings: list[CompetitionStandingResponse]
    total_participants: int
    as_of: datetime = Field(..., description="When standings were calculated")

    model_config = ConfigDict(from_attributes=True)


class ParticipantResponse(BaseModel):
    """Schema for competition participant."""

    id: UUID
    competition_id: UUID
    user_id: UUID
    portfolio_id: UUID
    registered_at: datetime
    is_active: bool
    disqualified: bool
    disqualification_reason: str | None
    final_rank: int | None
    final_return_pct: Decimal | None
    final_sharpe: Decimal | None
    prize_awarded: str | None
    current_rank: int | None = Field(default=None, description="Current rank in competition")
    current_return_pct: Decimal | None = Field(default=None, description="Current return percentage")

    model_config = ConfigDict(from_attributes=True)


class CompetitionRegistrationRequest(BaseModel):
    """Schema for registering for competition."""

    model_config = ConfigDict(from_attributes=True)


class CompetitionWithdrawalRequest(BaseModel):
    """Schema for withdrawing from competition."""

    model_config = ConfigDict(from_attributes=True)


class CompetitionDisqualificationRequest(BaseModel):
    """Schema for disqualifying a participant."""

    user_id: UUID = Field(..., description="User to disqualify")
    reason: str = Field(..., min_length=10, max_length=500, description="Reason for disqualification")

    model_config = ConfigDict(from_attributes=True)


class CompetitionListResponse(BaseModel):
    """Schema for list of competitions."""

    competitions: list[CompetitionResponse]
    total: int
    limit: int
    offset: int

    model_config = ConfigDict(from_attributes=True)


class UserCompetitionResponse(BaseModel):
    """Schema for user's competition participation."""

    competition: CompetitionResponse
    participant: ParticipantResponse

    model_config = ConfigDict(from_attributes=True)
