"""User preferences schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from signalforge.models.user import (
    ExperienceLevel,
    InvestmentHorizon,
    RiskTolerance,
)
from signalforge.models.user_activity import ActivityType


class UserPreferencesUpdate(BaseModel):
    """Schema for updating user preferences."""

    risk_tolerance: RiskTolerance | None = None
    investment_horizon: InvestmentHorizon | None = None
    experience_level: ExperienceLevel | None = None
    preferred_sectors: list[str] | None = Field(
        None,
        description="List of GICS sector codes or names",
    )
    notification_preferences: dict[str, Any] | None = Field(
        None,
        description="Dictionary of notification settings",
    )


class UserPreferencesResponse(BaseModel):
    """Schema for reading user preferences."""

    id: UUID
    email: str
    username: str
    risk_tolerance: RiskTolerance
    investment_horizon: InvestmentHorizon
    experience_level: ExperienceLevel
    preferred_sectors: list[str]
    watchlist: list[str]
    notification_preferences: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class WatchlistUpdate(BaseModel):
    """Schema for adding symbols to watchlist."""

    symbols: list[str] = Field(
        ...,
        min_length=1,
        description="List of stock symbols to add to watchlist",
    )


class WatchlistRemove(BaseModel):
    """Schema for removing a symbol from watchlist."""

    symbol: str = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Stock symbol to remove from watchlist",
    )


class UserActivityCreate(BaseModel):
    """Schema for creating user activity record."""

    activity_type: ActivityType
    symbol: str | None = Field(None, max_length=20)
    sector: str | None = Field(None, max_length=100)
    signal_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserActivityResponse(BaseModel):
    """Schema for reading user activity."""

    id: UUID
    user_id: UUID
    activity_type: ActivityType
    symbol: str | None
    sector: str | None
    signal_id: UUID | None
    metadata: dict[str, Any] = Field(validation_alias="metadata_")
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}
