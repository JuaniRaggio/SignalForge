"""User profile schemas for recommendation engine."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class UserType(str, Enum):
    """User type classification."""

    CASUAL = "casual"  # Follows markets occasionally
    ACTIVE = "active"  # Daily engagement
    PROFESSIONAL = "professional"  # Full-time trader
    INSTITUTIONAL = "institutional"


class RiskTolerance(str, Enum):
    """Risk tolerance levels."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class InvestmentHorizon(str, Enum):
    """Investment time horizon."""

    DAY_TRADE = "day_trade"  # < 1 day
    SWING = "swing"  # 1-10 days
    POSITION = "position"  # 10-90 days
    LONG_TERM = "long_term"  # > 90 days


class PortfolioPosition(BaseModel):
    """Portfolio position information."""

    symbol: str
    shares: float
    cost_basis: float
    acquired_at: datetime


class NotificationPrefs(BaseModel):
    """Notification preferences."""

    email_enabled: bool = True
    push_enabled: bool = True
    max_alerts_per_day: int = 5
    quiet_hours_start: int | None = None  # 0-23
    quiet_hours_end: int | None = None


class ExplicitProfile(BaseModel):
    """Explicit user profile data."""

    user_id: str
    portfolio: list[PortfolioPosition] = Field(default_factory=list)
    watchlist: list[str] = Field(default_factory=list)  # symbols
    preferred_sectors: list[str] = Field(default_factory=list)
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    investment_horizon: InvestmentHorizon = InvestmentHorizon.POSITION
    notification_preferences: NotificationPrefs = Field(
        default_factory=NotificationPrefs
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EngagementPattern(BaseModel):
    """User engagement pattern metrics."""

    avg_session_duration: float = 0.0
    sessions_per_week: float = 0.0
    preferred_hours: list[int] = Field(default_factory=list)  # 0-23
    preferred_days: list[int] = Field(default_factory=list)  # 0-6 (Mon-Sun)
    content_type_preferences: dict[str, float] = Field(default_factory=dict)  # type -> score


class ImplicitProfile(BaseModel):
    """Implicit user behavior profile."""

    user_id: str
    viewed_symbols: list[tuple[str, datetime, float]] = Field(
        default_factory=list
    )  # symbol, timestamp, duration
    clicked_signals: list[tuple[str, datetime]] = Field(default_factory=list)
    search_history: list[tuple[str, datetime]] = Field(default_factory=list)
    sector_affinity: dict[str, float] = Field(default_factory=dict)  # sector -> affinity score
    engagement_pattern: EngagementPattern = Field(default_factory=EngagementPattern)


class UserEmbedding(BaseModel):
    """User embedding vector for recommendations."""

    user_id: str
    embedding: list[float]  # 64-dimensional vector
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)


class OnboardingAnswers(BaseModel):
    """Onboarding quiz answers."""

    time_following_markets: str  # "rarely", "daily", "constantly"
    recent_trading_activity: str  # "none", "occasional", "frequent"
    financial_knowledge: str  # "beginner", "intermediate", "advanced"
    preference_style: str  # "guidance", "raw_data", "both"
    api_interest: bool
