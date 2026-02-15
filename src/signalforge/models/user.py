"""User model with user types."""

import enum
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy import Enum, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from signalforge.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from signalforge.models.api_key import APIKey


class UserType(str, enum.Enum):
    """User type enum matching PROJECT_PLAN.md definitions."""

    CASUAL_OBSERVER = "casual_observer"
    INFORMED_INVESTOR = "informed_investor"
    ACTIVE_TRADER = "active_trader"
    QUANT_PROFESSIONAL = "quant_professional"


class RiskTolerance(str, enum.Enum):
    """User risk tolerance levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class InvestmentHorizon(str, enum.Enum):
    """User investment time horizon."""

    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class ExperienceLevel(str, enum.Enum):
    """User experience level with investing."""

    CASUAL = "casual"
    INFORMED = "informed"
    ACTIVE = "active"
    QUANT = "quant"


class User(Base, TimestampMixin):
    """User model for authentication and personalization."""

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        index=True,
        nullable=False,
    )
    user_type: Mapped[UserType] = mapped_column(
        Enum(UserType, values_callable=lambda x: [e.value for e in x]),
        default=UserType.CASUAL_OBSERVER,
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        default=True,
        nullable=False,
    )
    is_verified: Mapped[bool] = mapped_column(
        default=False,
        nullable=False,
    )
    risk_tolerance: Mapped[RiskTolerance] = mapped_column(
        Enum(RiskTolerance, values_callable=lambda x: [e.value for e in x]),
        default=RiskTolerance.MEDIUM,
        nullable=False,
    )
    investment_horizon: Mapped[InvestmentHorizon] = mapped_column(
        Enum(InvestmentHorizon, values_callable=lambda x: [e.value for e in x]),
        default=InvestmentHorizon.MEDIUM,
        nullable=False,
    )
    experience_level: Mapped[ExperienceLevel] = mapped_column(
        Enum(ExperienceLevel, values_callable=lambda x: [e.value for e in x]),
        default=ExperienceLevel.INFORMED,
        nullable=False,
    )
    preferred_sectors: Mapped[list[str] | None] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    watchlist: Mapped[list[str] | None] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    notification_preferences: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB,
        default=dict,
        nullable=False,
    )

    api_keys: Mapped[list["APIKey"]] = relationship(  # type: ignore[name-defined]
        "APIKey",
        back_populates="user",
        cascade="all, delete-orphan",
    )
