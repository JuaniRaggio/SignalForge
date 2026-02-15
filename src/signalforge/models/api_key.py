"""API Key model for gateway authentication."""

import enum
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from signalforge.models.base import Base, TimestampMixin

if False:
    from signalforge.models.user import User


class SubscriptionTier(str, enum.Enum):
    """Subscription tiers matching PROJECT_PLAN.md definitions."""

    FREE = "free"
    PROSUMER = "prosumer"
    PROFESSIONAL = "professional"


class APIKey(Base, TimestampMixin):
    """
    API Key model for secure API access with tier-based rate limiting.

    Stores hashed API keys for authentication and authorization.
    Each key is associated with a user and has configurable scopes and rate limits.
    """

    __tablename__ = "api_keys"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    key_hash: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    tier: Mapped[SubscriptionTier] = mapped_column(
        String(50),
        default=SubscriptionTier.FREE,
        nullable=False,
        index=True,
    )
    scopes: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    rate_limit_override: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    burst_limit_override: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    last_used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        index=True,
    )

    user: Mapped["User"] = relationship("User", back_populates="api_keys")

    def __repr__(self) -> str:
        """String representation of APIKey."""
        return f"<APIKey(id={self.id}, name={self.name}, tier={self.tier}, is_active={self.is_active})>"
