"""User model with user types."""

import enum
from uuid import UUID, uuid4

from sqlalchemy import Enum, String
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from signalforge.models.base import Base, TimestampMixin


class UserType(str, enum.Enum):
    """User type enum matching PROJECT_PLAN.md definitions."""

    CASUAL_OBSERVER = "casual_observer"
    INFORMED_INVESTOR = "informed_investor"
    ACTIVE_TRADER = "active_trader"
    QUANT_PROFESSIONAL = "quant_professional"


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
