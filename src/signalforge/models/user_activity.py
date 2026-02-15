"""User activity model for tracking implicit behavior."""

import enum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Enum, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from signalforge.models.base import Base, TimestampMixin


class ActivityType(str, enum.Enum):
    """Types of user activities that can be tracked."""

    VIEW = "view"
    CLICK = "click"
    FOLLOW = "follow"
    DISMISS = "dismiss"


class UserActivity(Base, TimestampMixin):
    """Model for tracking user behavior and interactions."""

    __tablename__ = "user_activities"

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
    activity_type: Mapped[ActivityType] = mapped_column(
        Enum(ActivityType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        index=True,
    )
    symbol: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
        index=True,
    )
    sector: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        index=True,
    )
    signal_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=True,
        index=True,
    )
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSONB,
        default=dict,
        nullable=False,
    )
