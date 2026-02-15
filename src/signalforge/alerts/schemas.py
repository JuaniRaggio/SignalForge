"""Alert schemas and data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AlertType(str, Enum):
    """Types of alerts that can be generated."""

    SIGNAL = "signal"
    PRICE_TARGET = "price_target"
    EARNINGS = "earnings"
    NEWS = "news"
    PORTFOLIO = "portfolio"
    WATCHLIST = "watchlist"
    SYSTEM = "system"


class AlertPriority(str, Enum):
    """Alert priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertChannel(str, Enum):
    """Available alert delivery channels."""

    WEBSOCKET = "websocket"
    EMAIL = "email"
    PUSH = "push"
    SMS = "sms"


class Alert(BaseModel):
    """Alert model."""

    alert_id: str
    user_id: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    symbol: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    channels: list[AlertChannel]
    created_at: datetime
    delivered_at: datetime | None = None
    read_at: datetime | None = None


class AlertPreferences(BaseModel):
    """User alert preferences."""

    user_id: str
    enabled_types: list[AlertType]
    enabled_channels: list[AlertChannel]
    max_alerts_per_day: int = 5
    quiet_hours_start: int | None = None  # 0-23
    quiet_hours_end: int | None = None
    min_priority: AlertPriority = AlertPriority.LOW
    symbol_filters: list[str] | None = None  # Only these symbols


class ThrottleStatus(BaseModel):
    """Current throttle status for a user."""

    user_id: str
    alerts_today: int
    remaining_today: int
    is_throttled: bool
    next_reset: datetime
    suppressed_count: int
