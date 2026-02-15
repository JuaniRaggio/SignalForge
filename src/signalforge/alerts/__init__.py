"""Smart Alerts System for SignalForge."""

from signalforge.alerts.channels import (
    AlertChannel,
    ChannelRouter,
    EmailChannel,
    PushChannel,
    WebSocketChannel,
)
from signalforge.alerts.manager import AlertManager
from signalforge.alerts.schemas import (
    Alert,
    AlertPreferences,
    AlertPriority,
    AlertType,
    ThrottleStatus,
)
from signalforge.alerts.schemas import (
    AlertChannel as AlertChannelEnum,
)
from signalforge.alerts.templates import AlertTemplates
from signalforge.alerts.throttler import AlertThrottler

__all__ = [
    # Main components
    "AlertManager",
    "AlertThrottler",
    "AlertTemplates",
    "ChannelRouter",
    # Channels
    "AlertChannel",
    "WebSocketChannel",
    "EmailChannel",
    "PushChannel",
    # Schemas
    "Alert",
    "AlertPreferences",
    "ThrottleStatus",
    "AlertType",
    "AlertPriority",
    "AlertChannelEnum",
]
