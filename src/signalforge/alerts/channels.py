"""Alert delivery channels."""

from __future__ import annotations

import smtplib
from abc import ABC, abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING, Any

import structlog

from signalforge.alerts.schemas import Alert, AlertPreferences
from signalforge.alerts.schemas import AlertChannel as AlertChannelEnum

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger(__name__)


class AlertChannel(ABC):
    """Base class for alert delivery channels."""

    @abstractmethod
    async def send(self, alert: Alert, user_id: str) -> bool:
        """Send alert through this channel.

        Args:
            alert: Alert to send
            user_id: Target user identifier

        Returns:
            True if successfully sent
        """
        ...

    @abstractmethod
    async def is_available(self, user_id: str) -> bool:
        """Check if channel is available for user.

        Args:
            user_id: User identifier

        Returns:
            True if channel is available
        """
        ...


class WebSocketChannel(AlertChannel):
    """Real-time WebSocket alert delivery."""

    def __init__(self, connection_manager: Any) -> None:
        """Initialize WebSocket channel.

        Args:
            connection_manager: WebSocket connection manager
        """
        self.manager = connection_manager

    async def send(self, alert: Alert, user_id: str) -> bool:
        """Send alert via WebSocket.

        Args:
            alert: Alert to send
            user_id: Target user identifier

        Returns:
            True if successfully sent
        """
        try:
            message = {
                "type": "alert",
                "data": alert.model_dump(mode="json"),
            }
            await self.manager.send_personal_message(message, user_id)
            logger.info("websocket_alert_sent", user_id=user_id, alert_id=alert.alert_id)
            return True
        except Exception as e:
            logger.error("websocket_alert_failed", user_id=user_id, error=str(e))
            return False

    async def is_available(self, user_id: str) -> bool:
        """Check if user has active WebSocket connection.

        Args:
            user_id: User identifier

        Returns:
            True if user is connected
        """
        result = self.manager.is_user_connected(user_id)
        return bool(result)


class EmailChannel(AlertChannel):
    """Email alert delivery."""

    def __init__(self, smtp_config: dict[str, Any]) -> None:
        """Initialize email channel.

        Args:
            smtp_config: SMTP configuration with host, port, username, password
        """
        self.smtp_host = smtp_config.get("host", "localhost")
        self.smtp_port = smtp_config.get("port", 587)
        self.smtp_username = smtp_config.get("username")
        self.smtp_password = smtp_config.get("password")
        self.from_email = smtp_config.get("from_email", "alerts@signalforge.com")
        self.use_tls = smtp_config.get("use_tls", True)
        # User email lookup function
        self._email_lookup: Callable[[str], str | None] | None = None

    def set_email_lookup(self, lookup_fn: Callable[[str], str | None]) -> None:
        """Set function to lookup user email addresses.

        Args:
            lookup_fn: Function that takes user_id and returns email
        """
        self._email_lookup = lookup_fn

    async def send(self, alert: Alert, user_id: str) -> bool:
        """Send alert via email.

        Args:
            alert: Alert to send
            user_id: Target user identifier

        Returns:
            True if successfully sent
        """
        if not self._email_lookup:
            logger.error("email_lookup_not_configured")
            return False

        user_email = self._email_lookup(user_id)
        if not user_email:
            logger.warning("user_email_not_found", user_id=user_id)
            return False

        try:
            subject, body = self._render_email(alert)

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_email
            msg["To"] = user_email

            html_part = MIMEText(body, "html")
            msg.attach(html_part)

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)

            logger.info("email_alert_sent", user_id=user_id, alert_id=alert.alert_id, to=user_email)
            return True
        except Exception as e:
            logger.error("email_alert_failed", user_id=user_id, error=str(e))
            return False

    async def is_available(self, user_id: str) -> bool:
        """Check if user has verified email.

        Args:
            user_id: User identifier

        Returns:
            True if user has verified email
        """
        if not self._email_lookup:
            return False
        email = self._email_lookup(user_id)
        return email is not None

    def _render_email(self, alert: Alert) -> tuple[str, str]:
        """Render email subject and body.

        Args:
            alert: Alert to render

        Returns:
            Tuple of (subject, html_body)
        """
        subject = f"[SignalForge] {alert.title}"

        priority_colors = {
            "low": "#6c757d",
            "medium": "#0d6efd",
            "high": "#fd7e14",
            "critical": "#dc3545",
        }

        color = priority_colors.get(alert.priority.value, "#6c757d")

        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background-color: {color}; color: white; padding: 20px; }}
                .content {{ padding: 20px; }}
                .footer {{ background-color: #f8f9fa; padding: 10px; text-align: center; font-size: 12px; }}
                .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; background-color: {color}; color: white; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>{alert.title}</h2>
                <span class="badge">{alert.priority.value.upper()}</span>
                {f'<span class="badge">{alert.symbol}</span>' if alert.symbol else ''}
            </div>
            <div class="content">
                <p>{alert.message}</p>
                {f'<p><em>Created at: {alert.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")}</em></p>'}
            </div>
            <div class="footer">
                <p>This is an automated alert from SignalForge. Manage your alert preferences in settings.</p>
            </div>
        </body>
        </html>
        """

        return subject, body


class PushChannel(AlertChannel):
    """Push notification delivery."""

    def __init__(self, push_service: Any) -> None:
        """Initialize push channel.

        Args:
            push_service: Push notification service (e.g., FCM, APNS)
        """
        self.push_service = push_service

    async def send(self, alert: Alert, user_id: str) -> bool:
        """Send push notification.

        Args:
            alert: Alert to send
            user_id: Target user identifier

        Returns:
            True if successfully sent
        """
        try:
            notification_data = {
                "title": alert.title,
                "body": alert.message,
                "data": {
                    "alert_id": alert.alert_id,
                    "alert_type": alert.alert_type.value,
                    "priority": alert.priority.value,
                    "symbol": alert.symbol or "",
                },
            }

            await self.push_service.send_to_user(user_id, notification_data)
            logger.info("push_alert_sent", user_id=user_id, alert_id=alert.alert_id)
            return True
        except Exception as e:
            logger.error("push_alert_failed", user_id=user_id, error=str(e))
            return False

    async def is_available(self, user_id: str) -> bool:
        """Check if user has registered device.

        Args:
            user_id: User identifier

        Returns:
            True if user has registered push device
        """
        try:
            result = await self.push_service.has_registered_device(user_id)
            return bool(result)
        except Exception:
            return False


class ChannelRouter:
    """Route alerts to appropriate channels."""

    def __init__(self) -> None:
        """Initialize channel router."""
        self.channels: dict[AlertChannelEnum, AlertChannel] = {}

    def register_channel(self, channel_type: AlertChannelEnum, handler: AlertChannel) -> None:
        """Register a channel handler.

        Args:
            channel_type: Type of channel
            handler: Channel handler instance
        """
        self.channels[channel_type] = handler
        logger.info("channel_registered", channel_type=channel_type.value)

    async def route(
        self,
        alert: Alert,
        user_id: str,
        preferences: AlertPreferences,
    ) -> dict[str, bool]:
        """Route alert to all enabled channels.

        Args:
            alert: Alert to route
            user_id: Target user identifier
            preferences: User preferences

        Returns:
            Dictionary mapping channel name to success status
        """
        results: dict[str, bool] = {}

        for channel_enum in alert.channels:
            # Check if channel is enabled in preferences
            if channel_enum not in preferences.enabled_channels:
                logger.debug(
                    "channel_disabled_in_preferences",
                    user_id=user_id,
                    channel=channel_enum.value,
                )
                results[channel_enum.value] = False
                continue

            # Get channel handler
            handler = self.channels.get(channel_enum)
            if not handler:
                logger.warning("channel_not_registered", channel=channel_enum.value)
                results[channel_enum.value] = False
                continue

            # Check if channel is available for user
            if not await handler.is_available(user_id):
                logger.debug(
                    "channel_not_available",
                    user_id=user_id,
                    channel=channel_enum.value,
                )
                results[channel_enum.value] = False
                continue

            # Send through channel
            success = await handler.send(alert, user_id)
            results[channel_enum.value] = success

        return results
