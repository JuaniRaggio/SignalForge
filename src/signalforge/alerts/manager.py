"""Alert management system."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

from signalforge.alerts.channels import ChannelRouter
from signalforge.alerts.schemas import (
    Alert,
    AlertChannel,
    AlertPreferences,
    AlertPriority,
    AlertType,
)
from signalforge.alerts.templates import AlertTemplates
from signalforge.alerts.throttler import AlertThrottler

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from signalforge.integration.schemas import IntegratedSignal

logger = structlog.get_logger(__name__)


class AlertManager:
    """Central alert management system."""

    def __init__(
        self,
        throttler: AlertThrottler,
        channel_router: ChannelRouter,
        session: AsyncSession | None = None,
    ) -> None:
        """Initialize alert manager.

        Args:
            throttler: Alert throttler instance
            channel_router: Channel router instance
            session: Optional database session
        """
        self.throttler = throttler
        self.router = channel_router
        self.session = session
        self.templates = AlertTemplates()
        # In-memory storage for alerts (in production, use database)
        self._alerts: dict[str, list[Alert]] = {}
        # In-memory storage for preferences (in production, use database)
        self._preferences: dict[str, AlertPreferences] = {}

    async def create_alert(
        self,
        user_id: str,
        alert_type: AlertType,
        title: str,
        message: str,
        symbol: str | None = None,
        priority: AlertPriority = AlertPriority.MEDIUM,
        data: dict[str, Any] | None = None,
        channels: list[AlertChannel] | None = None,
    ) -> Alert:
        """Create and send an alert.

        Args:
            user_id: Target user identifier
            alert_type: Type of alert
            title: Alert title
            message: Alert message
            symbol: Optional stock symbol
            priority: Alert priority
            data: Additional alert data
            channels: Target channels (defaults to all enabled)

        Returns:
            Created alert
        """
        # Get user preferences
        preferences = await self.get_preferences(user_id)

        # Determine channels
        if channels is None:
            channels = preferences.enabled_channels

        # Create alert
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            user_id=user_id,
            alert_type=alert_type,
            priority=priority,
            title=title,
            message=message,
            symbol=symbol,
            data=data or {},
            channels=channels,
            created_at=datetime.now(UTC),
        )

        # Check throttling
        can_send, reason = await self.throttler.can_send(user_id, alert, preferences)

        if not can_send:
            logger.info(
                "alert_throttled",
                user_id=user_id,
                alert_id=alert.alert_id,
                reason=reason,
            )
            await self.throttler.record_suppressed(user_id, alert)
            # Still store the alert, but don't send
            await self._store_alert(alert)
            return alert

        # Route to channels
        delivery_results = await self.router.route(alert, user_id, preferences)

        # Mark as delivered if at least one channel succeeded
        if any(delivery_results.values()):
            alert.delivered_at = datetime.now(UTC)
            await self.throttler.record_sent(user_id, alert)
            logger.info(
                "alert_delivered",
                user_id=user_id,
                alert_id=alert.alert_id,
                channels=delivery_results,
            )
        else:
            logger.warning(
                "alert_delivery_failed",
                user_id=user_id,
                alert_id=alert.alert_id,
                channels=delivery_results,
            )

        # Store alert
        await self._store_alert(alert)

        return alert

    async def create_signal_alert(
        self,
        user_id: str,
        signal: IntegratedSignal,
    ) -> Alert:
        """Create contextual signal alert.

        Example: "AAPL +5% post earnings. Your model predicted this with 78% confidence"

        Args:
            user_id: Target user identifier
            signal: Integrated signal

        Returns:
            Created alert
        """
        confidence_pct = int(signal.confidence * 100)

        # Determine if bullish or bearish
        is_bullish = signal.strength > 0
        template_name = "signal_bullish" if is_bullish else "signal_bearish"

        # Create contextual message
        message = self.templates.render(
            template_name,
            symbol=signal.symbol,
            confidence=confidence_pct,
            reason=signal.explanation,
        )

        # Determine priority based on confidence and strength
        if signal.confidence >= 0.8 and abs(signal.strength) >= 0.7:
            priority = AlertPriority.CRITICAL
        elif signal.confidence >= 0.7 and abs(signal.strength) >= 0.5:
            priority = AlertPriority.HIGH
        elif signal.confidence >= 0.5:
            priority = AlertPriority.MEDIUM
        else:
            priority = AlertPriority.LOW

        title = self.templates.get_title(AlertType.SIGNAL, priority)

        return await self.create_alert(
            user_id=user_id,
            alert_type=AlertType.SIGNAL,
            title=title,
            message=message,
            symbol=signal.symbol,
            priority=priority,
            data={
                "direction": signal.direction.value,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "recommendation": signal.recommendation,
                "position_size_pct": signal.position_size_pct,
                "stop_loss_pct": signal.stop_loss_pct,
                "take_profit_pct": signal.take_profit_pct,
            },
        )

    async def create_price_alert(
        self,
        user_id: str,
        symbol: str,
        current_price: float,
        target_price: float,
        direction: str,
    ) -> Alert:
        """Create price target alert.

        Args:
            user_id: Target user identifier
            symbol: Stock symbol
            current_price: Current price
            target_price: Target price
            direction: "above" or "below"

        Returns:
            Created alert
        """
        message = self.templates.render(
            "price_target_hit",
            symbol=symbol,
            target=f"{target_price:.2f}",
            current=f"{current_price:.2f}",
        )

        # Calculate distance from target
        pct_diff = abs((current_price - target_price) / target_price) * 100

        # Determine priority based on how significant the move is
        if pct_diff >= 10:
            priority = AlertPriority.CRITICAL
        elif pct_diff >= 5:
            priority = AlertPriority.HIGH
        else:
            priority = AlertPriority.MEDIUM

        title = self.templates.get_title(AlertType.PRICE_TARGET, priority)

        return await self.create_alert(
            user_id=user_id,
            alert_type=AlertType.PRICE_TARGET,
            title=title,
            message=message,
            symbol=symbol,
            priority=priority,
            data={
                "current_price": current_price,
                "target_price": target_price,
                "direction": direction,
                "pct_diff": pct_diff,
            },
        )

    async def create_portfolio_alert(
        self,
        user_id: str,
        alert_subtype: str,
        details: dict[str, Any],
    ) -> Alert:
        """Create portfolio-related alert.

        Args:
            user_id: Target user identifier
            alert_subtype: Type of portfolio alert ("gain", "loss", "rebalance")
            details: Alert details

        Returns:
            Created alert
        """
        symbol = details.get("symbol", "")
        template_map = {
            "gain": "portfolio_gain",
            "loss": "portfolio_loss",
            "rebalance": "portfolio_rebalance",
        }

        template_name = template_map.get(alert_subtype, "portfolio_gain")

        # Create message based on subtype
        if alert_subtype == "gain":
            message = self.templates.render(
                template_name,
                symbol=symbol,
                gain=f"{details.get('gain_pct', 0):.2f}",
                value=f"{details.get('current_value', 0):.2f}",
            )
            priority = AlertPriority.MEDIUM
        elif alert_subtype == "loss":
            loss_pct = details.get("loss_pct", 0)
            message = self.templates.render(
                template_name,
                symbol=symbol,
                loss=f"{loss_pct:.2f}",
            )
            # Higher priority for larger losses
            if loss_pct >= 10:
                priority = AlertPriority.CRITICAL
            elif loss_pct >= 5:
                priority = AlertPriority.HIGH
            else:
                priority = AlertPriority.MEDIUM
        else:  # rebalance
            message = self.templates.render(
                template_name,
                reason=details.get("reason", "Portfolio drift detected"),
            )
            priority = AlertPriority.LOW

        title = self.templates.get_title(AlertType.PORTFOLIO, priority)

        return await self.create_alert(
            user_id=user_id,
            alert_type=AlertType.PORTFOLIO,
            title=title,
            message=message,
            symbol=symbol if symbol else None,
            priority=priority,
            data=details,
        )

    async def get_user_alerts(
        self,
        user_id: str,
        limit: int = 50,
        unread_only: bool = False,
    ) -> list[Alert]:
        """Get alerts for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of alerts to return
            unread_only: Only return unread alerts

        Returns:
            List of alerts
        """
        if user_id not in self._alerts:
            return []

        alerts = self._alerts[user_id]

        if unread_only:
            alerts = [a for a in alerts if a.read_at is None]

        # Sort by created_at descending (newest first)
        alerts = sorted(alerts, key=lambda a: a.created_at, reverse=True)

        return alerts[:limit]

    async def mark_read(self, alert_id: str, user_id: str) -> None:
        """Mark alert as read.

        Args:
            alert_id: Alert identifier
            user_id: User identifier
        """
        if user_id not in self._alerts:
            return

        for alert in self._alerts[user_id]:
            if alert.alert_id == alert_id:
                alert.read_at = datetime.now(UTC)
                logger.info("alert_marked_read", alert_id=alert_id, user_id=user_id)
                break

    async def get_preferences(self, user_id: str) -> AlertPreferences:
        """Get user's alert preferences.

        Args:
            user_id: User identifier

        Returns:
            User preferences (defaults if not set)
        """
        if user_id in self._preferences:
            return self._preferences[user_id]

        # Return defaults
        return AlertPreferences(
            user_id=user_id,
            enabled_types=[
                AlertType.SIGNAL,
                AlertType.PRICE_TARGET,
                AlertType.PORTFOLIO,
                AlertType.EARNINGS,
            ],
            enabled_channels=[AlertChannel.WEBSOCKET, AlertChannel.EMAIL],
            max_alerts_per_day=5,
            min_priority=AlertPriority.LOW,
        )

    async def update_preferences(
        self,
        user_id: str,
        updates: dict[str, Any],
    ) -> AlertPreferences:
        """Update alert preferences.

        Args:
            user_id: User identifier
            updates: Dictionary of preference updates

        Returns:
            Updated preferences
        """
        current = await self.get_preferences(user_id)

        # Update fields
        for key, value in updates.items():
            if hasattr(current, key):
                setattr(current, key, value)

        self._preferences[user_id] = current
        logger.info("preferences_updated", user_id=user_id, updates=list(updates.keys()))

        return current

    async def _store_alert(self, alert: Alert) -> None:
        """Store alert (in-memory or database).

        Args:
            alert: Alert to store
        """
        if alert.user_id not in self._alerts:
            self._alerts[alert.user_id] = []

        self._alerts[alert.user_id].append(alert)
