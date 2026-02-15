"""Alert throttling to prevent notification fatigue."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from signalforge.alerts.schemas import (
    Alert,
    AlertPreferences,
    AlertPriority,
    ThrottleStatus,
)

if TYPE_CHECKING:
    import redis.asyncio

logger = structlog.get_logger(__name__)


class AlertThrottler:
    """Throttle alerts to prevent notification fatigue."""

    def __init__(
        self,
        default_max_per_day: int = 5,
        redis_client: redis.asyncio.Redis | None = None,
    ) -> None:
        """Initialize throttler.

        Args:
            default_max_per_day: Default maximum alerts per day
            redis_client: Redis client for persistent storage
        """
        self.default_max_per_day = default_max_per_day
        self.redis = redis_client
        # In-memory fallback if Redis is not available
        self._memory_store: dict[str, list[datetime]] = {}
        self._suppressed_store: dict[str, list[Alert]] = {}

    async def can_send(
        self,
        user_id: str,
        alert: Alert,
        preferences: AlertPreferences,
    ) -> tuple[bool, str | None]:
        """Check if alert can be sent.

        Checks:
        1. Daily limit
        2. Quiet hours
        3. Priority threshold
        4. Symbol filters

        Args:
            user_id: User identifier
            alert: Alert to check
            preferences: User preferences

        Returns:
            Tuple of (can_send, reason_if_blocked)
        """
        # Check priority threshold
        if not self._meets_priority_threshold(alert, preferences.min_priority):
            reason = f"Alert priority {alert.priority} below threshold {preferences.min_priority}"
            logger.debug("alert_blocked_priority", user_id=user_id, reason=reason)
            return False, reason

        # Check quiet hours
        if self._is_quiet_hours(preferences):
            reason = "Currently in quiet hours"
            logger.debug("alert_blocked_quiet_hours", user_id=user_id)
            return False, reason

        # Check symbol filters
        if preferences.symbol_filters and (
            not alert.symbol or alert.symbol not in preferences.symbol_filters
        ):
            reason = f"Symbol {alert.symbol} not in user filters"
            logger.debug("alert_blocked_symbol_filter", user_id=user_id, symbol=alert.symbol)
            return False, reason

        # Check daily limit
        alerts_today = await self._get_alerts_today(user_id)
        max_alerts = preferences.max_alerts_per_day or self.default_max_per_day

        if alerts_today >= max_alerts:
            reason = f"Daily limit reached ({alerts_today}/{max_alerts})"
            logger.info("alert_throttled", user_id=user_id, alerts_today=alerts_today, max_alerts=max_alerts)
            return False, reason

        return True, None

    async def record_sent(self, user_id: str, alert: Alert) -> None:
        """Record that an alert was sent.

        Args:
            user_id: User identifier
            alert: Alert that was sent
        """
        if self.redis:
            key = f"alerts:sent:{user_id}:{self._get_date_key()}"
            rpush_result = self.redis.rpush(key, alert.alert_id)
            if hasattr(rpush_result, "__await__"):
                await rpush_result
            expire_result = self.redis.expire(key, 86400 * 2)  # Keep for 2 days
            if hasattr(expire_result, "__await__"):
                await expire_result
        else:
            # In-memory fallback
            if user_id not in self._memory_store:
                self._memory_store[user_id] = []
            self._memory_store[user_id].append(datetime.now(UTC))
            # Clean old entries
            self._memory_store[user_id] = [
                ts for ts in self._memory_store[user_id]
                if ts.date() == datetime.now(UTC).date()
            ]

        logger.info("alert_sent_recorded", user_id=user_id, alert_id=alert.alert_id)

    async def record_suppressed(self, user_id: str, alert: Alert) -> None:
        """Record that an alert was suppressed.

        Args:
            user_id: User identifier
            alert: Alert that was suppressed
        """
        if self.redis:
            key = f"alerts:suppressed:{user_id}:{self._get_date_key()}"
            await self.redis.incr(key)
            await self.redis.expire(key, 86400 * 2)
        else:
            # In-memory fallback
            if user_id not in self._suppressed_store:
                self._suppressed_store[user_id] = []
            self._suppressed_store[user_id].append(alert)

        logger.info("alert_suppressed", user_id=user_id, alert_id=alert.alert_id)

    async def get_status(self, user_id: str) -> ThrottleStatus:
        """Get current throttle status for user.

        Args:
            user_id: User identifier

        Returns:
            Current throttle status
        """
        alerts_today = await self._get_alerts_today(user_id)
        suppressed_count = await self._get_suppressed_count(user_id)

        # Calculate next reset (midnight UTC)
        now = datetime.now(UTC)
        next_reset = datetime.combine(
            now.date() + timedelta(days=1),
            datetime.min.time(),
            tzinfo=UTC,
        )

        return ThrottleStatus(
            user_id=user_id,
            alerts_today=alerts_today,
            remaining_today=max(0, self.default_max_per_day - alerts_today),
            is_throttled=alerts_today >= self.default_max_per_day,
            next_reset=next_reset,
            suppressed_count=suppressed_count,
        )

    async def get_suppressed_alerts(
        self,
        user_id: str,
        since: datetime,
    ) -> list[Alert]:
        """Get alerts that were suppressed due to throttling.

        Args:
            user_id: User identifier
            since: Get alerts suppressed since this time

        Returns:
            List of suppressed alerts
        """
        if user_id not in self._suppressed_store:
            return []

        return [
            alert for alert in self._suppressed_store[user_id]
            if alert.created_at >= since
        ]

    def _is_quiet_hours(self, preferences: AlertPreferences) -> bool:
        """Check if currently in quiet hours.

        Args:
            preferences: User preferences

        Returns:
            True if in quiet hours
        """
        if preferences.quiet_hours_start is None or preferences.quiet_hours_end is None:
            return False

        now = datetime.now(UTC)
        current_hour = now.hour

        start = preferences.quiet_hours_start
        end = preferences.quiet_hours_end

        # Handle overnight quiet hours (e.g., 22:00 to 06:00)
        if start > end:
            return current_hour >= start or current_hour < end
        else:
            return start <= current_hour < end

    def _meets_priority_threshold(
        self,
        alert: Alert,
        min_priority: AlertPriority,
    ) -> bool:
        """Check if alert meets minimum priority.

        Args:
            alert: Alert to check
            min_priority: Minimum required priority

        Returns:
            True if alert meets threshold
        """
        priority_levels = {
            AlertPriority.LOW: 0,
            AlertPriority.MEDIUM: 1,
            AlertPriority.HIGH: 2,
            AlertPriority.CRITICAL: 3,
        }

        return priority_levels[alert.priority] >= priority_levels[min_priority]

    async def _get_alerts_today(self, user_id: str) -> int:
        """Get count of alerts sent today.

        Args:
            user_id: User identifier

        Returns:
            Count of alerts sent today
        """
        if self.redis:
            key = f"alerts:sent:{user_id}:{self._get_date_key()}"
            count_result = self.redis.llen(key)
            if hasattr(count_result, "__await__"):
                count = await count_result
            else:
                count = count_result
            return int(count)
        else:
            # In-memory fallback
            if user_id not in self._memory_store:
                return 0
            today = datetime.now(UTC).date()
            return len([ts for ts in self._memory_store[user_id] if ts.date() == today])

    async def _get_suppressed_count(self, user_id: str) -> int:
        """Get count of suppressed alerts today.

        Args:
            user_id: User identifier

        Returns:
            Count of suppressed alerts
        """
        if self.redis:
            key = f"alerts:suppressed:{user_id}:{self._get_date_key()}"
            count = await self.redis.get(key)
            return int(count) if count else 0
        else:
            # In-memory fallback
            if user_id not in self._suppressed_store:
                return 0
            return len(self._suppressed_store[user_id])

    def _get_date_key(self) -> str:
        """Get current date key for Redis.

        Returns:
            Date key in YYYY-MM-DD format
        """
        return datetime.now(UTC).strftime("%Y-%m-%d")
