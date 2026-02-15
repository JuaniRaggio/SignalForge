"""Intelligent throttling with graceful degradation."""

from dataclasses import dataclass
from enum import Enum
from uuid import UUID

import structlog
from redis.asyncio import Redis

from signalforge.models.api_key import SubscriptionTier

logger = structlog.get_logger(__name__)


class ThrottleLevel(str, Enum):
    """Throttling levels for graceful degradation."""

    NORMAL = "normal"
    DEGRADED = "degraded"
    SEVERELY_DEGRADED = "severely_degraded"
    CRITICAL = "critical"


@dataclass
class ThrottleStatus:
    """Throttle status information."""

    level: ThrottleLevel
    priority: int
    delay_ms: int
    should_process: bool
    message: str


class IntelligentThrottler:
    """
    Intelligent throttling with graceful degradation.

    Implements priority-based throttling that degrades service gracefully
    under load, prioritizing paying users.
    """

    TIER_PRIORITIES: dict[SubscriptionTier, int] = {
        SubscriptionTier.PROFESSIONAL: 100,
        SubscriptionTier.PROSUMER: 50,
        SubscriptionTier.FREE: 10,
    }

    DEGRADATION_THRESHOLDS = {
        "normal": 70,
        "degraded": 85,
        "severely_degraded": 95,
        "critical": 100,
    }

    def __init__(self, redis_client: Redis) -> None:
        """
        Initialize the intelligent throttler.

        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client

    async def get_system_load(self) -> float:
        """
        Get current system load percentage.

        Uses Redis to track concurrent requests across all instances.

        Returns:
            Load percentage (0-100)
        """
        load_key = "system:load:current"
        load_str = await self.redis.get(load_key)

        if load_str:
            return float(load_str)

        return 0.0

    async def increment_load(self, weight: float = 1.0) -> None:
        """
        Increment system load counter.

        Args:
            weight: Load weight (higher for expensive operations)
        """
        load_key = "system:load:current"
        max_load_key = "system:load:max"

        max_load_str = await self.redis.get(max_load_key)
        max_load = float(max_load_str) if max_load_str else 100.0

        await self.redis.incrbyfloat(load_key, weight)

        current_load = await self.get_system_load()
        load_percentage = (current_load / max_load) * 100

        await self.redis.setex(load_key, 60, int(current_load))

        logger.debug(
            "load_incremented",
            current_load=current_load,
            max_load=max_load,
            load_percentage=load_percentage,
        )

    async def decrement_load(self, weight: float = 1.0) -> None:
        """
        Decrement system load counter.

        Args:
            weight: Load weight (should match increment weight)
        """
        load_key = "system:load:current"
        current_load = await self.get_system_load()

        new_load = max(0, current_load - weight)
        await self.redis.setex(load_key, 60, int(new_load))

        logger.debug("load_decremented", new_load=new_load)

    def _get_throttle_level(self, load_percentage: float) -> ThrottleLevel:
        """
        Determine throttle level based on system load.

        Args:
            load_percentage: System load percentage

        Returns:
            ThrottleLevel
        """
        if load_percentage < self.DEGRADATION_THRESHOLDS["normal"]:
            return ThrottleLevel.NORMAL
        elif load_percentage < self.DEGRADATION_THRESHOLDS["degraded"]:
            return ThrottleLevel.DEGRADED
        elif load_percentage < self.DEGRADATION_THRESHOLDS["severely_degraded"]:
            return ThrottleLevel.SEVERELY_DEGRADED
        else:
            return ThrottleLevel.CRITICAL

    def _calculate_delay(
        self,
        throttle_level: ThrottleLevel,
        priority: int,
    ) -> int:
        """
        Calculate delay in milliseconds based on throttle level and priority.

        Args:
            throttle_level: Current throttle level
            priority: User priority (higher = less delay)

        Returns:
            Delay in milliseconds
        """
        base_delays = {
            ThrottleLevel.NORMAL: 0,
            ThrottleLevel.DEGRADED: 100,
            ThrottleLevel.SEVERELY_DEGRADED: 500,
            ThrottleLevel.CRITICAL: 2000,
        }

        base_delay = base_delays.get(throttle_level, 0)

        priority_factor = max(0.1, 1.0 - (priority / 100.0))

        return int(base_delay * priority_factor)

    async def check_throttle(
        self,
        user_id: UUID,
        tier: SubscriptionTier,
        endpoint: str,
    ) -> ThrottleStatus:
        """
        Check if request should be throttled.

        Args:
            user_id: User ID
            tier: Subscription tier
            endpoint: API endpoint

        Returns:
            ThrottleStatus with decision and metadata
        """
        load_percentage = await self.get_system_load()
        throttle_level = self._get_throttle_level(load_percentage)
        priority = self.TIER_PRIORITIES.get(tier, 0)

        if throttle_level == ThrottleLevel.NORMAL:
            return ThrottleStatus(
                level=throttle_level,
                priority=priority,
                delay_ms=0,
                should_process=True,
                message="System operating normally",
            )

        if throttle_level == ThrottleLevel.CRITICAL and tier == SubscriptionTier.FREE:
            logger.warning(
                "request_rejected_critical_load",
                user_id=str(user_id),
                tier=tier.value,
                endpoint=endpoint,
                load_percentage=load_percentage,
            )
            return ThrottleStatus(
                level=throttle_level,
                priority=priority,
                delay_ms=0,
                should_process=False,
                message="System under heavy load. Please try again later.",
            )

        delay_ms = self._calculate_delay(throttle_level, priority)

        logger.info(
            "request_throttled",
            user_id=str(user_id),
            tier=tier.value,
            endpoint=endpoint,
            throttle_level=throttle_level.value,
            delay_ms=delay_ms,
            load_percentage=load_percentage,
        )

        return ThrottleStatus(
            level=throttle_level,
            priority=priority,
            delay_ms=delay_ms,
            should_process=True,
            message=f"System under load ({throttle_level.value}). Request will be processed with {delay_ms}ms delay.",
        )

    async def get_queue_position(
        self,
        tier: SubscriptionTier,
    ) -> int:
        """
        Get estimated queue position for a tier.

        Args:
            tier: Subscription tier

        Returns:
            Estimated queue position (0 means no queue)
        """
        load_percentage = await self.get_system_load()

        if load_percentage < self.DEGRADATION_THRESHOLDS["degraded"]:
            return 0

        queue_key = f"throttle:queue:{tier.value}"
        queue_length = await self.redis.llen(queue_key)  # type: ignore[misc]

        return int(queue_length) if queue_length is not None else 0

    async def enqueue_request(
        self,
        user_id: UUID,
        tier: SubscriptionTier,
        request_id: str,
    ) -> None:
        """
        Enqueue a request for later processing.

        Args:
            user_id: User ID
            tier: Subscription tier
            request_id: Unique request identifier
        """
        queue_key = f"throttle:queue:{tier.value}"

        await self.redis.rpush(queue_key, f"{user_id}:{request_id}")  # type: ignore[misc]
        await self.redis.expire(queue_key, 300)

        logger.info(
            "request_enqueued",
            user_id=str(user_id),
            tier=tier.value,
            request_id=request_id,
        )

    async def dequeue_request(
        self,
        tier: SubscriptionTier,
    ) -> tuple[UUID, str] | None:
        """
        Dequeue next request for processing.

        Processes higher priority queues first.

        Args:
            tier: Subscription tier to dequeue from

        Returns:
            Tuple of (user_id, request_id) or None if queue is empty
        """
        for priority_tier in [
            SubscriptionTier.PROFESSIONAL,
            SubscriptionTier.PROSUMER,
            SubscriptionTier.FREE,
        ]:
            if priority_tier != tier and self.TIER_PRIORITIES[priority_tier] < self.TIER_PRIORITIES[tier]:
                continue

            queue_key = f"throttle:queue:{priority_tier.value}"
            item = await self.redis.lpop(queue_key)  # type: ignore[misc]

            if item:
                user_id_str, request_id = item.split(":", 1)
                user_id = UUID(user_id_str)

                logger.info(
                    "request_dequeued",
                    user_id=str(user_id),
                    tier=priority_tier.value,
                    request_id=request_id,
                )

                return user_id, request_id

        return None
