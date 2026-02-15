"""Tier-based rate limiting with burst support using Redis."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

import structlog
from redis.asyncio import Redis

from signalforge.models.api_key import SubscriptionTier

logger = structlog.get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a tier."""

    requests_per_minute: int
    burst: int
    window_seconds: int = 60


@dataclass
class RateLimitStatus:
    """Rate limit status information."""

    limit: int
    remaining: int
    used: int
    reset_in_seconds: int
    is_burst: bool
    tier: str


class TierRateLimiter:
    """
    Tier-based rate limiter with burst support.

    Implements token bucket algorithm with separate limits for sustained
    and burst traffic. Uses Redis for distributed rate limiting.
    """

    TIER_LIMITS: dict[SubscriptionTier, RateLimitConfig] = {
        SubscriptionTier.FREE: RateLimitConfig(
            requests_per_minute=5,
            burst=10,
        ),
        SubscriptionTier.PROSUMER: RateLimitConfig(
            requests_per_minute=60,
            burst=100,
        ),
        SubscriptionTier.PROFESSIONAL: RateLimitConfig(
            requests_per_minute=300,
            burst=500,
        ),
    }

    def __init__(self, redis_client: Redis) -> None:
        """
        Initialize the tier rate limiter.

        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client

    @staticmethod
    def _normalize_tier(tier: SubscriptionTier | str) -> SubscriptionTier:
        """Convert tier to SubscriptionTier enum if it's a string."""
        if isinstance(tier, str):
            return SubscriptionTier(tier)
        return tier

    @staticmethod
    def _get_tier_value(tier: SubscriptionTier | str) -> str:
        """Get string value from tier, handling both enum and string."""
        if isinstance(tier, str):
            return tier
        return tier.value

    def _get_redis_key(
        self,
        user_id: UUID,
        limit_type: Literal["sustained", "burst"],
    ) -> str:
        """
        Generate Redis key for rate limit tracking.

        Args:
            user_id: User ID
            limit_type: Type of limit (sustained or burst)

        Returns:
            Redis key string
        """
        return f"rate_limit:{limit_type}:{user_id}"

    async def check_rate_limit(
        self,
        user_id: UUID,
        tier: SubscriptionTier | str,
        rate_limit_override: int | None = None,
        burst_limit_override: int | None = None,
    ) -> tuple[bool, RateLimitStatus]:
        """
        Check if request is within rate limit.

        Uses a two-tier approach:
        1. First checks burst limit (short-term)
        2. Then checks sustained rate limit (per-minute)

        Args:
            user_id: User ID to check
            tier: Subscription tier (enum or string)
            rate_limit_override: Optional override for sustained limit
            burst_limit_override: Optional override for burst limit

        Returns:
            Tuple of (allowed, status)
            allowed is True if request is within limits
        """
        tier_enum = self._normalize_tier(tier)
        tier_value = self._get_tier_value(tier)
        config = self.TIER_LIMITS.get(tier_enum)
        if not config:
            logger.error("invalid_tier", tier=tier_value)
            return False, self._create_error_status(tier)

        sustained_limit = rate_limit_override or config.requests_per_minute
        burst_limit = burst_limit_override or config.burst

        burst_key = self._get_redis_key(user_id, "burst")
        sustained_key = self._get_redis_key(user_id, "sustained")

        burst_count_str = await self.redis.get(burst_key)
        burst_count = int(burst_count_str) if burst_count_str else 0

        if burst_count >= burst_limit:
            ttl = await self.redis.ttl(burst_key)
            logger.warning(
                "burst_limit_exceeded",
                user_id=str(user_id),
                tier=tier_value,
                burst_count=burst_count,
                burst_limit=burst_limit,
            )
            return False, RateLimitStatus(
                limit=burst_limit,
                remaining=0,
                used=burst_count,
                reset_in_seconds=max(ttl, 0) if ttl > 0 else 60,
                is_burst=True,
                tier=tier_value,
            )

        sustained_count_str = await self.redis.get(sustained_key)
        sustained_count = int(sustained_count_str) if sustained_count_str else 0

        if sustained_count >= sustained_limit:
            ttl = await self.redis.ttl(sustained_key)
            logger.warning(
                "sustained_limit_exceeded",
                user_id=str(user_id),
                tier=tier_value,
                sustained_count=sustained_count,
                sustained_limit=sustained_limit,
            )
            return False, RateLimitStatus(
                limit=sustained_limit,
                remaining=0,
                used=sustained_count,
                reset_in_seconds=max(ttl, 0) if ttl > 0 else config.window_seconds,
                is_burst=False,
                tier=tier_value,
            )

        if burst_count_str is None:
            await self.redis.setex(burst_key, 60, 1)
        else:
            await self.redis.incr(burst_key)

        if sustained_count_str is None:
            await self.redis.setex(sustained_key, config.window_seconds, 1)
        else:
            await self.redis.incr(sustained_key)

        logger.debug(
            "rate_limit_check_passed",
            user_id=str(user_id),
            tier=tier_value,
            burst_count=burst_count + 1,
            sustained_count=sustained_count + 1,
        )

        return True, RateLimitStatus(
            limit=sustained_limit,
            remaining=sustained_limit - sustained_count - 1,
            used=sustained_count + 1,
            reset_in_seconds=config.window_seconds,
            is_burst=False,
            tier=tier_value,
        )

    async def get_rate_limit_status(
        self,
        user_id: UUID,
        tier: SubscriptionTier | str,
        rate_limit_override: int | None = None,
    ) -> RateLimitStatus:
        """
        Get current rate limit status without consuming a request.

        Args:
            user_id: User ID to check
            tier: Subscription tier (enum or string)
            rate_limit_override: Optional override for sustained limit

        Returns:
            RateLimitStatus with current usage information
        """
        tier_enum = self._normalize_tier(tier)
        tier_value = self._get_tier_value(tier)
        config = self.TIER_LIMITS.get(tier_enum)
        if not config:
            return self._create_error_status(tier)

        sustained_limit = rate_limit_override or config.requests_per_minute
        sustained_key = self._get_redis_key(user_id, "sustained")

        sustained_count_str = await self.redis.get(sustained_key)
        sustained_count = int(sustained_count_str) if sustained_count_str else 0

        ttl = await self.redis.ttl(sustained_key)
        reset_in_seconds = ttl if ttl > 0 else config.window_seconds

        return RateLimitStatus(
            limit=sustained_limit,
            remaining=max(0, sustained_limit - sustained_count),
            used=sustained_count,
            reset_in_seconds=reset_in_seconds,
            is_burst=False,
            tier=tier_value,
        )

    async def reset_rate_limit(self, user_id: UUID) -> None:
        """
        Reset rate limits for a user (admin function).

        Args:
            user_id: User ID to reset
        """
        burst_key = self._get_redis_key(user_id, "burst")
        sustained_key = self._get_redis_key(user_id, "sustained")

        await self.redis.delete(burst_key)
        await self.redis.delete(sustained_key)

        logger.info("rate_limit_reset", user_id=str(user_id))

    def _create_error_status(self, tier: SubscriptionTier | str) -> RateLimitStatus:
        """
        Create an error status for invalid tier.

        Args:
            tier: The invalid tier (enum or string)

        Returns:
            RateLimitStatus with zero limits
        """
        return RateLimitStatus(
            limit=0,
            remaining=0,
            used=0,
            reset_in_seconds=0,
            is_burst=False,
            tier=self._get_tier_value(tier),
        )
