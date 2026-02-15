"""Quota management service for subscription tiers."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.billing.models import Subscription, SubscriptionStatus, UsageRecord
from signalforge.billing.tier_features import (
    SubscriptionTier,
    get_quota_limit,
    has_unlimited_quota,
)
from signalforge.core.logging import LoggerMixin


class QuotaExceededError(Exception):
    """Exception raised when quota is exceeded."""

    def __init__(
        self,
        resource_type: str,
        current_usage: int,
        limit: int,
    ) -> None:
        """Initialize quota exceeded error.

        Args:
            resource_type: Type of resource that exceeded quota
            current_usage: Current usage count
            limit: Quota limit
        """
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit
        super().__init__(
            f"Quota exceeded for {resource_type}: {current_usage}/{limit}",
        )


class QuotaManager(LoggerMixin):
    """Manager for user quota tracking and enforcement."""

    def __init__(
        self,
        db: AsyncSession,
        redis: Redis,
    ) -> None:
        """Initialize quota manager.

        Args:
            db: Database session
            redis: Redis client for caching
        """
        self.db = db
        self.redis = redis

    async def get_user_tier(self, user_id: UUID) -> SubscriptionTier:
        """Get the current subscription tier for a user.

        Args:
            user_id: User ID

        Returns:
            User's current subscription tier (defaults to FREE)
        """
        cache_key = f"user_tier:{user_id}"
        cached_tier = await self.redis.get(cache_key)

        if cached_tier:
            return SubscriptionTier(cached_tier)

        result = await self.db.execute(
            select(Subscription)
            .join(Subscription.plan)
            .where(
                Subscription.user_id == user_id,
                Subscription.status == SubscriptionStatus.ACTIVE,
            )
            .order_by(Subscription.created_at.desc()),
        )
        subscription = result.scalar_one_or_none()

        if subscription and subscription.plan:
            tier = subscription.plan.tier
        else:
            tier = SubscriptionTier.FREE

        await self.redis.setex(cache_key, 300, tier.value)
        return tier

    async def check_quota(
        self,
        user_id: UUID,
        resource_type: str,
        count: int = 1,
    ) -> bool:
        """Check if user has available quota for a resource.

        Args:
            user_id: User ID
            resource_type: Type of resource to check
            count: Number of resources to check for

        Returns:
            True if quota is available, False otherwise
        """
        tier = await self.get_user_tier(user_id)

        if has_unlimited_quota(tier, resource_type):
            return True

        limit = get_quota_limit(tier, resource_type)

        if limit == 0:
            return False

        current_usage = await self.get_current_usage(user_id, resource_type)
        return (current_usage + count) <= limit

    async def increment_usage(
        self,
        user_id: UUID,
        resource_type: str,
        count: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> UsageRecord:
        """Increment usage for a resource.

        Args:
            user_id: User ID
            resource_type: Type of resource
            count: Number to increment by
            metadata: Optional metadata to store with usage record

        Returns:
            Created usage record

        Raises:
            QuotaExceededError: If quota would be exceeded
        """
        tier = await self.get_user_tier(user_id)

        if not has_unlimited_quota(tier, resource_type):
            limit = get_quota_limit(tier, resource_type)
            current_usage = await self.get_current_usage(user_id, resource_type)

            if (current_usage + count) > limit:
                raise QuotaExceededError(resource_type, current_usage, limit)

        usage_record = UsageRecord(
            user_id=user_id,
            resource_type=resource_type,
            count=count,
            extra_data=metadata or {},
        )
        self.db.add(usage_record)
        await self.db.flush()

        await self._invalidate_usage_cache(user_id, resource_type)

        self.logger.info(
            "usage_incremented",
            user_id=str(user_id),
            resource_type=resource_type,
            count=count,
        )

        return usage_record

    async def get_current_usage(
        self,
        user_id: UUID,
        resource_type: str,
    ) -> int:
        """Get current usage for a resource.

        Args:
            user_id: User ID
            resource_type: Type of resource

        Returns:
            Current usage count
        """
        cache_key = f"usage:{user_id}:{resource_type}:{self._get_period_key()}"
        cached_usage = await self.redis.get(cache_key)

        if cached_usage is not None:
            return int(cached_usage)

        start_time = self._get_period_start()
        result = await self.db.execute(
            select(func.coalesce(func.sum(UsageRecord.count), 0)).where(
                UsageRecord.user_id == user_id,
                UsageRecord.resource_type == resource_type,
                UsageRecord.recorded_at >= start_time,
            ),
        )
        usage = int(result.scalar_one())

        await self.redis.setex(cache_key, 3600, usage)
        return usage

    async def get_remaining_quota(
        self,
        user_id: UUID,
        resource_type: str,
    ) -> int:
        """Get remaining quota for a resource.

        Args:
            user_id: User ID
            resource_type: Type of resource

        Returns:
            Remaining quota (-1 for unlimited)
        """
        tier = await self.get_user_tier(user_id)

        if has_unlimited_quota(tier, resource_type):
            return -1

        limit = get_quota_limit(tier, resource_type)
        current_usage = await self.get_current_usage(user_id, resource_type)

        return max(0, limit - current_usage)

    async def reset_daily_quotas(self) -> int:
        """Reset daily quotas for all users.

        This should be called by a scheduled task (e.g., Celery).

        Returns:
            Number of cache keys cleared
        """
        self.logger.info("resetting_daily_quotas")

        pattern = f"usage:*:{self._get_period_key()}"
        cursor = 0
        deleted_count = 0

        while True:
            cursor, keys = await self.redis.scan(
                cursor=cursor,
                match=pattern,
                count=100,
            )
            if keys:
                deleted = await self.redis.delete(*keys)
                deleted_count += deleted

            if cursor == 0:
                break

        self.logger.info("daily_quotas_reset", deleted_keys=deleted_count)
        return deleted_count

    async def get_usage_stats(
        self,
        user_id: UUID,
        resource_types: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Get usage statistics for a user.

        Args:
            user_id: User ID
            resource_types: List of resource types (None for all)

        Returns:
            Dictionary of resource type to usage stats
        """
        tier = await self.get_user_tier(user_id)

        if resource_types is None:
            resource_types = [
                "predictions_per_day",
                "api_calls_per_minute",
            ]

        stats: dict[str, dict[str, Any]] = {}
        for resource_type in resource_types:
            try:
                is_unlimited = has_unlimited_quota(tier, resource_type)
                limit = get_quota_limit(tier, resource_type) if not is_unlimited else -1
                current = await self.get_current_usage(user_id, resource_type)
                remaining = await self.get_remaining_quota(user_id, resource_type)

                stats[resource_type] = {
                    "current_usage": current,
                    "quota_limit": limit,
                    "remaining": remaining,
                    "is_unlimited": is_unlimited,
                    "reset_at": self._get_next_reset_time(),
                }
            except (KeyError, ValueError):
                continue

        return stats

    async def _invalidate_usage_cache(
        self,
        user_id: UUID,
        resource_type: str,
    ) -> None:
        """Invalidate cached usage data.

        Args:
            user_id: User ID
            resource_type: Type of resource
        """
        cache_key = f"usage:{user_id}:{resource_type}:{self._get_period_key()}"
        await self.redis.delete(cache_key)

    def _get_period_key(self) -> str:
        """Get the current period key for daily quotas.

        Returns:
            Period key in format YYYY-MM-DD
        """
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def _get_period_start(self) -> datetime:
        """Get the start time of the current period.

        Returns:
            Start of current day in UTC
        """
        now = datetime.now(UTC)
        return datetime(now.year, now.month, now.day, tzinfo=UTC)

    def _get_next_reset_time(self) -> datetime:
        """Get the next quota reset time.

        Returns:
            Start of next day in UTC
        """
        return self._get_period_start() + timedelta(days=1)
