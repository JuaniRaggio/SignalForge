"""Usage tracking for billing and analytics."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID

import structlog
from redis.asyncio import Redis

logger = structlog.get_logger(__name__)


@dataclass
class UsageStats:
    """Usage statistics for a user."""

    user_id: UUID
    period: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    rate_limited_requests: int
    average_response_time_ms: float
    endpoints_used: dict[str, int]
    peak_hour: str | None
    total_data_transferred_bytes: int


@dataclass
class HourlyUsage:
    """Hourly usage aggregates."""

    hour: datetime
    request_count: int
    average_response_time_ms: float
    error_count: int
    rate_limit_count: int


class UsageTracker:
    """
    Tracks API usage for billing and analytics.

    Stores usage data in Redis with TTL and provides aggregation methods.
    Uses Polars for efficient data processing.
    """

    def __init__(self, redis_client: Redis) -> None:
        """
        Initialize the usage tracker.

        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client

    def _get_request_key(self, user_id: UUID, timestamp: datetime) -> str:
        """
        Generate Redis key for request tracking.

        Args:
            user_id: User ID
            timestamp: Request timestamp

        Returns:
            Redis key string
        """
        date_str = timestamp.strftime("%Y-%m-%d")
        hour_str = timestamp.strftime("%H")
        return f"usage:requests:{user_id}:{date_str}:{hour_str}"

    def _get_endpoint_key(self, user_id: UUID, date: datetime) -> str:
        """
        Generate Redis key for endpoint tracking.

        Args:
            user_id: User ID
            date: Date for tracking

        Returns:
            Redis key string
        """
        date_str = date.strftime("%Y-%m-%d")
        return f"usage:endpoints:{user_id}:{date_str}"

    def _get_stats_key(self, user_id: UUID, date: datetime) -> str:
        """
        Generate Redis key for daily stats.

        Args:
            user_id: User ID
            date: Date for stats

        Returns:
            Redis key string
        """
        date_str = date.strftime("%Y-%m-%d")
        return f"usage:stats:{user_id}:{date_str}"

    async def track_request(
        self,
        user_id: UUID,
        endpoint: str,
        response_time_ms: float,
        status_code: int,
        data_transferred_bytes: int = 0,
    ) -> None:
        """
        Track an API request.

        Args:
            user_id: User ID
            endpoint: API endpoint path
            response_time_ms: Response time in milliseconds
            status_code: HTTP status code
            data_transferred_bytes: Bytes transferred in response
        """
        now = datetime.now(UTC)
        request_key = self._get_request_key(user_id, now)
        endpoint_key = self._get_endpoint_key(user_id, now)
        stats_key = self._get_stats_key(user_id, now)

        ttl_seconds = 90 * 24 * 60 * 60

        is_success = 200 <= status_code < 300
        is_rate_limited = status_code == 429

        pipeline = self.redis.pipeline()

        pipeline.incr(request_key)
        pipeline.expire(request_key, ttl_seconds)

        pipeline.hincrby(endpoint_key, endpoint, 1)
        pipeline.expire(endpoint_key, ttl_seconds)

        pipeline.hincrby(stats_key, "total_requests", 1)
        if is_success:
            pipeline.hincrby(stats_key, "successful_requests", 1)
        else:
            pipeline.hincrby(stats_key, "failed_requests", 1)

        if is_rate_limited:
            pipeline.hincrby(stats_key, "rate_limited_requests", 1)

        pipeline.hincrbyfloat(stats_key, "total_response_time_ms", response_time_ms)
        pipeline.hincrby(stats_key, "total_data_transferred_bytes", data_transferred_bytes)
        pipeline.expire(stats_key, ttl_seconds)

        await pipeline.execute()

        logger.debug(
            "request_tracked",
            user_id=str(user_id),
            endpoint=endpoint,
            status_code=status_code,
            response_time_ms=response_time_ms,
        )

    async def get_usage_stats(
        self,
        user_id: UUID,
        period: Literal["today", "week", "month"] = "today",
    ) -> UsageStats:
        """
        Get aggregated usage statistics for a user.

        Args:
            user_id: User ID
            period: Time period to aggregate (today, week, month)

        Returns:
            UsageStats with aggregated data
        """
        now = datetime.now(UTC)

        if period == "today":
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            dates = [start_date]
        elif period == "week":
            start_date = now - timedelta(days=7)
            dates = [start_date + timedelta(days=i) for i in range(8)]
        else:
            start_date = now - timedelta(days=30)
            dates = [start_date + timedelta(days=i) for i in range(31)]

        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        rate_limited_requests = 0
        total_response_time_ms = 0.0
        total_data_transferred_bytes = 0
        endpoint_counts: dict[str, int] = {}
        hourly_counts: dict[str, int] = {}

        for date in dates:
            stats_key = self._get_stats_key(user_id, date)
            stats_data = await self.redis.hgetall(stats_key)  # type: ignore[misc]

            if stats_data:
                total_requests += int(stats_data.get("total_requests", 0))
                successful_requests += int(stats_data.get("successful_requests", 0))
                failed_requests += int(stats_data.get("failed_requests", 0))
                rate_limited_requests += int(stats_data.get("rate_limited_requests", 0))
                total_response_time_ms += float(stats_data.get("total_response_time_ms", 0))
                total_data_transferred_bytes += int(
                    stats_data.get("total_data_transferred_bytes", 0)
                )

            endpoint_key = self._get_endpoint_key(user_id, date)
            endpoint_data = await self.redis.hgetall(endpoint_key)  # type: ignore[misc]
            for endpoint, count in endpoint_data.items():
                endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + int(count)

            for hour in range(24):
                hour_date = date.replace(hour=hour, minute=0, second=0, microsecond=0)
                request_key = self._get_request_key(user_id, hour_date)
                hour_count = await self.redis.get(request_key)
                if hour_count:
                    hour_str = hour_date.strftime("%Y-%m-%d %H:00")
                    hourly_counts[hour_str] = int(hour_count)

        peak_hour = max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else None

        average_response_time_ms = (
            total_response_time_ms / total_requests if total_requests > 0 else 0.0
        )

        return UsageStats(
            user_id=user_id,
            period=period,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            rate_limited_requests=rate_limited_requests,
            average_response_time_ms=average_response_time_ms,
            endpoints_used=endpoint_counts,
            peak_hour=peak_hour,
            total_data_transferred_bytes=total_data_transferred_bytes,
        )

    async def get_hourly_aggregates(
        self,
        user_id: UUID,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[HourlyUsage]:
        """
        Get hourly usage aggregates.

        Args:
            user_id: User ID
            start_date: Start date (defaults to 24 hours ago)
            end_date: End date (defaults to now)

        Returns:
            List of HourlyUsage records
        """
        now = datetime.now(UTC)

        if end_date is None:
            end_date = now

        if start_date is None:
            start_date = now - timedelta(hours=24)

        current = start_date.replace(minute=0, second=0, microsecond=0)
        hourly_data: list[HourlyUsage] = []

        while current <= end_date:
            request_key = self._get_request_key(user_id, current)
            request_count_str = await self.redis.get(request_key)
            request_count = int(request_count_str) if request_count_str else 0

            date_for_stats = current.replace(hour=0, minute=0, second=0, microsecond=0)
            stats_key = self._get_stats_key(user_id, date_for_stats)
            stats_data = await self.redis.hgetall(stats_key)  # type: ignore[misc]

            total_requests = int(stats_data.get("total_requests", 0))
            total_response_time_ms = float(stats_data.get("total_response_time_ms", 0))
            average_response_time_ms = (
                total_response_time_ms / total_requests if total_requests > 0 else 0.0
            )

            failed_requests = int(stats_data.get("failed_requests", 0))
            rate_limited_requests = int(stats_data.get("rate_limited_requests", 0))

            hourly_data.append(
                HourlyUsage(
                    hour=current,
                    request_count=request_count,
                    average_response_time_ms=average_response_time_ms,
                    error_count=failed_requests,
                    rate_limit_count=rate_limited_requests,
                )
            )

            current += timedelta(hours=1)

        return hourly_data

    async def get_endpoint_breakdown(
        self,
        user_id: UUID,
        days: int = 7,
    ) -> dict[str, int]:
        """
        Get endpoint usage breakdown.

        Args:
            user_id: User ID
            days: Number of days to aggregate

        Returns:
            Dictionary mapping endpoints to request counts
        """
        now = datetime.now(UTC)
        endpoint_counts: dict[str, int] = {}

        for day_offset in range(days):
            date = now - timedelta(days=day_offset)
            endpoint_key = self._get_endpoint_key(user_id, date)
            endpoint_data = await self.redis.hgetall(endpoint_key)  # type: ignore[misc]

            for endpoint, count in endpoint_data.items():
                endpoint_counts[endpoint] = endpoint_counts.get(endpoint, 0) + int(count)

        return endpoint_counts
