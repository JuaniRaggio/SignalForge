"""Rate limiting utilities using Redis."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from redis.asyncio import Redis

from signalforge.core.config import get_settings
from signalforge.core.redis import get_redis

settings = get_settings()


async def check_rate_limit(
    request: Request,
    redis_client: Annotated[Redis, Depends(get_redis)],
    limit: int | None = None,
    window_seconds: int | None = None,
) -> None:
    """
    Check rate limit for the given request.

    Args:
        request: The incoming HTTP request
        redis_client: Redis client instance
        limit: Maximum number of requests allowed (defaults to settings.rate_limit_per_minute)
        window_seconds: Time window in seconds (defaults to settings.rate_limit_window_seconds)

    Raises:
        HTTPException: If rate limit is exceeded
    """
    if limit is None:
        limit = settings.rate_limit_per_minute
    if window_seconds is None:
        window_seconds = settings.rate_limit_window_seconds

    client_ip = request.client.host if request.client else "unknown"
    endpoint = request.url.path

    key = f"rate_limit:{endpoint}:{client_ip}"

    current_count = await redis_client.get(key)

    if current_count is None:
        await redis_client.setex(key, window_seconds, 1)
        return

    count = int(current_count)
    if count >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {limit} requests per {window_seconds} seconds allowed.",
            headers={"Retry-After": str(window_seconds)},
        )

    await redis_client.incr(key)


async def get_rate_limit_info(
    request: Request,
    redis_client: Redis,
    limit: int | None = None,
    window_seconds: int | None = None,
) -> dict[str, int]:
    """
    Get current rate limit information for a request.

    Args:
        request: The incoming HTTP request
        redis_client: Redis client instance
        limit: Maximum number of requests allowed
        window_seconds: Time window in seconds

    Returns:
        Dictionary with rate limit information
    """
    if limit is None:
        limit = settings.rate_limit_per_minute
    if window_seconds is None:
        window_seconds = settings.rate_limit_window_seconds

    client_ip = request.client.host if request.client else "unknown"
    endpoint = request.url.path

    key = f"rate_limit:{endpoint}:{client_ip}"

    current_count = await redis_client.get(key)
    ttl = await redis_client.ttl(key)

    count = int(current_count) if current_count else 0
    remaining = max(0, limit - count)

    return {
        "limit": limit,
        "remaining": remaining,
        "used": count,
        "reset_in_seconds": ttl if ttl > 0 else window_seconds,
    }
