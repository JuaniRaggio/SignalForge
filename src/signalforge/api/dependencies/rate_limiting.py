"""Rate limiting dependencies for API endpoints."""

from typing import Annotated

from fastapi import Depends, Request
from redis.asyncio import Redis

from signalforge.core.rate_limiter import check_rate_limit
from signalforge.core.redis import get_redis


async def rate_limit_auth_endpoints(
    request: Request,
    redis_client: Annotated[Redis, Depends(get_redis)],
) -> None:
    """Rate limit dependency for authentication endpoints."""
    await check_rate_limit(request, redis_client, limit=5, window_seconds=60)
