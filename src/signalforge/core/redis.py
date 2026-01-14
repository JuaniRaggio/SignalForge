"""Redis connection management."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis.asyncio as redis
from redis.asyncio import Redis

from signalforge.core.config import get_settings

settings = get_settings()

_redis_client: Redis | None = None


async def get_redis() -> Redis:
    """Get Redis client instance."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(  # type: ignore[no-untyped-call]
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


@asynccontextmanager
async def get_redis_context() -> AsyncGenerator[Redis, None]:
    """Context manager for Redis client (for use outside of FastAPI)."""
    client = await get_redis()
    try:
        yield client
    finally:
        pass


async def close_redis() -> None:
    """Close Redis connection."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None


async def check_redis_connection() -> bool:
    """Check if Redis connection is healthy."""
    try:
        client = await get_redis()
        await client.ping()
        return True
    except Exception:
        return False
