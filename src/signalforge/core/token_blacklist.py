"""Token blacklist management using Redis."""


from redis.asyncio import Redis

from signalforge.core.config import get_settings

settings = get_settings()


async def add_token_to_blacklist(
    redis_client: Redis,
    token: str,
    expire_seconds: int | None = None,
) -> None:
    """
    Add a token to the blacklist.

    Args:
        redis_client: Redis client instance
        token: The JWT token to blacklist
        expire_seconds: Optional expiration time in seconds (defaults to token expiry)
    """
    if expire_seconds is None:
        expire_seconds = settings.jwt_access_token_expire_minutes * 60

    key = f"blacklist:token:{token}"
    await redis_client.setex(key, expire_seconds, "1")


async def is_token_blacklisted(redis_client: Redis, token: str) -> bool:
    """
    Check if a token is blacklisted.

    Args:
        redis_client: Redis client instance
        token: The JWT token to check

    Returns:
        True if token is blacklisted, False otherwise
    """
    key = f"blacklist:token:{token}"
    result = await redis_client.exists(key)
    return bool(result > 0)


async def remove_token_from_blacklist(redis_client: Redis, token: str) -> None:
    """
    Remove a token from the blacklist (if needed for testing or admin operations).

    Args:
        redis_client: Redis client instance
        token: The JWT token to remove from blacklist
    """
    key = f"blacklist:token:{token}"
    await redis_client.delete(key)
