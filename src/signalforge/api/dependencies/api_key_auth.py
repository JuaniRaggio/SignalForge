"""API Key authentication dependencies."""

import time
from typing import Annotated

import structlog
from fastapi import Depends, Header, HTTPException, Request, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.api.dependencies.database import get_db
from signalforge.core.redis import get_redis
from signalforge.gateway.api_key_manager import APIKeyManager
from signalforge.gateway.throttling import IntelligentThrottler
from signalforge.gateway.tier_rate_limiter import TierRateLimiter
from signalforge.models.api_key import APIKey

logger = structlog.get_logger(__name__)


async def get_api_key_from_header(
    x_api_key: Annotated[str | None, Header()] = None,
) -> str:
    """
    Extract API key from X-API-Key header.

    Args:
        x_api_key: API key from header

    Returns:
        API key string

    Raises:
        HTTPException: If API key is missing
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return x_api_key


async def validate_api_key(
    api_key: Annotated[str, Depends(get_api_key_from_header)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> APIKey:
    """
    Validate API key and return the associated record.

    Args:
        api_key: Plain API key from header
        db: Database session

    Returns:
        APIKey record

    Raises:
        HTTPException: If API key is invalid or inactive
    """
    api_key_record = await APIKeyManager.validate_api_key(db, api_key)

    if not api_key_record:
        logger.warning("invalid_api_key_attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not api_key_record.is_active:
        logger.warning(
            "inactive_api_key_attempt",
            key_id=str(api_key_record.id),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has been revoked",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    await db.commit()

    return api_key_record


async def check_api_key_rate_limit(
    request: Request,
    api_key: Annotated[APIKey, Depends(validate_api_key)],
    redis_client: Annotated[Redis, Depends(get_redis)],
) -> APIKey:
    """
    Check rate limits for API key requests.

    Args:
        request: FastAPI request
        api_key: Validated API key
        redis_client: Redis client

    Returns:
        APIKey record

    Raises:
        HTTPException: If rate limit is exceeded
    """
    rate_limiter = TierRateLimiter(redis_client)

    allowed, rate_status = await rate_limiter.check_rate_limit(
        user_id=api_key.user_id,
        tier=api_key.tier,
        rate_limit_override=api_key.rate_limit_override,
        burst_limit_override=api_key.burst_limit_override,
    )

    request.state.rate_limit_status = rate_status

    if not allowed:
        logger.warning(
            "api_key_rate_limit_exceeded",
            user_id=str(api_key.user_id),
            key_id=str(api_key.id),
            tier=api_key.tier.value,
            limit=rate_status.limit,
        )

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {api_key.tier.value} tier. "
            f"Limit: {rate_status.limit} requests per minute.",
            headers={
                "X-RateLimit-Limit": str(rate_status.limit),
                "X-RateLimit-Remaining": str(rate_status.remaining),
                "X-RateLimit-Reset": str(rate_status.reset_in_seconds),
                "Retry-After": str(rate_status.reset_in_seconds),
            },
        )

    request.state.rate_limit_remaining = rate_status.remaining

    return api_key


async def check_api_key_throttle(
    request: Request,
    api_key: Annotated[APIKey, Depends(check_api_key_rate_limit)],
    redis_client: Annotated[Redis, Depends(get_redis)],
) -> APIKey:
    """
    Check throttling status for API key requests.

    Implements graceful degradation under system load.

    Args:
        request: FastAPI request
        api_key: Validated API key
        redis_client: Redis client

    Returns:
        APIKey record

    Raises:
        HTTPException: If request should be rejected due to system load
    """
    throttler = IntelligentThrottler(redis_client)

    throttle_status = await throttler.check_throttle(
        user_id=api_key.user_id,
        tier=api_key.tier,
        endpoint=request.url.path,
    )

    request.state.throttle_status = throttle_status

    if not throttle_status.should_process:
        logger.warning(
            "api_key_request_rejected_throttle",
            user_id=str(api_key.user_id),
            key_id=str(api_key.id),
            tier=api_key.tier.value,
            throttle_level=throttle_status.level.value,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=throttle_status.message,
            headers={
                "Retry-After": str(throttle_status.delay_ms // 1000 + 1),
            },
        )

    if throttle_status.delay_ms > 0:
        await throttler.increment_load()
        time.sleep(throttle_status.delay_ms / 1000)
        await throttler.decrement_load()

    return api_key


async def track_api_key_usage(
    request: Request,
    api_key: Annotated[APIKey, Depends(check_api_key_throttle)],
) -> APIKey:
    """
    Track API key usage for billing and analytics.

    This should be called after the request is processed.
    Use as a dependency to automatically track all API key requests.

    Args:
        request: FastAPI request
        api_key: Validated API key

    Returns:
        APIKey record
    """
    request.state.api_key = api_key
    request.state.usage_start_time = time.time()

    return api_key


async def require_api_key_scope(
    api_key: APIKey,
    required_scope: str,
) -> None:
    """
    Require a specific scope for an API key.

    Args:
        api_key: Validated API key
        required_scope: Required scope

    Raises:
        HTTPException: If API key lacks the required scope
    """
    if not await APIKeyManager.check_scope(api_key, required_scope):
        logger.warning(
            "api_key_insufficient_scope",
            key_id=str(api_key.id),
            required_scope=required_scope,
            available_scopes=api_key.scopes,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"API key lacks required scope: {required_scope}",
        )


def require_scope(scope: str):  # type: ignore[no-untyped-def]
    """
    Factory function to create a scope requirement dependency.

    Usage:
        @router.get("/endpoint", dependencies=[Depends(require_scope("admin"))])

    Args:
        scope: Required scope

    Returns:
        Dependency function
    """

    async def _check_scope(
        api_key: Annotated[APIKey, Depends(track_api_key_usage)],
    ) -> None:
        await require_api_key_scope(api_key, scope)

    return _check_scope


APIKeyDep = Annotated[APIKey, Depends(track_api_key_usage)]
