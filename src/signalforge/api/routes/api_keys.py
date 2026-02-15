"""API Key management routes."""

from typing import Annotated, Literal
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.api.dependencies.auth import get_current_active_user
from signalforge.api.dependencies.database import get_db
from signalforge.core.redis import get_redis
from signalforge.gateway.api_key_manager import APIKeyManager
from signalforge.gateway.tier_rate_limiter import TierRateLimiter
from signalforge.gateway.usage_tracker import UsageTracker
from signalforge.models.user import User
from signalforge.schemas.api_key import (
    APIKeyCreate,
    APIKeyCreatedResponse,
    APIKeyResponse,
    RateLimitStatusResponse,
    UsageStatsResponse,
)
from signalforge.schemas.base import BaseResponse, EmptyResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api-keys", tags=["api-keys"])


@router.post(
    "",
    response_model=BaseResponse[APIKeyCreatedResponse],
    status_code=status.HTTP_201_CREATED,
)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BaseResponse[APIKeyCreatedResponse]:
    """
    Create a new API key for the authenticated user.

    The plain API key is returned only once and should be stored securely.
    """
    try:
        plain_key, api_key = await APIKeyManager.generate_api_key(
            db=db,
            user_id=current_user.id,
            name=key_data.name,
            tier=key_data.tier,
            scopes=key_data.scopes,
            rate_limit_override=key_data.rate_limit_override,
            burst_limit_override=key_data.burst_limit_override,
            expires_at=key_data.expires_at,
        )

        await db.commit()

        logger.info(
            "api_key_created",
            user_id=str(current_user.id),
            key_id=str(api_key.id),
            tier=key_data.tier.value,
        )

        return BaseResponse(
            data=APIKeyCreatedResponse(
                api_key=plain_key,
                key_info=APIKeyResponse.model_validate(api_key),
            ),
            message="API key created successfully",
        )

    except ValueError as e:
        logger.warning(
            "api_key_creation_failed",
            user_id=str(current_user.id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            "api_key_creation_error",
            user_id=str(current_user.id),
            error=str(e),
        )
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key",
        )


@router.get(
    "",
    response_model=BaseResponse[list[APIKeyResponse]],
)
async def list_api_keys(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BaseResponse[list[APIKeyResponse]]:
    """
    List all API keys for the authenticated user.

    Returns metadata for all keys (never returns the actual key values).
    """
    try:
        api_keys = await APIKeyManager.list_user_keys(db, current_user.id)

        return BaseResponse(
            data=[APIKeyResponse.model_validate(key) for key in api_keys],
            message=f"Retrieved {len(api_keys)} API keys",
        )

    except Exception as e:
        logger.error(
            "api_key_list_error",
            user_id=str(current_user.id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API keys",
        )


@router.get(
    "/{key_id}",
    response_model=BaseResponse[APIKeyResponse],
)
async def get_api_key(
    key_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BaseResponse[APIKeyResponse]:
    """
    Get details of a specific API key.

    Returns metadata only (never returns the actual key value).
    """
    try:
        api_key = await APIKeyManager.get_key_by_id(db, key_id, current_user.id)

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )

        return BaseResponse(
            data=APIKeyResponse.model_validate(api_key),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "api_key_get_error",
            user_id=str(current_user.id),
            key_id=str(key_id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API key",
        )


@router.delete(
    "/{key_id}",
    response_model=EmptyResponse,
)
async def revoke_api_key(
    key_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EmptyResponse:
    """
    Revoke an API key.

    The key will be deactivated and can no longer be used for authentication.
    """
    try:
        success = await APIKeyManager.revoke_api_key(db, key_id, current_user.id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )

        await db.commit()

        logger.info(
            "api_key_revoked",
            user_id=str(current_user.id),
            key_id=str(key_id),
        )

        return EmptyResponse(
            message="API key revoked successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "api_key_revoke_error",
            user_id=str(current_user.id),
            key_id=str(key_id),
            error=str(e),
        )
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key",
        )


@router.get(
    "/{key_id}/usage",
    response_model=BaseResponse[UsageStatsResponse],
)
async def get_api_key_usage(
    key_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    redis_client: Annotated[Redis, Depends(get_redis)],
    period: Literal["today", "week", "month"] = "today",
) -> BaseResponse[UsageStatsResponse]:
    """
    Get usage statistics for an API key.

    Provides metrics including request counts, response times, and endpoint usage.
    """
    try:
        api_key = await APIKeyManager.get_key_by_id(db, key_id, current_user.id)

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )

        tracker = UsageTracker(redis_client)
        usage_stats = await tracker.get_usage_stats(current_user.id, period)

        return BaseResponse(
            data=UsageStatsResponse(
                user_id=usage_stats.user_id,
                period=usage_stats.period,
                total_requests=usage_stats.total_requests,
                successful_requests=usage_stats.successful_requests,
                failed_requests=usage_stats.failed_requests,
                rate_limited_requests=usage_stats.rate_limited_requests,
                average_response_time_ms=usage_stats.average_response_time_ms,
                endpoints_used=usage_stats.endpoints_used,
                peak_hour=usage_stats.peak_hour,
                total_data_transferred_bytes=usage_stats.total_data_transferred_bytes,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "api_key_usage_error",
            user_id=str(current_user.id),
            key_id=str(key_id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve usage statistics",
        )


@router.get(
    "/{key_id}/rate-limit-status",
    response_model=BaseResponse[RateLimitStatusResponse],
)
async def get_rate_limit_status(
    key_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    redis_client: Annotated[Redis, Depends(get_redis)],
) -> BaseResponse[RateLimitStatusResponse]:
    """
    Get current rate limit status for an API key.

    Shows remaining requests and reset time.
    """
    try:
        api_key = await APIKeyManager.get_key_by_id(db, key_id, current_user.id)

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )

        rate_limiter = TierRateLimiter(redis_client)
        status_info = await rate_limiter.get_rate_limit_status(
            current_user.id,
            api_key.tier,
            api_key.rate_limit_override,
        )

        return BaseResponse(
            data=RateLimitStatusResponse(
                limit=status_info.limit,
                remaining=status_info.remaining,
                used=status_info.used,
                reset_in_seconds=status_info.reset_in_seconds,
                is_burst=status_info.is_burst,
                tier=status_info.tier,
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "rate_limit_status_error",
            user_id=str(current_user.id),
            key_id=str(key_id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve rate limit status",
        )
