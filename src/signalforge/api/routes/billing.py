"""Billing and subscription API routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.api.dependencies.auth import get_current_active_user
from signalforge.api.dependencies.database import get_db
from signalforge.billing.quota_manager import QuotaManager
from signalforge.billing.schemas import (
    InvoiceResponse,
    QuotaCheckRequest,
    QuotaCheckResponse,
    SubscriptionCancelRequest,
    SubscriptionCreate,
    SubscriptionPlanResponse,
    SubscriptionResponse,
    UsageSummary,
    UserUsageResponse,
)
from signalforge.billing.service import (
    BillingService,
    InvalidSubscriptionError,
    PlanNotFoundError,
    SubscriptionNotFoundError,
)
from signalforge.core.redis import get_redis
from signalforge.models.user import User

router = APIRouter()


async def get_billing_service(
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BillingService:
    """Dependency to get billing service.

    Args:
        db: Database session

    Returns:
        BillingService instance
    """
    return BillingService(db)


async def get_quota_manager(
    db: Annotated[AsyncSession, Depends(get_db)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> QuotaManager:
    """Dependency to get quota manager.

    Args:
        db: Database session
        redis: Redis client

    Returns:
        QuotaManager instance
    """
    return QuotaManager(db, redis)


@router.get("/plans", response_model=list[SubscriptionPlanResponse])
async def list_subscription_plans(
    billing_service: Annotated[BillingService, Depends(get_billing_service)],
    active_only: bool = True,
) -> list[SubscriptionPlanResponse]:
    """List all available subscription plans.

    Args:
        billing_service: Billing service instance
        active_only: Only return active plans

    Returns:
        List of subscription plans
    """
    plans = await billing_service.list_plans(active_only=active_only)
    return [SubscriptionPlanResponse.model_validate(plan) for plan in plans]


@router.get("/subscription", response_model=SubscriptionResponse | None)
async def get_current_subscription(
    current_user: Annotated[User, Depends(get_current_active_user)],
    billing_service: Annotated[BillingService, Depends(get_billing_service)],
) -> SubscriptionResponse | None:
    """Get current user's active subscription.

    Args:
        current_user: Current authenticated user
        billing_service: Billing service instance

    Returns:
        Current subscription or None if no active subscription
    """
    subscription = await billing_service.get_user_active_subscription(current_user.id)
    if subscription is None:
        return None

    return SubscriptionResponse.model_validate(subscription)


@router.post(
    "/subscribe",
    response_model=SubscriptionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def subscribe_to_plan(
    subscription_data: SubscriptionCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    billing_service: Annotated[BillingService, Depends(get_billing_service)],
) -> SubscriptionResponse:
    """Subscribe current user to a plan.

    Args:
        subscription_data: Subscription data
        current_user: Current authenticated user
        billing_service: Billing service instance

    Returns:
        Created subscription

    Raises:
        HTTPException: If plan not found or user already has subscription
    """
    try:
        subscription = await billing_service.subscribe_user(
            current_user.id,
            subscription_data,
        )
        return SubscriptionResponse.model_validate(subscription)
    except PlanNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except InvalidSubscriptionError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/cancel", response_model=SubscriptionResponse)
async def cancel_subscription(
    cancel_request: SubscriptionCancelRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    billing_service: Annotated[BillingService, Depends(get_billing_service)],
) -> SubscriptionResponse:
    """Cancel current user's subscription.

    Args:
        cancel_request: Cancellation request data
        current_user: Current authenticated user
        billing_service: Billing service instance

    Returns:
        Updated subscription

    Raises:
        HTTPException: If no active subscription or cannot cancel
    """
    subscription = await billing_service.get_user_active_subscription(current_user.id)
    if subscription is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active subscription found",
        )

    try:
        subscription = await billing_service.cancel_subscription(
            current_user.id,
            subscription.id,
            cancel_immediately=cancel_request.cancel_immediately,
            reason=cancel_request.reason,
        )
        return SubscriptionResponse.model_validate(subscription)
    except (SubscriptionNotFoundError, InvalidSubscriptionError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post(
    "/subscription/{subscription_id}/reactivate",
    response_model=SubscriptionResponse,
)
async def reactivate_subscription(
    subscription_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    billing_service: Annotated[BillingService, Depends(get_billing_service)],
) -> SubscriptionResponse:
    """Reactivate a canceled subscription.

    Args:
        subscription_id: Subscription ID
        current_user: Current authenticated user
        billing_service: Billing service instance

    Returns:
        Updated subscription

    Raises:
        HTTPException: If subscription not found or cannot reactivate
    """
    try:
        subscription = await billing_service.reactivate_subscription(
            current_user.id,
            subscription_id,
        )
        return SubscriptionResponse.model_validate(subscription)
    except (SubscriptionNotFoundError, InvalidSubscriptionError) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/usage", response_model=UserUsageResponse)
async def get_usage_stats(
    current_user: Annotated[User, Depends(get_current_active_user)],
    quota_manager: Annotated[QuotaManager, Depends(get_quota_manager)],
) -> UserUsageResponse:
    """Get current usage statistics for the user.

    Args:
        current_user: Current authenticated user
        quota_manager: Quota manager instance

    Returns:
        User usage statistics
    """
    tier = await quota_manager.get_user_tier(current_user.id)
    usage_stats = await quota_manager.get_usage_stats(current_user.id)

    usage_summaries = [
        UsageSummary(
            resource_type=resource_type,
            total_usage=stats["current_usage"],
            quota_limit=stats["quota_limit"],
            remaining=stats["remaining"],
            is_unlimited=stats["is_unlimited"],
            reset_at=stats.get("reset_at"),
        )
        for resource_type, stats in usage_stats.items()
    ]

    return UserUsageResponse(
        user_id=current_user.id,
        tier=tier,
        usage=usage_summaries,
    )


@router.post("/quota/check", response_model=QuotaCheckResponse)
async def check_quota(
    quota_check: QuotaCheckRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    quota_manager: Annotated[QuotaManager, Depends(get_quota_manager)],
) -> QuotaCheckResponse:
    """Check if user has available quota for a resource.

    Args:
        quota_check: Quota check request
        current_user: Current authenticated user
        quota_manager: Quota manager instance

    Returns:
        Quota check response
    """
    allowed = await quota_manager.check_quota(
        current_user.id,
        quota_check.resource_type,
    )
    current_usage = await quota_manager.get_current_usage(
        current_user.id,
        quota_check.resource_type,
    )
    remaining = await quota_manager.get_remaining_quota(
        current_user.id,
        quota_check.resource_type,
    )
    tier = await quota_manager.get_user_tier(current_user.id)

    from signalforge.billing.tier_features import (
        get_quota_limit,
        has_unlimited_quota,
    )

    is_unlimited = has_unlimited_quota(tier, quota_check.resource_type)
    quota_limit = get_quota_limit(tier, quota_check.resource_type)

    message = None
    if not allowed:
        if quota_limit == 0:
            message = f"Feature '{quota_check.resource_type}' not available in your tier"
        else:
            message = f"Quota exceeded for '{quota_check.resource_type}'"

    return QuotaCheckResponse(
        allowed=allowed,
        resource_type=quota_check.resource_type,
        current_usage=current_usage,
        quota_limit=quota_limit,
        remaining=remaining,
        is_unlimited=is_unlimited,
        message=message,
    )


@router.get("/invoices", response_model=list[InvoiceResponse])
async def get_user_invoices(
    current_user: Annotated[User, Depends(get_current_active_user)],
    billing_service: Annotated[BillingService, Depends(get_billing_service)],
    limit: int = 50,
    offset: int = 0,
) -> list[InvoiceResponse]:
    """Get user's invoice history.

    Args:
        current_user: Current authenticated user
        billing_service: Billing service instance
        limit: Maximum number of invoices to return
        offset: Number of invoices to skip

    Returns:
        List of invoices
    """
    invoices = await billing_service.get_user_invoices(
        current_user.id,
        limit=limit,
        offset=offset,
    )
    return [InvoiceResponse.model_validate(invoice) for invoice in invoices]
