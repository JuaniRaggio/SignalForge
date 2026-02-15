"""User preferences and activity routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.api.dependencies.auth import get_current_active_user
from signalforge.api.dependencies.database import get_db
from signalforge.models.user import User
from signalforge.models.user_activity import UserActivity
from signalforge.schemas.user_preferences import (
    UserActivityCreate,
    UserActivityResponse,
    UserPreferencesResponse,
    UserPreferencesUpdate,
    WatchlistUpdate,
)

router = APIRouter()


@router.get("/me/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> User:
    """
    Get current user's preferences and profile information.

    Returns complete user profile including risk tolerance, investment horizon,
    experience level, preferred sectors, watchlist, and notification settings.
    """
    return current_user


@router.put("/me/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(
    preferences: UserPreferencesUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """
    Update current user's preferences.

    Only provided fields will be updated. Omitted fields remain unchanged.
    """
    if preferences.risk_tolerance is not None:
        current_user.risk_tolerance = preferences.risk_tolerance

    if preferences.investment_horizon is not None:
        current_user.investment_horizon = preferences.investment_horizon

    if preferences.experience_level is not None:
        current_user.experience_level = preferences.experience_level

    if preferences.preferred_sectors is not None:
        current_user.preferred_sectors = preferences.preferred_sectors

    if preferences.notification_preferences is not None:
        current_user.notification_preferences = preferences.notification_preferences

    await db.flush()
    await db.refresh(current_user)

    return current_user


@router.post(
    "/me/watchlist",
    response_model=UserPreferencesResponse,
    status_code=status.HTTP_200_OK,
)
async def add_to_watchlist(
    watchlist_update: WatchlistUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """
    Add symbols to user's watchlist.

    Duplicate symbols are automatically ignored. Symbols are stored in uppercase.
    """
    current_watchlist = set(current_user.watchlist or [])

    normalized_symbols = [s.upper() for s in watchlist_update.symbols]

    current_watchlist.update(normalized_symbols)

    current_user.watchlist = sorted(current_watchlist)

    await db.flush()
    await db.refresh(current_user)

    return current_user


@router.delete(
    "/me/watchlist/{symbol}",
    response_model=UserPreferencesResponse,
    status_code=status.HTTP_200_OK,
)
async def remove_from_watchlist(
    symbol: Annotated[str, Path(..., description="Stock symbol to remove")],
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """
    Remove a symbol from user's watchlist.

    Returns 404 if the symbol is not in the watchlist.
    """
    current_watchlist = current_user.watchlist or []

    normalized_symbol = symbol.upper()

    if normalized_symbol not in current_watchlist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Symbol {normalized_symbol} not found in watchlist",
        )

    updated_watchlist = [s for s in current_watchlist if s != normalized_symbol]
    current_user.watchlist = updated_watchlist

    await db.flush()
    await db.refresh(current_user)

    return current_user


@router.post(
    "/me/activity",
    response_model=UserActivityResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_user_activity(
    activity: UserActivityCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UserActivity:
    """
    Record user activity for implicit preference learning.

    This endpoint tracks user interactions like views, clicks, follows, and dismissals
    to build a better understanding of user preferences over time.
    """
    user_activity = UserActivity(
        user_id=current_user.id,
        activity_type=activity.activity_type,
        symbol=activity.symbol.upper() if activity.symbol else None,
        sector=activity.sector,
        signal_id=activity.signal_id,
        metadata_=activity.metadata,
    )

    db.add(user_activity)
    await db.flush()
    await db.refresh(user_activity)

    return user_activity


@router.get("/me/activity", response_model=list[UserActivityResponse])
async def get_user_activity(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[UserActivity]:
    """
    Get user activity history.

    Returns paginated list of user activities ordered by most recent first.
    Maximum 100 records per request.
    """
    result = await db.execute(
        select(UserActivity)
        .where(UserActivity.user_id == current_user.id)
        .order_by(desc(UserActivity.created_at))
        .limit(limit)
        .offset(offset)
    )

    activities = result.scalars().all()

    return list(activities)
