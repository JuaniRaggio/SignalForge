"""Leaderboard API routes for paper trading rankings."""

from decimal import Decimal
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.api.dependencies.auth import get_current_user
from signalforge.api.dependencies.database import get_db
from signalforge.core.logging import get_logger
from signalforge.models.leaderboard import LeaderboardPeriod, LeaderboardType
from signalforge.models.user import User
from signalforge.paper_trading.leaderboard_schemas import (
    LeaderboardEntryResponse,
    LeaderboardResponse,
    RankingHistoryEntry,
    RankingHistoryResponse,
    UserRankingResponse,
)
from signalforge.paper_trading.leaderboard_service import LeaderboardService

router = APIRouter(prefix="/api/v1/leaderboard", tags=["leaderboard"])
logger = get_logger(__name__)


@router.get("", response_model=LeaderboardResponse)
async def get_leaderboard(
    *,
    db: Annotated[AsyncSession, Depends(get_db)],
    period: str = Query(
        "weekly",
        pattern="^(daily|weekly|monthly|all_time)$",
        description="Time period for leaderboard",
    ),
    type: str = Query(
        "total_return",
        pattern="^(total_return|sharpe_ratio|risk_adjusted)$",
        description="Ranking criteria",
    ),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of entries"),
    offset: int = Query(0, ge=0, description="Number of entries to skip"),
) -> LeaderboardResponse:
    """Get leaderboard rankings.

    Retrieves the current leaderboard for the specified period and ranking type.
    Results are cached for 5 minutes to improve performance.

    Args:
        period: Time period (daily, weekly, monthly, all_time)
        type: Ranking criteria (total_return, sharpe_ratio, risk_adjusted)
        limit: Maximum number of entries to return (1-500)
        offset: Number of entries to skip for pagination
        db: Database session

    Returns:
        Leaderboard with metadata and rankings

    Raises:
        HTTPException: If invalid parameters are provided
    """
    try:
        leaderboard_period = LeaderboardPeriod(period)
        leaderboard_type = LeaderboardType(type)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid period or type: {e}",
        )

    service = LeaderboardService(db)

    rankings = await service.get_leaderboard(
        period=leaderboard_period,
        leaderboard_type=leaderboard_type,
        limit=limit,
        offset=offset,
    )

    period_start, period_end = service._get_period_dates(leaderboard_period)

    entries = [
        LeaderboardEntryResponse(
            rank=r.rank,
            user_id=r.user_id,
            username=r.username,
            portfolio_name=r.portfolio_name,
            score=r.score,
            total_return_pct=r.total_return_pct,
            sharpe_ratio=r.sharpe_ratio,
            max_drawdown_pct=r.max_drawdown_pct,
            total_trades=r.total_trades,
        )
        for r in rankings
    ]

    return LeaderboardResponse(
        period=period,
        leaderboard_type=type,
        period_start=period_start,
        period_end=period_end,
        total_entries=len(entries),
        entries=entries,
    )


@router.get("/me", response_model=UserRankingResponse)
async def get_my_ranking(
    *,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    period: str = Query(
        "weekly",
        pattern="^(daily|weekly|monthly|all_time)$",
        description="Time period",
    ),
    type: str = Query(
        "total_return",
        pattern="^(total_return|sharpe_ratio|risk_adjusted)$",
        description="Ranking criteria",
    ),
) -> UserRankingResponse:
    """Get current user's ranking.

    Retrieves the authenticated user's ranking in the specified leaderboard.

    Args:
        period: Time period for leaderboard
        type: Ranking criteria
        db: Database session
        current_user: Authenticated user

    Returns:
        User's ranking information including percentile

    Raises:
        HTTPException: If user has no ranking in the leaderboard
    """
    try:
        leaderboard_period = LeaderboardPeriod(period)
        leaderboard_type = LeaderboardType(type)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid period or type: {e}",
        )

    service = LeaderboardService(db)

    ranking = await service.get_user_ranking(
        user_id=current_user.id,
        period=leaderboard_period,
        leaderboard_type=leaderboard_type,
    )

    if not ranking:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User has no ranking in this leaderboard",
        )

    total_participants = await service.get_total_participants(
        period=leaderboard_period,
        leaderboard_type=leaderboard_type,
    )

    percentile = Decimal("0")
    if total_participants > 0:
        percentile = (
            (Decimal(total_participants) - Decimal(ranking.rank) + Decimal("1"))
            / Decimal(total_participants)
        ) * Decimal("100")

    return UserRankingResponse(
        user_id=ranking.user_id,
        period=period,
        leaderboard_type=type,
        rank=ranking.rank,
        total_participants=total_participants,
        percentile=percentile,
        score=ranking.score,
        total_return_pct=ranking.total_return_pct,
        sharpe_ratio=ranking.sharpe_ratio,
        max_drawdown_pct=ranking.max_drawdown_pct,
        total_trades=ranking.total_trades,
    )


@router.get("/me/history", response_model=RankingHistoryResponse)
async def get_my_ranking_history(
    *,
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: Annotated[User, Depends(get_current_user)],
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
) -> RankingHistoryResponse:
    """Get user's ranking history.

    Retrieves the authenticated user's ranking history over the specified time period.

    Args:
        days: Number of days of history to retrieve (1-365)
        db: Database session
        current_user: Authenticated user

    Returns:
        User's historical rankings
    """
    service = LeaderboardService(db)

    history_data = await service.get_user_rankings_history(
        user_id=current_user.id,
        days=days,
    )

    history = [
        RankingHistoryEntry(
            date=item["date"],
            period=item["period"],
            leaderboard_type=item["leaderboard_type"],
            rank=item["rank"],
            score=item["score"],
            total_return_pct=item["total_return_pct"],
        )
        for item in history_data
    ]

    return RankingHistoryResponse(
        user_id=current_user.id,
        history=history,
    )


@router.get("/users/{user_id}", response_model=UserRankingResponse)
async def get_user_ranking(
    user_id: UUID,
    *,
    db: Annotated[AsyncSession, Depends(get_db)],
    period: str = Query(
        "weekly",
        pattern="^(daily|weekly|monthly|all_time)$",
        description="Time period",
    ),
    type: str = Query(
        "total_return",
        pattern="^(total_return|sharpe_ratio|risk_adjusted)$",
        description="Ranking criteria",
    ),
) -> UserRankingResponse:
    """Get specific user's ranking.

    Retrieves the ranking of any user in the leaderboard. Public endpoint.

    Args:
        user_id: User ID to look up
        period: Time period for leaderboard
        type: Ranking criteria
        db: Database session

    Returns:
        User's ranking information

    Raises:
        HTTPException: If user not found or has no ranking
    """
    try:
        leaderboard_period = LeaderboardPeriod(period)
        leaderboard_type = LeaderboardType(type)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid period or type: {e}",
        )

    service = LeaderboardService(db)

    ranking = await service.get_user_ranking(
        user_id=user_id,
        period=leaderboard_period,
        leaderboard_type=leaderboard_type,
    )

    if not ranking:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or has no ranking in this leaderboard",
        )

    total_participants = await service.get_total_participants(
        period=leaderboard_period,
        leaderboard_type=leaderboard_type,
    )

    percentile = Decimal("0")
    if total_participants > 0:
        percentile = (
            (Decimal(total_participants) - Decimal(ranking.rank) + Decimal("1"))
            / Decimal(total_participants)
        ) * Decimal("100")

    return UserRankingResponse(
        user_id=ranking.user_id,
        period=period,
        leaderboard_type=type,
        rank=ranking.rank,
        total_participants=total_participants,
        percentile=percentile,
        score=ranking.score,
        total_return_pct=ranking.total_return_pct,
        sharpe_ratio=ranking.sharpe_ratio,
        max_drawdown_pct=ranking.max_drawdown_pct,
        total_trades=ranking.total_trades,
    )
