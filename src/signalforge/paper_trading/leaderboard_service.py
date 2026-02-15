"""Leaderboard service for paper trading rankings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.logging import get_logger
from signalforge.core.redis import get_redis
from signalforge.models.leaderboard import (
    LeaderboardEntry,
    LeaderboardPeriod,
    LeaderboardType,
)
from signalforge.models.paper_trading import (
    PaperPortfolio,
    PaperPortfolioSnapshot,
    PortfolioStatus,
)
from signalforge.models.user import User

logger = get_logger(__name__)

LEADERBOARD_CACHE_TTL = 300  # 5 minutes


@dataclass
class LeaderboardRanking:
    """Single leaderboard entry for display."""

    rank: int
    user_id: UUID
    username: str
    portfolio_id: UUID
    portfolio_name: str
    score: Decimal
    total_return_pct: Decimal
    sharpe_ratio: Decimal | None
    max_drawdown_pct: Decimal | None
    total_trades: int


@dataclass
class PortfolioPerformance:
    """Performance metrics for a portfolio."""

    portfolio_id: UUID
    user_id: UUID
    total_return_pct: Decimal
    sharpe_ratio: Decimal | None
    max_drawdown_pct: Decimal | None
    total_trades: int


class LeaderboardService:
    """Manage leaderboard calculations and retrieval."""

    def __init__(self, session: AsyncSession):
        """Initialize leaderboard service.

        Args:
            session: Database session
        """
        self.session = session

    async def calculate_leaderboard(
        self,
        period: LeaderboardPeriod,
        leaderboard_type: LeaderboardType = LeaderboardType.TOTAL_RETURN,
    ) -> list[LeaderboardEntry]:
        """Calculate and store leaderboard for a period.

        Args:
            period: Time period for leaderboard
            leaderboard_type: Ranking criteria

        Returns:
            List of leaderboard entries
        """
        logger.info(
            "calculating_leaderboard",
            period=period.value,
            leaderboard_type=leaderboard_type.value,
        )

        period_start, period_end = self._get_period_dates(period)

        # Delete existing entries for this period and type
        await self.session.execute(
            delete(LeaderboardEntry).where(
                LeaderboardEntry.period == period,
                LeaderboardEntry.leaderboard_type == leaderboard_type,
            )
        )

        # Fetch all active portfolios
        portfolios_result = await self.session.execute(
            select(PaperPortfolio).where(
                PaperPortfolio.status == PortfolioStatus.ACTIVE
            )
        )
        portfolios = portfolios_result.scalars().all()

        # Calculate performance for each portfolio
        performances: list[PortfolioPerformance] = []
        for portfolio in portfolios:
            perf = await self._calculate_portfolio_performance(
                portfolio, period_start, period_end
            )
            if perf is not None:
                performances.append(perf)

        # Rank portfolios based on leaderboard type
        ranked_performances = self._rank_performances(performances, leaderboard_type)

        # Create leaderboard entries
        entries: list[LeaderboardEntry] = []
        for rank, perf in enumerate(ranked_performances, start=1):
            score = self._get_score_for_type(perf, leaderboard_type)

            entry = LeaderboardEntry(
                user_id=perf.user_id,
                portfolio_id=perf.portfolio_id,
                period=period,
                period_start=period_start,
                period_end=period_end,
                leaderboard_type=leaderboard_type,
                rank=rank,
                score=score,
                total_return_pct=perf.total_return_pct,
                sharpe_ratio=perf.sharpe_ratio,
                max_drawdown_pct=perf.max_drawdown_pct,
                total_trades=perf.total_trades,
            )
            entries.append(entry)
            self.session.add(entry)

        await self.session.flush()

        # Invalidate cache
        await self._invalidate_cache(period, leaderboard_type)

        logger.info(
            "leaderboard_calculated",
            period=period.value,
            leaderboard_type=leaderboard_type.value,
            entries_count=len(entries),
        )

        return entries

    async def get_leaderboard(
        self,
        period: LeaderboardPeriod,
        leaderboard_type: LeaderboardType = LeaderboardType.TOTAL_RETURN,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LeaderboardRanking]:
        """Get cached leaderboard with user details.

        Args:
            period: Time period for leaderboard
            leaderboard_type: Ranking criteria
            limit: Maximum number of entries to return
            offset: Number of entries to skip

        Returns:
            List of leaderboard rankings
        """
        # Check Redis cache first
        cache_key = f"leaderboard:{period.value}:{leaderboard_type.value}:{limit}:{offset}"
        redis = await get_redis()

        cached_data = await redis.get(cache_key)
        if cached_data:
            logger.debug("leaderboard_cache_hit", cache_key=cache_key)
            data = json.loads(cached_data)
            return [
                LeaderboardRanking(
                    rank=item["rank"],
                    user_id=UUID(item["user_id"]),
                    username=item["username"],
                    portfolio_id=UUID(item["portfolio_id"]),
                    portfolio_name=item["portfolio_name"],
                    score=Decimal(item["score"]),
                    total_return_pct=Decimal(item["total_return_pct"]),
                    sharpe_ratio=Decimal(item["sharpe_ratio"]) if item["sharpe_ratio"] else None,
                    max_drawdown_pct=Decimal(item["max_drawdown_pct"]) if item["max_drawdown_pct"] else None,
                    total_trades=item["total_trades"],
                )
                for item in data
            ]

        # Fall back to database
        logger.debug("leaderboard_cache_miss", cache_key=cache_key)

        query = (
            select(LeaderboardEntry, User, PaperPortfolio)
            .join(User, LeaderboardEntry.user_id == User.id)
            .join(PaperPortfolio, LeaderboardEntry.portfolio_id == PaperPortfolio.id)
            .where(
                LeaderboardEntry.period == period,
                LeaderboardEntry.leaderboard_type == leaderboard_type,
            )
            .order_by(LeaderboardEntry.rank)
            .limit(limit)
            .offset(offset)
        )

        result = await self.session.execute(query)
        rows = result.all()

        rankings = [
            LeaderboardRanking(
                rank=entry.rank,
                user_id=entry.user_id,
                username=user.username,
                portfolio_id=entry.portfolio_id,
                portfolio_name=portfolio.name,
                score=entry.score,
                total_return_pct=entry.total_return_pct,
                sharpe_ratio=entry.sharpe_ratio,
                max_drawdown_pct=entry.max_drawdown_pct,
                total_trades=entry.total_trades,
            )
            for entry, user, portfolio in rows
        ]

        # Cache the results
        await self._cache_leaderboard(cache_key, rankings)

        return rankings

    async def get_user_ranking(
        self,
        user_id: UUID,
        period: LeaderboardPeriod,
        leaderboard_type: LeaderboardType = LeaderboardType.TOTAL_RETURN,
    ) -> LeaderboardRanking | None:
        """Get specific user's ranking.

        Args:
            user_id: User ID
            period: Time period for leaderboard
            leaderboard_type: Ranking criteria

        Returns:
            User's ranking or None if not found
        """
        query = (
            select(LeaderboardEntry, User, PaperPortfolio)
            .join(User, LeaderboardEntry.user_id == User.id)
            .join(PaperPortfolio, LeaderboardEntry.portfolio_id == PaperPortfolio.id)
            .where(
                LeaderboardEntry.user_id == user_id,
                LeaderboardEntry.period == period,
                LeaderboardEntry.leaderboard_type == leaderboard_type,
            )
        )

        result = await self.session.execute(query)
        row = result.first()

        if not row:
            return None

        entry, user, portfolio = row

        return LeaderboardRanking(
            rank=entry.rank,
            user_id=entry.user_id,
            username=user.username,
            portfolio_id=entry.portfolio_id,
            portfolio_name=portfolio.name,
            score=entry.score,
            total_return_pct=entry.total_return_pct,
            sharpe_ratio=entry.sharpe_ratio,
            max_drawdown_pct=entry.max_drawdown_pct,
            total_trades=entry.total_trades,
        )

    async def get_user_rankings_history(
        self,
        user_id: UUID,
        days: int = 30,
    ) -> list[dict]:
        """Get user's ranking history over time.

        Args:
            user_id: User ID
            days: Number of days of history to fetch

        Returns:
            List of historical rankings
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=days)

        query = (
            select(LeaderboardEntry)
            .where(
                LeaderboardEntry.user_id == user_id,
                LeaderboardEntry.created_at >= cutoff_date,
            )
            .order_by(LeaderboardEntry.created_at.desc())
        )

        result = await self.session.execute(query)
        entries = result.scalars().all()

        return [
            {
                "date": entry.created_at,
                "period": entry.period.value,
                "leaderboard_type": entry.leaderboard_type.value,
                "rank": entry.rank,
                "score": float(entry.score),
                "total_return_pct": float(entry.total_return_pct),
            }
            for entry in entries
        ]

    async def get_total_participants(
        self,
        period: LeaderboardPeriod,
        leaderboard_type: LeaderboardType,
    ) -> int:
        """Get total number of participants in leaderboard.

        Args:
            period: Time period
            leaderboard_type: Ranking criteria

        Returns:
            Total number of participants
        """
        query = select(func.count(LeaderboardEntry.id)).where(
            LeaderboardEntry.period == period,
            LeaderboardEntry.leaderboard_type == leaderboard_type,
        )

        result = await self.session.execute(query)
        count = result.scalar_one()
        return count or 0

    def _get_period_dates(
        self, period: LeaderboardPeriod
    ) -> tuple[datetime, datetime]:
        """Get start and end dates for period.

        Args:
            period: Time period

        Returns:
            Tuple of (start_date, end_date)
        """
        now = datetime.now(UTC)

        if period == LeaderboardPeriod.DAILY:
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == LeaderboardPeriod.WEEKLY:
            start = now - timedelta(days=now.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=7)
        elif period == LeaderboardPeriod.MONTHLY:
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end = now.replace(year=now.year + 1, month=1, day=1)
            else:
                end = now.replace(month=now.month + 1, day=1)
        else:  # ALL_TIME
            start = datetime(2020, 1, 1, tzinfo=UTC)
            end = now + timedelta(days=1)

        return start, end

    async def _calculate_portfolio_performance(
        self,
        portfolio: PaperPortfolio,
        period_start: datetime,
        period_end: datetime,
    ) -> PortfolioPerformance | None:
        """Calculate performance metrics for a portfolio in a given period.

        Args:
            portfolio: Portfolio to analyze
            period_start: Start of period
            period_end: End of period

        Returns:
            Performance metrics or None if insufficient data
        """
        # Fetch snapshots in the period
        snapshots_result = await self.session.execute(
            select(PaperPortfolioSnapshot)
            .where(
                PaperPortfolioSnapshot.portfolio_id == portfolio.id,
                PaperPortfolioSnapshot.snapshot_date >= period_start,
                PaperPortfolioSnapshot.snapshot_date < period_end,
            )
            .order_by(PaperPortfolioSnapshot.snapshot_date)
        )
        snapshots = snapshots_result.scalars().all()

        if not snapshots:
            return None

        # Calculate total return
        initial_value = snapshots[0].equity_value
        final_value = snapshots[-1].equity_value

        if initial_value == 0:
            return None

        total_return_pct = ((final_value - initial_value) / initial_value) * 100

        # Calculate Sharpe ratio (simplified version)
        sharpe_ratio = self._calculate_sharpe_ratio(list(snapshots))

        # Calculate max drawdown
        max_drawdown_pct = self._calculate_max_drawdown(list(snapshots))

        # Count trades (simplified - would need to query trades table)
        total_trades = 0

        return PortfolioPerformance(
            portfolio_id=portfolio.id,
            user_id=portfolio.user_id,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            total_trades=total_trades,
        )

    def _calculate_sharpe_ratio(
        self, snapshots: list[PaperPortfolioSnapshot]
    ) -> Decimal | None:
        """Calculate Sharpe ratio from snapshots.

        Args:
            snapshots: List of portfolio snapshots

        Returns:
            Sharpe ratio or None if insufficient data
        """
        if len(snapshots) < 2:
            return None

        returns = [
            float(snapshot.daily_return_pct or 0) / 100
            for snapshot in snapshots
            if snapshot.daily_return_pct is not None
        ]

        if not returns:
            return None

        # Calculate mean and standard deviation
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        std_dev = variance**0.5

        if std_dev == 0:
            return None

        # Annualized Sharpe ratio (assuming daily returns)
        # Risk-free rate assumed to be 0 for simplicity
        sharpe = (mean_return / std_dev) * (252**0.5)

        return Decimal(str(round(sharpe, 4)))

    def _calculate_max_drawdown(
        self, snapshots: list[PaperPortfolioSnapshot]
    ) -> Decimal | None:
        """Calculate maximum drawdown from snapshots.

        Args:
            snapshots: List of portfolio snapshots

        Returns:
            Maximum drawdown percentage or None if insufficient data
        """
        if len(snapshots) < 2:
            return None

        equity_values = [float(snapshot.equity_value) for snapshot in snapshots]

        peak = equity_values[0]
        max_dd = 0.0

        for value in equity_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        return Decimal(str(round(max_dd * 100, 4)))

    def _rank_performances(
        self,
        performances: list[PortfolioPerformance],
        leaderboard_type: LeaderboardType,
    ) -> list[PortfolioPerformance]:
        """Rank performances by leaderboard type.

        Args:
            performances: List of portfolio performances
            leaderboard_type: Ranking criteria

        Returns:
            Sorted list of performances
        """
        if leaderboard_type == LeaderboardType.TOTAL_RETURN:
            return sorted(
                performances,
                key=lambda p: (p.total_return_pct, p.portfolio_id),
                reverse=True,
            )
        elif leaderboard_type == LeaderboardType.SHARPE_RATIO:
            # Filter out None sharpe ratios
            valid_performances = [p for p in performances if p.sharpe_ratio is not None]
            return sorted(
                valid_performances,
                key=lambda p: (p.sharpe_ratio or Decimal("0"), p.portfolio_id),
                reverse=True,
            )
        else:  # RISK_ADJUSTED (using Sortino-like approach)
            # Prefer portfolios with better risk-adjusted returns
            # For simplicity, using Sharpe ratio as proxy
            valid_performances = [p for p in performances if p.sharpe_ratio is not None]
            return sorted(
                valid_performances,
                key=lambda p: (p.sharpe_ratio or Decimal("0"), p.portfolio_id),
                reverse=True,
            )

    def _get_score_for_type(
        self,
        performance: PortfolioPerformance,
        leaderboard_type: LeaderboardType,
    ) -> Decimal:
        """Get the score value for a performance based on leaderboard type.

        Args:
            performance: Portfolio performance
            leaderboard_type: Ranking criteria

        Returns:
            Score value
        """
        if leaderboard_type == LeaderboardType.TOTAL_RETURN:
            return performance.total_return_pct
        elif leaderboard_type == LeaderboardType.SHARPE_RATIO:
            return performance.sharpe_ratio or Decimal("0")
        else:  # RISK_ADJUSTED
            return performance.sharpe_ratio or Decimal("0")

    async def _cache_leaderboard(
        self,
        cache_key: str,
        rankings: list[LeaderboardRanking],
    ) -> None:
        """Cache leaderboard in Redis.

        Args:
            cache_key: Redis cache key
            rankings: List of rankings to cache
        """
        redis = await get_redis()

        data = [
            {
                "rank": r.rank,
                "user_id": str(r.user_id),
                "username": r.username,
                "portfolio_id": str(r.portfolio_id),
                "portfolio_name": r.portfolio_name,
                "score": str(r.score),
                "total_return_pct": str(r.total_return_pct),
                "sharpe_ratio": str(r.sharpe_ratio) if r.sharpe_ratio else None,
                "max_drawdown_pct": str(r.max_drawdown_pct) if r.max_drawdown_pct else None,
                "total_trades": r.total_trades,
            }
            for r in rankings
        ]

        await redis.setex(cache_key, LEADERBOARD_CACHE_TTL, json.dumps(data))

    async def _invalidate_cache(
        self,
        period: LeaderboardPeriod,
        leaderboard_type: LeaderboardType,
    ) -> None:
        """Invalidate all cache entries for a period and type.

        Args:
            period: Time period
            leaderboard_type: Ranking criteria
        """
        redis = await get_redis()
        pattern = f"leaderboard:{period.value}:{leaderboard_type.value}:*"

        cursor = 0
        while True:
            cursor, keys = await redis.scan(cursor, match=pattern, count=100)
            if keys:
                await redis.delete(*keys)
            if cursor == 0:
                break

        logger.debug(
            "leaderboard_cache_invalidated",
            period=period.value,
            leaderboard_type=leaderboard_type.value,
        )
