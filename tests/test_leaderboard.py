"""Comprehensive tests for leaderboard system."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

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
from signalforge.models.user import User, UserType
from signalforge.paper_trading.leaderboard_service import (
    LeaderboardService,
    PortfolioPerformance,
)


@pytest.fixture
def mock_redis():
    """Create a mock Redis client for tests that need caching."""
    mock = MagicMock()
    mock.get = AsyncMock(return_value=None)  # Cache miss by default
    mock.setex = AsyncMock(return_value=True)
    mock.scan = AsyncMock(return_value=(0, []))  # No keys to invalidate
    mock.delete = AsyncMock(return_value=1)
    return mock


@pytest.fixture
async def test_users(db_session: AsyncSession) -> list[User]:
    """Create test users."""
    users = []
    for i in range(5):
        user = User(
            email=f"user{i}@test.com",
            username=f"testuser{i}",
            hashed_password="hashed_password",
            user_type=UserType.ACTIVE_TRADER,
        )
        db_session.add(user)
        users.append(user)

    await db_session.flush()
    return users


@pytest.fixture
async def test_portfolios(
    db_session: AsyncSession, test_users: list[User]
) -> list[PaperPortfolio]:
    """Create test portfolios."""
    portfolios = []
    for user in test_users:
        portfolio = PaperPortfolio(
            user_id=user.id,
            name=f"{user.username}'s Portfolio",
            initial_capital=Decimal("100000"),
            current_cash=Decimal("80000"),
            status=PortfolioStatus.ACTIVE,
        )
        db_session.add(portfolio)
        portfolios.append(portfolio)

    await db_session.flush()
    return portfolios


@pytest.fixture
async def test_snapshots(
    db_session: AsyncSession, test_portfolios: list[PaperPortfolio]
) -> list[PaperPortfolioSnapshot]:
    """Create test snapshots with varying performance."""
    snapshots = []
    base_date = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    performance_data = [
        {"initial": 100000, "final": 115000, "returns": [0.5, 1.0, 0.3, 0.8, 0.5]},
        {"initial": 100000, "final": 108000, "returns": [0.3, 0.5, 0.2, 0.4, 0.3]},
        {"initial": 100000, "final": 105000, "returns": [0.2, 0.3, 0.1, 0.2, 0.2]},
        {"initial": 100000, "final": 102000, "returns": [0.1, 0.2, 0.0, 0.1, 0.1]},
        {"initial": 100000, "final": 98000, "returns": [-0.1, 0.0, -0.2, 0.1, -0.1]},
    ]

    for idx, portfolio in enumerate(test_portfolios):
        perf = performance_data[idx]
        initial_value = Decimal(str(perf["initial"]))
        final_value = Decimal(str(perf["final"]))
        daily_returns = perf["returns"]

        for day in range(5):
            snapshot_date = base_date - timedelta(days=4 - day)
            progress = (day + 1) / 5
            current_value = initial_value + (final_value - initial_value) * Decimal(
                str(progress)
            )

            daily_return_pct = (
                Decimal(str(daily_returns[day])) if day < len(daily_returns) else None
            )

            cumulative_return = (
                (current_value - initial_value) / initial_value
            ) * Decimal("100")

            snapshot = PaperPortfolioSnapshot(
                portfolio_id=portfolio.id,
                snapshot_date=snapshot_date,
                equity_value=current_value,
                cash_balance=Decimal("80000"),
                positions_value=current_value - Decimal("80000"),
                positions_count=3,
                daily_return_pct=daily_return_pct,
                cumulative_return_pct=cumulative_return,
            )
            db_session.add(snapshot)
            snapshots.append(snapshot)

    await db_session.flush()
    return snapshots


class TestLeaderboardService:
    """Tests for LeaderboardService."""

    @pytest.mark.asyncio
    async def test_get_period_dates_daily(self, db_session: AsyncSession) -> None:
        """Test period date calculation for daily leaderboard."""
        service = LeaderboardService(db_session)

        start, end = service._get_period_dates(LeaderboardPeriod.DAILY)

        assert start.hour == 0
        assert start.minute == 0
        assert start.second == 0
        assert end == start + timedelta(days=1)

    @pytest.mark.asyncio
    async def test_get_period_dates_weekly(self, db_session: AsyncSession) -> None:
        """Test period date calculation for weekly leaderboard."""
        service = LeaderboardService(db_session)

        start, end = service._get_period_dates(LeaderboardPeriod.WEEKLY)

        assert start.weekday() == 0  # Monday
        assert end == start + timedelta(days=7)

    @pytest.mark.asyncio
    async def test_get_period_dates_monthly(self, db_session: AsyncSession) -> None:
        """Test period date calculation for monthly leaderboard."""
        service = LeaderboardService(db_session)

        start, end = service._get_period_dates(LeaderboardPeriod.MONTHLY)

        assert start.day == 1
        assert end.day == 1
        assert (end.month == start.month + 1) or (
            end.month == 1 and start.month == 12
        )

    @pytest.mark.asyncio
    async def test_calculate_sharpe_ratio(self, db_session: AsyncSession) -> None:
        """Test Sharpe ratio calculation."""
        service = LeaderboardService(db_session)

        # Use varying returns to ensure non-zero standard deviation
        return_values = [
            Decimal("0.5"),
            Decimal("1.0"),
            Decimal("-0.2"),
            Decimal("0.8"),
            Decimal("0.3"),
            Decimal("1.2"),
            Decimal("-0.1"),
            Decimal("0.6"),
            Decimal("0.4"),
            Decimal("0.9"),
        ]
        snapshots = []
        for i in range(10):
            snapshot = PaperPortfolioSnapshot(
                portfolio_id=uuid4(),
                snapshot_date=datetime.now(UTC) - timedelta(days=9 - i),
                equity_value=Decimal("100000"),
                cash_balance=Decimal("50000"),
                positions_value=Decimal("50000"),
                positions_count=2,
                daily_return_pct=return_values[i],
            )
            snapshots.append(snapshot)

        sharpe = service._calculate_sharpe_ratio(snapshots)

        assert sharpe is not None
        assert isinstance(sharpe, Decimal)
        assert sharpe > 0

    @pytest.mark.asyncio
    async def test_calculate_sharpe_ratio_insufficient_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test Sharpe ratio with insufficient data."""
        service = LeaderboardService(db_session)

        snapshot = PaperPortfolioSnapshot(
            portfolio_id=uuid4(),
            snapshot_date=datetime.now(UTC),
            equity_value=Decimal("100000"),
            cash_balance=Decimal("50000"),
            positions_value=Decimal("50000"),
            positions_count=2,
        )

        sharpe = service._calculate_sharpe_ratio([snapshot])

        assert sharpe is None

    @pytest.mark.asyncio
    async def test_calculate_max_drawdown(self, db_session: AsyncSession) -> None:
        """Test maximum drawdown calculation."""
        service = LeaderboardService(db_session)

        snapshots = []
        equity_values = [100000, 105000, 103000, 108000, 102000, 107000]

        for i, equity in enumerate(equity_values):
            snapshot = PaperPortfolioSnapshot(
                portfolio_id=uuid4(),
                snapshot_date=datetime.now(UTC) - timedelta(days=len(equity_values) - i - 1),
                equity_value=Decimal(str(equity)),
                cash_balance=Decimal("50000"),
                positions_value=Decimal(str(equity - 50000)),
                positions_count=2,
            )
            snapshots.append(snapshot)

        max_dd = service._calculate_max_drawdown(snapshots)

        assert max_dd is not None
        assert isinstance(max_dd, Decimal)
        assert max_dd > 0
        # Max drawdown from 108000 to 102000 = 5.56%
        assert abs(max_dd - Decimal("5.56")) < Decimal("0.1")

    @pytest.mark.asyncio
    async def test_calculate_max_drawdown_no_drawdown(
        self, db_session: AsyncSession
    ) -> None:
        """Test max drawdown when equity only increases."""
        service = LeaderboardService(db_session)

        snapshots = []
        for i in range(5):
            snapshot = PaperPortfolioSnapshot(
                portfolio_id=uuid4(),
                snapshot_date=datetime.now(UTC) - timedelta(days=4 - i),
                equity_value=Decimal(str(100000 + i * 1000)),
                cash_balance=Decimal("50000"),
                positions_value=Decimal(str(50000 + i * 1000)),
                positions_count=2,
            )
            snapshots.append(snapshot)

        max_dd = service._calculate_max_drawdown(snapshots)

        assert max_dd is not None
        assert max_dd == Decimal("0")

    @pytest.mark.asyncio
    async def test_rank_performances_total_return(
        self, db_session: AsyncSession
    ) -> None:
        """Test ranking by total return."""
        service = LeaderboardService(db_session)

        performances = [
            PortfolioPerformance(
                portfolio_id=uuid4(),
                user_id=uuid4(),
                total_return_pct=Decimal("15"),
                sharpe_ratio=Decimal("1.5"),
                max_drawdown_pct=Decimal("5"),
                total_trades=10,
            ),
            PortfolioPerformance(
                portfolio_id=uuid4(),
                user_id=uuid4(),
                total_return_pct=Decimal("8"),
                sharpe_ratio=Decimal("2.0"),
                max_drawdown_pct=Decimal("3"),
                total_trades=5,
            ),
            PortfolioPerformance(
                portfolio_id=uuid4(),
                user_id=uuid4(),
                total_return_pct=Decimal("20"),
                sharpe_ratio=Decimal("1.2"),
                max_drawdown_pct=Decimal("8"),
                total_trades=15,
            ),
        ]

        ranked = service._rank_performances(
            performances, LeaderboardType.TOTAL_RETURN
        )

        assert len(ranked) == 3
        assert ranked[0].total_return_pct == Decimal("20")
        assert ranked[1].total_return_pct == Decimal("15")
        assert ranked[2].total_return_pct == Decimal("8")

    @pytest.mark.asyncio
    async def test_rank_performances_sharpe_ratio(
        self, db_session: AsyncSession
    ) -> None:
        """Test ranking by Sharpe ratio."""
        service = LeaderboardService(db_session)

        performances = [
            PortfolioPerformance(
                portfolio_id=uuid4(),
                user_id=uuid4(),
                total_return_pct=Decimal("15"),
                sharpe_ratio=Decimal("1.5"),
                max_drawdown_pct=Decimal("5"),
                total_trades=10,
            ),
            PortfolioPerformance(
                portfolio_id=uuid4(),
                user_id=uuid4(),
                total_return_pct=Decimal("8"),
                sharpe_ratio=Decimal("2.0"),
                max_drawdown_pct=Decimal("3"),
                total_trades=5,
            ),
            PortfolioPerformance(
                portfolio_id=uuid4(),
                user_id=uuid4(),
                total_return_pct=Decimal("20"),
                sharpe_ratio=None,
                max_drawdown_pct=Decimal("8"),
                total_trades=15,
            ),
        ]

        ranked = service._rank_performances(
            performances, LeaderboardType.SHARPE_RATIO
        )

        assert len(ranked) == 2  # Third performance filtered out (None sharpe)
        assert ranked[0].sharpe_ratio == Decimal("2.0")
        assert ranked[1].sharpe_ratio == Decimal("1.5")

    @pytest.mark.asyncio
    async def test_get_score_for_type(self, db_session: AsyncSession) -> None:
        """Test score extraction for different leaderboard types."""
        service = LeaderboardService(db_session)

        performance = PortfolioPerformance(
            portfolio_id=uuid4(),
            user_id=uuid4(),
            total_return_pct=Decimal("15"),
            sharpe_ratio=Decimal("1.5"),
            max_drawdown_pct=Decimal("5"),
            total_trades=10,
        )

        total_return_score = service._get_score_for_type(
            performance, LeaderboardType.TOTAL_RETURN
        )
        assert total_return_score == Decimal("15")

        sharpe_score = service._get_score_for_type(
            performance, LeaderboardType.SHARPE_RATIO
        )
        assert sharpe_score == Decimal("1.5")

        risk_adjusted_score = service._get_score_for_type(
            performance, LeaderboardType.RISK_ADJUSTED
        )
        assert risk_adjusted_score == Decimal("1.5")

    @pytest.mark.asyncio
    async def test_calculate_leaderboard(
        self,
        db_session: AsyncSession,
        test_users: list[User],
        test_portfolios: list[PaperPortfolio],
        test_snapshots: list[PaperPortfolioSnapshot],
    ) -> None:
        """Test full leaderboard calculation."""
        service = LeaderboardService(db_session)

        entries = await service.calculate_leaderboard(
            period=LeaderboardPeriod.WEEKLY,
            leaderboard_type=LeaderboardType.TOTAL_RETURN,
        )

        assert len(entries) > 0
        # Verify rankings are assigned
        ranks = [entry.rank for entry in entries]
        assert ranks == sorted(ranks)
        # Verify first rank has highest total return
        assert entries[0].rank == 1

    @pytest.mark.asyncio
    async def test_get_leaderboard(
        self,
        db_session: AsyncSession,
        test_users: list[User],
        test_portfolios: list[PaperPortfolio],
        test_snapshots: list[PaperPortfolioSnapshot],
        mock_redis,
    ) -> None:
        """Test retrieving leaderboard."""
        with patch(
            "signalforge.paper_trading.leaderboard_service.get_redis",
            return_value=mock_redis,
        ):
            service = LeaderboardService(db_session)

            # First calculate the leaderboard
            await service.calculate_leaderboard(
                period=LeaderboardPeriod.WEEKLY,
                leaderboard_type=LeaderboardType.TOTAL_RETURN,
            )

            # Then retrieve it
            rankings = await service.get_leaderboard(
                period=LeaderboardPeriod.WEEKLY,
                leaderboard_type=LeaderboardType.TOTAL_RETURN,
                limit=10,
                offset=0,
            )

            assert len(rankings) > 0
            assert all(hasattr(r, "username") for r in rankings)
            assert all(hasattr(r, "portfolio_name") for r in rankings)

    @pytest.mark.asyncio
    async def test_get_leaderboard_pagination(
        self,
        db_session: AsyncSession,
        test_users: list[User],
        test_portfolios: list[PaperPortfolio],
        test_snapshots: list[PaperPortfolioSnapshot],
        mock_redis,
    ) -> None:
        """Test leaderboard pagination."""
        with patch(
            "signalforge.paper_trading.leaderboard_service.get_redis",
            return_value=mock_redis,
        ):
            service = LeaderboardService(db_session)

            await service.calculate_leaderboard(
                period=LeaderboardPeriod.WEEKLY,
                leaderboard_type=LeaderboardType.TOTAL_RETURN,
            )

            first_page = await service.get_leaderboard(
                period=LeaderboardPeriod.WEEKLY,
                leaderboard_type=LeaderboardType.TOTAL_RETURN,
                limit=2,
                offset=0,
            )

            second_page = await service.get_leaderboard(
                period=LeaderboardPeriod.WEEKLY,
                leaderboard_type=LeaderboardType.TOTAL_RETURN,
                limit=2,
                offset=2,
            )

            # Ensure pages are different
            if len(first_page) > 0 and len(second_page) > 0:
                assert first_page[0].user_id != second_page[0].user_id

    @pytest.mark.asyncio
    async def test_get_user_ranking(
        self,
        db_session: AsyncSession,
        test_users: list[User],
        test_portfolios: list[PaperPortfolio],
        test_snapshots: list[PaperPortfolioSnapshot],
        mock_redis,
    ) -> None:
        """Test retrieving specific user's ranking."""
        with patch(
            "signalforge.paper_trading.leaderboard_service.get_redis",
            return_value=mock_redis,
        ):
            service = LeaderboardService(db_session)

            await service.calculate_leaderboard(
                period=LeaderboardPeriod.WEEKLY,
                leaderboard_type=LeaderboardType.TOTAL_RETURN,
            )

            user_id = test_users[0].id
            ranking = await service.get_user_ranking(
                user_id=user_id,
                period=LeaderboardPeriod.WEEKLY,
                leaderboard_type=LeaderboardType.TOTAL_RETURN,
            )

            assert ranking is not None
            assert ranking.user_id == user_id
            assert ranking.rank > 0

    @pytest.mark.asyncio
    async def test_get_user_ranking_not_found(
        self, db_session: AsyncSession
    ) -> None:
        """Test retrieving ranking for user not in leaderboard."""
        service = LeaderboardService(db_session)

        ranking = await service.get_user_ranking(
            user_id=uuid4(),
            period=LeaderboardPeriod.WEEKLY,
            leaderboard_type=LeaderboardType.TOTAL_RETURN,
        )

        assert ranking is None

    @pytest.mark.asyncio
    async def test_get_user_rankings_history(
        self,
        db_session: AsyncSession,
        test_users: list[User],
        test_portfolios: list[PaperPortfolio],
        test_snapshots: list[PaperPortfolioSnapshot],
        mock_redis,
    ) -> None:
        """Test retrieving user's ranking history."""
        with patch(
            "signalforge.paper_trading.leaderboard_service.get_redis",
            return_value=mock_redis,
        ):
            service = LeaderboardService(db_session)

            # Calculate leaderboard multiple times to create history
            await service.calculate_leaderboard(
                period=LeaderboardPeriod.DAILY,
                leaderboard_type=LeaderboardType.TOTAL_RETURN,
            )

            user_id = test_users[0].id
            history = await service.get_user_rankings_history(
                user_id=user_id,
                days=30,
            )

            assert isinstance(history, list)
            if len(history) > 0:
                assert "rank" in history[0]
                assert "period" in history[0]
                assert "score" in history[0]

    @pytest.mark.asyncio
    async def test_get_total_participants(
        self,
        db_session: AsyncSession,
        test_users: list[User],
        test_portfolios: list[PaperPortfolio],
        test_snapshots: list[PaperPortfolioSnapshot],
        mock_redis,
    ) -> None:
        """Test getting total number of participants."""
        with patch(
            "signalforge.paper_trading.leaderboard_service.get_redis",
            return_value=mock_redis,
        ):
            service = LeaderboardService(db_session)

            await service.calculate_leaderboard(
                period=LeaderboardPeriod.WEEKLY,
                leaderboard_type=LeaderboardType.TOTAL_RETURN,
            )

            total = await service.get_total_participants(
                period=LeaderboardPeriod.WEEKLY,
                leaderboard_type=LeaderboardType.TOTAL_RETURN,
            )

            assert total > 0
            assert total <= len(test_users)

    @pytest.mark.asyncio
    async def test_leaderboard_entry_deterministic_ranking(
        self,
        db_session: AsyncSession,
        test_users: list[User],
        test_portfolios: list[PaperPortfolio],
        test_snapshots: list[PaperPortfolioSnapshot],
        mock_redis,
    ) -> None:
        """Test that rankings are deterministic with ties."""
        with patch(
            "signalforge.paper_trading.leaderboard_service.get_redis",
            return_value=mock_redis,
        ):
            service = LeaderboardService(db_session)

            # Calculate twice
            entries1 = await service.calculate_leaderboard(
                period=LeaderboardPeriod.WEEKLY,
                leaderboard_type=LeaderboardType.TOTAL_RETURN,
            )

            entries2 = await service.calculate_leaderboard(
                period=LeaderboardPeriod.WEEKLY,
                leaderboard_type=LeaderboardType.TOTAL_RETURN,
            )

            # Rankings should be identical
            for e1, e2 in zip(entries1, entries2, strict=False):
                assert e1.rank == e2.rank
                assert e1.user_id == e2.user_id


class TestLeaderboardModels:
    """Tests for leaderboard models."""

    @pytest.mark.asyncio
    async def test_create_leaderboard_entry(
        self, db_session: AsyncSession, test_users: list[User]
    ) -> None:
        """Test creating a leaderboard entry."""
        user = test_users[0]
        portfolio_id = uuid4()

        entry = LeaderboardEntry(
            user_id=user.id,
            portfolio_id=portfolio_id,
            period=LeaderboardPeriod.WEEKLY,
            period_start=datetime.now(UTC),
            period_end=datetime.now(UTC) + timedelta(days=7),
            leaderboard_type=LeaderboardType.TOTAL_RETURN,
            rank=1,
            score=Decimal("15.5"),
            total_return_pct=Decimal("15.5"),
            sharpe_ratio=Decimal("1.8"),
            max_drawdown_pct=Decimal("5.2"),
            total_trades=25,
        )

        db_session.add(entry)
        await db_session.flush()

        assert entry.id is not None
        assert entry.rank == 1
        assert entry.score == Decimal("15.5")

    @pytest.mark.asyncio
    async def test_leaderboard_period_enum(self) -> None:
        """Test LeaderboardPeriod enum values."""
        assert LeaderboardPeriod.DAILY.value == "daily"
        assert LeaderboardPeriod.WEEKLY.value == "weekly"
        assert LeaderboardPeriod.MONTHLY.value == "monthly"
        assert LeaderboardPeriod.ALL_TIME.value == "all_time"

    @pytest.mark.asyncio
    async def test_leaderboard_type_enum(self) -> None:
        """Test LeaderboardType enum values."""
        assert LeaderboardType.TOTAL_RETURN.value == "total_return"
        assert LeaderboardType.SHARPE_RATIO.value == "sharpe_ratio"
        assert LeaderboardType.RISK_ADJUSTED.value == "risk_adjusted"
