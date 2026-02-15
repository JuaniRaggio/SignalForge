"""Tests for paper trading performance metrics and snapshots."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from uuid import uuid4

import polars as pl
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.paper_trading import (
    OrderSide,
    OrderStatus,
    OrderType,
    PaperOrder,
    PaperPortfolio,
    PaperPortfolioSnapshot,
    PaperPosition,
    PaperTrade,
    PortfolioStatus,
)
from signalforge.models.user import User
from signalforge.paper_trading.performance_service import PerformanceService
from signalforge.paper_trading.snapshot_service import SnapshotService


@pytest.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user."""
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password="hashed",
        is_active=True,
        is_verified=True,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def test_portfolio(db_session: AsyncSession, test_user: User) -> PaperPortfolio:
    """Create a test portfolio."""
    portfolio = PaperPortfolio(
        user_id=test_user.id,
        name="Test Portfolio",
        initial_capital=Decimal("100000.00"),
        current_cash=Decimal("100000.00"),
        status=PortfolioStatus.ACTIVE,
    )
    db_session.add(portfolio)
    await db_session.commit()
    await db_session.refresh(portfolio)
    return portfolio


@pytest.fixture
async def portfolio_with_snapshots(
    db_session: AsyncSession, test_portfolio: PaperPortfolio
) -> PaperPortfolio:
    """Create a portfolio with historical snapshots."""
    base_date = datetime.now(UTC) - timedelta(days=30)

    # Create 30 days of snapshots with varying returns
    returns = [
        0.5, -0.2, 0.3, 1.2, -0.5, 0.8, 0.1, -0.3, 0.6, 0.4,
        -0.1, 0.7, -0.4, 0.9, 0.2, -0.6, 0.5, 0.3, -0.2, 0.8,
        0.1, -0.3, 0.4, 0.6, -0.5, 0.7, 0.2, -0.1, 0.5, 0.3,
    ]

    equity = 100000.0
    cumulative_return = 0.0

    for i, daily_return in enumerate(returns):
        equity = equity * (1 + daily_return / 100)
        cumulative_return = ((equity - 100000.0) / 100000.0) * 100

        snapshot = PaperPortfolioSnapshot(
            portfolio_id=test_portfolio.id,
            snapshot_date=base_date + timedelta(days=i),
            equity_value=Decimal(str(round(equity, 2))),
            cash_balance=Decimal(str(round(equity * 0.3, 2))),
            positions_value=Decimal(str(round(equity * 0.7, 2))),
            positions_count=5,
            daily_return_pct=Decimal(str(daily_return)),
            cumulative_return_pct=Decimal(str(round(cumulative_return, 6))),
        )
        db_session.add(snapshot)

    await db_session.commit()
    return test_portfolio


@pytest.fixture
async def portfolio_with_trades(
    db_session: AsyncSession, test_portfolio: PaperPortfolio
) -> PaperPortfolio:
    """Create a portfolio with historical trades."""
    base_time = datetime.now(UTC) - timedelta(days=30)

    # Create sample trades with wins and losses
    trades_data = [
        ("AAPL", OrderSide.BUY, 100, Decimal("150.00"), 0),
        ("AAPL", OrderSide.SELL, 100, Decimal("155.00"), 2),  # 3.33% win
        ("GOOGL", OrderSide.BUY, 50, Decimal("120.00"), 3),
        ("GOOGL", OrderSide.SELL, 50, Decimal("115.00"), 8),  # 4.17% loss
        ("MSFT", OrderSide.BUY, 75, Decimal("300.00"), 9),
        ("MSFT", OrderSide.SELL, 75, Decimal("315.00"), 14),  # 5% win
        ("TSLA", OrderSide.BUY, 30, Decimal("200.00"), 15),
        ("TSLA", OrderSide.SELL, 30, Decimal("190.00"), 20),  # 5% loss
        ("NVDA", OrderSide.BUY, 40, Decimal("400.00"), 21),
        ("NVDA", OrderSide.SELL, 40, Decimal("420.00"), 25),  # 5% win
    ]

    for symbol, side, quantity, price, day_offset in trades_data:
        # Create order first (FK constraint)
        order = PaperOrder(
            portfolio_id=test_portfolio.id,
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=side,
            quantity=quantity,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_price=price,
            filled_at=base_time + timedelta(days=day_offset),
        )
        db_session.add(order)
        await db_session.flush()

        # Create trade referencing the order
        trade = PaperTrade(
            portfolio_id=test_portfolio.id,
            order_id=order.id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            executed_at=base_time + timedelta(days=day_offset),
        )
        db_session.add(trade)

    await db_session.commit()
    return test_portfolio


class TestPerformanceService:
    """Test suite for PerformanceService."""

    async def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        service = PerformanceService(risk_free_rate=0.05)

        # Create sample returns (daily percentages)
        returns = pl.Series("returns", [0.5, -0.2, 0.3, 0.8, -0.1, 0.6, 0.2, -0.3, 0.4, 0.5])

        sharpe = service.calculate_sharpe_ratio(returns, 0.05)

        # Sharpe should be a reasonable value
        assert isinstance(sharpe, Decimal)
        assert sharpe > 0  # Positive returns should give positive Sharpe

    async def test_calculate_sortino_ratio(self):
        """Test Sortino ratio calculation."""
        service = PerformanceService(risk_free_rate=0.05)

        # Create sample returns with some negative values
        returns = pl.Series("returns", [0.5, -0.3, 0.4, 0.6, -0.2, 0.7, 0.3, -0.1, 0.5, 0.4])

        sortino = service.calculate_sortino_ratio(returns, 0.05)

        # Sortino should be higher than Sharpe for same data (only downside risk)
        assert isinstance(sortino, Decimal)
        assert sortino > 0

    async def test_calculate_max_drawdown(self):
        """Test maximum drawdown calculation."""
        service = PerformanceService()

        # Create equity curve with a known drawdown
        equity_curve = pl.Series(
            "equity",
            [100000, 102000, 101000, 99000, 98000, 100000, 103000, 102000],
        )

        max_dd, duration = service.calculate_max_drawdown(equity_curve)

        # Maximum drawdown should be from peak (102000) to trough (98000) = 3.92%
        assert isinstance(max_dd, Decimal)
        assert max_dd > 0
        assert max_dd < Decimal("5")  # Should be around 3.92%

        # Duration should be greater than 0
        assert duration > 0

    async def test_calculate_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        service = PerformanceService()

        annualized_return = Decimal("15.0")
        max_drawdown = Decimal("5.0")

        calmar = service.calculate_calmar_ratio(annualized_return, max_drawdown)

        # Calmar = 15 / 5 = 3.0
        assert isinstance(calmar, Decimal)
        assert calmar == Decimal("3.0")

    async def test_calculate_calmar_ratio_zero_drawdown(self):
        """Test Calmar ratio with zero drawdown."""
        service = PerformanceService()

        annualized_return = Decimal("15.0")
        max_drawdown = Decimal("0")

        calmar = service.calculate_calmar_ratio(annualized_return, max_drawdown)

        # Should return 0 when drawdown is 0 to avoid division by zero
        assert calmar == Decimal("0")

    async def test_get_trade_statistics(
        self, db_session: AsyncSession, portfolio_with_trades: PaperPortfolio
    ):
        """Test trade statistics calculation."""
        service = PerformanceService()

        stats = await service.get_trade_statistics(portfolio_with_trades.id, db_session)

        # We have 5 round-trip trades: 3 wins, 2 losses
        assert stats["total_trades"] == 5
        assert stats["winning_trades"] == 3
        assert stats["losing_trades"] == 2
        assert stats["win_rate"] > Decimal("0")  # Should be 60%
        assert stats["profit_factor"] > Decimal("0")

    async def test_calculate_metrics(
        self, db_session: AsyncSession, portfolio_with_snapshots: PaperPortfolio
    ):
        """Test comprehensive metrics calculation."""
        service = PerformanceService()

        metrics = await service.calculate_metrics(portfolio_with_snapshots.id, db_session)

        # Verify all metrics are calculated
        assert metrics.portfolio_id == portfolio_with_snapshots.id
        assert isinstance(metrics.total_return_pct, Decimal)
        assert isinstance(metrics.annualized_return_pct, Decimal)
        assert isinstance(metrics.sharpe_ratio, Decimal)
        assert isinstance(metrics.sortino_ratio, Decimal)
        assert isinstance(metrics.max_drawdown_pct, Decimal)
        assert metrics.max_drawdown_duration_days >= 0
        assert isinstance(metrics.volatility_annualized, Decimal)
        assert isinstance(metrics.calmar_ratio, Decimal)

    async def test_calculate_metrics_no_data(
        self, db_session: AsyncSession, test_portfolio: PaperPortfolio
    ):
        """Test metrics calculation with no snapshots."""
        service = PerformanceService()

        metrics = await service.calculate_metrics(test_portfolio.id, db_session)

        # Should return zero metrics
        assert metrics.portfolio_id == test_portfolio.id
        assert metrics.total_return_pct == Decimal("0")
        assert metrics.total_trades == 0


class TestSnapshotService:
    """Test suite for SnapshotService."""

    async def test_create_daily_snapshot(
        self, db_session: AsyncSession, test_portfolio: PaperPortfolio
    ):
        """Test creating a daily snapshot."""
        service = SnapshotService()

        # Add a position to the portfolio
        position = PaperPosition(
            portfolio_id=test_portfolio.id,
            symbol="AAPL",
            quantity=100,
            avg_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
            unrealized_pnl=Decimal("500.00"),
            unrealized_pnl_pct=Decimal("3.33"),
        )
        db_session.add(position)
        await db_session.commit()

        # Create snapshot
        snapshot = await service.create_daily_snapshot(test_portfolio.id, db_session)

        # Verify snapshot was created correctly
        assert snapshot.portfolio_id == test_portfolio.id
        assert snapshot.positions_count == 1
        assert snapshot.equity_value > test_portfolio.current_cash  # Has positions
        assert snapshot.cumulative_return_pct is not None

    async def test_create_daily_snapshot_with_previous(
        self, db_session: AsyncSession, portfolio_with_snapshots: PaperPortfolio
    ):
        """Test creating a snapshot when previous snapshots exist."""
        service = SnapshotService()

        # Create new snapshot
        snapshot = await service.create_daily_snapshot(
            portfolio_with_snapshots.id, db_session
        )

        # Should have calculated daily return from previous snapshot
        assert snapshot.daily_return_pct is not None

    async def test_create_snapshots_for_all_portfolios(
        self, db_session: AsyncSession, test_portfolio: PaperPortfolio
    ):
        """Test creating snapshots for all active portfolios."""
        service = SnapshotService()

        snapshots = await service.create_snapshots_for_all_portfolios(db_session)

        # Should create at least one snapshot
        assert len(snapshots) >= 1
        assert any(s.portfolio_id == test_portfolio.id for s in snapshots)

    async def test_get_equity_curve(
        self, db_session: AsyncSession, portfolio_with_snapshots: PaperPortfolio
    ):
        """Test retrieving equity curve."""
        service = SnapshotService()

        equity_df = await service.get_equity_curve(portfolio_with_snapshots.id, db_session, days=30)

        # Should return a DataFrame with data
        assert not equity_df.is_empty()
        assert len(equity_df) > 0
        assert "equity" in equity_df.columns
        assert "daily_return_pct" in equity_df.columns

    async def test_get_equity_curve_empty(
        self, db_session: AsyncSession, test_portfolio: PaperPortfolio
    ):
        """Test retrieving equity curve with no snapshots."""
        service = SnapshotService()

        equity_df = await service.get_equity_curve(test_portfolio.id, db_session, days=30)

        # Should return empty DataFrame with correct schema
        assert equity_df.is_empty()
        assert "equity" in equity_df.columns

    async def test_get_returns_series(
        self, db_session: AsyncSession, portfolio_with_snapshots: PaperPortfolio
    ):
        """Test retrieving returns series."""
        service = SnapshotService()

        returns = await service.get_returns_series(portfolio_with_snapshots.id, db_session, days=30)

        # Should return series with returns data
        assert len(returns) > 0
        assert all(isinstance(r, float) for r in returns.to_list())

    async def test_get_latest_snapshot(
        self, db_session: AsyncSession, portfolio_with_snapshots: PaperPortfolio
    ):
        """Test retrieving latest snapshot."""
        service = SnapshotService()

        snapshot = await service.get_latest_snapshot(portfolio_with_snapshots.id, db_session)

        assert snapshot is not None
        assert snapshot.portfolio_id == portfolio_with_snapshots.id

    async def test_get_latest_snapshot_none(
        self, db_session: AsyncSession, test_portfolio: PaperPortfolio
    ):
        """Test retrieving latest snapshot when none exist."""
        service = SnapshotService()

        snapshot = await service.get_latest_snapshot(test_portfolio.id, db_session)

        assert snapshot is None


class TestPerformanceMetricsIntegration:
    """Integration tests for performance metrics."""

    async def test_full_performance_workflow(
        self, db_session: AsyncSession, test_portfolio: PaperPortfolio
    ):
        """Test complete workflow: trades -> snapshots -> metrics."""
        snapshot_service = SnapshotService()
        perf_service = PerformanceService()

        # Add some positions
        position = PaperPosition(
            portfolio_id=test_portfolio.id,
            symbol="AAPL",
            quantity=100,
            avg_entry_price=Decimal("150.00"),
            current_price=Decimal("155.00"),
            unrealized_pnl=Decimal("500.00"),
            unrealized_pnl_pct=Decimal("3.33"),
        )
        db_session.add(position)

        # Create order first (FK constraint)
        order1 = PaperOrder(
            portfolio_id=test_portfolio.id,
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,
            status=OrderStatus.FILLED,
            filled_quantity=100,
            filled_price=Decimal("150.00"),
            filled_at=datetime.now(UTC) - timedelta(days=5),
        )
        db_session.add(order1)
        await db_session.flush()

        # Add trade referencing the order
        trade1 = PaperTrade(
            portfolio_id=test_portfolio.id,
            order_id=order1.id,
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            price=Decimal("150.00"),
            executed_at=datetime.now(UTC) - timedelta(days=5),
        )
        db_session.add(trade1)
        await db_session.commit()

        # Create snapshot
        snapshot = await snapshot_service.create_daily_snapshot(
            test_portfolio.id, db_session
        )
        assert snapshot is not None

        # Calculate metrics
        metrics = await perf_service.calculate_metrics(test_portfolio.id, db_session)

        # Verify metrics
        assert metrics.portfolio_id == test_portfolio.id
        assert metrics.total_trades >= 0

    async def test_sharpe_ratio_consistency(self):
        """Test that Sharpe ratio is consistent for known data."""
        service = PerformanceService(risk_free_rate=0.05)

        # Create returns with known mean and std
        # Mean = 0.5%, Std = 0.3%
        returns = pl.Series("returns", [0.5, 0.8, 0.3, 0.6, 0.4, 0.5, 0.7, 0.2, 0.6, 0.5])

        sharpe = service.calculate_sharpe_ratio(returns, 0.05)

        # Sharpe should be positive
        # Note: With the test data (mean ~0.5%, std ~0.16%), the annualized Sharpe can be high
        assert sharpe > 0
        assert sharpe < Decimal("100")  # Sanity check
