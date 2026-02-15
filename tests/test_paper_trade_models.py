"""Tests for paper trading models."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.paper_trade import LegacyPaperPosition as PaperPosition, PortfolioSnapshot


@pytest.mark.asyncio
class TestPaperPosition:
    """Tests for PaperPosition model."""

    async def test_create_paper_position(self, db_session: AsyncSession) -> None:
        """Test creating a paper position."""
        position = PaperPosition(
            signal_id="test-signal-uuid-123",
            symbol="AAPL",
            direction="long",
            entry_date=datetime.now(UTC),
            entry_price=Decimal("150.50"),
            shares=100,
            stop_loss=Decimal("145.00"),
            take_profit=Decimal("160.00"),
            status="open",
        )

        db_session.add(position)
        await db_session.commit()
        await db_session.refresh(position)

        assert position.id is not None
        assert position.signal_id == "test-signal-uuid-123"
        assert position.symbol == "AAPL"
        assert position.direction == "long"
        assert position.entry_price == Decimal("150.50")
        assert position.shares == 100
        assert position.stop_loss == Decimal("145.00")
        assert position.take_profit == Decimal("160.00")
        assert position.status == "open"
        assert position.exit_date is None
        assert position.exit_price is None
        assert position.exit_reason is None
        assert position.pnl is None
        assert position.pnl_pct is None
        assert position.created_at is not None
        assert position.updated_at is not None

    async def test_paper_position_default_status(self, db_session: AsyncSession) -> None:
        """Test that status defaults to 'open'."""
        position = PaperPosition(
            signal_id="test-signal-uuid-456",
            symbol="TSLA",
            direction="short",
            entry_date=datetime.now(UTC),
            entry_price=Decimal("200.00"),
            shares=50,
            stop_loss=Decimal("210.00"),
            take_profit=Decimal("180.00"),
        )

        db_session.add(position)
        await db_session.commit()
        await db_session.refresh(position)

        assert position.status == "open"

    async def test_close_paper_position(self, db_session: AsyncSession) -> None:
        """Test closing a paper position with P&L."""
        entry_price = Decimal("150.00")
        shares = 100
        exit_price = Decimal("155.00")

        position = PaperPosition(
            signal_id="test-signal-uuid-789",
            symbol="AAPL",
            direction="long",
            entry_date=datetime.now(UTC) - timedelta(days=2),
            entry_price=entry_price,
            shares=shares,
            stop_loss=Decimal("145.00"),
            take_profit=Decimal("160.00"),
            status="open",
        )

        db_session.add(position)
        await db_session.commit()

        # Close the position
        position.status = "closed"
        position.exit_date = datetime.now(UTC)
        position.exit_price = exit_price
        position.exit_reason = "take_profit"
        position.pnl = (exit_price - entry_price) * shares
        position.pnl_pct = ((exit_price - entry_price) / entry_price) * 100

        await db_session.commit()
        await db_session.refresh(position)

        assert position.status == "closed"
        assert position.exit_date is not None
        assert position.exit_price == exit_price
        assert position.exit_reason == "take_profit"
        assert position.pnl == Decimal("500.00")
        assert position.pnl_pct == Decimal("3.3333")

    async def test_query_positions_by_symbol(self, db_session: AsyncSession) -> None:
        """Test querying positions by symbol."""
        positions = [
            PaperPosition(
                signal_id=f"signal-{i}",
                symbol="AAPL" if i % 2 == 0 else "TSLA",
                direction="long",
                entry_date=datetime.now(UTC),
                entry_price=Decimal("150.00"),
                shares=100,
                stop_loss=Decimal("145.00"),
                take_profit=Decimal("160.00"),
            )
            for i in range(4)
        ]

        db_session.add_all(positions)
        await db_session.commit()

        result = await db_session.execute(
            select(PaperPosition).where(PaperPosition.symbol == "AAPL")
        )
        aapl_positions = result.scalars().all()

        assert len(aapl_positions) == 2
        assert all(p.symbol == "AAPL" for p in aapl_positions)

    async def test_query_positions_by_status(self, db_session: AsyncSession) -> None:
        """Test querying positions by status."""
        positions = [
            PaperPosition(
                signal_id=f"signal-{i}",
                symbol="AAPL",
                direction="long",
                entry_date=datetime.now(UTC),
                entry_price=Decimal("150.00"),
                shares=100,
                stop_loss=Decimal("145.00"),
                take_profit=Decimal("160.00"),
                status="open" if i % 2 == 0 else "closed",
            )
            for i in range(4)
        ]

        db_session.add_all(positions)
        await db_session.commit()

        result = await db_session.execute(
            select(PaperPosition).where(PaperPosition.status == "open")
        )
        open_positions = result.scalars().all()

        assert len(open_positions) == 2
        assert all(p.status == "open" for p in open_positions)

    async def test_query_positions_by_symbol_and_status(self, db_session: AsyncSession) -> None:
        """Test querying positions by symbol and status (composite index)."""
        positions = [
            PaperPosition(
                signal_id=f"signal-{i}",
                symbol="AAPL" if i < 2 else "TSLA",
                direction="long",
                entry_date=datetime.now(UTC),
                entry_price=Decimal("150.00"),
                shares=100,
                stop_loss=Decimal("145.00"),
                take_profit=Decimal("160.00"),
                status="open" if i % 2 == 0 else "closed",
            )
            for i in range(4)
        ]

        db_session.add_all(positions)
        await db_session.commit()

        result = await db_session.execute(
            select(PaperPosition).where(
                PaperPosition.symbol == "AAPL",
                PaperPosition.status == "open",
            )
        )
        filtered_positions = result.scalars().all()

        assert len(filtered_positions) == 1
        assert filtered_positions[0].symbol == "AAPL"
        assert filtered_positions[0].status == "open"


@pytest.mark.asyncio
class TestPortfolioSnapshot:
    """Tests for PortfolioSnapshot model."""

    async def test_create_portfolio_snapshot(self, db_session: AsyncSession) -> None:
        """Test creating a portfolio snapshot."""
        snapshot = PortfolioSnapshot(
            snapshot_date=datetime.now(UTC),
            equity_value=Decimal("100000.00"),
            cash_balance=Decimal("50000.00"),
            positions_value=Decimal("50000.00"),
            positions_count=5,
            daily_return_pct=Decimal("1.25"),
        )

        db_session.add(snapshot)
        await db_session.commit()
        await db_session.refresh(snapshot)

        assert snapshot.id is not None
        assert snapshot.equity_value == Decimal("100000.00")
        assert snapshot.cash_balance == Decimal("50000.00")
        assert snapshot.positions_value == Decimal("50000.00")
        assert snapshot.positions_count == 5
        assert snapshot.daily_return_pct == Decimal("1.25")
        assert snapshot.created_at is not None

    async def test_portfolio_snapshot_nullable_return(self, db_session: AsyncSession) -> None:
        """Test that daily_return_pct can be None (first snapshot)."""
        snapshot = PortfolioSnapshot(
            snapshot_date=datetime.now(UTC),
            equity_value=Decimal("100000.00"),
            cash_balance=Decimal("100000.00"),
            positions_value=Decimal("0.00"),
            positions_count=0,
            daily_return_pct=None,
        )

        db_session.add(snapshot)
        await db_session.commit()
        await db_session.refresh(snapshot)

        assert snapshot.daily_return_pct is None

    async def test_portfolio_snapshot_unique_date(self, db_session: AsyncSession) -> None:
        """Test that snapshot_date must be unique."""
        snapshot_date = datetime.now(UTC)

        snapshot1 = PortfolioSnapshot(
            snapshot_date=snapshot_date,
            equity_value=Decimal("100000.00"),
            cash_balance=Decimal("50000.00"),
            positions_value=Decimal("50000.00"),
            positions_count=5,
        )

        db_session.add(snapshot1)
        await db_session.commit()

        snapshot2 = PortfolioSnapshot(
            snapshot_date=snapshot_date,
            equity_value=Decimal("105000.00"),
            cash_balance=Decimal("52500.00"),
            positions_value=Decimal("52500.00"),
            positions_count=5,
        )

        db_session.add(snapshot2)

        with pytest.raises(IntegrityError):
            await db_session.commit()

    async def test_query_snapshots_by_date_range(self, db_session: AsyncSession) -> None:
        """Test querying snapshots by date range."""
        base_date = datetime.now(UTC) - timedelta(days=10)

        snapshots = [
            PortfolioSnapshot(
                snapshot_date=base_date + timedelta(days=i),
                equity_value=Decimal("100000.00") + Decimal(i * 1000),
                cash_balance=Decimal("50000.00"),
                positions_value=Decimal("50000.00") + Decimal(i * 1000),
                positions_count=5,
                daily_return_pct=Decimal("1.00"),
            )
            for i in range(10)
        ]

        db_session.add_all(snapshots)
        await db_session.commit()

        start_date = base_date + timedelta(days=3)
        end_date = base_date + timedelta(days=7)

        result = await db_session.execute(
            select(PortfolioSnapshot)
            .where(
                PortfolioSnapshot.snapshot_date >= start_date,
                PortfolioSnapshot.snapshot_date <= end_date,
            )
            .order_by(PortfolioSnapshot.snapshot_date)
        )
        filtered_snapshots = result.scalars().all()

        assert len(filtered_snapshots) == 5
        assert filtered_snapshots[0].equity_value == Decimal("103000.00")
        assert filtered_snapshots[-1].equity_value == Decimal("107000.00")

    async def test_portfolio_snapshot_equity_calculation(self, db_session: AsyncSession) -> None:
        """Test that equity_value equals cash_balance plus positions_value."""
        cash = Decimal("60000.00")
        positions = Decimal("40000.00")

        snapshot = PortfolioSnapshot(
            snapshot_date=datetime.now(UTC),
            equity_value=cash + positions,
            cash_balance=cash,
            positions_value=positions,
            positions_count=3,
        )

        db_session.add(snapshot)
        await db_session.commit()
        await db_session.refresh(snapshot)

        assert snapshot.equity_value == snapshot.cash_balance + snapshot.positions_value
        assert snapshot.equity_value == Decimal("100000.00")
