"""Daily snapshot management for paper portfolios."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

import polars as pl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.logging import get_logger
from signalforge.models.paper_trading import (
    PaperPortfolio,
    PaperPortfolioSnapshot,
    PaperPosition,
    PortfolioStatus,
)

logger = get_logger(__name__)


class SnapshotService:
    """Manage portfolio snapshots."""

    async def create_daily_snapshot(
        self, portfolio_id: UUID, session: AsyncSession
    ) -> PaperPortfolioSnapshot:
        """Create end-of-day snapshot for portfolio.

        Args:
            portfolio_id: UUID of portfolio to snapshot
            session: Database session

        Returns:
            Created snapshot record
        """
        logger.info("creating_daily_snapshot", portfolio_id=str(portfolio_id))

        # Fetch portfolio
        portfolio_result = await session.execute(
            select(PaperPortfolio).where(PaperPortfolio.id == portfolio_id)
        )
        portfolio = portfolio_result.scalar_one_or_none()

        if not portfolio:
            raise ValueError(f"Portfolio {portfolio_id} not found")

        # Fetch all open positions
        positions_result = await session.execute(
            select(PaperPosition).where(PaperPosition.portfolio_id == portfolio_id)
        )
        positions = positions_result.scalars().all()

        # Calculate positions value
        positions_value = Decimal("0")
        for position in positions:
            position_value = Decimal(str(position.quantity)) * position.current_price
            positions_value += position_value

        # Calculate total equity
        equity_value = portfolio.current_cash + positions_value
        cash_balance = portfolio.current_cash
        positions_count = len(positions)

        # Get previous snapshot to calculate daily return
        prev_snapshot_result = await session.execute(
            select(PaperPortfolioSnapshot)
            .where(PaperPortfolioSnapshot.portfolio_id == portfolio_id)
            .order_by(PaperPortfolioSnapshot.snapshot_date.desc())
            .limit(1)
        )
        prev_snapshot = prev_snapshot_result.scalar_one_or_none()

        daily_return_pct = None
        cumulative_return_pct = None

        # Calculate daily return from previous snapshot
        if prev_snapshot and prev_snapshot.equity_value > 0:
            daily_return = (equity_value - prev_snapshot.equity_value) / prev_snapshot.equity_value
            daily_return_pct = daily_return * Decimal("100")

        # Calculate cumulative return from initial capital
        if portfolio.initial_capital > 0:
            cumulative_return = (equity_value - portfolio.initial_capital) / portfolio.initial_capital
            cumulative_return_pct = cumulative_return * Decimal("100")

        # Create snapshot
        snapshot = PaperPortfolioSnapshot(
            portfolio_id=portfolio_id,
            snapshot_date=datetime.now(UTC),
            equity_value=equity_value,
            cash_balance=cash_balance,
            positions_value=positions_value,
            positions_count=positions_count,
            daily_return_pct=daily_return_pct,
            cumulative_return_pct=cumulative_return_pct,
        )

        session.add(snapshot)
        await session.commit()
        await session.refresh(snapshot)

        logger.info(
            "snapshot_created",
            portfolio_id=str(portfolio_id),
            equity_value=str(equity_value),
            daily_return_pct=str(daily_return_pct) if daily_return_pct else None,
        )

        return snapshot

    async def create_snapshots_for_all_portfolios(
        self, session: AsyncSession
    ) -> list[PaperPortfolioSnapshot]:
        """Create snapshots for all active portfolios.

        Args:
            session: Database session

        Returns:
            List of created snapshots
        """
        logger.info("creating_snapshots_for_all_active_portfolios")

        # Fetch all active portfolios
        portfolios_result = await session.execute(
            select(PaperPortfolio).where(PaperPortfolio.status == PortfolioStatus.ACTIVE)
        )
        portfolios = portfolios_result.scalars().all()

        snapshots = []
        for portfolio in portfolios:
            try:
                snapshot = await self.create_daily_snapshot(portfolio.id, session)
                snapshots.append(snapshot)
            except Exception as e:
                logger.error(
                    "snapshot_creation_failed",
                    portfolio_id=str(portfolio.id),
                    error=str(e),
                )

        logger.info("snapshots_created", count=len(snapshots))
        return snapshots

    async def get_equity_curve(
        self, portfolio_id: UUID, session: AsyncSession, days: int = 30
    ) -> pl.DataFrame:
        """Get equity curve as Polars DataFrame.

        Args:
            portfolio_id: UUID of portfolio
            session: Database session
            days: Number of days to retrieve (default 30)

        Returns:
            Polars DataFrame with equity curve data
        """
        logger.info("fetching_equity_curve", portfolio_id=str(portfolio_id), days=days)

        # Fetch snapshots
        snapshots_result = await session.execute(
            select(PaperPortfolioSnapshot)
            .where(PaperPortfolioSnapshot.portfolio_id == portfolio_id)
            .order_by(PaperPortfolioSnapshot.snapshot_date.desc())
            .limit(days)
        )
        snapshots = snapshots_result.scalars().all()

        if not snapshots:
            # Return empty DataFrame with correct schema
            return pl.DataFrame(
                {
                    "date": [],
                    "equity": [],
                    "cash_balance": [],
                    "positions_value": [],
                    "positions_count": [],
                    "daily_return_pct": [],
                    "cumulative_return_pct": [],
                },
                schema={
                    "date": pl.Datetime,
                    "equity": pl.Float64,
                    "cash_balance": pl.Float64,
                    "positions_value": pl.Float64,
                    "positions_count": pl.Int64,
                    "daily_return_pct": pl.Float64,
                    "cumulative_return_pct": pl.Float64,
                },
            )

        # Convert to DataFrame (reverse to get chronological order)
        snapshots_list = list(snapshots)
        snapshots_list.reverse()

        data = {
            "date": [s.snapshot_date for s in snapshots_list],
            "equity": [float(s.equity_value) for s in snapshots_list],
            "cash_balance": [float(s.cash_balance) for s in snapshots_list],
            "positions_value": [float(s.positions_value) for s in snapshots_list],
            "positions_count": [s.positions_count for s in snapshots_list],
            "daily_return_pct": [
                float(s.daily_return_pct) if s.daily_return_pct is not None else None
                for s in snapshots_list
            ],
            "cumulative_return_pct": [
                float(s.cumulative_return_pct) if s.cumulative_return_pct is not None else None
                for s in snapshots_list
            ],
        }

        df = pl.DataFrame(data)
        return df

    async def get_returns_series(
        self, portfolio_id: UUID, session: AsyncSession, days: int = 30
    ) -> pl.Series:
        """Get daily returns series.

        Args:
            portfolio_id: UUID of portfolio
            session: Database session
            days: Number of days to retrieve (default 30)

        Returns:
            Polars Series with daily returns
        """
        logger.info("fetching_returns_series", portfolio_id=str(portfolio_id), days=days)

        df = await self.get_equity_curve(portfolio_id, session, days)

        if df.is_empty():
            return pl.Series("daily_return_pct", [], dtype=pl.Float64)

        # Filter out None values and return series
        returns = df.filter(pl.col("daily_return_pct").is_not_null()).select("daily_return_pct")

        if returns.is_empty():
            return pl.Series("daily_return_pct", [], dtype=pl.Float64)

        return returns["daily_return_pct"]

    async def get_latest_snapshot(
        self, portfolio_id: UUID, session: AsyncSession
    ) -> PaperPortfolioSnapshot | None:
        """Get the most recent snapshot for a portfolio.

        Args:
            portfolio_id: UUID of portfolio
            session: Database session

        Returns:
            Latest snapshot or None if no snapshots exist
        """
        result = await session.execute(
            select(PaperPortfolioSnapshot)
            .where(PaperPortfolioSnapshot.portfolio_id == portfolio_id)
            .order_by(PaperPortfolioSnapshot.snapshot_date.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
