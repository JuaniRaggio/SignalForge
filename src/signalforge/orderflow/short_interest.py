"""Short interest tracking and analysis."""

from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.orderflow import ShortInterest
from signalforge.orderflow.schemas import ShortInterestChange, ShortInterestRecord

logger = structlog.get_logger(__name__)


class ShortInterestTracker:
    """Track and analyze short interest data."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize tracker.

        Args:
            session: Database session for queries.
        """
        self.session = session
        self.logger = logger.bind(component="short_interest_tracker")

    async def get_current_short_interest(self, symbol: str) -> ShortInterestRecord:
        """Get most recent short interest data.

        Args:
            symbol: Stock symbol.

        Returns:
            Most recent short interest record.

        Raises:
            ValueError: If no data found.
        """
        stmt = select(ShortInterest).where(
            ShortInterest.symbol == symbol
        ).order_by(ShortInterest.report_date.desc()).limit(1)

        result = await self.session.execute(stmt)
        record = result.scalar_one_or_none()

        if not record:
            raise ValueError(f"No short interest data found for {symbol}")

        return ShortInterestRecord(
            symbol=record.symbol,
            report_date=record.report_date,
            short_interest=record.short_interest,
            shares_outstanding=record.shares_outstanding,
            short_percent=record.short_percent,
            days_to_cover=record.days_to_cover,
            change_percent=record.change_percent,
        )

    async def get_short_interest_history(
        self, symbol: str, reports: int = 12
    ) -> list[ShortInterestRecord]:
        """Get historical short interest data.

        Args:
            symbol: Stock symbol.
            reports: Number of historical reports to retrieve.

        Returns:
            List of short interest records.
        """
        stmt = select(ShortInterest).where(
            ShortInterest.symbol == symbol
        ).order_by(ShortInterest.report_date.desc()).limit(reports)

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        return [
            ShortInterestRecord(
                symbol=r.symbol,
                report_date=r.report_date,
                short_interest=r.short_interest,
                shares_outstanding=r.shares_outstanding,
                short_percent=r.short_percent,
                days_to_cover=r.days_to_cover,
                change_percent=r.change_percent,
            )
            for r in records
        ]

    async def detect_short_squeeze_candidates(
        self,
        min_short_percent: float = 20.0,
        min_days_to_cover: float = 5.0,
    ) -> list[str]:
        """Detect potential short squeeze candidates.

        Args:
            min_short_percent: Minimum short interest percentage.
            min_days_to_cover: Minimum days to cover threshold.

        Returns:
            List of symbols meeting squeeze criteria.
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=30)

        stmt = select(ShortInterest).where(
            ShortInterest.report_date >= cutoff_date,
            ShortInterest.short_percent >= min_short_percent,
            ShortInterest.days_to_cover >= min_days_to_cover,
        ).order_by(ShortInterest.short_percent.desc())

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        candidates = list({r.symbol for r in records})

        self.logger.info(
            "detected_squeeze_candidates",
            count=len(candidates),
            min_short_pct=min_short_percent,
            min_days_to_cover=min_days_to_cover,
        )

        return candidates

    async def calculate_short_interest_change(
        self, symbol: str
    ) -> ShortInterestChange:
        """Calculate short interest change from previous report.

        Args:
            symbol: Stock symbol.

        Returns:
            Short interest change analysis.

        Raises:
            ValueError: If insufficient data available.
        """
        history = await self.get_short_interest_history(symbol, reports=2)

        if len(history) < 1:
            raise ValueError(f"No short interest data for {symbol}")

        current = history[0]
        previous = history[1] if len(history) > 1 else None

        if previous:
            change_shares = current.short_interest - previous.short_interest
            change_percent = (
                (current.short_interest - previous.short_interest)
                / previous.short_interest
                * 100
            )
            is_increasing = change_shares > 0
            is_significant = abs(change_percent) > 10.0
        else:
            change_shares = 0
            change_percent = 0.0
            is_increasing = False
            is_significant = False

        return ShortInterestChange(
            symbol=symbol,
            current=current,
            previous=previous,
            change_percent=change_percent,
            change_shares=change_shares,
            is_increasing=is_increasing,
            is_significant=is_significant,
        )

    async def get_most_shorted_stocks(self, top_n: int = 20) -> list[ShortInterestRecord]:
        """Get most heavily shorted stocks.

        Args:
            top_n: Number of top stocks to return.

        Returns:
            List of most shorted stocks.
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=30)

        # Get the latest report date per symbol using proper GROUP BY with MAX
        latest_dates = (
            select(
                ShortInterest.symbol,
                func.max(ShortInterest.report_date).label("max_date"),
            )
            .where(ShortInterest.report_date >= cutoff_date)
            .group_by(ShortInterest.symbol)
            .subquery()
        )

        # Join to get full records for the latest date per symbol
        stmt = (
            select(ShortInterest)
            .join(
                latest_dates,
                (ShortInterest.symbol == latest_dates.c.symbol) &
                (ShortInterest.report_date == latest_dates.c.max_date)
            )
            .order_by(ShortInterest.short_percent.desc())
            .limit(top_n)
        )

        result = await self.session.execute(stmt)
        records = result.scalars().all()

        return [
            ShortInterestRecord(
                symbol=r.symbol,
                report_date=r.report_date,
                short_interest=r.short_interest,
                shares_outstanding=r.shares_outstanding,
                short_percent=r.short_percent,
                days_to_cover=r.days_to_cover,
                change_percent=r.change_percent,
            )
            for r in records
        ]
