"""Earnings tracking and analysis module."""

import contextlib
from datetime import UTC, datetime, timedelta

import structlog
import yfinance as yf
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.events.schemas import EarningsEvent
from signalforge.models.event import Event, EventImportance, EventType
from signalforge.models.price import Price

logger = structlog.get_logger(__name__)


class EarningsTracker:
    """Service for tracking and analyzing earnings events."""

    def __init__(self, db_session: AsyncSession) -> None:
        """Initialize the earnings tracker.

        Args:
            db_session: Async database session for queries.
        """
        self.db = db_session
        self.logger = logger.bind(service="earnings_tracker")

    async def get_earnings_calendar(
        self,
        symbols: list[str],
        days: int = 30,
    ) -> list[EarningsEvent]:
        """Get earnings calendar for specified symbols.

        Args:
            symbols: List of stock symbols to query.
            days: Number of days to look ahead.

        Returns:
            List of earnings events for the symbols.
        """
        now = datetime.now(UTC)
        end_date = now + timedelta(days=days)

        query = (
            select(Event)
            .where(
                Event.event_type == EventType.EARNINGS,
                Event.symbol.in_(symbols),
                Event.event_date >= now,
                Event.event_date <= end_date,
            )
            .order_by(Event.event_date.asc())
        )

        result = await self.db.execute(query)
        events = list(result.scalars().all())

        earnings_events = []
        for event in events:
            metadata = event.metadata_json or {}
            earnings_event = EarningsEvent(
                symbol=event.symbol or "",
                event_date=event.event_date,
                importance=event.importance,
                eps_estimate=metadata.get("eps_estimate"),
                eps_actual=metadata.get("eps_actual"),
                revenue_estimate=metadata.get("revenue_estimate"),
                revenue_actual=metadata.get("revenue_actual"),
                title=event.title,
                description=event.description,
                source=event.source,
            )
            earnings_events.append(earnings_event)

        self.logger.info(
            "Retrieved earnings calendar",
            symbols=symbols,
            days=days,
            count=len(earnings_events),
        )
        return earnings_events

    async def fetch_earnings_from_yfinance(self, symbol: str) -> list[EarningsEvent]:
        """Fetch earnings data from Yahoo Finance.

        Args:
            symbol: Stock symbol to fetch earnings for.

        Returns:
            List of earnings events fetched from Yahoo Finance.
        """
        self.logger.info("Fetching earnings from yfinance", symbol=symbol)

        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar

            if calendar is None or calendar.empty:
                self.logger.warning("No earnings calendar found", symbol=symbol)
                return []

            earnings_events = []

            # Get upcoming earnings date
            if "Earnings Date" in calendar.index:
                earnings_dates = calendar.loc["Earnings Date"]
                if not earnings_dates.empty:
                    # Handle both single date and date range
                    if isinstance(earnings_dates, str):
                        earnings_date = datetime.fromisoformat(earnings_dates.replace("Z", "+00:00"))
                    else:
                        # Take the first date if range is provided
                        earnings_date = datetime.fromisoformat(
                            str(earnings_dates.iloc[0]).replace("Z", "+00:00")
                        )

                    # Get EPS estimate if available
                    eps_estimate = None
                    if "EPS Estimate" in calendar.index:
                        eps_data = calendar.loc["EPS Estimate"]
                        if not eps_data.empty:
                            with contextlib.suppress(ValueError, TypeError):
                                eps_estimate = float(eps_data.iloc[0])

                    earnings_event = EarningsEvent(
                        symbol=symbol,
                        event_date=earnings_date,
                        importance=EventImportance.HIGH,
                        eps_estimate=eps_estimate,
                        title=f"{symbol} Earnings Report",
                        description=f"Quarterly earnings release for {symbol}",
                        source="yfinance",
                    )
                    earnings_events.append(earnings_event)

            self.logger.info(
                "Fetched earnings from yfinance",
                symbol=symbol,
                count=len(earnings_events),
            )
            return earnings_events

        except Exception as e:
            self.logger.error("Error fetching earnings from yfinance", symbol=symbol, error=str(e))
            return []

    def calculate_earnings_surprise(self, eps_actual: float, eps_estimate: float) -> float:
        """Calculate earnings surprise percentage.

        Earnings surprise is the percentage difference between actual and estimated EPS.

        Args:
            eps_actual: Actual earnings per share.
            eps_estimate: Estimated earnings per share.

        Returns:
            Earnings surprise as a percentage.
        """
        if eps_estimate == 0:
            self.logger.warning("EPS estimate is zero, cannot calculate surprise")
            return 0.0

        surprise = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100
        self.logger.debug(
            "Calculated earnings surprise",
            eps_actual=eps_actual,
            eps_estimate=eps_estimate,
            surprise=surprise,
        )
        return surprise

    async def get_earnings_history(
        self,
        symbol: str,
        quarters: int = 8,
    ) -> list[EarningsEvent]:
        """Get historical earnings events for a symbol.

        Args:
            symbol: Stock symbol to query.
            quarters: Number of past quarters to retrieve (default 8 = 2 years).

        Returns:
            List of historical earnings events.
        """
        # Approximate quarters as 90-day periods
        days_back = quarters * 90
        cutoff_date = datetime.now(UTC) - timedelta(days=days_back)

        query = (
            select(Event)
            .where(
                Event.event_type == EventType.EARNINGS,
                Event.symbol == symbol,
                Event.event_date >= cutoff_date,
                Event.event_date < datetime.now(UTC),
            )
            .order_by(Event.event_date.desc())
            .limit(quarters)
        )

        result = await self.db.execute(query)
        events = list(result.scalars().all())

        earnings_events = []
        for event in events:
            metadata = event.metadata_json or {}
            earnings_event = EarningsEvent(
                symbol=event.symbol or "",
                event_date=event.event_date,
                importance=event.importance,
                eps_estimate=metadata.get("eps_estimate"),
                eps_actual=metadata.get("eps_actual"),
                revenue_estimate=metadata.get("revenue_estimate"),
                revenue_actual=metadata.get("revenue_actual"),
                title=event.title,
                description=event.description,
                source=event.source,
            )
            earnings_events.append(earnings_event)

        self.logger.info(
            "Retrieved earnings history",
            symbol=symbol,
            quarters=quarters,
            count=len(earnings_events),
        )
        return earnings_events

    async def get_earnings_reaction(
        self,
        symbol: str,
        event_id: int,
    ) -> dict[str, float | str | None]:
        """Get price reaction to an earnings event.

        Analyzes stock price movement before and after an earnings event.

        Args:
            symbol: Stock symbol.
            event_id: ID of the earnings event.

        Returns:
            Dictionary containing price reaction metrics.
        """
        # Get the earnings event
        event_query = select(Event).where(Event.id == event_id)
        event_result = await self.db.execute(event_query)
        event = event_result.scalar_one_or_none()

        if event is None:
            self.logger.error("Earnings event not found", event_id=event_id)
            return {
                "error": "Event not found",
                "event_id": event_id,
            }

        # Get prices around the earnings date
        # Look at 1 day before and 1 day after for immediate reaction
        before_date = event.event_date - timedelta(days=1)
        after_date = event.event_date + timedelta(days=1)

        price_query = select(Price).where(
            Price.symbol == symbol,
            Price.timestamp >= before_date,
            Price.timestamp <= after_date,
        )

        price_result = await self.db.execute(price_query)
        prices = list(price_result.scalars().all())

        if len(prices) < 2:
            self.logger.warning(
                "Insufficient price data for earnings reaction",
                symbol=symbol,
                event_id=event_id,
                price_count=len(prices),
            )
            return {
                "error": "Insufficient price data",
                "symbol": symbol,
                "event_id": event_id,
            }

        # Sort prices by timestamp
        prices_sorted = sorted(prices, key=lambda p: p.timestamp)
        price_before = prices_sorted[0]
        price_after = prices_sorted[-1]

        # Calculate metrics
        price_change = float(price_after.close - price_before.close)
        price_change_pct = (price_change / float(price_before.close)) * 100
        volume_change_pct = (
            ((float(price_after.volume) - float(price_before.volume)) / float(price_before.volume))
            * 100
            if price_before.volume > 0
            else 0
        )

        # Get earnings surprise if available
        metadata = event.metadata_json or {}
        eps_actual = metadata.get("eps_actual")
        eps_estimate = metadata.get("eps_estimate")
        earnings_surprise = (
            self.calculate_earnings_surprise(float(eps_actual), float(eps_estimate))
            if eps_actual is not None and eps_estimate is not None
            else None
        )

        reaction = {
            "symbol": symbol,
            "event_id": event_id,
            "event_date": event.event_date.isoformat(),
            "price_before": float(price_before.close),
            "price_after": float(price_after.close),
            "price_change": price_change,
            "price_change_pct": price_change_pct,
            "volume_before": float(price_before.volume),
            "volume_after": float(price_after.volume),
            "volume_change_pct": volume_change_pct,
            "earnings_surprise": earnings_surprise,
            "eps_actual": eps_actual,
            "eps_estimate": eps_estimate,
        }

        self.logger.info(
            "Calculated earnings reaction",
            symbol=symbol,
            event_id=event_id,
            price_change_pct=price_change_pct,
        )
        return reaction

    async def store_earnings_event(self, earnings_event: EarningsEvent) -> Event:
        """Store an earnings event in the database.

        Args:
            earnings_event: Earnings event to store.

        Returns:
            The stored Event.
        """
        metadata = {
            "eps_estimate": earnings_event.eps_estimate,
            "eps_actual": earnings_event.eps_actual,
            "revenue_estimate": earnings_event.revenue_estimate,
            "revenue_actual": earnings_event.revenue_actual,
        }

        event = Event(
            symbol=earnings_event.symbol,
            event_type=EventType.EARNINGS,
            event_date=earnings_event.event_date,
            importance=earnings_event.importance,
            title=earnings_event.title,
            description=earnings_event.description,
            expected_value=earnings_event.eps_estimate,
            actual_value=earnings_event.eps_actual,
            metadata_json=metadata,
            source=earnings_event.source,
        )

        self.db.add(event)
        await self.db.commit()
        await self.db.refresh(event)

        self.logger.info(
            "Stored earnings event",
            event_id=event.id,
            symbol=earnings_event.symbol,
        )
        return event
