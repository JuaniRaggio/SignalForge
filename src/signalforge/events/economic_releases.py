"""Economic releases tracking and analysis module."""

from datetime import UTC, datetime, timedelta

import httpx
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.events.schemas import EconomicEvent
from signalforge.models.event import Event, EventType

logger = structlog.get_logger(__name__)


class EconomicCalendar:
    """Service for tracking economic releases and indicators."""

    def __init__(self, db_session: AsyncSession) -> None:
        """Initialize the economic calendar.

        Args:
            db_session: Async database session for queries.
        """
        self.db = db_session
        self.logger = logger.bind(service="economic_calendar")
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.http_client.aclose()

    async def get_upcoming_releases(self, days: int = 14) -> list[EconomicEvent]:
        """Get upcoming economic releases for the next N days.

        Args:
            days: Number of days to look ahead.

        Returns:
            List of upcoming economic events.
        """
        now = datetime.now(UTC)
        end_date = now + timedelta(days=days)

        query = (
            select(Event)
            .where(
                Event.event_type == EventType.ECONOMIC,
                Event.event_date >= now,
                Event.event_date <= end_date,
            )
            .order_by(Event.event_date.asc())
        )

        result = await self.db.execute(query)
        events = list(result.scalars().all())

        economic_events = []
        for event in events:
            metadata = event.metadata_json or {}
            economic_event = EconomicEvent(
                event_date=event.event_date,
                importance=event.importance,
                indicator_name=metadata.get("indicator_name", "Unknown"),
                forecast=metadata.get("forecast"),
                actual=metadata.get("actual"),
                prior=metadata.get("prior"),
                title=event.title,
                description=event.description,
                source=event.source,
            )
            economic_events.append(economic_event)

        self.logger.info("Retrieved upcoming economic releases", days=days, count=len(economic_events))
        return economic_events

    async def get_cpi_calendar(self) -> list[EconomicEvent]:
        """Get Consumer Price Index (CPI) release calendar.

        Returns:
            List of CPI events.
        """
        query = (
            select(Event)
            .where(Event.event_type == EventType.CPI)
            .order_by(Event.event_date.desc())
            .limit(12)
        )

        result = await self.db.execute(query)
        events = list(result.scalars().all())

        cpi_events = []
        for event in events:
            cpi_event = EconomicEvent(
                event_date=event.event_date,
                importance=event.importance,
                indicator_name="CPI",
                forecast=event.expected_value,
                actual=event.actual_value,
                prior=event.previous_value,
                title=event.title,
                description=event.description,
                source=event.source,
            )
            cpi_events.append(cpi_event)

        self.logger.info("Retrieved CPI calendar", count=len(cpi_events))
        return cpi_events

    async def get_nfp_calendar(self) -> list[EconomicEvent]:
        """Get Non-Farm Payrolls (NFP) release calendar.

        Returns:
            List of NFP events.
        """
        query = (
            select(Event)
            .where(Event.event_type == EventType.NFP)
            .order_by(Event.event_date.desc())
            .limit(12)
        )

        result = await self.db.execute(query)
        events = list(result.scalars().all())

        nfp_events = []
        for event in events:
            nfp_event = EconomicEvent(
                event_date=event.event_date,
                importance=event.importance,
                indicator_name="Non-Farm Payrolls",
                forecast=event.expected_value,
                actual=event.actual_value,
                prior=event.previous_value,
                title=event.title,
                description=event.description,
                source=event.source,
            )
            nfp_events.append(nfp_event)

        self.logger.info("Retrieved NFP calendar", count=len(nfp_events))
        return nfp_events

    async def get_gdp_calendar(self) -> list[EconomicEvent]:
        """Get Gross Domestic Product (GDP) release calendar.

        Returns:
            List of GDP events.
        """
        query = (
            select(Event)
            .where(Event.event_type == EventType.GDP)
            .order_by(Event.event_date.desc())
            .limit(8)
        )

        result = await self.db.execute(query)
        events = list(result.scalars().all())

        gdp_events = []
        for event in events:
            gdp_event = EconomicEvent(
                event_date=event.event_date,
                importance=event.importance,
                indicator_name="GDP",
                forecast=event.expected_value,
                actual=event.actual_value,
                prior=event.previous_value,
                title=event.title,
                description=event.description,
                source=event.source,
            )
            gdp_events.append(gdp_event)

        self.logger.info("Retrieved GDP calendar", count=len(gdp_events))
        return gdp_events

    async def fetch_from_fred(
        self,
        series_id: str,
        api_key: str | None = None,
    ) -> list[dict[str, str | float]]:
        """Fetch economic data from Federal Reserve Economic Data (FRED).

        Args:
            series_id: FRED series ID to fetch.
            api_key: FRED API key. If None, returns empty list.

        Returns:
            List of data points with dates and values.
        """
        if not api_key:
            self.logger.warning("FRED API key not provided", series_id=series_id)
            return []

        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
        }

        try:
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            observations = data.get("observations", [])

            results = []
            for obs in observations:
                try:
                    value = float(obs["value"])
                    results.append({
                        "date": obs["date"],
                        "value": value,
                        "series_id": series_id,
                    })
                except (ValueError, KeyError) as e:
                    self.logger.warning(
                        "Skipping invalid observation",
                        series_id=series_id,
                        error=str(e),
                    )
                    continue

            self.logger.info(
                "Fetched data from FRED",
                series_id=series_id,
                count=len(results),
            )
            return results

        except httpx.HTTPError as e:
            self.logger.error(
                "Error fetching from FRED",
                series_id=series_id,
                error=str(e),
            )
            return []

    def calculate_surprise(self, actual: float, forecast: float) -> float:
        """Calculate economic surprise percentage.

        Args:
            actual: Actual reported value.
            forecast: Forecasted value.

        Returns:
            Surprise as a percentage.
        """
        if forecast == 0:
            self.logger.warning("Forecast is zero, cannot calculate surprise")
            return 0.0

        surprise = ((actual - forecast) / abs(forecast)) * 100
        self.logger.debug(
            "Calculated economic surprise",
            actual=actual,
            forecast=forecast,
            surprise=surprise,
        )
        return surprise

    async def store_economic_event(self, economic_event: EconomicEvent) -> Event:
        """Store an economic event in the database.

        Args:
            economic_event: Economic event to store.

        Returns:
            The stored Event.
        """
        # Map indicator name to event type
        type_mapping = {
            "CPI": EventType.CPI,
            "Non-Farm Payrolls": EventType.NFP,
            "NFP": EventType.NFP,
            "GDP": EventType.GDP,
        }

        event_type = type_mapping.get(economic_event.indicator_name, EventType.ECONOMIC)

        metadata = {
            "indicator_name": economic_event.indicator_name,
            "forecast": economic_event.forecast,
            "actual": economic_event.actual,
            "prior": economic_event.prior,
        }

        event = Event(
            symbol=None,
            event_type=event_type,
            event_date=economic_event.event_date,
            importance=economic_event.importance,
            title=economic_event.title,
            description=economic_event.description,
            expected_value=economic_event.forecast,
            actual_value=economic_event.actual,
            previous_value=economic_event.prior,
            metadata_json=metadata,
            source=economic_event.source,
        )

        self.db.add(event)
        await self.db.commit()
        await self.db.refresh(event)

        self.logger.info(
            "Stored economic event",
            event_id=event.id,
            indicator=economic_event.indicator_name,
        )
        return event

    async def get_indicator_history(
        self,
        indicator_name: str,
        months: int = 12,
    ) -> list[EconomicEvent]:
        """Get historical data for a specific economic indicator.

        Args:
            indicator_name: Name of the economic indicator.
            months: Number of months of history to retrieve.

        Returns:
            List of historical economic events.
        """
        cutoff_date = datetime.now(UTC) - timedelta(days=months * 30)

        query = (
            select(Event)
            .where(
                Event.event_type == EventType.ECONOMIC,
                Event.event_date >= cutoff_date,
                Event.event_date < datetime.now(UTC),
            )
            .order_by(Event.event_date.desc())
        )

        result = await self.db.execute(query)
        events = list(result.scalars().all())

        # Filter by indicator name from metadata
        filtered_events = []
        for event in events:
            metadata = event.metadata_json or {}
            if metadata.get("indicator_name") == indicator_name:
                economic_event = EconomicEvent(
                    event_date=event.event_date,
                    importance=event.importance,
                    indicator_name=indicator_name,
                    forecast=event.expected_value,
                    actual=event.actual_value,
                    prior=event.previous_value,
                    title=event.title,
                    description=event.description,
                    source=event.source,
                )
                filtered_events.append(economic_event)

        self.logger.info(
            "Retrieved indicator history",
            indicator=indicator_name,
            months=months,
            count=len(filtered_events),
        )
        return filtered_events

    async def analyze_indicator_trend(
        self,
        indicator_name: str,
        months: int = 6,
    ) -> dict[str, object]:
        """Analyze trend for an economic indicator.

        Args:
            indicator_name: Name of the economic indicator.
            months: Number of months to analyze.

        Returns:
            Dictionary containing trend analysis.
        """
        history = await self.get_indicator_history(indicator_name, months)

        if not history:
            return {
                "indicator": indicator_name,
                "trend": "insufficient_data",
                "average_surprise": None,
                "data_points": 0,
            }

        # Calculate average surprise
        surprises = []
        for event in history:
            if event.actual is not None and event.forecast is not None:
                surprise = self.calculate_surprise(event.actual, event.forecast)
                surprises.append(surprise)

        avg_surprise = sum(surprises) / len(surprises) if surprises else 0.0

        # Determine trend
        if len(history) >= 3:
            recent_actuals = [
                e.actual for e in history[:3] if e.actual is not None
            ]
            if len(recent_actuals) >= 3:
                if all(recent_actuals[i] > recent_actuals[i + 1] for i in range(len(recent_actuals) - 1)):
                    trend = "increasing"
                elif all(recent_actuals[i] < recent_actuals[i + 1] for i in range(len(recent_actuals) - 1)):
                    trend = "decreasing"
                else:
                    trend = "mixed"
            else:
                trend = "insufficient_data"
        else:
            trend = "insufficient_data"

        analysis = {
            "indicator": indicator_name,
            "trend": trend,
            "average_surprise": avg_surprise,
            "data_points": len(history),
            "latest_value": history[0].actual if history else None,
            "latest_date": history[0].event_date.isoformat() if history else None,
        }

        self.logger.info(
            "Analyzed indicator trend",
            indicator=indicator_name,
            trend=trend,
            avg_surprise=avg_surprise,
        )

        return analysis
