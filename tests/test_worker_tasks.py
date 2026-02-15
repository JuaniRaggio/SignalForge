"""Tests for worker tasks."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from signalforge.workers.event_tasks import (
    _sync_earnings_calendar_async,
    _sync_economic_calendar_async,
    _sync_fed_schedule_async,
    run_async,
    sync_earnings_calendar,
    sync_economic_calendar,
    sync_fed_schedule,
)
from signalforge.workers.orderflow_tasks import (
    _aggregate_daily_flow_async,
    _detect_flow_anomalies_async,
    _update_short_interest_async,
    aggregate_daily_flow,
    detect_flow_anomalies,
    update_short_interest,
)


class TestRunAsync:
    """Test the run_async helper function."""

    def test_run_async_executes_coroutine(self) -> None:
        """Test that run_async executes a coroutine correctly."""

        async def sample_coro() -> str:
            return "test_result"

        result = run_async(sample_coro())
        assert result == "test_result"

    def test_run_async_handles_exceptions(self) -> None:
        """Test that run_async propagates exceptions."""

        async def failing_coro() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            run_async(failing_coro())

    def test_run_async_with_arguments(self) -> None:
        """Test that run_async works with coroutines that take arguments."""

        async def coro_with_args(x: int, y: int) -> int:
            return x + y

        result = run_async(coro_with_args(5, 3))
        assert result == 8


class TestEventTasks:
    """Test event tasks."""

    @patch("signalforge.workers.event_tasks.get_session_context")
    @patch("signalforge.workers.event_tasks.EarningsTracker")
    async def test_sync_earnings_calendar_async(
        self,
        mock_tracker_class: MagicMock,
        mock_session_context: MagicMock,
    ) -> None:
        """Test async earnings calendar sync."""
        mock_session = AsyncMock()
        mock_session_context.return_value.__aenter__.return_value = mock_session

        mock_tracker = AsyncMock()
        mock_tracker_class.return_value = mock_tracker

        from signalforge.events.schemas import EarningsEvent
        from signalforge.models.event import EventImportance

        mock_earnings_event = EarningsEvent(
            symbol="AAPL",
            event_date=asyncio.get_event_loop().time(),
            importance=EventImportance.HIGH,
            title="Test Earnings",
            description="Test",
            source="test",
        )
        mock_tracker.fetch_earnings_from_yfinance.return_value = [mock_earnings_event]
        mock_tracker.store_earnings_event.return_value = None

        result = await _sync_earnings_calendar_async(["AAPL"])

        assert "success" in result
        assert "AAPL" in result["success"]
        assert result["events_stored"] == 1
        mock_tracker.fetch_earnings_from_yfinance.assert_called_once_with("AAPL")

    @patch("signalforge.workers.event_tasks.get_session_context")
    @patch("signalforge.workers.event_tasks.FedTracker")
    async def test_sync_fed_schedule_async(
        self,
        mock_tracker_class: MagicMock,
        mock_session_context: MagicMock,
    ) -> None:
        """Test async Fed schedule sync."""
        mock_session = AsyncMock()
        mock_session_context.return_value.__aenter__.return_value = mock_session

        mock_tracker = AsyncMock()
        mock_tracker_class.return_value = mock_tracker

        from datetime import UTC, datetime, timedelta

        from signalforge.events.schemas import FedEvent
        from signalforge.models.event import EventImportance

        future_date = datetime.now(UTC) + timedelta(days=30)
        mock_fed_event = FedEvent(
            event_date=future_date,
            importance=EventImportance.CRITICAL,
            title="FOMC Meeting",
            description="Test",
            source="test",
        )
        mock_tracker.get_fomc_schedule.return_value = [mock_fed_event]
        mock_tracker.store_fed_event.return_value = None

        result = await _sync_fed_schedule_async()

        assert result["success"] is True
        assert result["events_stored"] == 1
        mock_tracker.get_fomc_schedule.assert_called_once()

    @patch("signalforge.workers.event_tasks.get_session_context")
    @patch("signalforge.workers.event_tasks.EconomicCalendar")
    async def test_sync_economic_calendar_async(
        self,
        mock_calendar_class: MagicMock,
        mock_session_context: MagicMock,
    ) -> None:
        """Test async economic calendar sync."""
        mock_session = AsyncMock()
        mock_session_context.return_value.__aenter__.return_value = mock_session

        mock_calendar = AsyncMock()
        mock_calendar_class.return_value = mock_calendar

        from datetime import UTC, datetime

        from signalforge.events.schemas import EconomicEvent
        from signalforge.models.event import EventImportance

        mock_economic_event = EconomicEvent(
            event_date=datetime.now(UTC),
            importance=EventImportance.HIGH,
            indicator_name="CPI",
            title="CPI Release",
            description="Test",
            source="test",
        )
        mock_calendar.get_upcoming_releases.return_value = [mock_economic_event]
        mock_calendar.store_economic_event.return_value = None
        mock_calendar.close.return_value = None

        result = await _sync_economic_calendar_async()

        assert result["success"] is True
        assert result["events_stored"] == 1
        mock_calendar.close.assert_called_once()


class TestOrderFlowTasks:
    """Test orderflow tasks."""

    @patch("signalforge.workers.orderflow_tasks.get_session_context")
    @patch("signalforge.workers.orderflow_tasks.ShortInterestTracker")
    async def test_update_short_interest_async(
        self,
        mock_tracker_class: MagicMock,
        mock_session_context: MagicMock,
    ) -> None:
        """Test async short interest update."""
        mock_session = AsyncMock()
        mock_session_context.return_value.__aenter__.return_value = mock_session

        mock_tracker = AsyncMock()
        mock_tracker_class.return_value = mock_tracker

        from datetime import UTC, datetime

        from signalforge.orderflow.schemas import ShortInterestChange, ShortInterestRecord

        mock_current = ShortInterestRecord(
            symbol="AAPL",
            report_date=datetime.now(UTC),
            short_interest=1000000,
            shares_outstanding=10000000,
            short_percent=25.0,
            days_to_cover=6.0,
            change_percent=15.0,
        )
        mock_change = ShortInterestChange(
            symbol="AAPL",
            current=mock_current,
            previous=None,
            change_percent=15.0,
            change_shares=100000,
            is_increasing=True,
            is_significant=True,
        )

        mock_tracker.get_current_short_interest.return_value = mock_current
        mock_tracker.calculate_short_interest_change.return_value = mock_change

        result = await _update_short_interest_async(["AAPL"])

        assert result["success"] is True
        assert result["symbols_updated"] == 1
        assert len(result["squeeze_candidates"]) == 1
        assert len(result["significant_changes"]) == 1

    @patch("signalforge.workers.orderflow_tasks.get_session_context")
    @patch("signalforge.workers.orderflow_tasks.FlowAnomalyDetector")
    async def test_detect_flow_anomalies_async(
        self,
        mock_detector_class: MagicMock,
        mock_session_context: MagicMock,
    ) -> None:
        """Test async flow anomaly detection."""
        mock_session = AsyncMock()
        mock_session_context.return_value.__aenter__.return_value = mock_session

        mock_detector = AsyncMock()
        mock_detector_class.return_value = mock_detector

        from datetime import UTC, datetime

        from signalforge.orderflow.schemas import AnomalySeverity, FlowAnomaly

        mock_anomaly = FlowAnomaly(
            symbol="AAPL",
            timestamp=datetime.now(UTC),
            anomaly_type="volume_spike",
            severity=AnomalySeverity.HIGH,
            description="Volume spike detected",
            z_score=3.5,
        )

        mock_detector.detect_anomalies.return_value = [mock_anomaly]
        mock_detector.get_anomaly_score.return_value = 75.0

        result = await _detect_flow_anomalies_async(["AAPL"], sensitivity=2.0)

        assert result["success"] is True
        assert result["symbols_analyzed"] == 1
        assert result["total_anomalies"] == 1
        assert len(result["anomalies_detected"]) == 1

    @patch("signalforge.workers.orderflow_tasks.get_session_context")
    @patch("signalforge.workers.orderflow_tasks.FlowAggregator")
    async def test_aggregate_daily_flow_async(
        self,
        mock_aggregator_class: MagicMock,
        mock_session_context: MagicMock,
    ) -> None:
        """Test async daily flow aggregation."""
        mock_session = AsyncMock()
        mock_session_context.return_value.__aenter__.return_value = mock_session

        mock_aggregator = AsyncMock()
        mock_aggregator_class.return_value = mock_aggregator

        from signalforge.models.orderflow import FlowDirection
        from signalforge.orderflow.schemas import FlowAggregation

        mock_aggregation = FlowAggregation(
            symbol="AAPL",
            period_days=5,
            net_flow=1000000.0,
            bullish_flow=1500000.0,
            bearish_flow=500000.0,
            bias=FlowDirection.BULLISH,
            z_score=2.5,
            flow_momentum=0.8,
            dark_pool_volume=100000,
            options_premium=50000.0,
            short_interest_change=5.0,
        )

        mock_aggregator.aggregate_flows.return_value = mock_aggregation

        result = await _aggregate_daily_flow_async(["AAPL"], days=5)

        assert result["success"] is True
        assert result["symbols_aggregated"] == 1
        assert len(result["aggregations"]) == 1
        assert result["aggregations"][0]["symbol"] == "AAPL"
        assert result["aggregations"][0]["bias"] == "bullish"


class TestTaskSignatures:
    """Test that task functions have correct signatures."""

    def test_sync_earnings_calendar_signature(self) -> None:
        """Test sync_earnings_calendar task signature."""
        assert callable(sync_earnings_calendar)
        assert hasattr(sync_earnings_calendar, "delay")

    def test_sync_fed_schedule_signature(self) -> None:
        """Test sync_fed_schedule task signature."""
        assert callable(sync_fed_schedule)
        assert hasattr(sync_fed_schedule, "delay")

    def test_sync_economic_calendar_signature(self) -> None:
        """Test sync_economic_calendar task signature."""
        assert callable(sync_economic_calendar)
        assert hasattr(sync_economic_calendar, "delay")

    def test_update_short_interest_signature(self) -> None:
        """Test update_short_interest task signature."""
        assert callable(update_short_interest)
        assert hasattr(update_short_interest, "delay")

    def test_detect_flow_anomalies_signature(self) -> None:
        """Test detect_flow_anomalies task signature."""
        assert callable(detect_flow_anomalies)
        assert hasattr(detect_flow_anomalies, "delay")

    def test_aggregate_daily_flow_signature(self) -> None:
        """Test aggregate_daily_flow task signature."""
        assert callable(aggregate_daily_flow)
        assert hasattr(aggregate_daily_flow, "delay")
