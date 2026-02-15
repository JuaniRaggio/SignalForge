"""Tests for dark pool processing and analysis."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.orderflow import FlowDirection, FlowType, OrderFlowRecord
from signalforge.orderflow.dark_pools import DarkPoolProcessor
from signalforge.orderflow.schemas import DarkPoolPrint


@pytest.fixture
def sample_ats_data() -> list[dict[str, str | int | float]]:
    """Sample ATS data for testing with some large outliers."""
    base_time = datetime.now(UTC)
    # Create 17 normal trades and 3 large outliers
    normal_trades = [
        {
            "symbol": "AAPL",
            "timestamp": base_time - timedelta(hours=i),
            "shares": 10000,  # Consistent small size
            "price": 150.0,
            "venue": f"DARKPOOL_{i % 3}",
        }
        for i in range(17)
    ]
    # Add large outliers that will be > mean + 2*std
    large_trades = [
        {
            "symbol": "AAPL",
            "timestamp": base_time - timedelta(hours=17 + i),
            "shares": 100000,  # 10x normal size
            "price": 150.0,
            "venue": "DARKPOOL_BLOCK",
        }
        for i in range(3)
    ]
    return normal_trades + large_trades


@pytest.fixture
def large_print_data() -> list[dict[str, str | int | float]]:
    """Data with large prints."""
    base_time = datetime.now(UTC)
    return [
        {
            "symbol": "TSLA",
            "timestamp": base_time - timedelta(hours=1),
            "shares": 100000,
            "price": 200.0,
            "venue": "DARKPOOL_A",
        },
        {
            "symbol": "TSLA",
            "timestamp": base_time - timedelta(hours=2),
            "shares": 10000,
            "price": 200.0,
            "venue": "DARKPOOL_B",
        },
    ]


class TestDarkPoolProcessor:
    """Test suite for DarkPoolProcessor."""

    async def test_process_ats_data_success(
        self, db_session: AsyncSession, sample_ats_data: list[dict[str, str | int | float]]
    ) -> None:
        """Test successful ATS data processing."""
        processor = DarkPoolProcessor(db_session)
        prints = await processor.process_ats_data(sample_ats_data)

        assert len(prints) == 20
        assert all(isinstance(p, DarkPoolPrint) for p in prints)
        assert all(p.symbol == "AAPL" for p in prints)
        assert all(p.value > 0 for p in prints)
        assert any(p.is_large for p in prints)

    async def test_process_ats_data_empty(self, db_session: AsyncSession) -> None:
        """Test processing empty data."""
        processor = DarkPoolProcessor(db_session)
        prints = await processor.process_ats_data([])

        assert prints == []

    async def test_process_ats_data_missing_columns(
        self, db_session: AsyncSession
    ) -> None:
        """Test handling missing required columns."""
        processor = DarkPoolProcessor(db_session)
        invalid_data = [{"symbol": "AAPL", "shares": 1000}]

        with pytest.raises(ValueError, match="Missing required columns"):
            await processor.process_ats_data(invalid_data)

    async def test_process_ats_data_z_scores(
        self, db_session: AsyncSession, sample_ats_data: list[dict[str, str | int | float]]
    ) -> None:
        """Test z-score calculation."""
        processor = DarkPoolProcessor(db_session)
        prints = await processor.process_ats_data(sample_ats_data)

        assert all(p.z_score is not None for p in prints)
        z_scores = [p.z_score for p in prints if p.z_score is not None]
        assert max(z_scores) > 1.0
        assert min(z_scores) < 0

    async def test_get_dark_pool_summary_no_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test summary with no data."""
        processor = DarkPoolProcessor(db_session)
        summary = await processor.get_dark_pool_summary("NVDA", days=30)

        assert summary.symbol == "NVDA"
        assert summary.total_volume == 0
        assert summary.total_value == 0.0
        assert summary.trade_count == 0
        assert summary.institutional_bias == FlowDirection.NEUTRAL

    async def test_get_dark_pool_summary_with_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test summary with dark pool data."""
        base_time = datetime.now(UTC)

        records = [
            OrderFlowRecord(
                symbol="MSFT",
                flow_type=FlowType.DARK_POOL,
                direction=FlowDirection.BULLISH,
                timestamp=base_time - timedelta(days=i),
                value=1_000_000 + i * 100_000,
                volume=10000 + i * 1000,
                price=300.0,
                source="TEST_VENUE",
            )
            for i in range(10)
        ]

        db_session.add_all(records)
        await db_session.commit()

        processor = DarkPoolProcessor(db_session)
        summary = await processor.get_dark_pool_summary("MSFT", days=30)

        assert summary.symbol == "MSFT"
        assert summary.total_volume > 0
        assert summary.total_value > 0
        assert summary.trade_count == 10
        assert summary.institutional_bias == FlowDirection.BULLISH

    async def test_get_dark_pool_summary_mixed_direction(
        self, db_session: AsyncSession
    ) -> None:
        """Test summary with mixed bullish/bearish flow."""
        base_time = datetime.now(UTC)

        records = [
            OrderFlowRecord(
                symbol="GOOGL",
                flow_type=FlowType.DARK_POOL,
                direction=FlowDirection.BULLISH,
                timestamp=base_time - timedelta(days=i),
                value=1_000_000,
                volume=10000,
                price=150.0,
                source="VENUE_A",
            )
            for i in range(5)
        ] + [
            OrderFlowRecord(
                symbol="GOOGL",
                flow_type=FlowType.DARK_POOL,
                direction=FlowDirection.BEARISH,
                timestamp=base_time - timedelta(days=i + 5),
                value=900_000,
                volume=9000,
                price=150.0,
                source="VENUE_B",
            )
            for i in range(5)
        ]

        db_session.add_all(records)
        await db_session.commit()

        processor = DarkPoolProcessor(db_session)
        summary = await processor.get_dark_pool_summary("GOOGL", days=30)

        assert summary.institutional_bias in [
            FlowDirection.BULLISH,
            FlowDirection.NEUTRAL,
        ]

    async def test_detect_large_prints(self, db_session: AsyncSession) -> None:
        """Test large print detection."""
        base_time = datetime.now(UTC)

        records = [
            OrderFlowRecord(
                symbol="AMZN",
                flow_type=FlowType.DARK_POOL,
                direction=FlowDirection.BULLISH,
                timestamp=base_time - timedelta(hours=i),
                value=2_000_000 if i < 3 else 500_000,
                volume=10000,
                price=200.0,
                source="VENUE",
            )
            for i in range(10)
        ]

        db_session.add_all(records)
        await db_session.commit()

        processor = DarkPoolProcessor(db_session)
        large_prints = await processor.detect_large_prints("AMZN", threshold_usd=1_000_000)

        assert len(large_prints) == 3
        assert all(p.value >= 1_000_000 for p in large_prints)

    async def test_calculate_dark_pool_ratio_no_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test dark pool ratio with no data."""
        processor = DarkPoolProcessor(db_session)
        ratio = await processor.calculate_dark_pool_ratio("NFLX")

        assert ratio == 0.0

    async def test_calculate_dark_pool_ratio_with_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test dark pool ratio calculation."""
        base_time = datetime.now(UTC)

        dark_records = [
            OrderFlowRecord(
                symbol="META",
                flow_type=FlowType.DARK_POOL,
                direction=FlowDirection.BULLISH,
                timestamp=base_time - timedelta(days=i),
                value=1_000_000,
                volume=10000,
                price=300.0,
                source="DARK",
            )
            for i in range(5)
        ]

        lit_records = [
            OrderFlowRecord(
                symbol="META",
                flow_type=FlowType.BLOCK_TRADE,
                direction=FlowDirection.BULLISH,
                timestamp=base_time - timedelta(days=i),
                value=1_000_000,
                volume=30000,
                price=300.0,
                source="LIT",
            )
            for i in range(5)
        ]

        db_session.add_all(dark_records + lit_records)
        await db_session.commit()

        processor = DarkPoolProcessor(db_session)
        ratio = await processor.calculate_dark_pool_ratio("META")

        assert 0.0 <= ratio <= 1.0
        assert ratio == 0.25

    async def test_get_institutional_bias_bullish(
        self, db_session: AsyncSession
    ) -> None:
        """Test institutional bias detection (bullish)."""
        base_time = datetime.now(UTC)

        records = [
            OrderFlowRecord(
                symbol="AMD",
                flow_type=FlowType.DARK_POOL,
                direction=FlowDirection.BULLISH,
                timestamp=base_time - timedelta(days=i),
                value=2_000_000,
                volume=10000,
                price=150.0,
                source="VENUE",
            )
            for i in range(8)
        ] + [
            OrderFlowRecord(
                symbol="AMD",
                flow_type=FlowType.DARK_POOL,
                direction=FlowDirection.BEARISH,
                timestamp=base_time - timedelta(days=i + 8),
                value=500_000,
                volume=3000,
                price=150.0,
                source="VENUE",
            )
            for i in range(2)
        ]

        db_session.add_all(records)
        await db_session.commit()

        processor = DarkPoolProcessor(db_session)
        bias = await processor.get_institutional_bias("AMD")

        assert bias == FlowDirection.BULLISH

    async def test_get_institutional_bias_bearish(
        self, db_session: AsyncSession
    ) -> None:
        """Test institutional bias detection (bearish)."""
        base_time = datetime.now(UTC)

        records = [
            OrderFlowRecord(
                symbol="INTC",
                flow_type=FlowType.DARK_POOL,
                direction=FlowDirection.BEARISH,
                timestamp=base_time - timedelta(days=i),
                value=2_000_000,
                volume=10000,
                price=40.0,
                source="VENUE",
            )
            for i in range(8)
        ] + [
            OrderFlowRecord(
                symbol="INTC",
                flow_type=FlowType.DARK_POOL,
                direction=FlowDirection.BULLISH,
                timestamp=base_time - timedelta(days=i + 8),
                value=500_000,
                volume=3000,
                price=40.0,
                source="VENUE",
            )
            for i in range(2)
        ]

        db_session.add_all(records)
        await db_session.commit()

        processor = DarkPoolProcessor(db_session)
        bias = await processor.get_institutional_bias("INTC")

        assert bias == FlowDirection.BEARISH
