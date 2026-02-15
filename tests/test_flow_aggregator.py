"""Tests for flow aggregation."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.orderflow import FlowDirection, FlowType, OrderFlowRecord
from signalforge.orderflow.aggregator import FlowAggregator


@pytest.fixture
def sample_flow_records(db_session: AsyncSession) -> list[OrderFlowRecord]:
    """Create sample flow records."""
    base_time = datetime.now(UTC)

    return [
        OrderFlowRecord(
            symbol="AAPL",
            flow_type=FlowType.DARK_POOL if i % 2 == 0 else FlowType.OPTIONS,
            direction=FlowDirection.BULLISH if i < 6 else FlowDirection.BEARISH,
            timestamp=base_time - timedelta(days=i),
            value=1_000_000 + i * 100_000,
            volume=10000 + i * 1000,
            price=150.0,
            source="TEST_SOURCE",
        )
        for i in range(10)
    ]


class TestFlowAggregator:
    """Test suite for FlowAggregator."""

    async def test_aggregate_flows_no_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test aggregation with no data."""
        aggregator = FlowAggregator(db_session)
        result = await aggregator.aggregate_flows("NVDA", days=5)

        assert result.symbol == "NVDA"
        assert result.net_flow == 0.0
        assert result.bullish_flow == 0.0
        assert result.bearish_flow == 0.0
        assert result.bias == FlowDirection.NEUTRAL
        assert result.z_score == 0.0
        assert result.dark_pool_volume == 0
        assert result.options_premium == 0.0

    async def test_aggregate_flows_with_data(
        self, db_session: AsyncSession, sample_flow_records: list[OrderFlowRecord]
    ) -> None:
        """Test aggregation with flow data."""
        db_session.add_all(sample_flow_records)
        await db_session.commit()

        aggregator = FlowAggregator(db_session)
        result = await aggregator.aggregate_flows("AAPL", days=15)

        assert result.symbol == "AAPL"
        assert result.net_flow != 0.0
        assert result.bullish_flow > 0
        assert result.bearish_flow > 0
        assert result.dark_pool_volume > 0
        assert result.options_premium > 0

    async def test_aggregate_flows_bullish_bias(
        self, db_session: AsyncSession
    ) -> None:
        """Test aggregation with bullish bias."""
        base_time = datetime.now(UTC)

        records = [
            OrderFlowRecord(
                symbol="TSLA",
                flow_type=FlowType.DARK_POOL,
                direction=FlowDirection.BULLISH,
                timestamp=base_time - timedelta(days=i),
                value=2_000_000,
                volume=20000,
                price=200.0,
                source="BULLISH_SOURCE",
            )
            for i in range(8)
        ] + [
            OrderFlowRecord(
                symbol="TSLA",
                flow_type=FlowType.DARK_POOL,
                direction=FlowDirection.BEARISH,
                timestamp=base_time - timedelta(days=i + 8),
                value=500_000,
                volume=5000,
                price=200.0,
                source="BEARISH_SOURCE",
            )
            for i in range(2)
        ]

        db_session.add_all(records)
        await db_session.commit()

        aggregator = FlowAggregator(db_session)
        result = await aggregator.aggregate_flows("TSLA", days=15)

        assert result.bias == FlowDirection.BULLISH
        assert result.bullish_flow > result.bearish_flow
        assert result.net_flow > 0

    async def test_aggregate_flows_bearish_bias(
        self, db_session: AsyncSession
    ) -> None:
        """Test aggregation with bearish bias."""
        base_time = datetime.now(UTC)

        records = [
            OrderFlowRecord(
                symbol="MSFT",
                flow_type=FlowType.OPTIONS,
                direction=FlowDirection.BEARISH,
                timestamp=base_time - timedelta(days=i),
                value=2_000_000,
                volume=20000,
                price=300.0,
                source="BEARISH_SOURCE",
            )
            for i in range(8)
        ] + [
            OrderFlowRecord(
                symbol="MSFT",
                flow_type=FlowType.OPTIONS,
                direction=FlowDirection.BULLISH,
                timestamp=base_time - timedelta(days=i + 8),
                value=500_000,
                volume=5000,
                price=300.0,
                source="BULLISH_SOURCE",
            )
            for i in range(2)
        ]

        db_session.add_all(records)
        await db_session.commit()

        aggregator = FlowAggregator(db_session)
        result = await aggregator.aggregate_flows("MSFT", days=15)

        assert result.bias == FlowDirection.BEARISH
        assert result.bearish_flow > result.bullish_flow
        assert result.net_flow < 0

    async def test_calculate_net_flow(
        self, db_session: AsyncSession, sample_flow_records: list[OrderFlowRecord]
    ) -> None:
        """Test net flow calculation."""
        db_session.add_all(sample_flow_records)
        await db_session.commit()

        aggregator = FlowAggregator(db_session)
        net_flow = await aggregator.calculate_net_flow("AAPL")

        assert isinstance(net_flow, float)

    async def test_calculate_flow_z_score_insufficient_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test z-score with insufficient data."""
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
                source="SOURCE",
            )
            for i in range(5)
        ]

        db_session.add_all(records)
        await db_session.commit()

        aggregator = FlowAggregator(db_session)
        z_score = await aggregator.calculate_flow_z_score("GOOGL", lookback_days=20)

        assert z_score == 0.0

    async def test_calculate_flow_z_score_with_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test z-score calculation with sufficient data."""
        base_time = datetime.now(UTC)

        records = [
            OrderFlowRecord(
                symbol="AMZN",
                flow_type=FlowType.DARK_POOL,
                direction=FlowDirection.BULLISH if i % 2 == 0 else FlowDirection.BEARISH,
                timestamp=base_time - timedelta(days=i),
                value=1_000_000 if i < 25 else 5_000_000,
                volume=10000,
                price=150.0,
                source="SOURCE",
            )
            for i in range(30)
        ]

        db_session.add_all(records)
        await db_session.commit()

        aggregator = FlowAggregator(db_session)
        z_score = await aggregator.calculate_flow_z_score("AMZN", lookback_days=30)

        assert isinstance(z_score, float)

    async def test_get_flow_momentum_increasing(
        self, db_session: AsyncSession
    ) -> None:
        """Test flow momentum (increasing)."""
        base_time = datetime.now(UTC)

        records = [
            OrderFlowRecord(
                symbol="META",
                flow_type=FlowType.OPTIONS,
                direction=FlowDirection.BULLISH,
                timestamp=base_time - timedelta(days=i),
                value=1_000_000 * (10 - i) if i < 5 else 500_000,
                volume=10000,
                price=300.0,
                source="SOURCE",
            )
            for i in range(10)
        ]

        db_session.add_all(records)
        await db_session.commit()

        aggregator = FlowAggregator(db_session)
        momentum = await aggregator.get_flow_momentum("META")

        assert isinstance(momentum, float)
        assert -1.0 <= momentum <= 1.0

    async def test_get_flow_momentum_insufficient_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test momentum with insufficient data."""
        aggregator = FlowAggregator(db_session)
        momentum = await aggregator.get_flow_momentum("NFLX")

        assert momentum == 0.0

    async def test_rank_by_institutional_interest(
        self, db_session: AsyncSession
    ) -> None:
        """Test ranking by institutional interest."""
        base_time = datetime.now(UTC)

        symbols = ["AAPL", "GOOGL", "MSFT"]

        for idx, symbol in enumerate(symbols):
            records = [
                OrderFlowRecord(
                    symbol=symbol,
                    flow_type=FlowType.DARK_POOL,
                    direction=FlowDirection.BULLISH,
                    timestamp=base_time - timedelta(days=i),
                    value=(idx + 1) * 1_000_000,
                    volume=(idx + 1) * 10000,
                    price=150.0,
                    source="SOURCE",
                )
                for i in range(10)
            ]
            db_session.add_all(records)

        await db_session.commit()

        aggregator = FlowAggregator(db_session)
        ranked = await aggregator.rank_by_institutional_interest(symbols)

        assert len(ranked) == 3
        assert all(isinstance(r, tuple) for r in ranked)
        assert all(len(r) == 2 for r in ranked)
        assert ranked[0][1] >= ranked[-1][1]

    async def test_rank_by_institutional_interest_empty(
        self, db_session: AsyncSession
    ) -> None:
        """Test ranking with no symbols."""
        aggregator = FlowAggregator(db_session)
        ranked = await aggregator.rank_by_institutional_interest([])

        assert ranked == []

    async def test_rank_by_institutional_interest_with_errors(
        self, db_session: AsyncSession
    ) -> None:
        """Test ranking handles missing data gracefully."""
        symbols = ["AAPL", "INVALID", "GOOGL"]

        base_time = datetime.now(UTC)
        records = [
            OrderFlowRecord(
                symbol="AAPL",
                flow_type=FlowType.DARK_POOL,
                direction=FlowDirection.BULLISH,
                timestamp=base_time - timedelta(days=i),
                value=2_000_000,
                volume=20000,
                price=150.0,
                source="SOURCE",
            )
            for i in range(5)
        ]

        db_session.add_all(records)
        await db_session.commit()

        aggregator = FlowAggregator(db_session)
        ranked = await aggregator.rank_by_institutional_interest(symbols)

        assert len(ranked) == 3
        assert any(s[0] == "AAPL" for s in ranked)

    async def test_aggregate_flows_flow_types(
        self, db_session: AsyncSession
    ) -> None:
        """Test aggregation distinguishes flow types."""
        base_time = datetime.now(UTC)

        records = [
            OrderFlowRecord(
                symbol="NVDA",
                flow_type=FlowType.DARK_POOL,
                direction=FlowDirection.BULLISH,
                timestamp=base_time - timedelta(days=i),
                value=1_000_000,
                volume=10000,
                price=500.0,
                source="DARK",
            )
            for i in range(5)
        ] + [
            OrderFlowRecord(
                symbol="NVDA",
                flow_type=FlowType.OPTIONS,
                direction=FlowDirection.BULLISH,
                timestamp=base_time - timedelta(days=i + 5),
                value=500_000,
                volume=0,
                price=500.0,
                source="OPTIONS",
            )
            for i in range(5)
        ]

        db_session.add_all(records)
        await db_session.commit()

        aggregator = FlowAggregator(db_session)
        result = await aggregator.aggregate_flows("NVDA", days=15)

        assert result.dark_pool_volume > 0
        assert result.options_premium > 0
