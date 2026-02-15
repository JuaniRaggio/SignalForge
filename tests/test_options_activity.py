"""Tests for options activity detection."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.orderflow import FlowDirection, OptionsActivity
from signalforge.orderflow.options_activity import OptionsActivityDetector


@pytest.fixture
def sample_options_data(db_session: AsyncSession) -> list[OptionsActivity]:
    """Create sample options activity data."""
    base_time = datetime.now(UTC)
    expiry = base_time + timedelta(days=30)

    return [
        OptionsActivity(
            symbol="AAPL",
            timestamp=base_time - timedelta(hours=i),
            option_type="call" if i % 2 == 0 else "put",
            strike=150.0 + i * 5,
            expiry=expiry,
            volume=1000 * (i + 1),
            open_interest=5000 + i * 100,
            premium=50000.0 * (i + 1),
            implied_volatility=0.3 + i * 0.01,
            delta=0.5 - i * 0.02 if i % 2 == 0 else -0.5 + i * 0.02,
        )
        for i in range(10)
    ]


class TestOptionsActivityDetector:
    """Test suite for OptionsActivityDetector."""

    async def test_detect_unusual_activity_no_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test detection with no data."""
        detector = OptionsActivityDetector(db_session)
        unusual = await detector.detect_unusual_activity("NVDA", volume_threshold=2.0)

        assert unusual == []

    async def test_detect_unusual_activity_with_data(
        self, db_session: AsyncSession, sample_options_data: list[OptionsActivity]
    ) -> None:
        """Test unusual activity detection."""
        db_session.add_all(sample_options_data)
        await db_session.commit()

        detector = OptionsActivityDetector(db_session)
        unusual = await detector.detect_unusual_activity("AAPL", volume_threshold=0.15)

        assert len(unusual) > 0
        assert all(u.record.symbol == "AAPL" for u in unusual)
        assert all(u.volume_ratio > 0 for u in unusual)
        assert all(len(u.reason) > 0 for u in unusual)

    async def test_detect_unusual_activity_high_volume(
        self, db_session: AsyncSession
    ) -> None:
        """Test detection with high volume/OI ratio."""
        base_time = datetime.now(UTC)
        expiry = base_time + timedelta(days=30)

        high_volume_option = OptionsActivity(
            symbol="TSLA",
            timestamp=base_time,
            option_type="call",
            strike=200.0,
            expiry=expiry,
            volume=10000,
            open_interest=1000,
            premium=200000.0,
            implied_volatility=0.4,
            delta=0.6,
        )

        db_session.add(high_volume_option)
        await db_session.commit()

        detector = OptionsActivityDetector(db_session)
        unusual = await detector.detect_unusual_activity("TSLA", volume_threshold=5.0)

        assert len(unusual) > 0
        assert unusual[0].volume_ratio >= 5.0

    async def test_detect_unusual_activity_high_premium(
        self, db_session: AsyncSession
    ) -> None:
        """Test detection with unusually high premium."""
        base_time = datetime.now(UTC)
        expiry = base_time + timedelta(days=30)

        records = [
            OptionsActivity(
                symbol="MSFT",
                timestamp=base_time - timedelta(hours=i),
                option_type="call",
                strike=300.0,
                expiry=expiry,
                volume=1000,
                open_interest=5000,
                premium=50000.0 if i < 9 else 500000.0,
                implied_volatility=0.3,
                delta=0.5,
            )
            for i in range(10)
        ]

        db_session.add_all(records)
        await db_session.commit()

        detector = OptionsActivityDetector(db_session)
        unusual = await detector.detect_unusual_activity("MSFT", volume_threshold=2.0)

        assert len(unusual) > 0
        high_premium = [u for u in unusual if u.record.premium > 400000]
        assert len(high_premium) > 0

    async def test_get_put_call_ratio_no_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test put/call ratio with no data."""
        detector = OptionsActivityDetector(db_session)
        ratio = await detector.get_put_call_ratio("GOOGL")

        assert ratio == 1.0

    async def test_get_put_call_ratio_balanced(
        self, db_session: AsyncSession
    ) -> None:
        """Test put/call ratio with balanced data."""
        base_time = datetime.now(UTC)
        expiry = base_time + timedelta(days=30)

        records = [
            OptionsActivity(
                symbol="AMZN",
                timestamp=base_time - timedelta(hours=i),
                option_type="call" if i < 5 else "put",
                strike=150.0,
                expiry=expiry,
                volume=1000,
                open_interest=5000,
                premium=50000.0,
                implied_volatility=0.3,
                delta=0.5 if i < 5 else -0.5,
            )
            for i in range(10)
        ]

        db_session.add_all(records)
        await db_session.commit()

        detector = OptionsActivityDetector(db_session)
        ratio = await detector.get_put_call_ratio("AMZN")

        assert ratio == 1.0

    async def test_get_put_call_ratio_bearish(
        self, db_session: AsyncSession
    ) -> None:
        """Test put/call ratio indicating bearish sentiment."""
        base_time = datetime.now(UTC)
        expiry = base_time + timedelta(days=30)

        records = [
            OptionsActivity(
                symbol="META",
                timestamp=base_time - timedelta(hours=i),
                option_type="put",
                strike=150.0,
                expiry=expiry,
                volume=2000,
                open_interest=5000,
                premium=50000.0,
                implied_volatility=0.3,
                delta=-0.5,
            )
            for i in range(8)
        ] + [
            OptionsActivity(
                symbol="META",
                timestamp=base_time - timedelta(hours=i + 8),
                option_type="call",
                strike=150.0,
                expiry=expiry,
                volume=1000,
                open_interest=5000,
                premium=50000.0,
                implied_volatility=0.3,
                delta=0.5,
            )
            for i in range(2)
        ]

        db_session.add_all(records)
        await db_session.commit()

        detector = OptionsActivityDetector(db_session)
        ratio = await detector.get_put_call_ratio("META")

        assert ratio > 1.0

    async def test_get_options_flow_summary(
        self, db_session: AsyncSession, sample_options_data: list[OptionsActivity]
    ) -> None:
        """Test options flow summary."""
        db_session.add_all(sample_options_data)
        await db_session.commit()

        detector = OptionsActivityDetector(db_session)
        summary = await detector.get_options_flow_summary("AAPL", days=5)

        assert summary["total_volume"] > 0
        assert summary["total_premium"] > 0
        assert summary["call_volume"] > 0
        assert summary["put_volume"] > 0
        assert summary["put_call_ratio"] > 0
        assert summary["avg_iv"] > 0

    async def test_get_options_flow_summary_no_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test summary with no data."""
        detector = OptionsActivityDetector(db_session)
        summary = await detector.get_options_flow_summary("NFLX", days=5)

        assert summary["total_volume"] == 0
        assert summary["total_premium"] == 0.0
        assert summary["put_call_ratio"] == 1.0

    async def test_detect_large_premium_trades(
        self, db_session: AsyncSession
    ) -> None:
        """Test large premium trade detection."""
        base_time = datetime.now(UTC)
        expiry = base_time + timedelta(days=30)

        records = [
            OptionsActivity(
                symbol="AMD",
                timestamp=base_time - timedelta(hours=i),
                option_type="call",
                strike=150.0,
                expiry=expiry,
                volume=5000,
                open_interest=10000,
                premium=150000.0 if i < 3 else 50000.0,
                implied_volatility=0.4,
                delta=0.6,
            )
            for i in range(10)
        ]

        db_session.add_all(records)
        await db_session.commit()

        detector = OptionsActivityDetector(db_session)
        large_trades = await detector.detect_large_premium_trades("AMD", threshold_usd=100000)

        assert len(large_trades) == 3
        assert all(t.premium >= 100000 for t in large_trades)

    async def test_calculate_options_sentiment_bullish(
        self, db_session: AsyncSession
    ) -> None:
        """Test bullish options sentiment."""
        base_time = datetime.now(UTC)
        expiry = base_time + timedelta(days=30)

        records = [
            OptionsActivity(
                symbol="NVDA",
                timestamp=base_time - timedelta(hours=i),
                option_type="call",
                strike=500.0,
                expiry=expiry,
                volume=5000,
                open_interest=10000,
                premium=100000.0,
                implied_volatility=0.4,
                delta=0.6,
            )
            for i in range(10)
        ]

        db_session.add_all(records)
        await db_session.commit()

        detector = OptionsActivityDetector(db_session)
        sentiment = await detector.calculate_options_sentiment("NVDA")

        assert sentiment == FlowDirection.BULLISH

    async def test_calculate_options_sentiment_bearish(
        self, db_session: AsyncSession
    ) -> None:
        """Test bearish options sentiment."""
        base_time = datetime.now(UTC)
        expiry = base_time + timedelta(days=30)

        records = [
            OptionsActivity(
                symbol="INTC",
                timestamp=base_time - timedelta(hours=i),
                option_type="put",
                strike=40.0,
                expiry=expiry,
                volume=5000,
                open_interest=10000,
                premium=100000.0,
                implied_volatility=0.4,
                delta=-0.6,
            )
            for i in range(10)
        ]

        db_session.add_all(records)
        await db_session.commit()

        detector = OptionsActivityDetector(db_session)
        sentiment = await detector.calculate_options_sentiment("INTC")

        assert sentiment == FlowDirection.BEARISH

    async def test_calculate_options_sentiment_neutral(
        self, db_session: AsyncSession
    ) -> None:
        """Test neutral options sentiment."""
        base_time = datetime.now(UTC)
        expiry = base_time + timedelta(days=30)

        records = [
            OptionsActivity(
                symbol="SPY",
                timestamp=base_time - timedelta(hours=i),
                option_type="call" if i < 5 else "put",
                strike=450.0,
                expiry=expiry,
                volume=3000,
                open_interest=10000,
                premium=100000.0,
                implied_volatility=0.2,
                delta=0.5 if i < 5 else -0.5,
            )
            for i in range(10)
        ]

        db_session.add_all(records)
        await db_session.commit()

        detector = OptionsActivityDetector(db_session)
        sentiment = await detector.calculate_options_sentiment("SPY")

        assert sentiment == FlowDirection.NEUTRAL

    async def test_get_expiry_concentration(
        self, db_session: AsyncSession
    ) -> None:
        """Test expiry concentration analysis."""
        base_time = datetime.now(UTC)
        expiry1 = base_time + timedelta(days=30)
        expiry2 = base_time + timedelta(days=60)

        records = [
            OptionsActivity(
                symbol="QQQ",
                timestamp=base_time - timedelta(hours=i),
                option_type="call",
                strike=400.0,
                expiry=expiry1 if i < 7 else expiry2,
                volume=1000,
                open_interest=5000,
                premium=50000.0,
                implied_volatility=0.3,
                delta=0.5,
            )
            for i in range(10)
        ]

        db_session.add_all(records)
        await db_session.commit()

        detector = OptionsActivityDetector(db_session)
        concentration = await detector.get_expiry_concentration("QQQ")

        assert len(concentration) == 2
        assert all(v > 0 for v in concentration.values())

    async def test_get_expiry_concentration_no_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test expiry concentration with no data."""
        detector = OptionsActivityDetector(db_session)
        concentration = await detector.get_expiry_concentration("XYZ")

        assert concentration == {}
