"""Tests for short interest tracking."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.models.orderflow import ShortInterest
from signalforge.orderflow.short_interest import ShortInterestTracker


@pytest.fixture
def sample_short_interest_data(db_session: AsyncSession) -> list[ShortInterest]:
    """Create sample short interest data."""
    base_date = datetime.now(UTC)

    return [
        ShortInterest(
            symbol="GME",
            report_date=base_date - timedelta(days=i * 15),
            short_interest=50_000_000 + i * 1_000_000,
            shares_outstanding=100_000_000,
            short_percent=50.0 + i * 1.0,
            days_to_cover=5.0 + i * 0.5,
            change_percent=2.0 if i > 0 else None,
        )
        for i in range(12)
    ]


class TestShortInterestTracker:
    """Test suite for ShortInterestTracker."""

    async def test_get_current_short_interest_success(
        self, db_session: AsyncSession, sample_short_interest_data: list[ShortInterest]
    ) -> None:
        """Test getting current short interest."""
        db_session.add_all(sample_short_interest_data)
        await db_session.commit()

        tracker = ShortInterestTracker(db_session)
        current = await tracker.get_current_short_interest("GME")

        assert current.symbol == "GME"
        assert current.short_interest > 0
        assert 0 <= current.short_percent <= 100

    async def test_get_current_short_interest_no_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test getting short interest with no data."""
        tracker = ShortInterestTracker(db_session)

        with pytest.raises(ValueError, match="No short interest data found"):
            await tracker.get_current_short_interest("XYZ")

    async def test_get_short_interest_history(
        self, db_session: AsyncSession, sample_short_interest_data: list[ShortInterest]
    ) -> None:
        """Test getting historical short interest."""
        db_session.add_all(sample_short_interest_data)
        await db_session.commit()

        tracker = ShortInterestTracker(db_session)
        history = await tracker.get_short_interest_history("GME", reports=12)

        assert len(history) == 12
        assert all(h.symbol == "GME" for h in history)
        assert history[0].report_date >= history[-1].report_date

    async def test_get_short_interest_history_limited(
        self, db_session: AsyncSession, sample_short_interest_data: list[ShortInterest]
    ) -> None:
        """Test getting limited history."""
        db_session.add_all(sample_short_interest_data)
        await db_session.commit()

        tracker = ShortInterestTracker(db_session)
        history = await tracker.get_short_interest_history("GME", reports=5)

        assert len(history) == 5

    async def test_detect_short_squeeze_candidates(
        self, db_session: AsyncSession
    ) -> None:
        """Test detecting short squeeze candidates."""
        base_date = datetime.now(UTC)

        records = [
            ShortInterest(
                symbol="AMC",
                report_date=base_date - timedelta(days=5),
                short_interest=80_000_000,
                shares_outstanding=100_000_000,
                short_percent=80.0,
                days_to_cover=10.0,
            ),
            ShortInterest(
                symbol="BB",
                report_date=base_date - timedelta(days=5),
                short_interest=40_000_000,
                shares_outstanding=100_000_000,
                short_percent=40.0,
                days_to_cover=8.0,
            ),
            ShortInterest(
                symbol="AAPL",
                report_date=base_date - timedelta(days=5),
                short_interest=5_000_000,
                shares_outstanding=100_000_000,
                short_percent=5.0,
                days_to_cover=2.0,
            ),
        ]

        db_session.add_all(records)
        await db_session.commit()

        tracker = ShortInterestTracker(db_session)
        candidates = await tracker.detect_short_squeeze_candidates(
            min_short_percent=20.0, min_days_to_cover=5.0
        )

        assert "AMC" in candidates
        assert "BB" in candidates
        assert "AAPL" not in candidates

    async def test_detect_short_squeeze_candidates_no_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test squeeze detection with no data."""
        tracker = ShortInterestTracker(db_session)
        candidates = await tracker.detect_short_squeeze_candidates()

        assert candidates == []

    async def test_detect_short_squeeze_candidates_strict_criteria(
        self, db_session: AsyncSession
    ) -> None:
        """Test squeeze detection with strict criteria."""
        base_date = datetime.now(UTC)

        records = [
            ShortInterest(
                symbol="TSLA",
                report_date=base_date - timedelta(days=5),
                short_interest=30_000_000,
                shares_outstanding=100_000_000,
                short_percent=30.0,
                days_to_cover=6.0,
            ),
        ]

        db_session.add_all(records)
        await db_session.commit()

        tracker = ShortInterestTracker(db_session)
        candidates = await tracker.detect_short_squeeze_candidates(
            min_short_percent=50.0, min_days_to_cover=10.0
        )

        assert candidates == []

    async def test_calculate_short_interest_change_increasing(
        self, db_session: AsyncSession
    ) -> None:
        """Test short interest change calculation (increasing)."""
        base_date = datetime.now(UTC)

        records = [
            ShortInterest(
                symbol="NVDA",
                report_date=base_date,
                short_interest=60_000_000,
                shares_outstanding=100_000_000,
                short_percent=60.0,
                days_to_cover=5.0,
            ),
            ShortInterest(
                symbol="NVDA",
                report_date=base_date - timedelta(days=15),
                short_interest=50_000_000,
                shares_outstanding=100_000_000,
                short_percent=50.0,
                days_to_cover=4.5,
            ),
        ]

        db_session.add_all(records)
        await db_session.commit()

        tracker = ShortInterestTracker(db_session)
        change = await tracker.calculate_short_interest_change("NVDA")

        assert change.symbol == "NVDA"
        assert change.change_shares > 0
        assert change.change_percent > 0
        assert change.is_increasing
        assert change.is_significant

    async def test_calculate_short_interest_change_decreasing(
        self, db_session: AsyncSession
    ) -> None:
        """Test short interest change calculation (decreasing)."""
        base_date = datetime.now(UTC)

        records = [
            ShortInterest(
                symbol="MSFT",
                report_date=base_date,
                short_interest=40_000_000,
                shares_outstanding=100_000_000,
                short_percent=40.0,
                days_to_cover=4.0,
            ),
            ShortInterest(
                symbol="MSFT",
                report_date=base_date - timedelta(days=15),
                short_interest=50_000_000,
                shares_outstanding=100_000_000,
                short_percent=50.0,
                days_to_cover=5.0,
            ),
        ]

        db_session.add_all(records)
        await db_session.commit()

        tracker = ShortInterestTracker(db_session)
        change = await tracker.calculate_short_interest_change("MSFT")

        assert change.change_shares < 0
        assert change.change_percent < 0
        assert not change.is_increasing
        assert change.is_significant

    async def test_calculate_short_interest_change_single_record(
        self, db_session: AsyncSession
    ) -> None:
        """Test change calculation with single record."""
        base_date = datetime.now(UTC)

        record = ShortInterest(
            symbol="GOOGL",
            report_date=base_date,
            short_interest=30_000_000,
            shares_outstanding=100_000_000,
            short_percent=30.0,
            days_to_cover=3.0,
        )

        db_session.add(record)
        await db_session.commit()

        tracker = ShortInterestTracker(db_session)
        change = await tracker.calculate_short_interest_change("GOOGL")

        assert change.symbol == "GOOGL"
        assert change.previous is None
        assert change.change_shares == 0
        assert change.change_percent == 0.0
        assert not change.is_increasing
        assert not change.is_significant

    async def test_calculate_short_interest_change_no_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test change calculation with no data."""
        tracker = ShortInterestTracker(db_session)

        with pytest.raises(ValueError, match="No short interest data"):
            await tracker.calculate_short_interest_change("XYZ")

    async def test_get_most_shorted_stocks(
        self, db_session: AsyncSession
    ) -> None:
        """Test getting most shorted stocks."""
        base_date = datetime.now(UTC)

        records = [
            ShortInterest(
                symbol=f"STOCK{i}",
                report_date=base_date - timedelta(days=5),
                short_interest=(100 - i) * 1_000_000,
                shares_outstanding=100_000_000,
                short_percent=float(100 - i),
                days_to_cover=5.0,
            )
            for i in range(30)
        ]

        db_session.add_all(records)
        await db_session.commit()

        tracker = ShortInterestTracker(db_session)
        most_shorted = await tracker.get_most_shorted_stocks(top_n=10)

        assert len(most_shorted) == 10
        assert most_shorted[0].short_percent >= most_shorted[-1].short_percent
        assert all(s.short_percent > 0 for s in most_shorted)

    async def test_get_most_shorted_stocks_limited(
        self, db_session: AsyncSession
    ) -> None:
        """Test getting most shorted stocks with limited data."""
        base_date = datetime.now(UTC)

        records = [
            ShortInterest(
                symbol=f"STOCK{i}",
                report_date=base_date - timedelta(days=5),
                short_interest=(10 - i) * 1_000_000,
                shares_outstanding=100_000_000,
                short_percent=float(10 - i),
                days_to_cover=3.0,
            )
            for i in range(5)
        ]

        db_session.add_all(records)
        await db_session.commit()

        tracker = ShortInterestTracker(db_session)
        most_shorted = await tracker.get_most_shorted_stocks(top_n=20)

        assert len(most_shorted) == 5

    async def test_get_most_shorted_stocks_no_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test getting most shorted stocks with no data."""
        tracker = ShortInterestTracker(db_session)
        most_shorted = await tracker.get_most_shorted_stocks(top_n=10)

        assert most_shorted == []

    async def test_short_interest_validation(
        self, db_session: AsyncSession
    ) -> None:
        """Test short interest data validation."""
        base_date = datetime.now(UTC)

        record = ShortInterest(
            symbol="VALID",
            report_date=base_date,
            short_interest=25_000_000,
            shares_outstanding=100_000_000,
            short_percent=25.0,
            days_to_cover=4.0,
        )

        db_session.add(record)
        await db_session.commit()

        tracker = ShortInterestTracker(db_session)
        current = await tracker.get_current_short_interest("VALID")

        assert current.short_percent == record.short_percent
        assert current.short_interest == record.short_interest
        assert current.shares_outstanding == record.shares_outstanding
