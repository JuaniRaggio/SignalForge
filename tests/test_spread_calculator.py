"""Comprehensive tests for spread calculator module.

This module tests all aspects of the spread calculation functionality including:
- SpreadMetrics dataclass validation
- SpreadConfig validation
- Corwin-Schultz spread estimator
- SpreadCalculator class
- Spread calculation methods
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.execution.schemas import SpreadMetrics
from signalforge.execution.spread import (
    SpreadCalculator,
    SpreadConfig,
    calculate_corwin_schultz_spread,
)
from signalforge.models.price import Price


class TestSpreadMetrics:
    """Tests for SpreadMetrics dataclass."""

    def test_valid_metrics(self) -> None:
        """Test creating valid SpreadMetrics."""
        metrics = SpreadMetrics(
            symbol="AAPL",
            current_spread_bps=5.0,
            avg_spread_20d_bps=4.5,
            spread_volatility=0.5,
            percentile_rank=60.0,
        )

        assert metrics.symbol == "AAPL"
        assert metrics.current_spread_bps == 5.0
        assert metrics.avg_spread_20d_bps == 4.5
        assert metrics.spread_volatility == 0.5
        assert metrics.percentile_rank == 60.0

    def test_default_boundary_values(self) -> None:
        """Test metrics at boundary values."""
        metrics = SpreadMetrics(
            symbol="AAPL",
            current_spread_bps=0.0,
            avg_spread_20d_bps=0.0,
            spread_volatility=0.0,
            percentile_rank=0.0,
        )

        assert metrics.current_spread_bps == 0.0
        assert metrics.avg_spread_20d_bps == 0.0
        assert metrics.spread_volatility == 0.0
        assert metrics.percentile_rank == 0.0

    def test_invalid_negative_spread_bps(self) -> None:
        """Test error with negative current_spread_bps."""
        with pytest.raises(ValueError):
            SpreadMetrics(
                symbol="AAPL",
                current_spread_bps=-5.0,
                avg_spread_20d_bps=4.5,
                spread_volatility=0.5,
                percentile_rank=60.0,
            )

    def test_invalid_negative_avg_spread(self) -> None:
        """Test error with negative avg_spread_20d_bps."""
        with pytest.raises(ValueError):
            SpreadMetrics(
                symbol="AAPL",
                current_spread_bps=5.0,
                avg_spread_20d_bps=-4.5,
                spread_volatility=0.5,
                percentile_rank=60.0,
            )

    def test_invalid_negative_volatility(self) -> None:
        """Test error with negative spread_volatility."""
        with pytest.raises(ValueError):
            SpreadMetrics(
                symbol="AAPL",
                current_spread_bps=5.0,
                avg_spread_20d_bps=4.5,
                spread_volatility=-0.5,
                percentile_rank=60.0,
            )

    def test_invalid_percentile_rank_too_high(self) -> None:
        """Test error with percentile_rank > 100."""
        with pytest.raises(ValueError):
            SpreadMetrics(
                symbol="AAPL",
                current_spread_bps=5.0,
                avg_spread_20d_bps=4.5,
                spread_volatility=0.5,
                percentile_rank=150.0,
            )

    def test_invalid_percentile_rank_negative(self) -> None:
        """Test error with negative percentile_rank."""
        with pytest.raises(ValueError):
            SpreadMetrics(
                symbol="AAPL",
                current_spread_bps=5.0,
                avg_spread_20d_bps=4.5,
                spread_volatility=0.5,
                percentile_rank=-10.0,
            )

    def test_zero_spread(self) -> None:
        """Test metrics with zero spread (edge case)."""
        metrics = SpreadMetrics(
            symbol="AAPL",
            current_spread_bps=0.0,
            avg_spread_20d_bps=0.0,
            spread_volatility=0.0,
            percentile_rank=50.0,
        )

        assert metrics.current_spread_bps == 0.0
        assert metrics.avg_spread_20d_bps == 0.0

    def test_max_percentile(self) -> None:
        """Test metrics with max percentile rank."""
        metrics = SpreadMetrics(
            symbol="AAPL",
            current_spread_bps=5.0,
            avg_spread_20d_bps=4.5,
            spread_volatility=0.5,
            percentile_rank=100.0,
        )

        assert metrics.percentile_rank == 100.0


class TestSpreadConfig:
    """Tests for SpreadConfig dataclass."""

    def test_valid_config(self) -> None:
        """Test creating valid SpreadConfig."""
        config = SpreadConfig(
            lookback_days=30,
            min_data_points=10,
        )

        assert config.lookback_days == 30
        assert config.min_data_points == 10

    def test_default_values(self) -> None:
        """Test default config values."""
        config = SpreadConfig()

        assert config.lookback_days == 20
        assert config.min_data_points == 5

    def test_invalid_lookback_days_too_small(self) -> None:
        """Test error with lookback_days < 1."""
        with pytest.raises(ValueError):
            SpreadConfig(lookback_days=0)

    def test_invalid_min_data_points_too_small(self) -> None:
        """Test error with min_data_points < 1."""
        with pytest.raises(ValueError):
            SpreadConfig(min_data_points=0)

    def test_valid_boundary_values(self) -> None:
        """Test config with minimum valid values."""
        config = SpreadConfig(
            lookback_days=1,
            min_data_points=1,
        )

        assert config.lookback_days == 1
        assert config.min_data_points == 1


class TestCalculateCorwinSchultzSpread:
    """Tests for calculate_corwin_schultz_spread function."""

    def test_basic_calculation(self) -> None:
        """Test basic spread calculation."""
        spread = calculate_corwin_schultz_spread(
            high=101.0,
            low=99.0,
            high_prev=100.5,
            low_prev=99.5,
        )

        assert isinstance(spread, float)
        assert spread >= 0

    def test_spread_non_negative(self) -> None:
        """Test that spreads are non-negative."""
        spread = calculate_corwin_schultz_spread(
            high=100.0,
            low=100.0,
            high_prev=100.0,
            low_prev=100.0,
        )

        assert spread >= 0

    def test_tight_spread_produces_small_values(self) -> None:
        """Test that tight spreads produce small spread estimates."""
        # Very tight high-low ranges (5 cent range on $100)
        spread = calculate_corwin_schultz_spread(
            high=100.05,
            low=99.95,
            high_prev=100.05,
            low_prev=99.95,
        )

        # Tight spreads should produce very small estimates
        assert spread < 0.01  # Less than 1%

    def test_wide_spread_produces_large_values(self) -> None:
        """Test that wide spreads produce larger spread estimates."""
        # Wide high-low ranges ($10 range on $50 = 20%)
        spread = calculate_corwin_schultz_spread(
            high=55.0,
            low=45.0,
            high_prev=55.0,
            low_prev=45.0,
        )

        # Wide spreads should produce larger estimates
        assert spread > 0.001  # More than 0.1%

    def test_equal_high_low(self) -> None:
        """Test edge case where high equals low (no spread)."""
        # When high = low, spread should be zero
        spread = calculate_corwin_schultz_spread(
            high=100.0,
            low=100.0,
            high_prev=100.0,
            low_prev=100.0,
        )

        assert spread == 0.0

    def test_different_two_day_ranges(self) -> None:
        """Test with different ranges on two days."""
        spread = calculate_corwin_schultz_spread(
            high=102.0,
            low=98.0,
            high_prev=101.0,
            low_prev=99.0,
        )

        assert isinstance(spread, float)
        assert spread >= 0

    def test_realistic_values(self) -> None:
        """Test with realistic market data."""
        # Typical large-cap stock with 1-2% daily range
        spread = calculate_corwin_schultz_spread(
            high=151.50,
            low=149.00,
            high_prev=150.75,
            low_prev=148.50,
        )

        # Realistic spreads for large-cap stocks: 0.01% - 1%
        assert 0.0 <= spread < 0.10


class TestSpreadCalculator:
    """Tests for SpreadCalculator class."""

    def test_initialization_default(self) -> None:
        """Test calculator initialization with defaults."""
        calculator = SpreadCalculator()

        assert calculator.lookback_days == 20

    def test_initialization_custom_lookback(self) -> None:
        """Test calculator initialization with custom lookback."""
        calculator = SpreadCalculator(lookback_days=30)

        assert calculator.lookback_days == 30

    def test_initialization_invalid_lookback(self) -> None:
        """Test error with invalid lookback days."""
        with pytest.raises(ValueError, match="Lookback days must be at least 1"):
            SpreadCalculator(lookback_days=0)

    def test_calculate_spread_bps_valid(self) -> None:
        """Test spread calculation in basis points."""
        calculator = SpreadCalculator()

        # $2 spread on $100 midpoint = 2% = 200 bps
        spread_bps = calculator.calculate_spread_bps(bid=99.0, ask=101.0)

        assert isinstance(spread_bps, float)
        assert spread_bps == pytest.approx(200.0, rel=0.01)

    def test_calculate_spread_bps_tight(self) -> None:
        """Test tight spread calculation."""
        calculator = SpreadCalculator()

        # $0.01 spread on $100 midpoint = 0.01% = 1 bps
        spread_bps = calculator.calculate_spread_bps(bid=99.995, ask=100.005)

        assert spread_bps == pytest.approx(1.0, rel=0.01)

    def test_calculate_spread_bps_invalid_negative_bid(self) -> None:
        """Test error with negative bid."""
        calculator = SpreadCalculator()

        with pytest.raises(ValueError, match="Bid and ask must be positive"):
            calculator.calculate_spread_bps(bid=-100.0, ask=101.0)

    def test_calculate_spread_bps_invalid_negative_ask(self) -> None:
        """Test error with negative ask."""
        calculator = SpreadCalculator()

        with pytest.raises(ValueError, match="Bid and ask must be positive"):
            calculator.calculate_spread_bps(bid=100.0, ask=-101.0)

    def test_calculate_spread_bps_invalid_zero_bid(self) -> None:
        """Test error with zero bid."""
        calculator = SpreadCalculator()

        with pytest.raises(ValueError, match="Bid and ask must be positive"):
            calculator.calculate_spread_bps(bid=0.0, ask=101.0)

    def test_calculate_spread_bps_invalid_bid_greater_than_ask(self) -> None:
        """Test error when bid > ask."""
        calculator = SpreadCalculator()

        with pytest.raises(ValueError, match="Bid cannot be greater than ask"):
            calculator.calculate_spread_bps(bid=102.0, ask=100.0)

    def test_calculate_metrics_from_data_valid(self) -> None:
        """Test metrics calculation from spread data."""
        calculator = SpreadCalculator()

        spreads = [5.0, 6.0, 4.0, 5.5, 4.5]
        current_spread = 5.0

        metrics = calculator.calculate_metrics_from_data(
            symbol="AAPL",
            spreads=spreads,
            current_spread=current_spread,
        )

        assert isinstance(metrics, SpreadMetrics)
        assert metrics.symbol == "AAPL"
        assert metrics.current_spread_bps == current_spread
        assert metrics.avg_spread_20d_bps == pytest.approx(np.mean(spreads), rel=0.01)
        assert metrics.spread_volatility >= 0
        assert 0 <= metrics.percentile_rank <= 100

    def test_calculate_metrics_from_data_empty_spreads(self) -> None:
        """Test error with empty spreads list."""
        calculator = SpreadCalculator()

        with pytest.raises(ValueError, match="Spreads list cannot be empty"):
            calculator.calculate_metrics_from_data(
                symbol="AAPL",
                spreads=[],
                current_spread=5.0,
            )

    def test_calculate_metrics_from_data_single_spread(self) -> None:
        """Test metrics with single spread value."""
        calculator = SpreadCalculator()

        metrics = calculator.calculate_metrics_from_data(
            symbol="AAPL",
            spreads=[5.0],
            current_spread=5.0,
        )

        assert metrics.current_spread_bps == 5.0
        assert metrics.avg_spread_20d_bps == 5.0
        assert metrics.spread_volatility == 0.0
        assert metrics.percentile_rank == 100.0

    def test_calculate_metrics_percentile_calculation(self) -> None:
        """Test percentile rank calculation."""
        calculator = SpreadCalculator()

        # Current spread is the minimum - should be low percentile
        spreads = [1.0, 2.0, 3.0, 4.0, 5.0]
        current_spread = 1.0

        metrics = calculator.calculate_metrics_from_data(
            symbol="AAPL",
            spreads=spreads,
            current_spread=current_spread,
        )

        assert metrics.percentile_rank == 20.0  # 1 out of 5 <= current

    def test_calculate_metrics_high_percentile(self) -> None:
        """Test high percentile rank when spread is widest."""
        calculator = SpreadCalculator()

        # Current spread is the maximum - should be 100th percentile
        spreads = [1.0, 2.0, 3.0, 4.0, 5.0]
        current_spread = 5.0

        metrics = calculator.calculate_metrics_from_data(
            symbol="AAPL",
            spreads=spreads,
            current_spread=current_spread,
        )

        assert metrics.percentile_rank == 100.0  # All 5 <= current


class TestSpreadCalculatorAsync:
    """Tests for async database functionality."""

    async def test_get_spread_metrics_no_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test get_spread_metrics with no data raises error."""
        calculator = SpreadCalculator()

        with pytest.raises(ValueError, match="Insufficient data"):
            await calculator.get_spread_metrics("AAPL", db_session)

    async def test_get_spread_metrics_insufficient_data(
        self, db_session: AsyncSession
    ) -> None:
        """Test get_spread_metrics with insufficient data raises error."""
        base_time = datetime.now(UTC)

        # Only 3 records - less than minimum of 5
        prices = [
            Price(
                symbol="AAPL",
                timestamp=base_time - timedelta(days=i),
                open=150.0,
                high=152.0,
                low=148.0,
                close=151.0,
                volume=1000000,
            )
            for i in range(3)
        ]

        db_session.add_all(prices)
        await db_session.commit()

        calculator = SpreadCalculator()

        with pytest.raises(ValueError, match="Insufficient data"):
            await calculator.get_spread_metrics("AAPL", db_session)

    async def test_get_spread_metrics_success(
        self, db_session: AsyncSession
    ) -> None:
        """Test successful spread metrics retrieval."""
        base_time = datetime.now(UTC)

        # Create 20 days of price data
        prices = [
            Price(
                symbol="AAPL",
                timestamp=base_time - timedelta(days=i),
                open=150.0 + i * 0.5,
                high=152.0 + i * 0.5,
                low=148.0 + i * 0.5,
                close=151.0 + i * 0.5,
                volume=1000000,
            )
            for i in range(20)
        ]

        db_session.add_all(prices)
        await db_session.commit()

        calculator = SpreadCalculator()
        metrics = await calculator.get_spread_metrics("AAPL", db_session)

        assert isinstance(metrics, SpreadMetrics)
        assert metrics.symbol == "AAPL"
        assert metrics.current_spread_bps >= 0
        assert metrics.avg_spread_20d_bps >= 0
        assert metrics.spread_volatility >= 0
        assert 0 <= metrics.percentile_rank <= 100

    async def test_get_spread_metrics_custom_lookback(
        self, db_session: AsyncSession
    ) -> None:
        """Test spread metrics with custom lookback."""
        base_time = datetime.now(UTC)

        # Create 30 days of price data
        prices = [
            Price(
                symbol="MSFT",
                timestamp=base_time - timedelta(days=i),
                open=300.0,
                high=305.0,
                low=295.0,
                close=302.0,
                volume=2000000,
            )
            for i in range(30)
        ]

        db_session.add_all(prices)
        await db_session.commit()

        calculator = SpreadCalculator(lookback_days=10)
        metrics = await calculator.get_spread_metrics("MSFT", db_session)

        assert metrics.symbol == "MSFT"
        assert metrics.current_spread_bps >= 0

    async def test_get_spread_metrics_different_symbols(
        self, db_session: AsyncSession
    ) -> None:
        """Test spread metrics for different symbols."""
        base_time = datetime.now(UTC)

        # Create data for two symbols with different spreads
        aapl_prices = [
            Price(
                symbol="AAPL",
                timestamp=base_time - timedelta(days=i),
                open=150.0,
                high=151.0,  # Tight range
                low=149.0,
                close=150.5,
                volume=1000000,
            )
            for i in range(10)
        ]

        tsla_prices = [
            Price(
                symbol="TSLA",
                timestamp=base_time - timedelta(days=i),
                open=200.0,
                high=210.0,  # Wide range
                low=190.0,
                close=205.0,
                volume=500000,
            )
            for i in range(10)
        ]

        db_session.add_all(aapl_prices + tsla_prices)
        await db_session.commit()

        calculator = SpreadCalculator()

        aapl_metrics = await calculator.get_spread_metrics("AAPL", db_session)
        tsla_metrics = await calculator.get_spread_metrics("TSLA", db_session)

        assert aapl_metrics.symbol == "AAPL"
        assert tsla_metrics.symbol == "TSLA"
        # TSLA should have wider spreads
        assert tsla_metrics.current_spread_bps > aapl_metrics.current_spread_bps


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_corwin_schultz_very_small_range(self) -> None:
        """Test with extremely tight spreads."""
        # Sub-penny spreads
        spread = calculate_corwin_schultz_spread(
            high=100.001,
            low=99.999,
            high_prev=100.001,
            low_prev=99.999,
        )

        assert spread >= 0
        assert spread < 0.001  # Very small

    def test_corwin_schultz_large_range(self) -> None:
        """Test with very wide spreads."""
        # 50% range
        spread = calculate_corwin_schultz_spread(
            high=150.0,
            low=50.0,
            high_prev=150.0,
            low_prev=50.0,
        )

        assert spread >= 0

    def test_spread_bps_same_bid_ask(self) -> None:
        """Test spread when bid equals ask."""
        calculator = SpreadCalculator()

        spread_bps = calculator.calculate_spread_bps(bid=100.0, ask=100.0)

        assert spread_bps == 0.0

    def test_spread_bps_very_small_difference(self) -> None:
        """Test with sub-penny spread."""
        calculator = SpreadCalculator()

        # $0.001 spread on $100
        spread_bps = calculator.calculate_spread_bps(bid=99.9995, ask=100.0005)

        assert spread_bps == pytest.approx(0.1, rel=0.01)

    def test_spread_bps_large_prices(self) -> None:
        """Test with high-priced stocks."""
        calculator = SpreadCalculator()

        # $10 spread on $5000 midpoint (0.2% = 20 bps)
        spread_bps = calculator.calculate_spread_bps(bid=4995.0, ask=5005.0)

        assert spread_bps == pytest.approx(20.0, rel=0.01)

    def test_metrics_constant_spreads(self) -> None:
        """Test metrics with constant spread values."""
        calculator = SpreadCalculator()

        # All same values
        spreads = [5.0] * 20
        current_spread = 5.0

        metrics = calculator.calculate_metrics_from_data(
            symbol="CONST",
            spreads=spreads,
            current_spread=current_spread,
        )

        assert metrics.avg_spread_20d_bps == 5.0
        assert metrics.spread_volatility == 0.0
        assert metrics.percentile_rank == 100.0

    def test_metrics_many_spreads(self) -> None:
        """Test metrics with large number of spread values."""
        calculator = SpreadCalculator()

        # Many spreads with some variance
        np.random.seed(42)
        spreads = [float(5.0 + np.random.random()) for _ in range(1000)]
        current_spread = spreads[0]

        metrics = calculator.calculate_metrics_from_data(
            symbol="LARGE",
            spreads=spreads,
            current_spread=current_spread,
        )

        assert metrics.avg_spread_20d_bps == pytest.approx(np.mean(spreads), rel=0.01)
        assert metrics.spread_volatility == pytest.approx(np.std(spreads), rel=0.01)


class TestIntegration:
    """Integration tests for complete spread calculation workflows."""

    async def test_end_to_end_workflow(self, db_session: AsyncSession) -> None:
        """Test complete workflow: init -> fetch -> calculate."""
        base_time = datetime.now(UTC)

        # Create realistic price data with varying spreads
        prices = []
        for i in range(30):
            base_price = 150.0 + np.sin(i * 0.2) * 5  # Some variation
            spread = 2.0 + np.random.random()  # Variable spread
            prices.append(
                Price(
                    symbol="AAPL",
                    timestamp=base_time - timedelta(days=i),
                    open=base_price,
                    high=base_price + spread,
                    low=base_price - spread,
                    close=base_price + 0.5,
                    volume=1000000 + i * 10000,
                )
            )

        db_session.add_all(prices)
        await db_session.commit()

        # Initialize calculator
        calculator = SpreadCalculator(lookback_days=20)

        # Get spread metrics
        metrics = await calculator.get_spread_metrics("AAPL", db_session)

        # Verify results
        assert metrics.symbol == "AAPL"
        assert metrics.current_spread_bps > 0
        assert metrics.avg_spread_20d_bps > 0
        assert metrics.spread_volatility >= 0
        assert 0 <= metrics.percentile_rank <= 100

    async def test_multiple_symbols_workflow(
        self, db_session: AsyncSession
    ) -> None:
        """Test calculating spreads for multiple symbols."""
        base_time = datetime.now(UTC)
        symbols = ["AAPL", "MSFT", "GOOGL"]

        # Create data for each symbol
        all_prices = []
        for symbol in symbols:
            base_price = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 100.0}[symbol]
            for i in range(10):
                all_prices.append(
                    Price(
                        symbol=symbol,
                        timestamp=base_time - timedelta(days=i),
                        open=base_price,
                        high=base_price + 2.0,
                        low=base_price - 2.0,
                        close=base_price + 1.0,
                        volume=1000000,
                    )
                )

        db_session.add_all(all_prices)
        await db_session.commit()

        calculator = SpreadCalculator()
        results = {}

        for symbol in symbols:
            metrics = await calculator.get_spread_metrics(symbol, db_session)
            results[symbol] = metrics

        # All should have valid metrics
        assert len(results) == 3
        for symbol, metrics in results.items():
            assert metrics.symbol == symbol
            assert metrics.current_spread_bps > 0

    def test_corwin_schultz_formula_consistency(self) -> None:
        """Test Corwin-Schultz formula produces consistent results."""
        # Same inputs should produce same outputs
        inputs = (105.0, 95.0, 104.0, 96.0)

        spread1 = calculate_corwin_schultz_spread(*inputs)
        spread2 = calculate_corwin_schultz_spread(*inputs)

        assert spread1 == spread2

    def test_spread_calculator_reusable(self) -> None:
        """Test that calculator can be reused for multiple calculations."""
        calculator = SpreadCalculator()

        # Multiple spread bps calculations
        results = []
        for i in range(10):
            bid = 100.0 - i * 0.1
            ask = 100.0 + i * 0.1
            if bid < ask:
                results.append(calculator.calculate_spread_bps(bid=bid, ask=ask))

        # All should be valid
        assert all(r >= 0 for r in results)
        # Larger differences should give larger spreads
        assert results[-1] > results[0]
