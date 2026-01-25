"""Comprehensive tests for execution quality module.

This module tests all aspects of the execution quality functionality including:
- Liquidity assessment and scoring
- Slippage estimation and risk classification
- Edge cases and error handling
- Redis caching functionality
- Integration with realistic market scenarios
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import polars as pl
import pytest
import pytest_asyncio

from signalforge.execution.liquidity import (
    HIGH_LIQUIDITY_THRESHOLD,
    MEDIUM_LIQUIDITY_THRESHOLD,
    LiquidityMetrics,
    assess_liquidity,
    calculate_avg_daily_volume,
    calculate_liquidity_score,
    get_cached_liquidity_metrics,
)
from signalforge.execution.slippage import (
    LOW_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    SlippageEstimate,
    calculate_execution_risk,
    estimate_slippage,
)


@pytest.fixture
def sample_price_data() -> pl.DataFrame:
    """Create sample price data for testing.

    Returns 30 days of realistic OHLCV data with moderate volume.
    """
    n_days = 30
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    data = {
        "timestamp": dates,
        "open": [150.0 + i * 0.5 for i in range(n_days)],
        "high": [155.0 + i * 0.5 for i in range(n_days)],
        "low": [148.0 + i * 0.5 for i in range(n_days)],
        "close": [152.0 + i * 0.5 for i in range(n_days)],
        "volume": [5_000_000 + i * 10_000 for i in range(n_days)],
    }

    return pl.DataFrame(data)


@pytest.fixture
def high_volume_data() -> pl.DataFrame:
    """Create high volume data for testing liquid assets."""
    n_days = 30
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    data = {
        "timestamp": dates,
        "close": [100.0] * n_days,
        "volume": [50_000_000] * n_days,  # Very high volume
    }

    return pl.DataFrame(data)


@pytest.fixture
def low_volume_data() -> pl.DataFrame:
    """Create low volume data for testing illiquid assets."""
    n_days = 30
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    data = {
        "timestamp": dates,
        "close": [10.0] * n_days,
        "volume": [10_000] * n_days,  # Very low volume
    }

    return pl.DataFrame(data)


@pytest.fixture
def volatile_volume_data() -> pl.DataFrame:
    """Create data with volatile (inconsistent) volume."""
    n_days = 30
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_days)]

    # Alternating high and low volume
    volumes = [1_000_000 if i % 2 == 0 else 10_000_000 for i in range(n_days)]

    data = {
        "timestamp": dates,
        "close": [100.0] * n_days,
        "volume": volumes,
    }

    return pl.DataFrame(data)


class TestLiquidityMetrics:
    """Tests for LiquidityMetrics dataclass."""

    def test_valid_metrics(self) -> None:
        """Test creating valid LiquidityMetrics."""
        metrics = LiquidityMetrics(
            symbol="AAPL",
            avg_daily_volume=5_000_000.0,
            volume_volatility=500_000.0,
            liquidity_score=75.5,
            is_liquid=True,
        )

        assert metrics.symbol == "AAPL"
        assert metrics.avg_daily_volume == 5_000_000.0
        assert metrics.volume_volatility == 500_000.0
        assert metrics.liquidity_score == 75.5
        assert metrics.is_liquid is True

    def test_invalid_avg_daily_volume(self) -> None:
        """Test error with negative avg_daily_volume."""
        with pytest.raises(ValueError, match="avg_daily_volume cannot be negative"):
            LiquidityMetrics(
                symbol="AAPL",
                avg_daily_volume=-1000.0,
                volume_volatility=500_000.0,
                liquidity_score=75.5,
                is_liquid=True,
            )

    def test_invalid_volume_volatility(self) -> None:
        """Test error with negative volume_volatility."""
        with pytest.raises(ValueError, match="volume_volatility cannot be negative"):
            LiquidityMetrics(
                symbol="AAPL",
                avg_daily_volume=5_000_000.0,
                volume_volatility=-500_000.0,
                liquidity_score=75.5,
                is_liquid=True,
            )

    def test_invalid_liquidity_score_too_high(self) -> None:
        """Test error with liquidity_score > 100."""
        with pytest.raises(ValueError, match="liquidity_score must be between 0 and 100"):
            LiquidityMetrics(
                symbol="AAPL",
                avg_daily_volume=5_000_000.0,
                volume_volatility=500_000.0,
                liquidity_score=150.0,
                is_liquid=True,
            )

    def test_invalid_liquidity_score_negative(self) -> None:
        """Test error with negative liquidity_score."""
        with pytest.raises(ValueError, match="liquidity_score must be between 0 and 100"):
            LiquidityMetrics(
                symbol="AAPL",
                avg_daily_volume=5_000_000.0,
                volume_volatility=500_000.0,
                liquidity_score=-10.0,
                is_liquid=False,
            )

    def test_to_dict(self) -> None:
        """Test converting metrics to dictionary."""
        metrics = LiquidityMetrics(
            symbol="AAPL",
            avg_daily_volume=5_000_000.0,
            volume_volatility=500_000.0,
            liquidity_score=75.5,
            is_liquid=True,
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict["symbol"] == "AAPL"
        assert metrics_dict["avg_daily_volume"] == 5_000_000.0
        assert metrics_dict["volume_volatility"] == 500_000.0
        assert metrics_dict["liquidity_score"] == 75.5
        assert metrics_dict["is_liquid"] is True


class TestCalculateAvgDailyVolume:
    """Tests for calculate_avg_daily_volume function."""

    def test_basic_calculation(self, sample_price_data: pl.DataFrame) -> None:
        """Test basic ADV calculation."""
        adv = calculate_avg_daily_volume(sample_price_data, window=20)

        assert isinstance(adv, float)
        assert adv > 0

    def test_full_window(self) -> None:
        """Test ADV with exact window size."""
        df = pl.DataFrame({"volume": [1_000_000, 2_000_000, 3_000_000, 4_000_000]})

        adv = calculate_avg_daily_volume(df, window=4)

        # Average of [1M, 2M, 3M, 4M] = 2.5M
        assert adv == 2_500_000.0

    def test_partial_window(self) -> None:
        """Test ADV when data is less than window size."""
        df = pl.DataFrame({"volume": [1_000_000, 2_000_000]})

        # Request 20 days but only have 2
        adv = calculate_avg_daily_volume(df, window=20)

        # Should use all available data
        assert adv == 1_500_000.0

    def test_window_larger_than_data(self) -> None:
        """Test ADV with window larger than available data."""
        df = pl.DataFrame({"volume": [5_000_000] * 10})

        adv = calculate_avg_daily_volume(df, window=30)

        # Should use all 10 days available
        assert adv == 5_000_000.0

    def test_missing_volume_column(self) -> None:
        """Test error when volume column is missing."""
        df = pl.DataFrame({"close": [100.0, 101.0, 102.0]})

        with pytest.raises(ValueError, match="must contain 'volume' column"):
            calculate_avg_daily_volume(df)

    def test_empty_dataframe(self) -> None:
        """Test error with empty DataFrame."""
        df = pl.DataFrame({"volume": []})

        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            calculate_avg_daily_volume(df)

    def test_single_row(self) -> None:
        """Test ADV with single row."""
        df = pl.DataFrame({"volume": [3_000_000]})

        adv = calculate_avg_daily_volume(df, window=20)

        assert adv == 3_000_000.0


class TestCalculateLiquidityScore:
    """Tests for calculate_liquidity_score function."""

    def test_high_liquidity(self) -> None:
        """Test score calculation for highly liquid asset."""
        score = calculate_liquidity_score(
            avg_volume=10_000_000,  # 10M shares
            price=100.0,  # $100/share = $1B daily volume
            volume_volatility=500_000,  # Low volatility (5% CV)
        )

        assert score > HIGH_LIQUIDITY_THRESHOLD
        assert 0 <= score <= 100

    def test_medium_liquidity(self) -> None:
        """Test score calculation for medium liquidity asset."""
        score = calculate_liquidity_score(
            avg_volume=1_000_000,  # 1M shares
            price=50.0,  # $50M daily volume
            volume_volatility=200_000,  # Medium volatility (20% CV)
        )

        assert MEDIUM_LIQUIDITY_THRESHOLD <= score <= HIGH_LIQUIDITY_THRESHOLD
        assert 0 <= score <= 100

    def test_low_liquidity(self) -> None:
        """Test score calculation for low liquidity asset."""
        score = calculate_liquidity_score(
            avg_volume=10_000,  # 10K shares
            price=5.0,  # $50K daily volume
            volume_volatility=5_000,  # High volatility (50% CV)
        )

        assert score < MEDIUM_LIQUIDITY_THRESHOLD
        assert 0 <= score <= 100

    def test_zero_volume(self) -> None:
        """Test score with zero volume."""
        score = calculate_liquidity_score(
            avg_volume=0.0,
            price=100.0,
            volume_volatility=0.0,
        )

        assert score == 0.0

    def test_high_volume_volatility(self) -> None:
        """Test score with very high volume volatility."""
        score_low_vol = calculate_liquidity_score(
            avg_volume=5_000_000,
            price=100.0,
            volume_volatility=100_000,  # Low volatility
        )

        score_high_vol = calculate_liquidity_score(
            avg_volume=5_000_000,
            price=100.0,
            volume_volatility=5_000_000,  # High volatility (100% CV)
        )

        # Higher volatility should result in lower score
        assert score_low_vol > score_high_vol

    def test_invalid_negative_volume(self) -> None:
        """Test error with negative avg_volume."""
        with pytest.raises(ValueError, match="avg_volume cannot be negative"):
            calculate_liquidity_score(
                avg_volume=-1000.0,
                price=100.0,
                volume_volatility=100.0,
            )

    def test_invalid_zero_price(self) -> None:
        """Test error with zero price."""
        with pytest.raises(ValueError, match="price must be positive"):
            calculate_liquidity_score(
                avg_volume=1_000_000.0,
                price=0.0,
                volume_volatility=100.0,
            )

    def test_invalid_negative_price(self) -> None:
        """Test error with negative price."""
        with pytest.raises(ValueError, match="price must be positive"):
            calculate_liquidity_score(
                avg_volume=1_000_000.0,
                price=-100.0,
                volume_volatility=100.0,
            )

    def test_invalid_negative_volatility(self) -> None:
        """Test error with negative volume_volatility."""
        with pytest.raises(ValueError, match="volume_volatility cannot be negative"):
            calculate_liquidity_score(
                avg_volume=1_000_000.0,
                price=100.0,
                volume_volatility=-100.0,
            )


class TestAssessLiquidity:
    """Tests for assess_liquidity function."""

    def test_basic_assessment(self, sample_price_data: pl.DataFrame) -> None:
        """Test basic liquidity assessment."""
        metrics = assess_liquidity(sample_price_data, "AAPL")

        assert isinstance(metrics, LiquidityMetrics)
        assert metrics.symbol == "AAPL"
        assert metrics.avg_daily_volume > 0
        assert metrics.volume_volatility >= 0
        assert 0 <= metrics.liquidity_score <= 100
        assert isinstance(metrics.is_liquid, bool)

    def test_high_volume_assessment(self, high_volume_data: pl.DataFrame) -> None:
        """Test assessment of highly liquid asset."""
        metrics = assess_liquidity(high_volume_data, "AAPL")

        assert metrics.liquidity_score > HIGH_LIQUIDITY_THRESHOLD
        assert metrics.is_liquid is True

    def test_low_volume_assessment(self, low_volume_data: pl.DataFrame) -> None:
        """Test assessment of illiquid asset."""
        metrics = assess_liquidity(low_volume_data, "PENNY")

        assert metrics.liquidity_score < MEDIUM_LIQUIDITY_THRESHOLD
        assert metrics.is_liquid is False

    def test_volatile_volume_assessment(self, volatile_volume_data: pl.DataFrame) -> None:
        """Test assessment with inconsistent volume."""
        metrics = assess_liquidity(volatile_volume_data, "VOLATILE")

        # High volatility should reduce the score
        assert metrics.volume_volatility > 0

    def test_custom_window(self, sample_price_data: pl.DataFrame) -> None:
        """Test assessment with custom window size."""
        metrics_20 = assess_liquidity(sample_price_data, "AAPL", window=20)
        metrics_10 = assess_liquidity(sample_price_data, "AAPL", window=10)

        # Both should return valid metrics but may differ
        assert metrics_20.avg_daily_volume > 0
        assert metrics_10.avg_daily_volume > 0

    def test_missing_columns(self) -> None:
        """Test error with missing required columns."""
        df = pl.DataFrame({"volume": [1_000_000]})

        with pytest.raises(ValueError, match="Missing required columns"):
            assess_liquidity(df, "AAPL")

    def test_empty_dataframe(self) -> None:
        """Test error with empty DataFrame."""
        df = pl.DataFrame(
            {
                "timestamp": [],
                "close": [],
                "volume": [],
            }
        )

        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            assess_liquidity(df, "AAPL")


class TestGetCachedLiquidityMetrics:
    """Tests for get_cached_liquidity_metrics function."""

    @pytest_asyncio.fixture
    def mock_redis(self) -> MagicMock:
        """Create a mock Redis client."""
        mock = MagicMock()
        mock.get = AsyncMock(return_value=None)
        mock.setex = AsyncMock(return_value=True)
        return mock

    @pytest.mark.asyncio
    async def test_cache_miss(
        self,
        sample_price_data: pl.DataFrame,
        mock_redis: MagicMock,
    ) -> None:
        """Test behavior when cache is empty."""
        mock_redis.get.return_value = None

        metrics = await get_cached_liquidity_metrics(
            sample_price_data,
            "AAPL",
            mock_redis,
        )

        # Should calculate and cache
        assert isinstance(metrics, LiquidityMetrics)
        assert metrics.symbol == "AAPL"
        mock_redis.get.assert_called_once()
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit(
        self,
        sample_price_data: pl.DataFrame,
        mock_redis: MagicMock,
    ) -> None:
        """Test behavior when cache contains valid data."""
        import json

        cached_metrics = {
            "symbol": "AAPL",
            "avg_daily_volume": 5_000_000.0,
            "volume_volatility": 500_000.0,
            "liquidity_score": 75.5,
            "is_liquid": True,
        }
        mock_redis.get.return_value = json.dumps(cached_metrics)

        metrics = await get_cached_liquidity_metrics(
            sample_price_data,
            "AAPL",
            mock_redis,
        )

        # Should return cached data without recalculating
        assert metrics.symbol == "AAPL"
        assert metrics.avg_daily_volume == 5_000_000.0
        assert metrics.liquidity_score == 75.5
        mock_redis.get.assert_called_once()
        mock_redis.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_force_refresh(
        self,
        sample_price_data: pl.DataFrame,
        mock_redis: MagicMock,
    ) -> None:
        """Test force refresh bypasses cache."""
        import json

        cached_metrics = {
            "symbol": "AAPL",
            "avg_daily_volume": 1_000_000.0,
            "volume_volatility": 100_000.0,
            "liquidity_score": 50.0,
            "is_liquid": True,
        }
        mock_redis.get.return_value = json.dumps(cached_metrics)

        metrics = await get_cached_liquidity_metrics(
            sample_price_data,
            "AAPL",
            mock_redis,
            force_refresh=True,
        )

        # Should recalculate even with cached data
        assert isinstance(metrics, LiquidityMetrics)
        # Should not have called get since we forced refresh
        mock_redis.get.assert_not_called()
        mock_redis.setex.assert_called_once()


class TestSlippageEstimate:
    """Tests for SlippageEstimate dataclass."""

    def test_valid_estimate(self) -> None:
        """Test creating valid SlippageEstimate."""
        estimate = SlippageEstimate(
            symbol="AAPL",
            order_size=100_000.0,
            estimated_slippage_pct=0.15,
            estimated_slippage_usd=150.0,
            adv_ratio=0.02,
            execution_risk="medium",
        )

        assert estimate.symbol == "AAPL"
        assert estimate.order_size == 100_000.0
        assert estimate.estimated_slippage_pct == 0.15
        assert estimate.estimated_slippage_usd == 150.0
        assert estimate.adv_ratio == 0.02
        assert estimate.execution_risk == "medium"

    def test_invalid_negative_order_size(self) -> None:
        """Test error with negative order_size."""
        with pytest.raises(ValueError, match="order_size cannot be negative"):
            SlippageEstimate(
                symbol="AAPL",
                order_size=-1000.0,
                estimated_slippage_pct=0.15,
                estimated_slippage_usd=150.0,
                adv_ratio=0.02,
                execution_risk="medium",
            )

    def test_invalid_negative_slippage_pct(self) -> None:
        """Test error with negative slippage percentage."""
        with pytest.raises(ValueError, match="estimated_slippage_pct cannot be negative"):
            SlippageEstimate(
                symbol="AAPL",
                order_size=100_000.0,
                estimated_slippage_pct=-0.15,
                estimated_slippage_usd=150.0,
                adv_ratio=0.02,
                execution_risk="medium",
            )

    def test_invalid_risk_level(self) -> None:
        """Test error with invalid execution_risk."""
        with pytest.raises(ValueError, match="execution_risk must be"):
            SlippageEstimate(
                symbol="AAPL",
                order_size=100_000.0,
                estimated_slippage_pct=0.15,
                estimated_slippage_usd=150.0,
                adv_ratio=0.02,
                execution_risk="extreme",  # type: ignore
            )

    def test_to_dict(self) -> None:
        """Test converting estimate to dictionary."""
        estimate = SlippageEstimate(
            symbol="AAPL",
            order_size=100_000.0,
            estimated_slippage_pct=0.15,
            estimated_slippage_usd=150.0,
            adv_ratio=0.02,
            execution_risk="medium",
        )

        estimate_dict = estimate.to_dict()

        assert estimate_dict["symbol"] == "AAPL"
        assert estimate_dict["order_size"] == 100_000.0
        assert estimate_dict["estimated_slippage_pct"] == 0.15
        assert estimate_dict["estimated_slippage_usd"] == 150.0
        assert estimate_dict["adv_ratio"] == 0.02
        assert estimate_dict["execution_risk"] == "medium"


class TestCalculateExecutionRisk:
    """Tests for calculate_execution_risk function."""

    def test_low_risk(self) -> None:
        """Test low risk classification."""
        risk = calculate_execution_risk(0.005)  # 0.5% of ADV

        assert risk == "low"

    def test_low_risk_boundary(self) -> None:
        """Test boundary for low risk."""
        risk = calculate_execution_risk(LOW_RISK_THRESHOLD - 0.001)

        assert risk == "low"

    def test_medium_risk(self) -> None:
        """Test medium risk classification."""
        risk = calculate_execution_risk(0.03)  # 3% of ADV

        assert risk == "medium"

    def test_medium_risk_lower_boundary(self) -> None:
        """Test lower boundary for medium risk."""
        risk = calculate_execution_risk(LOW_RISK_THRESHOLD)

        assert risk == "medium"

    def test_medium_risk_upper_boundary(self) -> None:
        """Test upper boundary for medium risk."""
        risk = calculate_execution_risk(MEDIUM_RISK_THRESHOLD - 0.001)

        assert risk == "medium"

    def test_high_risk(self) -> None:
        """Test high risk classification."""
        risk = calculate_execution_risk(0.1)  # 10% of ADV

        assert risk == "high"

    def test_high_risk_boundary(self) -> None:
        """Test boundary for high risk."""
        risk = calculate_execution_risk(MEDIUM_RISK_THRESHOLD)

        assert risk == "high"

    def test_very_high_ratio(self) -> None:
        """Test with very high ADV ratio."""
        risk = calculate_execution_risk(0.5)  # 50% of ADV

        assert risk == "high"

    def test_zero_ratio(self) -> None:
        """Test with zero ADV ratio."""
        risk = calculate_execution_risk(0.0)

        assert risk == "low"

    def test_negative_ratio(self) -> None:
        """Test error with negative ADV ratio."""
        with pytest.raises(ValueError, match="adv_ratio cannot be negative"):
            calculate_execution_risk(-0.01)


class TestEstimateSlippage:
    """Tests for estimate_slippage function."""

    def test_small_order_low_risk(self) -> None:
        """Test slippage for small order (low risk)."""
        estimate = estimate_slippage(
            order_size_usd=10_000,
            avg_daily_volume=10_000_000,
            current_price=100.0,
            volatility=0.02,
        )

        assert estimate.execution_risk == "low"
        assert estimate.estimated_slippage_pct < 0.5  # Less than 0.5%
        assert estimate.estimated_slippage_usd < 100  # Less than $100

    def test_medium_order_medium_risk(self) -> None:
        """Test slippage for medium order (medium risk)."""
        # ADV in dollars = 1M shares * $100 = $100M
        # For medium risk (1-5% of ADV), need $1M-$5M order
        # Using $2M order = 2% of ADV
        estimate = estimate_slippage(
            order_size_usd=2_000_000,
            avg_daily_volume=1_000_000,
            current_price=100.0,
            volatility=0.02,
        )

        assert estimate.execution_risk == "medium"
        assert estimate.estimated_slippage_pct > 0
        assert estimate.estimated_slippage_usd > 0

    def test_large_order_high_risk(self) -> None:
        """Test slippage for large order (high risk)."""
        # ADV in dollars = 1M shares * $100 = $100M
        # For high risk (>5% of ADV), need >$5M order
        # Using $10M order = 10% of ADV
        estimate = estimate_slippage(
            order_size_usd=10_000_000,
            avg_daily_volume=1_000_000,
            current_price=100.0,
            volatility=0.02,
        )

        assert estimate.execution_risk == "high"
        assert estimate.estimated_slippage_pct > 0.01  # More than 0.01% (1 bps)

    def test_high_volatility_increases_slippage(self) -> None:
        """Test that higher volatility increases slippage."""
        estimate_low_vol = estimate_slippage(
            order_size_usd=100_000,
            avg_daily_volume=5_000_000,
            current_price=50.0,
            volatility=0.01,  # 1% volatility
        )

        estimate_high_vol = estimate_slippage(
            order_size_usd=100_000,
            avg_daily_volume=5_000_000,
            current_price=50.0,
            volatility=0.04,  # 4% volatility
        )

        # Higher volatility should result in higher slippage
        assert estimate_high_vol.estimated_slippage_pct > estimate_low_vol.estimated_slippage_pct

    def test_zero_order_size(self) -> None:
        """Test slippage with zero order size."""
        estimate = estimate_slippage(
            order_size_usd=0.0,
            avg_daily_volume=10_000_000,
            current_price=100.0,
            volatility=0.02,
        )

        assert estimate.order_size == 0.0
        assert estimate.estimated_slippage_pct == 0.0
        assert estimate.estimated_slippage_usd == 0.0
        assert estimate.adv_ratio == 0.0
        assert estimate.execution_risk == "low"

    def test_with_symbol(self) -> None:
        """Test estimate with symbol tracking."""
        estimate = estimate_slippage(
            order_size_usd=50_000,
            avg_daily_volume=5_000_000,
            current_price=100.0,
            volatility=0.02,
            symbol="AAPL",
        )

        assert estimate.symbol == "AAPL"

    def test_adv_ratio_calculation(self) -> None:
        """Test ADV ratio is calculated correctly."""
        estimate = estimate_slippage(
            order_size_usd=100_000,
            avg_daily_volume=1_000_000,
            current_price=100.0,
            volatility=0.02,
        )

        # ADV in dollars = 1M shares * $100 = $100M
        # Order / ADV = $100K / $100M = 0.001
        assert abs(estimate.adv_ratio - 0.001) < 0.0001

    def test_invalid_negative_order_size(self) -> None:
        """Test error with negative order size."""
        with pytest.raises(ValueError, match="order_size_usd cannot be negative"):
            estimate_slippage(
                order_size_usd=-1000,
                avg_daily_volume=10_000_000,
                current_price=100.0,
            )

    def test_invalid_zero_volume(self) -> None:
        """Test error with zero average daily volume."""
        with pytest.raises(ValueError, match="avg_daily_volume must be positive"):
            estimate_slippage(
                order_size_usd=10_000,
                avg_daily_volume=0,
                current_price=100.0,
            )

    def test_invalid_negative_volume(self) -> None:
        """Test error with negative average daily volume."""
        with pytest.raises(ValueError, match="avg_daily_volume must be positive"):
            estimate_slippage(
                order_size_usd=10_000,
                avg_daily_volume=-1_000_000,
                current_price=100.0,
            )

    def test_invalid_zero_price(self) -> None:
        """Test error with zero price."""
        with pytest.raises(ValueError, match="current_price must be positive"):
            estimate_slippage(
                order_size_usd=10_000,
                avg_daily_volume=10_000_000,
                current_price=0.0,
            )

    def test_invalid_negative_price(self) -> None:
        """Test error with negative price."""
        with pytest.raises(ValueError, match="current_price must be positive"):
            estimate_slippage(
                order_size_usd=10_000,
                avg_daily_volume=10_000_000,
                current_price=-100.0,
            )

    def test_invalid_negative_volatility(self) -> None:
        """Test error with negative volatility."""
        with pytest.raises(ValueError, match="volatility cannot be negative"):
            estimate_slippage(
                order_size_usd=10_000,
                avg_daily_volume=10_000_000,
                current_price=100.0,
                volatility=-0.02,
            )


class TestIntegration:
    """Integration tests for complete execution quality workflows."""

    def test_end_to_end_liquidity_and_slippage(self, sample_price_data: pl.DataFrame) -> None:
        """Test complete workflow: assess liquidity then estimate slippage."""
        # Step 1: Assess liquidity
        liquidity = assess_liquidity(sample_price_data, "AAPL")

        assert liquidity.is_liquid  # Should be liquid with sample data

        # Step 2: Use liquidity metrics for slippage estimation
        current_price = float(sample_price_data["close"][-1])
        order_size_usd = 50_000

        slippage = estimate_slippage(
            order_size_usd=order_size_usd,
            avg_daily_volume=liquidity.avg_daily_volume,
            current_price=current_price,
            volatility=0.02,
            symbol=liquidity.symbol,
        )

        # Verify results are consistent
        assert slippage.symbol == liquidity.symbol
        assert slippage.order_size == order_size_usd
        assert slippage.execution_risk in ("low", "medium", "high")

    def test_illiquid_asset_workflow(self, low_volume_data: pl.DataFrame) -> None:
        """Test workflow with illiquid asset."""
        # Assess liquidity
        liquidity = assess_liquidity(low_volume_data, "ILLIQUID")

        # Should be flagged as illiquid
        assert not liquidity.is_liquid
        assert liquidity.liquidity_score < MEDIUM_LIQUIDITY_THRESHOLD

        # Estimate slippage for same order size
        current_price = float(low_volume_data["close"][-1])
        slippage = estimate_slippage(
            order_size_usd=10_000,
            avg_daily_volume=liquidity.avg_daily_volume,
            current_price=current_price,
            symbol="ILLIQUID",
        )

        # Should have higher risk due to low volume
        # With low ADV, even small orders have higher relative impact
        assert slippage.adv_ratio > 0

    def test_liquid_asset_workflow(self, high_volume_data: pl.DataFrame) -> None:
        """Test workflow with highly liquid asset."""
        # Assess liquidity
        liquidity = assess_liquidity(high_volume_data, "LIQUID")

        # Should be highly liquid
        assert liquidity.is_liquid
        assert liquidity.liquidity_score > HIGH_LIQUIDITY_THRESHOLD

        # Estimate slippage
        current_price = float(high_volume_data["close"][-1])
        slippage = estimate_slippage(
            order_size_usd=100_000,
            avg_daily_volume=liquidity.avg_daily_volume,
            current_price=current_price,
            symbol="LIQUID",
        )

        # Should have low execution risk with high volume
        assert slippage.execution_risk == "low"
        assert slippage.estimated_slippage_pct < 0.5  # Less than 0.5%
