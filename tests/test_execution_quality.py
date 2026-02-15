"""Comprehensive tests for execution quality module.

This module tests all aspects of the execution quality functionality including:
- Liquidity scoring with various asset types
- Slippage estimation with square-root market impact model
- Spread calculation and metrics
- Volume filtering and position sizing
- Edge cases and error handling
"""

from datetime import datetime

import numpy as np
import pytest

from signalforge.execution import (
    LiquidityScore,
    LiquidityScorer,
    SlippageEstimate,
    SlippageEstimator,
    SpreadCalculator,
    SpreadMetrics,
    VolumeFilter,
    VolumeFilterResult,
)


class TestLiquidityScore:
    """Tests for LiquidityScore Pydantic schema."""

    def test_valid_score(self) -> None:
        """Test creating valid LiquidityScore."""
        score = LiquidityScore(
            symbol="AAPL",
            timestamp=datetime(2024, 1, 1),
            score=75.5,
            volume_score=80.0,
            spread_score=70.0,
            market_cap_score=90.0,
            adv_20=5_000_000.0,
            rating="good",
        )

        assert score.symbol == "AAPL"
        assert score.score == 75.5
        assert score.volume_score == 80.0
        assert score.rating == "good"

    def test_score_out_of_range_high(self) -> None:
        """Test validation fails for score > 100."""
        with pytest.raises(ValueError):
            LiquidityScore(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 1),
                score=150.0,
                volume_score=80.0,
                spread_score=70.0,
                market_cap_score=90.0,
                adv_20=5_000_000.0,
                rating="excellent",
            )

    def test_score_negative(self) -> None:
        """Test validation fails for negative score."""
        with pytest.raises(ValueError):
            LiquidityScore(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 1),
                score=-10.0,
                volume_score=80.0,
                spread_score=70.0,
                market_cap_score=90.0,
                adv_20=5_000_000.0,
                rating="poor",
            )

    def test_negative_adv(self) -> None:
        """Test validation fails for negative ADV."""
        with pytest.raises(ValueError):
            LiquidityScore(
                symbol="AAPL",
                timestamp=datetime(2024, 1, 1),
                score=75.0,
                volume_score=80.0,
                spread_score=70.0,
                market_cap_score=90.0,
                adv_20=-1000.0,
                rating="good",
            )


class TestLiquidityScorer:
    """Tests for LiquidityScorer class."""

    def test_initialization(self) -> None:
        """Test scorer initialization with default weights."""
        scorer = LiquidityScorer()

        assert scorer.volume_weight == 0.4
        assert scorer.spread_weight == 0.3
        assert scorer.market_cap_weight == 0.2
        assert scorer.depth_weight == 0.1

    def test_custom_weights(self) -> None:
        """Test scorer with custom weights."""
        scorer = LiquidityScorer(
            volume_weight=0.5, spread_weight=0.3, market_cap_weight=0.15, depth_weight=0.05
        )

        assert scorer.volume_weight == 0.5
        assert scorer.spread_weight == 0.3
        assert scorer.market_cap_weight == 0.15
        assert scorer.depth_weight == 0.05

    def test_weights_validation(self) -> None:
        """Test weights must sum to 1.0."""
        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            LiquidityScorer(
                volume_weight=0.5, spread_weight=0.3, market_cap_weight=0.3, depth_weight=0.1
            )

    def test_score_from_data_high_volume(self) -> None:
        """Test scoring with high volume data."""
        scorer = LiquidityScorer()

        score = scorer.score_from_data(
            symbol="AAPL",
            adv_20=50_000_000,  # 50M shares (very high)
            avg_spread_bps=3.0,  # 3 bps (tight)
            market_cap=3_000_000_000_000,  # $3T
        )

        assert score.score > 80  # Should be excellent
        assert score.rating == "excellent"
        assert score.volume_score == 100.0  # > 10M shares
        assert score.spread_score == 100.0  # < 5 bps

    def test_score_from_data_low_volume(self) -> None:
        """Test scoring with low volume data."""
        scorer = LiquidityScorer()

        score = scorer.score_from_data(
            symbol="PENNY",
            adv_20=5_000,  # 5K shares (very low)
            avg_spread_bps=80.0,  # 80 bps (wide)
            market_cap=20_000_000,  # $20M (nano cap)
        )

        assert score.score < 40  # Should be poor/illiquid
        assert score.rating in ("poor", "illiquid")
        assert score.volume_score == 20.0  # <= 10K shares
        assert score.spread_score == 20.0  # >= 50 bps

    def test_volume_score_boundaries(self) -> None:
        """Test volume score at boundaries."""
        scorer = LiquidityScorer()

        assert scorer._calculate_volume_score(15_000_000) == 100.0
        assert scorer._calculate_volume_score(10_000_001) == 100.0
        assert scorer._calculate_volume_score(10_000_000) == 80.0
        assert scorer._calculate_volume_score(1_500_000) == 80.0
        assert scorer._calculate_volume_score(1_000_000) == 60.0
        assert scorer._calculate_volume_score(150_000) == 60.0
        assert scorer._calculate_volume_score(100_000) == 40.0
        assert scorer._calculate_volume_score(15_000) == 40.0
        assert scorer._calculate_volume_score(10_000) == 20.0
        assert scorer._calculate_volume_score(5_000) == 20.0

    def test_spread_score_boundaries(self) -> None:
        """Test spread score at boundaries."""
        scorer = LiquidityScorer()

        assert scorer._calculate_spread_score(3.0) == 100.0
        assert scorer._calculate_spread_score(4.99) == 100.0
        assert scorer._calculate_spread_score(7.0) == 80.0
        assert scorer._calculate_spread_score(9.99) == 80.0
        assert scorer._calculate_spread_score(15.0) == 60.0
        assert scorer._calculate_spread_score(24.99) == 60.0
        assert scorer._calculate_spread_score(35.0) == 40.0
        assert scorer._calculate_spread_score(49.99) == 40.0
        assert scorer._calculate_spread_score(75.0) == 20.0

    def test_market_cap_score(self) -> None:
        """Test market cap scoring."""
        scorer = LiquidityScorer()

        assert scorer._calculate_market_cap_score(15_000_000_000) == 100.0  # Large cap
        assert scorer._calculate_market_cap_score(5_000_000_000) == 80.0  # Mid cap
        assert scorer._calculate_market_cap_score(500_000_000) == 60.0  # Small cap
        assert scorer._calculate_market_cap_score(100_000_000) == 40.0  # Micro cap
        assert scorer._calculate_market_cap_score(30_000_000) == 20.0  # Nano cap

    def test_rating_conversion(self) -> None:
        """Test score to rating conversion."""
        scorer = LiquidityScorer()

        assert scorer._get_rating(90.0) == "excellent"
        assert scorer._get_rating(80.0) == "excellent"
        assert scorer._get_rating(70.0) == "good"
        assert scorer._get_rating(60.0) == "good"
        assert scorer._get_rating(50.0) == "fair"
        assert scorer._get_rating(40.0) == "fair"
        assert scorer._get_rating(30.0) == "poor"
        assert scorer._get_rating(20.0) == "poor"
        assert scorer._get_rating(10.0) == "illiquid"

    def test_score_bounds(self) -> None:
        """Test score is always within 0-100."""
        scorer = LiquidityScorer()

        # Test extreme values
        score1 = scorer.score_from_data("TEST", adv_20=0, avg_spread_bps=1000, market_cap=1000)
        assert 0 <= score1.score <= 100

        score2 = scorer.score_from_data(
            "TEST", adv_20=1_000_000_000, avg_spread_bps=0.1, market_cap=10_000_000_000_000
        )
        assert 0 <= score2.score <= 100


class TestSlippageEstimate:
    """Tests for SlippageEstimate Pydantic schema."""

    def test_valid_estimate(self) -> None:
        """Test creating valid SlippageEstimate."""
        estimate = SlippageEstimate(
            symbol="AAPL",
            order_size=100_000.0,
            estimated_slippage_bps=15.0,
            estimated_slippage_pct=0.15,
            confidence=0.85,
            market_impact_cost=150.0,
        )

        assert estimate.symbol == "AAPL"
        assert estimate.order_size == 100_000.0
        assert estimate.estimated_slippage_bps == 15.0
        assert estimate.confidence == 0.85

    def test_zero_order_size_schema(self) -> None:
        """Test zero order size is allowed in schema."""
        estimate = SlippageEstimate(
            symbol="AAPL",
            order_size=0.0,
            estimated_slippage_bps=0.0,
            estimated_slippage_pct=0.0,
            confidence=1.0,
            market_impact_cost=0.0,
        )
        assert estimate.order_size == 0.0

    def test_confidence_bounds(self) -> None:
        """Test confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            SlippageEstimate(
                symbol="AAPL",
                order_size=100_000.0,
                estimated_slippage_bps=15.0,
                estimated_slippage_pct=0.15,
                confidence=1.5,
                market_impact_cost=150.0,
            )


class TestSlippageEstimator:
    """Tests for SlippageEstimator class."""

    def test_initialization(self) -> None:
        """Test estimator initialization."""
        estimator = SlippageEstimator()
        assert estimator.impact_coefficient == 0.1

        estimator2 = SlippageEstimator(impact_coefficient=0.2)
        assert estimator2.impact_coefficient == 0.2

    def test_invalid_coefficient(self) -> None:
        """Test negative impact coefficient is rejected."""
        with pytest.raises(ValueError):
            SlippageEstimator(impact_coefficient=-0.1)

    def test_small_order_estimate(self) -> None:
        """Test slippage estimation for small order."""
        estimator = SlippageEstimator()

        estimate = estimator.estimate(
            symbol="AAPL",
            order_size_shares=1000,
            adv_20=10_000_000,
            avg_spread_bps=5.0,
        )

        assert estimate.symbol == "AAPL"
        assert estimate.estimated_slippage_bps > 0
        assert estimate.confidence > 0.9  # High confidence for small order

    def test_large_order_estimate(self) -> None:
        """Test slippage estimation for large order."""
        estimator = SlippageEstimator()

        estimate = estimator.estimate(
            symbol="AAPL",
            order_size_shares=1_000_000,
            adv_20=5_000_000,
            avg_spread_bps=5.0,
        )

        assert estimate.estimated_slippage_bps > 5.0  # Should be > base spread
        assert estimate.confidence < 0.7  # Lower confidence for large order

    def test_zero_order_size(self) -> None:
        """Test zero order size edge case."""
        estimator = SlippageEstimator()

        estimate = estimator.estimate(
            symbol="AAPL", order_size_shares=0, adv_20=10_000_000, avg_spread_bps=5.0
        )

        assert estimate.order_size == 0.0
        assert estimate.estimated_slippage_bps == 0.0
        assert estimate.estimated_slippage_pct == 0.0
        assert estimate.confidence == 1.0

    def test_volatility_impact(self) -> None:
        """Test that higher volatility increases slippage."""
        estimator = SlippageEstimator()

        est_low_vol = estimator.estimate(
            symbol="AAPL",
            order_size_shares=10_000,
            adv_20=10_000_000,
            avg_spread_bps=5.0,
            volatility=0.01,
        )

        est_high_vol = estimator.estimate(
            symbol="AAPL",
            order_size_shares=10_000,
            adv_20=10_000_000,
            avg_spread_bps=5.0,
            volatility=0.05,
        )

        assert est_high_vol.estimated_slippage_bps > est_low_vol.estimated_slippage_bps

    def test_estimate_for_dollar_amount(self) -> None:
        """Test dollar-based estimation."""
        estimator = SlippageEstimator()

        estimate = estimator.estimate_for_dollar_amount(
            symbol="AAPL",
            dollar_amount=100_000,
            price=100.0,
            adv_20=10_000_000,
            avg_spread_bps=5.0,
        )

        assert estimate.order_size == 100_000
        assert estimate.market_impact_cost > 0

    def test_invalid_inputs(self) -> None:
        """Test invalid input validation."""
        estimator = SlippageEstimator()

        with pytest.raises(ValueError):
            estimator.estimate("AAPL", -1000, 10_000_000, 5.0)

        with pytest.raises(ValueError):
            estimator.estimate("AAPL", 1000, 0, 5.0)

        with pytest.raises(ValueError):
            estimator.estimate("AAPL", 1000, 10_000_000, -5.0)


class TestSpreadMetrics:
    """Tests for SpreadMetrics Pydantic schema."""

    def test_valid_metrics(self) -> None:
        """Test creating valid SpreadMetrics."""
        metrics = SpreadMetrics(
            symbol="AAPL",
            current_spread_bps=5.0,
            avg_spread_20d_bps=6.0,
            spread_volatility=1.5,
            percentile_rank=45.0,
        )

        assert metrics.symbol == "AAPL"
        assert metrics.current_spread_bps == 5.0
        assert metrics.avg_spread_20d_bps == 6.0

    def test_negative_spread(self) -> None:
        """Test validation fails for negative spread."""
        with pytest.raises(ValueError):
            SpreadMetrics(
                symbol="AAPL",
                current_spread_bps=-5.0,
                avg_spread_20d_bps=6.0,
                spread_volatility=1.5,
                percentile_rank=45.0,
            )


class TestSpreadCalculator:
    """Tests for SpreadCalculator class."""

    def test_initialization(self) -> None:
        """Test calculator initialization."""
        calc = SpreadCalculator()
        assert calc.lookback_days == 20

        calc2 = SpreadCalculator(lookback_days=30)
        assert calc2.lookback_days == 30

    def test_invalid_lookback(self) -> None:
        """Test invalid lookback days."""
        with pytest.raises(ValueError):
            SpreadCalculator(lookback_days=0)

    def test_calculate_spread_bps(self) -> None:
        """Test spread calculation in basis points."""
        calc = SpreadCalculator()

        spread = calc.calculate_spread_bps(bid=99.0, ask=101.0)

        # Midpoint = 100, spread = 2, percentage = 2/100 = 0.02 = 200 bps
        assert abs(spread - 200.0) < 0.1

    def test_tight_spread(self) -> None:
        """Test tight spread calculation."""
        calc = SpreadCalculator()

        spread = calc.calculate_spread_bps(bid=99.95, ask=100.05)

        # Very tight spread should be around 10 bps
        assert spread < 15.0

    def test_calculate_metrics_from_data(self) -> None:
        """Test metrics calculation from spread history."""
        calc = SpreadCalculator()

        spreads = [5.0, 6.0, 4.0, 7.0, 5.5, 6.5, 5.0, 6.0]
        metrics = calc.calculate_metrics_from_data("AAPL", spreads, current_spread=5.5)

        assert metrics.symbol == "AAPL"
        assert metrics.current_spread_bps == 5.5
        assert metrics.avg_spread_20d_bps == np.mean(spreads)
        assert 0 <= metrics.percentile_rank <= 100

    def test_empty_spreads_list(self) -> None:
        """Test error with empty spreads list."""
        calc = SpreadCalculator()

        with pytest.raises(ValueError, match="Spreads list cannot be empty"):
            calc.calculate_metrics_from_data("AAPL", [], current_spread=5.0)

    def test_invalid_bid_ask(self) -> None:
        """Test validation for invalid bid/ask."""
        calc = SpreadCalculator()

        with pytest.raises(ValueError):
            calc.calculate_spread_bps(bid=-1.0, ask=100.0)

        with pytest.raises(ValueError):
            calc.calculate_spread_bps(bid=100.0, ask=99.0)  # Bid > ask


class TestVolumeFilterResult:
    """Tests for VolumeFilterResult Pydantic schema."""

    def test_valid_result_pass(self) -> None:
        """Test creating valid passing result."""
        result = VolumeFilterResult(
            symbol="AAPL",
            passes_filter=True,
            position_pct_of_adv=0.5,
            max_position_size=500_000.0,
            reason=None,
        )

        assert result.passes_filter is True
        assert result.reason is None

    def test_valid_result_fail(self) -> None:
        """Test creating valid failing result."""
        result = VolumeFilterResult(
            symbol="AAPL",
            passes_filter=False,
            position_pct_of_adv=2.5,
            max_position_size=500_000.0,
            reason="Position exceeds ADV limit",
        )

        assert result.passes_filter is False
        assert result.reason is not None


class TestVolumeFilter:
    """Tests for VolumeFilter class."""

    def test_initialization(self) -> None:
        """Test filter initialization."""
        filter = VolumeFilter()

        assert filter.max_position_pct_of_adv == 0.01
        assert filter.min_adv_threshold == 100_000

    def test_custom_parameters(self) -> None:
        """Test filter with custom parameters."""
        filter = VolumeFilter(max_position_pct_of_adv=0.02, min_adv_threshold=200_000)

        assert filter.max_position_pct_of_adv == 0.02
        assert filter.min_adv_threshold == 200_000

    def test_invalid_parameters(self) -> None:
        """Test invalid parameter validation."""
        with pytest.raises(ValueError):
            VolumeFilter(max_position_pct_of_adv=0.0)

        with pytest.raises(ValueError):
            VolumeFilter(max_position_pct_of_adv=1.5)

        with pytest.raises(ValueError):
            VolumeFilter(min_adv_threshold=-1000)

    def test_check_passes(self) -> None:
        """Test check that passes all filters."""
        filter = VolumeFilter()

        result = filter.check(
            symbol="AAPL",
            position_size_shares=1000,
            adv_20=10_000_000,
            price=100.0,
        )

        assert result.passes_filter is True
        assert result.reason is None

    def test_check_fails_min_adv(self) -> None:
        """Test check fails on minimum ADV threshold."""
        filter = VolumeFilter()

        result = filter.check(
            symbol="PENNY", position_size_shares=100, adv_20=500, price=10.0  # ADV = $5,000
        )

        assert result.passes_filter is False
        assert "below minimum threshold" in result.reason

    def test_check_fails_max_position(self) -> None:
        """Test check fails on exceeding max position size."""
        filter = VolumeFilter()

        result = filter.check(
            symbol="AAPL",
            position_size_shares=200_000,  # $20M position
            adv_20=10_000_000,  # ADV = $1B
            price=100.0,
        )

        # Position is $20M, max is 1% of $1B = $10M
        assert result.passes_filter is False
        assert "exceeds max" in result.reason

    def test_get_max_position(self) -> None:
        """Test max position calculation."""
        filter = VolumeFilter()

        max_pos = filter.get_max_position(adv_20=10_000_000, price=100.0)

        # ADV = $1B, max = 1% = $10M
        assert max_pos == 10_000_000.0

    def test_edge_case_zero_adv(self) -> None:
        """Test behavior with zero ADV."""
        filter = VolumeFilter()

        result = filter.check(symbol="ZERO", position_size_shares=100, adv_20=0, price=100.0)

        assert result.passes_filter is False


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self) -> None:
        """Test complete execution quality assessment workflow."""
        # 1. Score liquidity
        scorer = LiquidityScorer()
        liquidity = scorer.score_from_data(
            symbol="AAPL", adv_20=10_000_000, avg_spread_bps=5.0, market_cap=3_000_000_000_000
        )

        assert liquidity.rating in ("excellent", "good")

        # 2. Estimate slippage
        estimator = SlippageEstimator()
        slippage = estimator.estimate(
            symbol="AAPL",
            order_size_shares=10_000,
            adv_20=liquidity.adv_20,
            avg_spread_bps=5.0,
        )

        assert slippage.estimated_slippage_bps > 0

        # 3. Check volume filter
        filter = VolumeFilter()
        result = filter.check(
            symbol="AAPL", position_size_shares=10_000, adv_20=liquidity.adv_20, price=150.0
        )

        assert result.passes_filter is True

    def test_illiquid_asset_workflow(self) -> None:
        """Test workflow with illiquid asset."""
        scorer = LiquidityScorer()
        liquidity = scorer.score_from_data(
            symbol="ILLIQUID", adv_20=5_000, avg_spread_bps=100.0, market_cap=10_000_000
        )

        assert liquidity.rating in ("poor", "illiquid")

        estimator = SlippageEstimator()
        slippage = estimator.estimate(
            symbol="ILLIQUID", order_size_shares=1000, adv_20=liquidity.adv_20, avg_spread_bps=100.0
        )

        # Large slippage expected
        assert slippage.estimated_slippage_bps > 50.0

        filter = VolumeFilter()
        result = filter.check(
            symbol="ILLIQUID", position_size_shares=1000, adv_20=liquidity.adv_20, price=10.0
        )

        # Likely fails due to low ADV
        assert result.passes_filter is False
