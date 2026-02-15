"""Comprehensive tests for signal aggregation components."""

from datetime import datetime
from decimal import Decimal

import pytest

from signalforge.integration import (
    AggregationConfig,
    ConfidenceCalculator,
    EnhancedSignalAggregator,
    IntegratedSignal,
    MarketRegime,
    MarketRegimeDetector,
    SignalDirection,
)
from signalforge.nlp.signals.generator import (
    AnalystConsensus,
    NLPSignalOutput,
    SectorSignal,
    SentimentOutput,
)
from signalforge.schemas.execution import (
    ExecutionQualityMetrics,
    ExecutionQualityResponse,
    ExecutionWarning,
    LiquidityMetrics,
    LiquidityScoreResponse,
    SlippageComponents,
    SlippageResponse,
    SpreadMetrics,
    SpreadMetricsResponse,
    VolumeAnalysis,
    VolumeFilterResponse,
)
from signalforge.schemas.ml import PredictionResult


# Fixtures for test data
@pytest.fixture
def ml_prediction_bullish() -> PredictionResult:
    """Create bullish ML prediction."""
    return PredictionResult(
        horizon=1,
        predicted_price=Decimal("150.00"),
        predicted_direction="up",
        confidence=0.85,
        lower_bound=Decimal("145.00"),
        upper_bound=Decimal("155.00"),
    )


@pytest.fixture
def ml_prediction_bearish() -> PredictionResult:
    """Create bearish ML prediction."""
    return PredictionResult(
        horizon=1,
        predicted_price=Decimal("140.00"),
        predicted_direction="down",
        confidence=0.75,
        lower_bound=Decimal("135.00"),
        upper_bound=Decimal("145.00"),
    )


@pytest.fixture
def ml_prediction_neutral() -> PredictionResult:
    """Create neutral ML prediction."""
    return PredictionResult(
        horizon=1,
        predicted_price=Decimal("147.50"),
        predicted_direction="neutral",
        confidence=0.60,
        lower_bound=Decimal("145.00"),
        upper_bound=Decimal("150.00"),
    )


@pytest.fixture
def nlp_signals_bullish() -> NLPSignalOutput:
    """Create bullish NLP signals."""
    return NLPSignalOutput(
        ticker="AAPL",
        sentiment=SentimentOutput(
            score=0.8,
            label="bullish",
            delta_vs_baseline=0.3,
            is_new_information=True,
        ),
        price_targets=[],
        analyst_consensus=AnalystConsensus(
            rating="buy",
            confidence=0.9,
            bullish_count=8,
            bearish_count=1,
            neutral_count=1,
        ),
        sector_signals=[
            SectorSignal(
                sector="Technology",
                signal="buy",
                confidence=0.85,
                reasoning="Strong sector momentum",
            )
        ],
        urgency="high",
        generated_at=datetime.now(),
    )


@pytest.fixture
def nlp_signals_bearish() -> NLPSignalOutput:
    """Create bearish NLP signals."""
    return NLPSignalOutput(
        ticker="AAPL",
        sentiment=SentimentOutput(
            score=-0.7,
            label="bearish",
            delta_vs_baseline=-0.4,
            is_new_information=True,
        ),
        price_targets=[],
        analyst_consensus=AnalystConsensus(
            rating="sell",
            confidence=0.8,
            bullish_count=1,
            bearish_count=7,
            neutral_count=2,
        ),
        sector_signals=[
            SectorSignal(
                sector="Technology",
                signal="sell",
                confidence=0.75,
                reasoning="Weak sector performance",
            )
        ],
        urgency="high",
        generated_at=datetime.now(),
    )


@pytest.fixture
def execution_quality_good() -> ExecutionQualityResponse:
    """Create good execution quality response."""
    return ExecutionQualityResponse(
        symbol="AAPL",
        order_size=100,
        side="buy",
        timestamp=datetime.now(),
        current_price=Decimal("147.50"),
        metrics=ExecutionQualityMetrics(
            liquidity_score=85.0,
            estimated_slippage_bps=Decimal("5.0"),
            spread_bps=Decimal("2.0"),
            volume_participation_pct=Decimal("0.01"),
            execution_difficulty="easy",
            overall_score=90.0,
        ),
        liquidity=LiquidityScoreResponse(
            symbol="AAPL",
            timestamp=datetime.now(),
            liquidity_score=85.0,
            liquidity_tier="high",
            metrics=LiquidityMetrics(
                avg_volume_1d=50000000,
                avg_volume_5d=48000000,
                avg_volume_20d=45000000,
                avg_dollar_volume_20d=Decimal("6500000000"),
                relative_volume=Decimal("1.1"),
                bid_ask_spread_pct=Decimal("0.02"),
            ),
            recommendation="Highly liquid - favorable for trading",
            warnings=None,
        ),
        slippage=SlippageResponse(
            symbol="AAPL",
            order_size=100,
            side="buy",
            current_price=Decimal("147.50"),
            estimated_execution_price=Decimal("147.57"),
            slippage=SlippageComponents(
                market_impact_bps=Decimal("3.0"),
                spread_cost_bps=Decimal("1.5"),
                timing_risk_bps=Decimal("0.5"),
                total_slippage_bps=Decimal("5.0"),
            ),
            estimated_cost=Decimal("7.38"),
            confidence=0.95,
            timestamp=datetime.now(),
        ),
        spread=SpreadMetricsResponse(
            symbol="AAPL",
            timestamp=datetime.now(),
            metrics=SpreadMetrics(
                current_bid=Decimal("147.49"),
                current_ask=Decimal("147.51"),
                spread_absolute=Decimal("0.02"),
                spread_bps=Decimal("1.36"),
                avg_spread_1h=Decimal("0.02"),
                avg_spread_1d=Decimal("0.02"),
                spread_percentile=45.0,
            ),
            is_favorable=True,
            recommendation="Spread is favorable",
        ),
        volume=VolumeFilterResponse(
            symbol="AAPL",
            order_size=100,
            passes_filter=True,
            analysis=VolumeAnalysis(
                current_volume=25000000,
                avg_volume_20d=45000000,
                order_as_pct_avg_volume=Decimal("0.0002"),
                estimated_time_to_fill_minutes=1,
                volume_profile="normal",
            ),
            warnings=[],
            recommendation="Order size is appropriate",
            timestamp=datetime.now(),
        ),
        overall_recommendation="Favorable execution conditions",
        warnings=[],
        is_tradeable=True,
    )


@pytest.fixture
def execution_quality_poor() -> ExecutionQualityResponse:
    """Create poor execution quality response."""
    return ExecutionQualityResponse(
        symbol="AAPL",
        order_size=1000000,
        side="buy",
        timestamp=datetime.now(),
        current_price=Decimal("147.50"),
        metrics=ExecutionQualityMetrics(
            liquidity_score=35.0,
            estimated_slippage_bps=Decimal("50.0"),
            spread_bps=Decimal("20.0"),
            volume_participation_pct=Decimal("0.2"),
            execution_difficulty="very_difficult",
            overall_score=30.0,
        ),
        liquidity=LiquidityScoreResponse(
            symbol="AAPL",
            timestamp=datetime.now(),
            liquidity_score=35.0,
            liquidity_tier="low",
            metrics=LiquidityMetrics(
                avg_volume_1d=1000000,
                avg_volume_5d=900000,
                avg_volume_20d=800000,
                avg_dollar_volume_20d=Decimal("120000000"),
                relative_volume=Decimal("0.8"),
                bid_ask_spread_pct=Decimal("0.5"),
            ),
            recommendation="Low liquidity - trade with caution",
            warnings=["Low average volume"],
        ),
        slippage=SlippageResponse(
            symbol="AAPL",
            order_size=1000000,
            side="buy",
            current_price=Decimal("147.50"),
            estimated_execution_price=Decimal("148.24"),
            slippage=SlippageComponents(
                market_impact_bps=Decimal("35.0"),
                spread_cost_bps=Decimal("10.0"),
                timing_risk_bps=Decimal("5.0"),
                total_slippage_bps=Decimal("50.0"),
            ),
            estimated_cost=Decimal("738.00"),
            confidence=0.65,
            timestamp=datetime.now(),
        ),
        spread=SpreadMetricsResponse(
            symbol="AAPL",
            timestamp=datetime.now(),
            metrics=SpreadMetrics(
                current_bid=Decimal("147.25"),
                current_ask=Decimal("147.75"),
                spread_absolute=Decimal("0.50"),
                spread_bps=Decimal("33.89"),
                avg_spread_1h=Decimal("0.45"),
                avg_spread_1d=Decimal("0.40"),
                spread_percentile=85.0,
            ),
            is_favorable=False,
            recommendation="Spread is unfavorable",
        ),
        volume=VolumeFilterResponse(
            symbol="AAPL",
            order_size=1000000,
            passes_filter=False,
            analysis=VolumeAnalysis(
                current_volume=500000,
                avg_volume_20d=800000,
                order_as_pct_avg_volume=Decimal("1.25"),
                estimated_time_to_fill_minutes=180,
                volume_profile="low",
            ),
            warnings=["Order size exceeds recommended limits"],
            recommendation="Consider reducing order size",
            timestamp=datetime.now(),
        ),
        overall_recommendation="Unfavorable execution conditions - exercise caution",
        warnings=[
            ExecutionWarning(
                severity="high",
                category="liquidity",
                message="Low liquidity may cause significant slippage",
            ),
            ExecutionWarning(
                severity="high",
                category="volume",
                message="Order size too large relative to average volume",
            ),
        ],
        is_tradeable=False,
    )


# MarketRegimeDetector Tests
class TestMarketRegimeDetector:
    """Tests for MarketRegimeDetector."""

    def test_init(self) -> None:
        """Test detector initialization."""
        detector = MarketRegimeDetector()
        assert detector.volatility_threshold_high == 0.25
        assert detector.volatility_threshold_low == 0.10
        assert detector.trend_threshold == 0.05

    def test_init_custom_thresholds(self) -> None:
        """Test detector with custom thresholds."""
        detector = MarketRegimeDetector(
            volatility_threshold_high=0.30,
            volatility_threshold_low=0.08,
            trend_threshold=0.03,
        )
        assert detector.volatility_threshold_high == 0.30
        assert detector.volatility_threshold_low == 0.08
        assert detector.trend_threshold == 0.03

    @pytest.mark.asyncio
    async def test_detect_regime_default(self) -> None:
        """Test regime detection returns valid regime."""
        detector = MarketRegimeDetector()
        regime = await detector.detect_regime(symbol="AAPL")
        assert isinstance(regime, MarketRegime)

    @pytest.mark.asyncio
    async def test_detect_regime_market_wide(self) -> None:
        """Test market-wide regime detection."""
        detector = MarketRegimeDetector()
        regime = await detector.detect_regime(symbol=None, lookback_days=30)
        assert isinstance(regime, MarketRegime)

    def test_calculate_volatility_empty_returns(self) -> None:
        """Test volatility calculation with empty returns."""
        detector = MarketRegimeDetector()
        vol = detector._calculate_volatility([])
        assert vol == 0.0

    def test_calculate_volatility_single_return(self) -> None:
        """Test volatility calculation with single return."""
        detector = MarketRegimeDetector()
        vol = detector._calculate_volatility([0.01])
        assert vol == 0.0

    def test_calculate_volatility_normal(self) -> None:
        """Test volatility calculation with normal returns."""
        detector = MarketRegimeDetector()
        returns = [0.01, -0.02, 0.015, -0.01, 0.005]
        vol = detector._calculate_volatility(returns)
        assert vol > 0.0
        assert isinstance(vol, float)

    def test_calculate_trend_empty_prices(self) -> None:
        """Test trend calculation with empty prices."""
        detector = MarketRegimeDetector()
        direction, strength = detector._calculate_trend([])
        assert direction == "sideways"
        assert strength == 0.0

    def test_calculate_trend_single_price(self) -> None:
        """Test trend calculation with single price."""
        detector = MarketRegimeDetector()
        direction, strength = detector._calculate_trend([100.0])
        assert direction == "sideways"
        assert strength == 0.0

    def test_calculate_trend_upward(self) -> None:
        """Test trend calculation with upward trend."""
        detector = MarketRegimeDetector()
        prices = [100.0, 102.0, 105.0, 108.0, 112.0]
        direction, strength = detector._calculate_trend(prices)
        assert direction == "up"
        assert strength > 0.0

    def test_calculate_trend_downward(self) -> None:
        """Test trend calculation with downward trend."""
        detector = MarketRegimeDetector()
        prices = [112.0, 108.0, 105.0, 102.0, 100.0]
        direction, strength = detector._calculate_trend(prices)
        assert direction == "down"
        assert strength > 0.0

    def test_calculate_trend_flat(self) -> None:
        """Test trend calculation with flat prices."""
        detector = MarketRegimeDetector()
        prices = [100.0, 100.0, 100.0, 100.0, 100.0]
        direction, strength = detector._calculate_trend(prices)
        assert direction == "sideways"
        assert strength == 0.0

    def test_classify_regime_crisis(self) -> None:
        """Test regime classification for crisis."""
        detector = MarketRegimeDetector()
        regime = detector._classify_regime(
            volatility=0.5,
            trend_direction="down",
            trend_strength=0.8,
        )
        assert regime == MarketRegime.CRISIS

    def test_classify_regime_volatile(self) -> None:
        """Test regime classification for volatile market."""
        detector = MarketRegimeDetector()
        regime = detector._classify_regime(
            volatility=0.30,
            trend_direction="sideways",
            trend_strength=0.02,
        )
        assert regime == MarketRegime.VOLATILE

    def test_classify_regime_bull(self) -> None:
        """Test regime classification for bull market."""
        detector = MarketRegimeDetector()
        regime = detector._classify_regime(
            volatility=0.15,
            trend_direction="up",
            trend_strength=0.10,
        )
        assert regime == MarketRegime.BULL

    def test_classify_regime_bear(self) -> None:
        """Test regime classification for bear market."""
        detector = MarketRegimeDetector()
        regime = detector._classify_regime(
            volatility=0.15,
            trend_direction="down",
            trend_strength=0.10,
        )
        assert regime == MarketRegime.BEAR

    def test_classify_regime_sideways(self) -> None:
        """Test regime classification for sideways market."""
        detector = MarketRegimeDetector()
        regime = detector._classify_regime(
            volatility=0.08,
            trend_direction="sideways",
            trend_strength=0.02,
        )
        assert regime == MarketRegime.SIDEWAYS

    def test_get_regime_characteristics_bull(self) -> None:
        """Test getting characteristics for bull regime."""
        detector = MarketRegimeDetector()
        chars = detector.get_regime_characteristics(MarketRegime.BULL)
        assert chars["risk_level"] == "low"
        assert chars["position_sizing_multiplier"] == 1.2

    def test_get_regime_characteristics_bear(self) -> None:
        """Test getting characteristics for bear regime."""
        detector = MarketRegimeDetector()
        chars = detector.get_regime_characteristics(MarketRegime.BEAR)
        assert chars["risk_level"] == "high"
        assert chars["position_sizing_multiplier"] == 0.8

    def test_get_regime_characteristics_crisis(self) -> None:
        """Test getting characteristics for crisis regime."""
        detector = MarketRegimeDetector()
        chars = detector.get_regime_characteristics(MarketRegime.CRISIS)
        assert chars["risk_level"] == "extreme"
        assert chars["position_sizing_multiplier"] == 0.5


# ConfidenceCalculator Tests
class TestConfidenceCalculator:
    """Tests for ConfidenceCalculator."""

    def test_init(self) -> None:
        """Test calculator initialization."""
        calculator = ConfidenceCalculator()
        assert calculator is not None

    def test_calculate_integrated_confidence_high(self) -> None:
        """Test integrated confidence calculation with high values."""
        calculator = ConfidenceCalculator()
        confidence = calculator.calculate_integrated_confidence(
            ml_confidence=0.9,
            nlp_confidence=0.85,
            execution_score=0.95,
            signal_agreement=0.8,
        )
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.7  # Should be high

    def test_calculate_integrated_confidence_low(self) -> None:
        """Test integrated confidence calculation with low values."""
        calculator = ConfidenceCalculator()
        confidence = calculator.calculate_integrated_confidence(
            ml_confidence=0.3,
            nlp_confidence=0.4,
            execution_score=0.2,
            signal_agreement=-0.5,
        )
        assert 0.0 <= confidence <= 1.0
        assert confidence < 0.4  # Should be low

    def test_calculate_integrated_confidence_bounds(self) -> None:
        """Test integrated confidence respects bounds."""
        calculator = ConfidenceCalculator()
        # Test with extreme values
        confidence = calculator.calculate_integrated_confidence(
            ml_confidence=1.5,  # Invalid, should be clamped
            nlp_confidence=-0.5,  # Invalid, should be clamped
            execution_score=1.0,
            signal_agreement=2.0,  # Invalid, should be clamped
        )
        assert 0.0 <= confidence <= 1.0

    def test_calibrate_confidence_no_history(self) -> None:
        """Test confidence calibration with no historical data."""
        calculator = ConfidenceCalculator()
        calibrated = calculator.calibrate_confidence(
            raw_confidence=0.8,
            historical_accuracy=None,
        )
        assert calibrated == 0.8

    def test_calibrate_confidence_high_accuracy(self) -> None:
        """Test confidence calibration with high historical accuracy."""
        calculator = ConfidenceCalculator()
        calibrated = calculator.calibrate_confidence(
            raw_confidence=0.8,
            historical_accuracy=0.9,
        )
        assert calibrated > 0.7
        assert calibrated <= 1.0

    def test_calibrate_confidence_low_accuracy(self) -> None:
        """Test confidence calibration with low historical accuracy."""
        calculator = ConfidenceCalculator()
        calibrated = calculator.calibrate_confidence(
            raw_confidence=0.8,
            historical_accuracy=0.3,
        )
        # Should be pulled toward 0.5
        assert 0.5 <= calibrated <= 0.8

    def test_calculate_agreement_perfect_positive(self) -> None:
        """Test agreement calculation with perfect positive agreement."""
        calculator = ConfidenceCalculator()
        agreement = calculator.calculate_agreement(
            ml_direction=1.0,
            nlp_direction=1.0,
        )
        assert agreement == 1.0

    def test_calculate_agreement_perfect_negative(self) -> None:
        """Test agreement calculation with perfect disagreement."""
        calculator = ConfidenceCalculator()
        agreement = calculator.calculate_agreement(
            ml_direction=1.0,
            nlp_direction=-1.0,
        )
        assert agreement == -1.0

    def test_calculate_agreement_neutral(self) -> None:
        """Test agreement calculation with neutral signals."""
        calculator = ConfidenceCalculator()
        agreement = calculator.calculate_agreement(
            ml_direction=0.0,
            nlp_direction=0.0,
        )
        assert agreement == 0.0

    def test_calculate_agreement_partial(self) -> None:
        """Test agreement calculation with partial agreement."""
        calculator = ConfidenceCalculator()
        agreement = calculator.calculate_agreement(
            ml_direction=0.8,
            nlp_direction=0.6,
        )
        assert 0.0 < agreement < 1.0

    def test_get_confidence_interval_high_confidence(self) -> None:
        """Test confidence interval with high confidence."""
        calculator = ConfidenceCalculator()
        lower, upper = calculator.get_confidence_interval(
            prediction=0.5,
            confidence=0.9,
        )
        assert lower < 0.5 < upper
        assert upper - lower < 0.1  # Narrow interval

    def test_get_confidence_interval_low_confidence(self) -> None:
        """Test confidence interval with low confidence."""
        calculator = ConfidenceCalculator()
        lower, upper = calculator.get_confidence_interval(
            prediction=0.5,
            confidence=0.2,
        )
        assert lower < 0.5 < upper
        assert upper - lower > 0.1  # Wide interval


# EnhancedSignalAggregator Tests
class TestEnhancedSignalAggregator:
    """Tests for EnhancedSignalAggregator."""

    def test_init_default(self) -> None:
        """Test aggregator initialization with defaults."""
        aggregator = EnhancedSignalAggregator()
        assert aggregator.config is not None
        assert aggregator.regime_detector is not None
        assert aggregator.confidence_calculator is not None

    def test_init_custom_config(self) -> None:
        """Test aggregator initialization with custom config."""
        config = AggregationConfig(
            ml_weight=0.5,
            nlp_weight=0.3,
            execution_weight=0.1,
            regime_weight=0.1,
        )
        aggregator = EnhancedSignalAggregator(config=config)
        assert aggregator.config.ml_weight == 0.5
        assert aggregator.config.nlp_weight == 0.3

    @pytest.mark.asyncio
    async def test_aggregate_all_sources(
        self,
        ml_prediction_bullish: PredictionResult,
        nlp_signals_bullish: NLPSignalOutput,
        execution_quality_good: ExecutionQualityResponse,
    ) -> None:
        """Test aggregation with all data sources."""
        aggregator = EnhancedSignalAggregator()
        signal = await aggregator.aggregate(
            symbol="AAPL",
            ml_prediction=ml_prediction_bullish,
            nlp_signals=nlp_signals_bullish,
            execution_quality=execution_quality_good,
        )
        assert isinstance(signal, IntegratedSignal)
        assert signal.symbol == "AAPL"
        assert signal.direction in SignalDirection
        assert -1.0 <= signal.strength <= 1.0
        assert 0.0 <= signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_aggregate_bullish_signal(
        self,
        ml_prediction_bullish: PredictionResult,
        nlp_signals_bullish: NLPSignalOutput,
        execution_quality_good: ExecutionQualityResponse,
    ) -> None:
        """Test aggregation produces bullish signal."""
        aggregator = EnhancedSignalAggregator()
        signal = await aggregator.aggregate(
            symbol="AAPL",
            ml_prediction=ml_prediction_bullish,
            nlp_signals=nlp_signals_bullish,
            execution_quality=execution_quality_good,
        )
        assert signal.strength > 0.0
        assert signal.direction in (SignalDirection.LONG, SignalDirection.STRONG_LONG)

    @pytest.mark.asyncio
    async def test_aggregate_bearish_signal(
        self,
        ml_prediction_bearish: PredictionResult,
        nlp_signals_bearish: NLPSignalOutput,
        execution_quality_good: ExecutionQualityResponse,
    ) -> None:
        """Test aggregation produces bearish signal."""
        aggregator = EnhancedSignalAggregator()
        signal = await aggregator.aggregate(
            symbol="AAPL",
            ml_prediction=ml_prediction_bearish,
            nlp_signals=nlp_signals_bearish,
            execution_quality=execution_quality_good,
        )
        assert signal.strength < 0.0
        assert signal.direction in (SignalDirection.SHORT, SignalDirection.STRONG_SHORT)

    @pytest.mark.asyncio
    async def test_aggregate_conflicting_signals(
        self,
        ml_prediction_bullish: PredictionResult,
        nlp_signals_bearish: NLPSignalOutput,
        execution_quality_good: ExecutionQualityResponse,
    ) -> None:
        """Test aggregation with conflicting ML and NLP signals."""
        aggregator = EnhancedSignalAggregator()
        signal = await aggregator.aggregate(
            symbol="AAPL",
            ml_prediction=ml_prediction_bullish,
            nlp_signals=nlp_signals_bearish,
            execution_quality=execution_quality_good,
        )
        # Should generate warnings about disagreement
        assert any("disagree" in w.lower() for w in signal.warnings)

    @pytest.mark.asyncio
    async def test_aggregate_no_ml(
        self,
        nlp_signals_bullish: NLPSignalOutput,
        execution_quality_good: ExecutionQualityResponse,
    ) -> None:
        """Test aggregation without ML prediction."""
        aggregator = EnhancedSignalAggregator()
        signal = await aggregator.aggregate(
            symbol="AAPL",
            ml_prediction=None,
            nlp_signals=nlp_signals_bullish,
            execution_quality=execution_quality_good,
        )
        assert signal.ml_prediction is None
        assert "ML prediction not available" in signal.warnings

    @pytest.mark.asyncio
    async def test_aggregate_no_nlp(
        self,
        ml_prediction_bullish: PredictionResult,
        execution_quality_good: ExecutionQualityResponse,
    ) -> None:
        """Test aggregation without NLP signals."""
        aggregator = EnhancedSignalAggregator()
        signal = await aggregator.aggregate(
            symbol="AAPL",
            ml_prediction=ml_prediction_bullish,
            nlp_signals=None,
            execution_quality=execution_quality_good,
        )
        assert signal.nlp_sentiment is None
        assert "NLP signals not available" in signal.warnings

    @pytest.mark.asyncio
    async def test_aggregate_no_execution(
        self,
        ml_prediction_bullish: PredictionResult,
        nlp_signals_bullish: NLPSignalOutput,
    ) -> None:
        """Test aggregation without execution quality."""
        aggregator = EnhancedSignalAggregator()
        signal = await aggregator.aggregate(
            symbol="AAPL",
            ml_prediction=ml_prediction_bullish,
            nlp_signals=nlp_signals_bullish,
            execution_quality=None,
        )
        assert signal.execution_score is None
        assert "Execution quality not available" in signal.warnings

    @pytest.mark.asyncio
    async def test_aggregate_poor_execution(
        self,
        ml_prediction_bullish: PredictionResult,
        nlp_signals_bullish: NLPSignalOutput,
        execution_quality_poor: ExecutionQualityResponse,
    ) -> None:
        """Test aggregation with poor execution quality."""
        aggregator = EnhancedSignalAggregator()
        signal = await aggregator.aggregate(
            symbol="AAPL",
            ml_prediction=ml_prediction_bullish,
            nlp_signals=nlp_signals_bullish,
            execution_quality=execution_quality_poor,
        )
        # Should have warnings about execution
        assert any("tradeable" in w.lower() or "execution" in w.lower() for w in signal.warnings)
        # Position size should be reduced
        assert signal.position_size_pct < 0.1

    @pytest.mark.asyncio
    async def test_aggregate_batch(self) -> None:
        """Test batch aggregation."""
        aggregator = EnhancedSignalAggregator()
        symbols = ["AAPL", "MSFT", "GOOGL"]
        signals = await aggregator.aggregate_batch(symbols)
        assert len(signals) <= len(symbols)  # May have failures
        for signal in signals:
            assert signal.symbol in symbols

    def test_normalize_ml_prediction_up(
        self, ml_prediction_bullish: PredictionResult
    ) -> None:
        """Test ML prediction normalization for up direction."""
        aggregator = EnhancedSignalAggregator()
        normalized = aggregator._normalize_ml_prediction(ml_prediction_bullish)
        assert 0.0 < normalized <= 1.0

    def test_normalize_ml_prediction_down(
        self, ml_prediction_bearish: PredictionResult
    ) -> None:
        """Test ML prediction normalization for down direction."""
        aggregator = EnhancedSignalAggregator()
        normalized = aggregator._normalize_ml_prediction(ml_prediction_bearish)
        assert -1.0 <= normalized < 0.0

    def test_normalize_ml_prediction_neutral(
        self, ml_prediction_neutral: PredictionResult
    ) -> None:
        """Test ML prediction normalization for neutral direction."""
        aggregator = EnhancedSignalAggregator()
        normalized = aggregator._normalize_ml_prediction(ml_prediction_neutral)
        assert normalized == 0.0

    def test_normalize_nlp_sentiment_bullish(
        self, nlp_signals_bullish: NLPSignalOutput
    ) -> None:
        """Test NLP sentiment normalization for bullish sentiment."""
        aggregator = EnhancedSignalAggregator()
        normalized = aggregator._normalize_nlp_sentiment(nlp_signals_bullish)
        assert 0.0 < normalized <= 1.2  # Can exceed 1.0 with new info boost

    def test_normalize_nlp_sentiment_bearish(
        self, nlp_signals_bearish: NLPSignalOutput
    ) -> None:
        """Test NLP sentiment normalization for bearish sentiment."""
        aggregator = EnhancedSignalAggregator()
        normalized = aggregator._normalize_nlp_sentiment(nlp_signals_bearish)
        assert -1.2 <= normalized < 0.0  # Can exceed -1.0 with new info boost

    def test_determine_direction_strong_long(self) -> None:
        """Test direction determination for strong long."""
        aggregator = EnhancedSignalAggregator()
        direction = aggregator._determine_direction(0.7)
        assert direction == SignalDirection.STRONG_LONG

    def test_determine_direction_long(self) -> None:
        """Test direction determination for long."""
        aggregator = EnhancedSignalAggregator()
        direction = aggregator._determine_direction(0.3)
        assert direction == SignalDirection.LONG

    def test_determine_direction_neutral(self) -> None:
        """Test direction determination for neutral."""
        aggregator = EnhancedSignalAggregator()
        direction = aggregator._determine_direction(0.05)
        assert direction == SignalDirection.NEUTRAL

    def test_determine_direction_short(self) -> None:
        """Test direction determination for short."""
        aggregator = EnhancedSignalAggregator()
        direction = aggregator._determine_direction(-0.3)
        assert direction == SignalDirection.SHORT

    def test_determine_direction_strong_short(self) -> None:
        """Test direction determination for strong short."""
        aggregator = EnhancedSignalAggregator()
        direction = aggregator._determine_direction(-0.7)
        assert direction == SignalDirection.STRONG_SHORT

    def test_calculate_position_size_strong_signal(self) -> None:
        """Test position sizing for strong signal."""
        aggregator = EnhancedSignalAggregator()
        size = aggregator._calculate_position_size(
            strength=0.8,
            confidence=0.9,
            execution_score=0.95,
        )
        assert 0.0 < size <= 0.25
        assert size > 0.1  # Should be significant

    def test_calculate_position_size_weak_signal(self) -> None:
        """Test position sizing for weak signal."""
        aggregator = EnhancedSignalAggregator()
        size = aggregator._calculate_position_size(
            strength=0.1,
            confidence=0.4,
            execution_score=0.3,
        )
        assert size < 0.05  # Should be small or zero

    def test_generate_recommendation_strong_buy(self) -> None:
        """Test recommendation generation for strong buy."""
        aggregator = EnhancedSignalAggregator()
        rec = aggregator._generate_recommendation(SignalDirection.STRONG_LONG, 0.8)
        assert rec == "strong_buy"

    def test_generate_recommendation_buy(self) -> None:
        """Test recommendation generation for buy."""
        aggregator = EnhancedSignalAggregator()
        rec = aggregator._generate_recommendation(SignalDirection.LONG, 0.7)
        assert rec == "buy"

    def test_generate_recommendation_hold(self) -> None:
        """Test recommendation generation for hold."""
        aggregator = EnhancedSignalAggregator()
        rec = aggregator._generate_recommendation(SignalDirection.NEUTRAL, 0.6)
        assert rec == "hold"

    def test_generate_recommendation_low_confidence(self) -> None:
        """Test recommendation downgrade for low confidence."""
        aggregator = EnhancedSignalAggregator()
        rec = aggregator._generate_recommendation(SignalDirection.STRONG_LONG, 0.3)
        assert rec == "hold"  # Downgraded due to low confidence

    def test_calculate_risk_levels_neutral(self) -> None:
        """Test risk level calculation for neutral signal."""
        aggregator = EnhancedSignalAggregator()
        stop, profit = aggregator._calculate_risk_levels(0.05, MarketRegime.SIDEWAYS)
        assert stop is None
        assert profit is None

    def test_calculate_risk_levels_strong_signal(self) -> None:
        """Test risk level calculation for strong signal."""
        aggregator = EnhancedSignalAggregator()
        stop, profit = aggregator._calculate_risk_levels(0.8, MarketRegime.BULL)
        assert stop is not None
        assert profit is not None
        assert stop > 0.0
        assert profit > stop  # Profit should exceed stop

    def test_generate_warnings_missing_sources(self) -> None:
        """Test warning generation for missing data sources."""
        aggregator = EnhancedSignalAggregator()
        warnings = aggregator._generate_warnings(None, None, None, 0.8)
        assert "ML prediction not available" in warnings
        assert "NLP signals not available" in warnings
        assert "Execution quality not available" in warnings

    def test_generate_warnings_low_confidence(self) -> None:
        """Test warning generation for low confidence."""
        aggregator = EnhancedSignalAggregator()
        warnings = aggregator._generate_warnings(None, None, None, 0.3)
        assert any("low confidence" in w.lower() for w in warnings)
