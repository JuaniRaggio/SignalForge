"""Enhanced signal aggregator combining ML, NLP, and execution quality."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from signalforge.core.logging import get_logger
from signalforge.integration.confidence import ConfidenceCalculator
from signalforge.integration.regime import MarketRegimeDetector
from signalforge.integration.schemas import (
    AggregationConfig,
    IntegratedSignal,
    MarketRegime,
    SignalDirection,
)

if TYPE_CHECKING:
    from signalforge.nlp.signals.generator import NLPSignalOutput
    from signalforge.schemas.execution import ExecutionQualityResponse
    from signalforge.schemas.ml import PredictionResult

logger = get_logger(__name__)


class EnhancedSignalAggregator:
    """Aggregate all signal sources into unified trading signal."""

    def __init__(
        self,
        config: AggregationConfig | None = None,
        regime_detector: MarketRegimeDetector | None = None,
        confidence_calculator: ConfidenceCalculator | None = None,
    ) -> None:
        """Initialize the enhanced signal aggregator.

        Args:
            config: Aggregation configuration. Uses defaults if None.
            regime_detector: Market regime detector. Creates new instance if None.
            confidence_calculator: Confidence calculator. Creates new instance if None.
        """
        self.config = config or AggregationConfig()
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.confidence_calculator = confidence_calculator or ConfidenceCalculator()

        logger.info(
            "Initialized EnhancedSignalAggregator",
            ml_weight=self.config.ml_weight,
            nlp_weight=self.config.nlp_weight,
            execution_weight=self.config.execution_weight,
            regime_weight=self.config.regime_weight,
        )

    async def aggregate(
        self,
        symbol: str,
        ml_prediction: PredictionResult | None = None,
        nlp_signals: NLPSignalOutput | None = None,
        execution_quality: ExecutionQualityResponse | None = None,
    ) -> IntegratedSignal:
        """Aggregate signals from all sources.

        This method performs:
        1. Detect current market regime
        2. Normalize all inputs to common scale
        3. Apply regime-specific adjustments
        4. Calculate weighted combination
        5. Determine direction and strength
        6. Calculate position sizing
        7. Generate explanation and warnings

        Args:
            symbol: Stock symbol to aggregate signals for.
            ml_prediction: ML model prediction (optional).
            nlp_signals: NLP analysis signals (optional).
            execution_quality: Execution quality assessment (optional).

        Returns:
            Integrated signal combining all sources.
        """
        logger.info(
            "Aggregating signals",
            symbol=symbol,
            has_ml=ml_prediction is not None,
            has_nlp=nlp_signals is not None,
            has_execution=execution_quality is not None,
        )

        # Step 1: Detect market regime
        regime = await self.regime_detector.detect_regime(symbol=symbol)

        # Step 2: Normalize inputs
        ml_normalized = (
            self._normalize_ml_prediction(ml_prediction) if ml_prediction else 0.0
        )
        nlp_normalized = self._normalize_nlp_sentiment(nlp_signals) if nlp_signals else 0.0
        execution_normalized = (
            self._normalize_execution_quality(execution_quality)
            if execution_quality
            else 0.5
        )

        # Step 3: Calculate component contributions
        ml_contribution = ml_normalized * self.config.ml_weight
        nlp_contribution = nlp_normalized * self.config.nlp_weight
        execution_contribution = execution_normalized * self.config.execution_weight

        # Step 4: Calculate base signal
        base_signal = ml_contribution + nlp_contribution

        # Step 5: Apply regime adjustment
        regime_adjustment = self._apply_regime_adjustment(base_signal, regime)

        # Step 6: Calculate final strength
        strength = base_signal + (regime_adjustment * self.config.regime_weight)
        strength = max(-1.0, min(1.0, strength))  # Clamp to valid range

        # Step 7: Calculate confidence
        ml_conf = ml_prediction.confidence if ml_prediction else 0.5
        nlp_conf = (
            nlp_signals.sentiment.is_new_information if nlp_signals else 0.5
        )  # Simplified
        exec_score = execution_normalized

        agreement = self.confidence_calculator.calculate_agreement(
            ml_normalized, nlp_normalized
        )
        confidence = self.confidence_calculator.calculate_integrated_confidence(
            ml_conf, nlp_conf, exec_score, agreement
        )

        # Step 8: Determine direction
        direction = self._determine_direction(strength)

        # Step 9: Calculate position sizing
        position_size = self._calculate_position_size(strength, confidence, execution_normalized)

        # Step 10: Generate recommendation
        recommendation = self._generate_recommendation(direction, confidence)

        # Step 11: Calculate stop loss and take profit
        stop_loss_pct, take_profit_pct = self._calculate_risk_levels(strength, regime)

        # Step 12: Generate explanation
        generated_at = datetime.now()
        explanation = self._generate_explanation(
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            regime=regime,
            ml_prediction=ml_prediction,
            nlp_signals=nlp_signals,
            execution_quality=execution_quality,
        )

        # Step 13: Generate warnings
        warnings = self._generate_warnings(
            ml_prediction, nlp_signals, execution_quality, confidence
        )

        # Step 14: Create integrated signal
        signal = IntegratedSignal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=confidence,
            ml_contribution=ml_contribution,
            nlp_contribution=nlp_contribution,
            execution_contribution=execution_contribution,
            regime_adjustment=regime_adjustment,
            ml_prediction=ml_normalized if ml_prediction else None,
            nlp_sentiment=nlp_normalized if nlp_signals else None,
            execution_score=execution_normalized if execution_quality else None,
            current_regime=regime,
            recommendation=recommendation,
            position_size_pct=position_size,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            explanation=explanation,
            generated_at=generated_at,
            valid_until=generated_at + timedelta(hours=4),  # Signals valid for 4 hours
            warnings=warnings,
        )

        logger.info(
            "Signal aggregation complete",
            symbol=symbol,
            direction=direction.value,
            strength=strength,
            confidence=confidence,
            recommendation=recommendation,
        )

        return signal

    async def aggregate_batch(
        self,
        symbols: list[str],
    ) -> list[IntegratedSignal]:
        """Aggregate signals for multiple symbols.

        Args:
            symbols: List of stock symbols.

        Returns:
            List of integrated signals.
        """
        logger.info("Batch aggregation started", symbol_count=len(symbols))

        signals = []
        for symbol in symbols:
            try:
                signal = await self.aggregate(symbol=symbol)
                signals.append(signal)
            except Exception as e:
                logger.error(
                    "Failed to aggregate signal for symbol",
                    symbol=symbol,
                    error=str(e),
                )

        logger.info("Batch aggregation complete", signals_generated=len(signals))
        return signals

    def _normalize_ml_prediction(self, prediction: PredictionResult) -> float:
        """Normalize ML prediction to -1 to 1 scale.

        Args:
            prediction: ML prediction result.

        Returns:
            Normalized value from -1 (strong sell) to 1 (strong buy).
        """
        # Map prediction direction to numeric value
        direction_map = {"up": 1.0, "down": -1.0, "neutral": 0.0}
        direction_value = direction_map.get(prediction.predicted_direction, 0.0)

        # Scale by confidence
        normalized = direction_value * prediction.confidence

        logger.debug(
            "Normalized ML prediction",
            direction=prediction.predicted_direction,
            confidence=prediction.confidence,
            normalized=normalized,
        )

        return normalized

    def _normalize_nlp_sentiment(self, nlp: NLPSignalOutput) -> float:
        """Normalize NLP sentiment to -1 to 1 scale.

        Args:
            nlp: NLP signal output.

        Returns:
            Normalized sentiment from -1 (bearish) to 1 (bullish).
        """
        # NLP sentiment score is already in -1 to 1 range
        # Scale by whether it's new information
        base_sentiment = nlp.sentiment.score
        information_multiplier = 1.2 if nlp.sentiment.is_new_information else 1.0

        normalized = base_sentiment * information_multiplier
        normalized = max(-1.0, min(1.0, normalized))  # Clamp

        logger.debug(
            "Normalized NLP sentiment",
            base_sentiment=base_sentiment,
            is_new_information=nlp.sentiment.is_new_information,
            normalized=normalized,
        )

        return normalized

    def _normalize_execution_quality(self, execution: ExecutionQualityResponse) -> float:
        """Normalize execution quality to 0 to 1 scale.

        Args:
            execution: Execution quality response.

        Returns:
            Normalized quality score from 0 (poor) to 1 (excellent).
        """
        # Overall score is already 0-100, normalize to 0-1
        normalized = execution.metrics.overall_score / 100.0

        logger.debug(
            "Normalized execution quality",
            overall_score=execution.metrics.overall_score,
            normalized=normalized,
        )

        return normalized

    def _apply_regime_adjustment(
        self,
        signal: float,
        regime: MarketRegime,
    ) -> float:
        """Apply regime-specific adjustments to signal.

        Args:
            signal: Base signal value.
            regime: Current market regime.

        Returns:
            Regime adjustment factor.
        """
        # Get regime multiplier
        multiplier = self.config.regime_multipliers.get(regime.value, 1.0)

        # Calculate adjustment
        # Positive signals get boosted in bull markets, dampened in bear markets
        # Negative signals get boosted in bear markets, dampened in bull markets
        if signal > 0:  # Long signal
            if regime in (MarketRegime.BULL, MarketRegime.SIDEWAYS):
                adjustment = signal * (multiplier - 1.0)
            else:
                adjustment = signal * (multiplier - 1.0)
        else:  # Short signal or neutral
            if regime in (MarketRegime.BEAR, MarketRegime.VOLATILE):
                adjustment = abs(signal) * (multiplier - 1.0)
            else:
                adjustment = abs(signal) * (multiplier - 1.0)

        logger.debug(
            "Applied regime adjustment",
            signal=signal,
            regime=regime.value,
            multiplier=multiplier,
            adjustment=adjustment,
        )

        return adjustment

    def _calculate_position_size(
        self,
        strength: float,
        confidence: float,
        execution_score: float,
    ) -> float:
        """Calculate suggested position size as % of portfolio.

        Args:
            strength: Signal strength (-1 to 1).
            confidence: Overall confidence (0 to 1).
            execution_score: Execution quality score (0 to 1).

        Returns:
            Position size as percentage (0 to 1).
        """
        # Base position size on signal strength
        base_size = abs(strength) * 0.2  # Max 20% base size

        # Scale by confidence
        size = base_size * confidence

        # Adjust for execution quality
        # Poor execution should reduce position size
        size = size * execution_score

        # Ensure minimum threshold
        if size < 0.01:  # Less than 1%
            size = 0.0

        # Cap maximum position size
        size = min(size, 0.25)  # Max 25% of portfolio

        logger.debug(
            "Calculated position size",
            strength=strength,
            confidence=confidence,
            execution_score=execution_score,
            position_size=size,
        )

        return size

    def _determine_direction(self, strength: float) -> SignalDirection:
        """Map strength to direction enum.

        Args:
            strength: Signal strength (-1 to 1).

        Returns:
            Signal direction classification.
        """
        if strength >= 0.5:
            return SignalDirection.STRONG_LONG
        elif strength >= 0.15:
            return SignalDirection.LONG
        elif strength <= -0.5:
            return SignalDirection.STRONG_SHORT
        elif strength <= -0.15:
            return SignalDirection.SHORT
        else:
            return SignalDirection.NEUTRAL

    def _generate_recommendation(self, direction: SignalDirection, confidence: float) -> str:
        """Generate recommendation text.

        Args:
            direction: Signal direction.
            confidence: Overall confidence.

        Returns:
            Recommendation string.
        """
        # Map direction to base recommendation
        direction_map = {
            SignalDirection.STRONG_LONG: "strong_buy",
            SignalDirection.LONG: "buy",
            SignalDirection.NEUTRAL: "hold",
            SignalDirection.SHORT: "sell",
            SignalDirection.STRONG_SHORT: "strong_sell",
        }

        recommendation = direction_map[direction]

        # Downgrade if confidence is low
        if (
            confidence < self.config.min_confidence_threshold
            and recommendation in ("strong_buy", "buy", "strong_sell", "sell")
        ):
            recommendation = "hold"

        return recommendation

    def _calculate_risk_levels(
        self, strength: float, regime: MarketRegime
    ) -> tuple[float | None, float | None]:
        """Calculate stop loss and take profit levels.

        Args:
            strength: Signal strength.
            regime: Current market regime.

        Returns:
            Tuple of (stop_loss_pct, take_profit_pct).
        """
        if abs(strength) < 0.15:  # Neutral signal
            return (None, None)

        # Base levels
        stop_loss = 0.05  # 5% stop loss
        take_profit = 0.10  # 10% take profit

        # Adjust for regime volatility
        if regime == MarketRegime.VOLATILE:
            stop_loss *= 1.5
            take_profit *= 1.3
        elif regime == MarketRegime.CRISIS:
            stop_loss *= 2.0
            take_profit *= 1.5
        elif regime == MarketRegime.SIDEWAYS:
            stop_loss *= 0.8
            take_profit *= 0.8

        # Adjust for signal strength
        # Stronger signals can have tighter stops
        strength_factor = abs(strength)
        stop_loss *= (2.0 - strength_factor)
        take_profit *= (1.0 + strength_factor * 0.5)

        return (stop_loss, take_profit)

    def _generate_explanation(
        self,
        symbol: str,
        direction: SignalDirection,
        strength: float,
        confidence: float,
        regime: MarketRegime,
        ml_prediction: PredictionResult | None,
        nlp_signals: NLPSignalOutput | None,
        execution_quality: ExecutionQualityResponse | None,
    ) -> str:
        """Generate human-readable explanation.

        Args:
            symbol: Stock symbol.
            direction: Signal direction.
            strength: Signal strength.
            confidence: Overall confidence.
            regime: Market regime.
            ml_prediction: ML prediction if available.
            nlp_signals: NLP signals if available.
            execution_quality: Execution quality if available.

        Returns:
            Human-readable explanation string.
        """
        parts = [f"Signal for {symbol}: {direction.value.upper().replace('_', ' ')}"]

        # Add strength and confidence
        parts.append(f"Strength: {strength:.2f}, Confidence: {confidence:.2%}")

        # Add regime context
        parts.append(f"Market regime: {regime.value}")

        # Add component contributions
        if ml_prediction:
            parts.append(
                f"ML predicts {ml_prediction.predicted_direction} "
                f"(confidence: {ml_prediction.confidence:.2%})"
            )

        if nlp_signals:
            parts.append(
                f"NLP sentiment: {nlp_signals.sentiment.label} "
                f"(score: {nlp_signals.sentiment.score:.2f})"
            )

        if execution_quality:
            parts.append(
                f"Execution quality: {execution_quality.metrics.execution_difficulty} "
                f"(score: {execution_quality.metrics.overall_score:.0f}/100)"
            )

        return " | ".join(parts)

    def _generate_warnings(
        self,
        ml_prediction: PredictionResult | None,
        nlp_signals: NLPSignalOutput | None,
        execution_quality: ExecutionQualityResponse | None,
        confidence: float,
    ) -> list[str]:
        """Generate warning messages.

        Args:
            ml_prediction: ML prediction if available.
            nlp_signals: NLP signals if available.
            execution_quality: Execution quality if available.
            confidence: Overall confidence.

        Returns:
            List of warning messages.
        """
        warnings = []

        # Check for missing data sources
        if ml_prediction is None:
            warnings.append("ML prediction not available")

        if nlp_signals is None:
            warnings.append("NLP signals not available")

        if execution_quality is None:
            warnings.append("Execution quality not available")

        # Check confidence level
        if confidence < self.config.min_confidence_threshold:
            warnings.append(
                f"Low confidence ({confidence:.2%}) - "
                f"below threshold ({self.config.min_confidence_threshold:.2%})"
            )

        # Check execution quality warnings
        if execution_quality and not execution_quality.is_tradeable:
            warnings.append("Symbol may not be tradeable - poor execution quality")

        if execution_quality and execution_quality.warnings:
            for warning in execution_quality.warnings:
                if warning.severity in ("high", "medium"):
                    warnings.append(f"Execution: {warning.message}")

        # Check signal agreement
        if ml_prediction and nlp_signals:
            ml_dir = 1 if ml_prediction.predicted_direction == "up" else -1
            nlp_dir = 1 if nlp_signals.sentiment.score > 0 else -1

            if ml_dir != nlp_dir:
                warnings.append("ML and NLP signals disagree on direction")

        return warnings
