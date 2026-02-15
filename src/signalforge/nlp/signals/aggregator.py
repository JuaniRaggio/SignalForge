"""Aggregate NLP signals with ML predictions.

This module provides functionality to combine NLP-based signals with ML model
predictions to generate unified trading recommendations.

Key Features:
- Weighted aggregation of ML and NLP signals
- Market regime conditioning
- Execution quality adjustments
- Confidence-based weighting
- Comprehensive recommendation generation

Examples:
    Aggregate ML and NLP signals:

    >>> from signalforge.nlp.signals.aggregator import SignalAggregator
    >>> aggregator = SignalAggregator(ml_weight=0.6, nlp_weight=0.4)
    >>> signal = aggregator.aggregate(ml_prediction, nlp_signals, execution_quality)
    >>> print(f"Direction: {signal.direction}, Strength: {signal.strength:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    from signalforge.nlp.signals.generator import NLPSignalOutput

logger = get_logger(__name__)


@dataclass
class AggregatedSignal:
    """Aggregated trading signal combining ML and NLP.

    Attributes:
        symbol: Stock ticker symbol.
        direction: Signal direction (long, short, neutral).
        strength: Signal strength from -1 (strong short) to 1 (strong long).
        confidence: Overall confidence in the signal (0.0 to 1.0).
        ml_contribution: ML model's contribution to the signal.
        nlp_contribution: NLP analysis contribution to the signal.
        execution_feasibility: Execution quality score (0.0 to 1.0).
        recommendation: Human-readable recommendation.
        explanation: Detailed explanation of the signal.
    """

    symbol: str
    direction: str
    strength: float
    confidence: float
    ml_contribution: float
    nlp_contribution: float
    execution_feasibility: float
    recommendation: str
    explanation: str

    def __post_init__(self) -> None:
        """Validate aggregated signal fields."""
        if self.direction not in ("long", "short", "neutral"):
            raise ValueError(f"Invalid direction: {self.direction}. Must be long, short, or neutral")

        if not -1.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be between -1.0 and 1.0, got {self.strength}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

        if not 0.0 <= self.execution_feasibility <= 1.0:
            raise ValueError(
                f"execution_feasibility must be between 0.0 and 1.0, got {self.execution_feasibility}"
            )


class SignalAggregator:
    """Aggregate NLP signals with ML predictions.

    This aggregator combines multiple signal sources using configurable weights
    and applies market regime conditioning.

    Examples:
        >>> aggregator = SignalAggregator(ml_weight=0.6, nlp_weight=0.4)
        >>> signal = aggregator.aggregate(ml_pred, nlp_signals)
        >>> print(signal.recommendation)
    """

    def __init__(
        self,
        ml_weight: float = 0.6,
        nlp_weight: float = 0.4,
    ) -> None:
        """Initialize the signal aggregator.

        Args:
            ml_weight: Weight for ML predictions (0.0 to 1.0).
            nlp_weight: Weight for NLP signals (0.0 to 1.0).

        Raises:
            ValueError: If weights don't sum to 1.0.
        """
        if not abs(ml_weight + nlp_weight - 1.0) < 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {ml_weight + nlp_weight}")

        self._ml_weight = ml_weight
        self._nlp_weight = nlp_weight

        logger.info(
            "signal_aggregator_initialized",
            ml_weight=ml_weight,
            nlp_weight=nlp_weight,
        )

    def aggregate(
        self,
        ml_prediction: Any | None,
        nlp_signals: NLPSignalOutput | None,
        execution_quality: Any | None = None,
    ) -> AggregatedSignal:
        """Combine ML + NLP signals into unified recommendation.

        This method performs the following steps:
        1. Weight ML prediction by confidence
        2. Weight NLP sentiment by recency and information novelty
        3. Apply execution quality adjustments
        4. Generate final recommendation

        Args:
            ml_prediction: ML model prediction (optional).
            nlp_signals: NLP signal output (optional).
            execution_quality: Execution quality assessment (optional).

        Returns:
            AggregatedSignal with unified recommendation.

        Raises:
            ValueError: If both ml_prediction and nlp_signals are None.

        Examples:
            >>> signal = aggregator.aggregate(ml_pred, nlp_signals, exec_quality)
            >>> print(f"Recommendation: {signal.recommendation}")
        """
        if ml_prediction is None and nlp_signals is None:
            raise ValueError("At least one of ml_prediction or nlp_signals must be provided")

        # Extract symbol
        symbol = "UNKNOWN"
        if ml_prediction is not None:
            symbol = getattr(ml_prediction, "symbol", symbol)
        elif nlp_signals is not None:
            symbol = nlp_signals.ticker

        logger.info(
            "aggregating_signals",
            symbol=symbol,
            has_ml=ml_prediction is not None,
            has_nlp=nlp_signals is not None,
            has_execution=execution_quality is not None,
        )

        # 1. Calculate ML contribution
        ml_score = 0.0
        ml_confidence = 0.0

        if ml_prediction is not None:
            ml_score, ml_confidence = self._extract_ml_signal(ml_prediction)

        # 2. Calculate NLP contribution
        nlp_score = 0.0
        nlp_confidence = 0.0

        if nlp_signals is not None:
            nlp_score, nlp_confidence = self._extract_nlp_signal(nlp_signals)

        # 3. Calculate weighted average
        if ml_prediction is not None and nlp_signals is not None:
            # Both signals available - use configured weights
            weighted_score = (
                ml_score * self._ml_weight * ml_confidence
                + nlp_score * self._nlp_weight * nlp_confidence
            )
            overall_confidence = (
                ml_confidence * self._ml_weight + nlp_confidence * self._nlp_weight
            )
        elif ml_prediction is not None:
            # Only ML signal
            weighted_score = ml_score * ml_confidence
            overall_confidence = ml_confidence
        else:
            # Only NLP signal
            weighted_score = nlp_score * nlp_confidence
            overall_confidence = nlp_confidence

        # 4. Apply execution quality adjustments
        execution_feasibility = 1.0
        if execution_quality is not None:
            execution_feasibility = self._extract_execution_feasibility(execution_quality)

            # Dampen signal strength if execution is difficult
            weighted_score *= execution_feasibility

        # 5. Determine direction and strength
        direction = self._determine_direction(weighted_score)
        strength = weighted_score

        # 6. Calculate contributions
        ml_contribution = ml_score * self._ml_weight if ml_prediction is not None else 0.0
        nlp_contribution = nlp_score * self._nlp_weight if nlp_signals is not None else 0.0

        # 7. Generate recommendation and explanation
        recommendation = self._generate_recommendation(
            direction,
            strength,
            overall_confidence,
            execution_feasibility,
        )

        explanation = self._generate_explanation(
            ml_prediction,
            nlp_signals,
            execution_quality,
            direction,
            strength,
        )

        signal = AggregatedSignal(
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=overall_confidence,
            ml_contribution=ml_contribution,
            nlp_contribution=nlp_contribution,
            execution_feasibility=execution_feasibility,
            recommendation=recommendation,
            explanation=explanation,
        )

        logger.info(
            "signals_aggregated",
            symbol=symbol,
            direction=direction,
            strength=strength,
            confidence=overall_confidence,
            recommendation=recommendation,
        )

        return signal

    def apply_regime_conditioning(
        self,
        signal: AggregatedSignal,
        market_regime: str,
    ) -> AggregatedSignal:
        """Adjust signal based on market regime.

        Args:
            signal: Original aggregated signal.
            market_regime: Market regime (bull, bear, sideways, volatile).

        Returns:
            Adjusted signal based on market regime.

        Examples:
            >>> signal = aggregator.aggregate(ml_pred, nlp_signals)
            >>> adjusted = aggregator.apply_regime_conditioning(signal, "bear")
        """
        valid_regimes = ["bull", "bear", "sideways", "volatile"]
        if market_regime not in valid_regimes:
            logger.warning(
                "invalid_market_regime",
                regime=market_regime,
                valid=valid_regimes,
            )
            return signal

        # Apply regime-specific adjustments
        adjusted_strength = signal.strength
        adjusted_confidence = signal.confidence

        if market_regime == "bear":
            # In bear markets, be more conservative with long signals
            if signal.direction == "long":
                adjusted_strength *= 0.7
                adjusted_confidence *= 0.9
            # Be more aggressive with short signals
            elif signal.direction == "short":
                adjusted_strength *= 1.2
                adjusted_confidence = min(1.0, adjusted_confidence * 1.1)

        elif market_regime == "volatile":
            # In volatile markets, reduce all signal strengths
            adjusted_strength *= 0.8
            adjusted_confidence *= 0.85

        elif market_regime == "sideways":
            # In sideways markets, favor mean reversion
            adjusted_strength *= 0.9

        # Bull market: no adjustments (default behavior is optimistic)

        # Re-determine direction based on adjusted strength
        adjusted_direction = self._determine_direction(adjusted_strength)

        # Update recommendation
        adjusted_recommendation = self._generate_recommendation(
            adjusted_direction,
            adjusted_strength,
            adjusted_confidence,
            signal.execution_feasibility,
        )

        # Add regime note to explanation
        adjusted_explanation = (
            f"{signal.explanation}\n\n"
            f"Market Regime Adjustment: Signal adjusted for {market_regime} market conditions."
        )

        adjusted_signal = AggregatedSignal(
            symbol=signal.symbol,
            direction=adjusted_direction,
            strength=adjusted_strength,
            confidence=adjusted_confidence,
            ml_contribution=signal.ml_contribution,
            nlp_contribution=signal.nlp_contribution,
            execution_feasibility=signal.execution_feasibility,
            recommendation=adjusted_recommendation,
            explanation=adjusted_explanation,
        )

        logger.info(
            "regime_conditioning_applied",
            symbol=signal.symbol,
            regime=market_regime,
            original_strength=signal.strength,
            adjusted_strength=adjusted_strength,
        )

        return adjusted_signal

    def _extract_ml_signal(self, ml_prediction: Any) -> tuple[float, float]:
        """Extract signal and confidence from ML prediction.

        Args:
            ml_prediction: ML prediction result.

        Returns:
            Tuple of (signal_score, confidence).
        """
        # Handle different ML prediction formats
        if hasattr(ml_prediction, "predictions") and ml_prediction.predictions:
            # PredictResponse format
            pred = ml_prediction.predictions[0]
            direction = pred.predicted_direction

            # Convert direction to score
            if direction == "up":
                score = 0.7
            elif direction == "down":
                score = -0.7
            else:
                score = 0.0

            confidence = pred.confidence
        else:
            # Fallback: assume neutral prediction
            score = 0.0
            confidence = 0.5

        return score, confidence

    def _extract_nlp_signal(self, nlp_signals: NLPSignalOutput) -> tuple[float, float]:
        """Extract signal and confidence from NLP signals.

        Args:
            nlp_signals: NLP signal output.

        Returns:
            Tuple of (signal_score, confidence).
        """
        sentiment_score = nlp_signals.sentiment.score

        # Boost confidence if information is new
        confidence_boost = 0.1 if nlp_signals.sentiment.is_new_information else 0.0

        # Base confidence on analyst consensus
        confidence = nlp_signals.analyst_consensus.confidence + confidence_boost
        confidence = min(1.0, confidence)

        # Adjust score based on urgency
        urgency_multiplier = {
            "critical": 1.3,
            "high": 1.1,
            "medium": 1.0,
            "low": 0.8,
        }.get(nlp_signals.urgency, 1.0)

        adjusted_score = sentiment_score * urgency_multiplier
        adjusted_score = max(-1.0, min(1.0, adjusted_score))

        return adjusted_score, confidence

    def _extract_execution_feasibility(self, execution_quality: Any) -> float:
        """Extract execution feasibility score.

        Args:
            execution_quality: Execution quality assessment.

        Returns:
            Feasibility score (0.0 to 1.0).
        """
        if hasattr(execution_quality, "metrics"):
            # ExecutionQualityResponse format
            overall_score: float = float(execution_quality.metrics.overall_score)
            return overall_score / 100.0
        else:
            # Default to good execution
            return 0.85

    @staticmethod
    def _determine_direction(score: float) -> str:
        """Determine signal direction from score.

        Args:
            score: Signal score (-1 to 1).

        Returns:
            Direction string (long, short, neutral).
        """
        if score >= 0.2:
            return "long"
        elif score <= -0.2:
            return "short"
        else:
            return "neutral"

    @staticmethod
    def _generate_recommendation(
        direction: str,
        strength: float,
        confidence: float,
        execution_feasibility: float,
    ) -> str:
        """Generate human-readable recommendation.

        Args:
            direction: Signal direction.
            strength: Signal strength.
            confidence: Signal confidence.
            execution_feasibility: Execution feasibility.

        Returns:
            Recommendation string.
        """
        # Determine strength label
        abs_strength = abs(strength)
        if abs_strength >= 0.7:
            strength_label = "Strong"
        elif abs_strength >= 0.4:
            strength_label = "Moderate"
        else:
            strength_label = "Weak"

        # Determine confidence label
        if confidence >= 0.8:
            confidence_label = "High confidence"
        elif confidence >= 0.6:
            confidence_label = "Medium confidence"
        else:
            confidence_label = "Low confidence"

        # Check execution feasibility
        if execution_feasibility < 0.6:
            execution_note = " (Caution: Execution may be difficult)"
        else:
            execution_note = ""

        if direction == "neutral":
            return f"Hold. {confidence_label}.{execution_note}"

        action = "Buy" if direction == "long" else "Sell"
        return f"{strength_label} {action}. {confidence_label}.{execution_note}"

    @staticmethod
    def _generate_explanation(
        ml_prediction: Any | None,
        nlp_signals: NLPSignalOutput | None,
        execution_quality: Any | None,
        direction: str,
        strength: float,
    ) -> str:
        """Generate detailed explanation of the signal.

        Args:
            ml_prediction: ML prediction.
            nlp_signals: NLP signals.
            execution_quality: Execution quality.
            direction: Signal direction.
            strength: Signal strength.

        Returns:
            Explanation string.
        """
        parts = []

        parts.append(f"Signal Direction: {direction.upper()}")
        parts.append(f"Signal Strength: {strength:.2f}")

        if ml_prediction is not None:
            parts.append(
                f"\nML Model: Predicts {getattr(ml_prediction, 'predictions', [{}])[0].predicted_direction if hasattr(ml_prediction, 'predictions') else 'unknown'} movement"
            )

        if nlp_signals is not None:
            parts.append(
                f"\nNLP Analysis: {nlp_signals.sentiment.label.upper()} sentiment "
                f"(score: {nlp_signals.sentiment.score:.2f})"
            )
            parts.append(f"Analyst Consensus: {nlp_signals.analyst_consensus.rating.upper()}")
            parts.append(f"Urgency: {nlp_signals.urgency.upper()}")

            if nlp_signals.price_targets:
                avg_target = sum(t.target_price for t in nlp_signals.price_targets) / len(
                    nlp_signals.price_targets
                )
                parts.append(
                    f"Average Price Target: ${avg_target:.2f} "
                    f"({len(nlp_signals.price_targets)} analysts)"
                )

        if execution_quality is not None:
            parts.append(
                f"\nExecution Quality: "
                f"{getattr(execution_quality, 'overall_recommendation', 'Unknown')}"
            )

        return "\n".join(parts)
