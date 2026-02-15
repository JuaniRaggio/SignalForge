"""Confidence calculation and calibration for signal aggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class ConfidenceCalculator:
    """Calculate and calibrate prediction confidence."""

    def __init__(self) -> None:
        """Initialize the confidence calculator."""
        logger.info("Initialized ConfidenceCalculator")

    def calculate_integrated_confidence(
        self,
        ml_confidence: float,
        nlp_confidence: float,
        execution_score: float,
        signal_agreement: float,
    ) -> float:
        """Calculate overall confidence from component confidences.

        The integrated confidence is calculated as:
        1. Weight individual component confidences
        2. Boost confidence if ML and NLP signals agree
        3. Penalize confidence if signals conflict
        4. Apply execution feasibility adjustment

        Args:
            ml_confidence: ML model confidence (0 to 1).
            nlp_confidence: NLP analysis confidence (0 to 1).
            execution_score: Execution quality score (0 to 1).
            signal_agreement: Agreement between ML and NLP signals (-1 to 1).

        Returns:
            Integrated confidence score (0 to 1).
        """
        # Validate inputs
        ml_confidence = max(0.0, min(1.0, ml_confidence))
        nlp_confidence = max(0.0, min(1.0, nlp_confidence))
        execution_score = max(0.0, min(1.0, execution_score))
        signal_agreement = max(-1.0, min(1.0, signal_agreement))

        # Weighted average of component confidences
        # ML and NLP get higher weights than execution
        base_confidence = (
            0.4 * ml_confidence + 0.4 * nlp_confidence + 0.2 * execution_score
        )

        # Agreement adjustment
        # If signals agree (agreement > 0), boost confidence
        # If signals conflict (agreement < 0), reduce confidence
        agreement_adjustment = signal_agreement * 0.2

        # Calculate final confidence
        integrated_confidence = base_confidence + agreement_adjustment

        # Ensure bounds
        integrated_confidence = max(0.0, min(1.0, integrated_confidence))

        logger.debug(
            "Calculated integrated confidence",
            ml_confidence=ml_confidence,
            nlp_confidence=nlp_confidence,
            execution_score=execution_score,
            signal_agreement=signal_agreement,
            base_confidence=base_confidence,
            agreement_adjustment=agreement_adjustment,
            integrated_confidence=integrated_confidence,
        )

        return integrated_confidence

    def calibrate_confidence(
        self,
        raw_confidence: float,
        historical_accuracy: float | None = None,
    ) -> float:
        """Calibrate confidence based on historical accuracy.

        This method adjusts raw confidence scores to better reflect actual
        prediction accuracy based on historical performance.

        Args:
            raw_confidence: Raw confidence score (0 to 1).
            historical_accuracy: Historical accuracy of predictions (0 to 1).
                If None, no calibration is applied.

        Returns:
            Calibrated confidence score (0 to 1).
        """
        # Validate raw confidence
        raw_confidence = max(0.0, min(1.0, raw_confidence))

        # If no historical accuracy, return raw confidence
        if historical_accuracy is None:
            logger.debug("No historical accuracy provided, returning raw confidence")
            return raw_confidence

        # Validate historical accuracy
        historical_accuracy = max(0.0, min(1.0, historical_accuracy))

        # Simple calibration: blend raw confidence with historical accuracy
        # If historical accuracy is high, trust raw confidence more
        # If historical accuracy is low, pull confidence toward 0.5 (uncertain)
        calibration_weight = historical_accuracy

        calibrated_confidence = (
            calibration_weight * raw_confidence + (1 - calibration_weight) * 0.5
        )

        logger.debug(
            "Calibrated confidence",
            raw_confidence=raw_confidence,
            historical_accuracy=historical_accuracy,
            calibrated_confidence=calibrated_confidence,
        )

        return calibrated_confidence

    def calculate_agreement(
        self,
        ml_direction: float,
        nlp_direction: float,
    ) -> float:
        """Calculate agreement between ML and NLP signals.

        Args:
            ml_direction: ML signal direction (-1 to 1).
            nlp_direction: NLP signal direction (-1 to 1).

        Returns:
            Agreement score (-1 to 1) where:
                1.0 = perfect agreement
                0.0 = no correlation
                -1.0 = perfect disagreement
        """
        # Validate inputs
        ml_direction = max(-1.0, min(1.0, ml_direction))
        nlp_direction = max(-1.0, min(1.0, nlp_direction))

        # Calculate cosine similarity
        # For 1D vectors, this is just the product normalized
        agreement = ml_direction * nlp_direction

        # If both signals are near zero (neutral), consider this low agreement
        # rather than perfect agreement
        ml_strength = abs(ml_direction)
        nlp_strength = abs(nlp_direction)
        min_strength = min(ml_strength, nlp_strength)

        # Scale agreement by minimum strength
        # If either signal is weak, agreement is less meaningful
        scaled_agreement = agreement * min_strength

        logger.debug(
            "Calculated signal agreement",
            ml_direction=ml_direction,
            nlp_direction=nlp_direction,
            raw_agreement=agreement,
            scaled_agreement=scaled_agreement,
        )

        return scaled_agreement

    def get_confidence_interval(
        self,
        prediction: float,
        confidence: float,
    ) -> tuple[float, float]:
        """Get confidence interval for prediction.

        Args:
            prediction: Central prediction value.
            confidence: Confidence level (0 to 1).

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        # Validate confidence
        confidence = max(0.0, min(1.0, confidence))

        # Calculate interval width based on confidence
        # Higher confidence = narrower interval
        # Lower confidence = wider interval
        # Use inverse relationship: width proportional to (1 - confidence)
        base_width = 0.2  # 20% base width at 0 confidence
        width = base_width * (1 - confidence)

        # Calculate bounds
        lower_bound = prediction - width
        upper_bound = prediction + width

        logger.debug(
            "Calculated confidence interval",
            prediction=prediction,
            confidence=confidence,
            width=width,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        return (lower_bound, upper_bound)
