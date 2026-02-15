"""Integration layer for signal aggregation.

This module provides components for combining ML predictions, NLP signals,
execution quality assessments, and market regime detection into unified
trading signals.

Key Components:
- EnhancedSignalAggregator: Main aggregation engine
- MarketRegimeDetector: Detects current market conditions
- ConfidenceCalculator: Calculates and calibrates signal confidence
- IntegratedSignal: Final aggregated signal output
- AggregationConfig: Configuration for signal weights and thresholds

Examples:
    Basic signal aggregation:

    >>> from signalforge.integration import EnhancedSignalAggregator
    >>> aggregator = EnhancedSignalAggregator()
    >>> signal = await aggregator.aggregate(
    ...     symbol="AAPL",
    ...     ml_prediction=ml_result,
    ...     nlp_signals=nlp_output,
    ...     execution_quality=exec_quality,
    ... )
    >>> print(f"Recommendation: {signal.recommendation}")

    Batch processing:

    >>> signals = await aggregator.aggregate_batch(["AAPL", "MSFT", "GOOGL"])
    >>> for signal in signals:
    ...     print(f"{signal.symbol}: {signal.direction.value}")

    Custom configuration:

    >>> from signalforge.integration import AggregationConfig
    >>> config = AggregationConfig(
    ...     ml_weight=0.5,
    ...     nlp_weight=0.3,
    ...     min_confidence_threshold=0.6,
    ... )
    >>> aggregator = EnhancedSignalAggregator(config=config)
"""

from signalforge.integration.aggregator import EnhancedSignalAggregator
from signalforge.integration.confidence import ConfidenceCalculator
from signalforge.integration.regime import MarketRegimeDetector
from signalforge.integration.schemas import (
    AggregationConfig,
    IntegratedSignal,
    MarketRegime,
    SignalDirection,
)

__all__ = [
    "EnhancedSignalAggregator",
    "MarketRegimeDetector",
    "ConfidenceCalculator",
    "IntegratedSignal",
    "AggregationConfig",
    "MarketRegime",
    "SignalDirection",
]
