"""Execution quality module for signal validation.

This module provides tools for assessing the executability of trading signals
under real market conditions, including liquidity analysis, slippage estimation,
spread calculation, and volume filtering.
"""

from signalforge.execution.liquidity import LiquidityScorer
from signalforge.execution.schemas import (
    LiquidityScore,
    SlippageEstimate,
    SpreadMetrics,
    VolumeFilterResult,
)
from signalforge.execution.slippage import SlippageEstimator
from signalforge.execution.spread import SpreadCalculator
from signalforge.execution.volume_filter import VolumeFilter

__all__ = [
    # Classes
    "LiquidityScorer",
    "SlippageEstimator",
    "SpreadCalculator",
    "VolumeFilter",
    # Schemas
    "LiquidityScore",
    "SlippageEstimate",
    "SpreadMetrics",
    "VolumeFilterResult",
]
