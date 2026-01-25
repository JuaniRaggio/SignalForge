"""Execution quality module for signal validation.

This module provides tools for assessing the executability of trading signals
under real market conditions, including liquidity analysis and slippage estimation.
"""

from signalforge.execution.liquidity import (
    LiquidityMetrics,
    assess_liquidity,
    calculate_avg_daily_volume,
    calculate_liquidity_score,
    get_cached_liquidity_metrics,
)
from signalforge.execution.slippage import (
    SlippageEstimate,
    calculate_execution_risk,
    estimate_slippage,
)

__all__ = [
    "LiquidityMetrics",
    "assess_liquidity",
    "calculate_avg_daily_volume",
    "calculate_liquidity_score",
    "get_cached_liquidity_metrics",
    "SlippageEstimate",
    "calculate_execution_risk",
    "estimate_slippage",
]
