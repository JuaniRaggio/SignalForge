"""Benchmark comparison module for SignalForge.

This module provides tools for comparing strategy performance against
standard market benchmarks and calculating relative performance metrics,
as well as comprehensive performance analysis tools and paper trading simulation.
"""

from signalforge.benchmark.comparator import (
    BenchmarkComparator,
    BenchmarkData,
    ComparisonResult,
)
from signalforge.benchmark.paper_portfolio import (
    PaperPortfolio,
    PortfolioConfig,
    PositionState,
)
from signalforge.benchmark.performance import (
    PerformanceCalculator,
    PerformanceMetrics,
    annualize_returns,
    annualize_volatility,
)
from signalforge.benchmark.publisher import (
    BenchmarkPublisher,
    PublishConfig,
    PublishedReport,
)

__all__ = [
    "BenchmarkComparator",
    "BenchmarkData",
    "ComparisonResult",
    "PaperPortfolio",
    "PerformanceCalculator",
    "PerformanceMetrics",
    "PortfolioConfig",
    "PositionState",
    "annualize_returns",
    "annualize_volatility",
    "BenchmarkPublisher",
    "PublishConfig",
    "PublishedReport",
]
