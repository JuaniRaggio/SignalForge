"""Backtesting module for validating trading signals.

This module provides a complete backtesting engine for testing trading strategies
against historical market data. It includes:

- BacktestEngine: Main engine for running backtests
- BacktestConfig: Configuration for backtest parameters
- BacktestResult: Complete results including metrics, trades, and equity curve
- BacktestMetrics: Performance metrics from backtesting
- Trade: Individual trade representation

The backtesting engine supports:
- Long-only and long/short strategies
- Transaction costs (commissions and slippage)
- Configurable position sizing
- Comprehensive performance metrics (Sharpe ratio, max drawdown, win rate, etc.)
- Integration with MLflow for experiment tracking

Examples:
    Basic backtest with default configuration:

    >>> from signalforge.ml.backtesting import BacktestEngine
    >>> import polars as pl
    >>>
    >>> # Create price and signal data
    >>> prices = pl.DataFrame({...})
    >>> signals = pl.DataFrame({...})
    >>>
    >>> # Run backtest
    >>> engine = BacktestEngine()
    >>> result = engine.run(prices, signals)
    >>> print(result.metrics)

    Custom configuration with transaction costs:

    >>> from signalforge.ml.backtesting import BacktestEngine, BacktestConfig
    >>>
    >>> config = BacktestConfig(
    ...     initial_capital=50000.0,
    ...     commission_pct=0.001,
    ...     slippage_pct=0.0005,
    ...     allow_short=True,
    ... )
    >>> engine = BacktestEngine(config)
    >>> result = engine.run(prices, signals)
"""

from __future__ import annotations

from signalforge.ml.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    Trade,
)
from signalforge.ml.backtesting.metrics import (
    BacktestMetrics,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_win_rate,
)

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "Trade",
    "BacktestMetrics",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "calculate_win_rate",
]
