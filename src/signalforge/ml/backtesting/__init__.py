"""Backtesting module for validating trading signals and ML models.

This module provides a complete backtesting engine for testing trading strategies
against historical market data. It includes:

- BacktestEngine: Main engine for running backtests with strategies
- BacktestConfig: Configuration for backtest parameters
- BacktestResult: Complete results including metrics, trades, and equity curve
- BacktestMetrics: Performance metrics from backtesting (deprecated, use BacktestResult)
- Trade: Individual trade representation
- TradingStrategy: Base class for implementing trading strategies
- ThresholdStrategy: Simple threshold-based strategy
- RankingStrategy: Rank stocks by predicted return
- LongShortStrategy: Long top N, short bottom N stocks

The backtesting engine supports:
- Long-only and long/short strategies
- Transaction costs (commissions and slippage)
- Configurable position sizing
- Comprehensive performance metrics (Sharpe ratio, max drawdown, win rate, etc.)
- Walk-forward analysis with model retraining
- Integration with MLflow for experiment tracking

Examples:
    Basic backtest with default configuration:

    >>> from signalforge.ml.backtesting import BacktestEngine
    >>> from signalforge.ml.backtesting.strategies import ThresholdStrategy
    >>> import polars as pl
    >>>
    >>> # Create prediction and price data
    >>> predictions = pl.DataFrame({
    ...     "symbol": ["AAPL", "GOOGL"],
    ...     "timestamp": [...],
    ...     "predicted_return": [0.02, 0.03],
    ...     "confidence": [0.8, 0.7],
    ... })
    >>> prices = pl.DataFrame({
    ...     "symbol": ["AAPL", "GOOGL"],
    ...     "timestamp": [...],
    ...     "open": [...], "high": [...], "low": [...], "close": [...], "volume": [...],
    ... })
    >>>
    >>> # Run backtest with strategy
    >>> engine = BacktestEngine()
    >>> strategy = ThresholdStrategy(buy_threshold=0.02)
    >>> result = engine.run_with_strategy(predictions, prices, strategy)
    >>> print(f"Total Return: {result.total_return:.2f}%")
    >>> print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")

    Walk-forward backtest with model retraining:

    >>> from signalforge.ml.models import LSTMPredictor
    >>> from signalforge.ml.backtesting import BacktestEngine
    >>> from signalforge.ml.backtesting.strategies import RankingStrategy
    >>>
    >>> model = LSTMPredictor()
    >>> strategy = RankingStrategy(top_n=10)
    >>> results = engine.run_walk_forward(model, data, strategy, n_splits=5)
    >>> avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results)
"""

from __future__ import annotations

from signalforge.ml.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    Position,
    Trade,
)
from signalforge.ml.backtesting.metrics import (
    BacktestMetrics,
    calculate_all,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_calmar_ratio,
    calculate_information_ratio,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_win_rate,
)
from signalforge.ml.backtesting.strategies import (
    LongShortStrategy,
    RankingStrategy,
    ThresholdStrategy,
    TradeSignal,
    TradingStrategy,
)

__all__ = [
    # Engine
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "Position",
    "Trade",
    # Strategies
    "TradingStrategy",
    "TradeSignal",
    "ThresholdStrategy",
    "RankingStrategy",
    "LongShortStrategy",
    # Metrics (deprecated, use BacktestResult attributes)
    "BacktestMetrics",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_calmar_ratio",
    "calculate_win_rate",
    "calculate_profit_factor",
    "calculate_information_ratio",
    "calculate_annualized_return",
    "calculate_annualized_volatility",
    "calculate_all",
]
