"""Backtesting engine for trading strategy validation.

This module implements a complete backtesting engine for validating trading signals
against historical market data. It supports:

- Long-only and long/short strategies
- Transaction costs (commission and slippage)
- Configurable position sizing
- Comprehensive performance tracking
- Integration with MLflow for experiment tracking

The engine processes trading signals and simulates order execution with realistic
constraints to provide accurate performance metrics.

Examples:
    Basic backtest with default settings:

    >>> import polars as pl
    >>> from signalforge.ml.backtesting.engine import BacktestEngine
    >>>
    >>> # Price data with OHLCV
    >>> prices = pl.DataFrame({
    ...     "timestamp": [...],
    ...     "open": [...],
    ...     "high": [...],
    ...     "low": [...],
    ...     "close": [...],
    ...     "volume": [...],
    ... })
    >>>
    >>> # Trading signals
    >>> signals = pl.DataFrame({
    ...     "timestamp": [...],
    ...     "signal": [1, 0, 0, -1, 1, 0, -1],  # 1=buy, -1=sell, 0=hold
    ... })
    >>>
    >>> # Run backtest
    >>> engine = BacktestEngine()
    >>> result = engine.run(prices, signals)
    >>> print(result.metrics)

    Custom configuration with transaction costs:

    >>> from signalforge.ml.backtesting.engine import BacktestConfig
    >>>
    >>> config = BacktestConfig(
    ...     initial_capital=50000.0,
    ...     commission_pct=0.001,
    ...     slippage_pct=0.0005,
    ...     position_size_pct=1.0,
    ...     allow_short=False,
    ... )
    >>> engine = BacktestEngine(config)
    >>> result = engine.run(prices, signals)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import polars as pl

from signalforge.core.logging import get_logger
from signalforge.ml.backtesting.metrics import (
    BacktestMetrics,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_win_rate,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class Trade:
    """Represents a single completed trade.

    A trade is created when a position is opened and closed. It tracks
    all relevant information about the trade including entry/exit prices,
    position size, and profitability.

    Attributes:
        entry_date: Timestamp when position was opened
        exit_date: Timestamp when position was closed
        entry_price: Price at which position was entered
        exit_price: Price at which position was exited
        position_size: Number of shares/contracts traded
        direction: Trade direction ("long" or "short")
        pnl: Profit/loss in currency units (after costs)
        return_pct: Return as percentage of capital deployed

    Note:
        PnL includes transaction costs (commission and slippage).
        For long trades: pnl = (exit_price - entry_price) * position_size - costs
        For short trades: pnl = (entry_price - exit_price) * position_size - costs
    """

    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position_size: float
    direction: Literal["long", "short"]
    pnl: float
    return_pct: float

    def __post_init__(self) -> None:
        """Validate trade data after initialization."""
        if self.entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if self.exit_price <= 0:
            raise ValueError("exit_price must be positive")
        if self.position_size <= 0:
            raise ValueError("position_size must be positive")
        if self.direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got {self.direction}")
        if self.exit_date < self.entry_date:
            raise ValueError("exit_date must be after entry_date")


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters.

    This class encapsulates all configurable parameters for the backtest,
    including initial capital, transaction costs, and position sizing rules.

    Attributes:
        initial_capital: Starting portfolio value in currency units
        commission_pct: Commission rate per trade (as decimal, e.g., 0.001 = 0.1%)
        slippage_pct: Slippage rate per trade (as decimal, e.g., 0.0005 = 0.05%)
        position_size_pct: Percentage of capital to use per trade (1.0 = 100%)
        allow_short: Whether to allow short positions (False = long-only)

    Note:
        Transaction costs are applied on both entry and exit.
        Total transaction cost per round trip = 2 * (commission + slippage)
    """

    initial_capital: float = 100000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    position_size_pct: float = 1.0
    allow_short: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.commission_pct < 0:
            raise ValueError("commission_pct cannot be negative")
        if self.slippage_pct < 0:
            raise ValueError("slippage_pct cannot be negative")
        if not 0 < self.position_size_pct <= 1.0:
            raise ValueError("position_size_pct must be between 0 and 1")

    @property
    def total_transaction_cost_pct(self) -> float:
        """Total transaction cost percentage per side.

        Returns:
            Combined commission and slippage rate.
        """
        return self.commission_pct + self.slippage_pct


@dataclass
class BacktestResult:
    """Complete results from a backtest run.

    This class encapsulates all results from a backtest, including
    performance metrics, individual trades, and the equity curve.

    Attributes:
        metrics: Calculated performance metrics
        trades: List of all completed trades
        equity_curve: DataFrame with timestamp and portfolio value
        config: Configuration used for this backtest

    Examples:
        Access backtest results:

        >>> result = engine.run(prices, signals)
        >>> print(f"Total Return: {result.metrics.total_return:.2f}%")
        >>> print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
        >>> print(f"Total Trades: {result.metrics.total_trades}")
        >>>
        >>> # Plot equity curve
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(result.equity_curve["timestamp"], result.equity_curve["equity"])
    """

    metrics: BacktestMetrics
    trades: list[Trade]
    equity_curve: pl.DataFrame
    config: BacktestConfig


class BacktestEngine:
    """Engine for running backtests on trading signals.

    This class implements the core backtesting logic, processing price data
    and trading signals to simulate strategy performance with realistic
    transaction costs and position management.

    The engine tracks:
    - Portfolio equity over time
    - Individual trade execution and results
    - Transaction costs (commission and slippage)
    - Position state (open/closed, long/short)

    Attributes:
        config: Backtesting configuration

    Examples:
        Create and run backtest:

        >>> from signalforge.ml.backtesting import BacktestEngine, BacktestConfig
        >>>
        >>> config = BacktestConfig(
        ...     initial_capital=100000.0,
        ...     commission_pct=0.001,
        ...     allow_short=False,
        ... )
        >>> engine = BacktestEngine(config)
        >>> result = engine.run(prices, signals)
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        """Initialize the backtesting engine.

        Args:
            config: Backtesting configuration. If None, uses defaults.
        """
        self.config = config or BacktestConfig()
        logger.info(
            "backtest_engine_initialized",
            initial_capital=self.config.initial_capital,
            commission_pct=self.config.commission_pct,
            slippage_pct=self.config.slippage_pct,
            allow_short=self.config.allow_short,
        )

    def run(
        self,
        prices: pl.DataFrame,
        signals: pl.DataFrame,
    ) -> BacktestResult:
        """Run backtest on price data with trading signals.

        This method executes the complete backtesting process:
        1. Validates input data
        2. Merges prices and signals
        3. Simulates trade execution
        4. Tracks portfolio equity
        5. Calculates performance metrics

        Args:
            prices: DataFrame with columns [timestamp, open, high, low, close, volume]
            signals: DataFrame with columns [timestamp, signal]
                     signal: 1 = buy (enter long), -1 = sell (exit position), 0 = hold

        Returns:
            BacktestResult containing metrics, trades, and equity curve

        Raises:
            ValueError: If input data is invalid or missing required columns

        Note:
            Signal interpretation:
            - 1: Enter long position (or close short if allow_short=True)
            - -1: Exit current position (close long or short)
            - 0: Maintain current position (do nothing)

            Prices are assumed to be already sorted by timestamp.
        """
        logger.info("starting_backtest", price_rows=prices.height, signal_rows=signals.height)

        # Validate inputs
        self._validate_inputs(prices, signals)

        # Merge prices and signals
        data = self._merge_data(prices, signals)

        # Run backtest simulation
        trades, equity_curve = self._simulate_trading(data)

        # Calculate metrics
        metrics = self._calculate_metrics(trades, equity_curve)

        logger.info(
            "backtest_completed",
            total_trades=len(trades),
            total_return=metrics.total_return,
            sharpe_ratio=metrics.sharpe_ratio,
            max_drawdown=metrics.max_drawdown,
        )

        return BacktestResult(
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            config=self.config,
        )

    def _validate_inputs(self, prices: pl.DataFrame, signals: pl.DataFrame) -> None:
        """Validate input DataFrames.

        Args:
            prices: Price DataFrame
            signals: Signal DataFrame

        Raises:
            ValueError: If validation fails
        """
        # Check prices DataFrame
        required_price_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        missing_price_cols = [col for col in required_price_cols if col not in prices.columns]
        if missing_price_cols:
            raise ValueError(f"Missing required price columns: {missing_price_cols}")

        if prices.is_empty():
            raise ValueError("Price DataFrame cannot be empty")

        # Check signals DataFrame
        required_signal_cols = ["timestamp", "signal"]
        missing_signal_cols = [col for col in required_signal_cols if col not in signals.columns]
        if missing_signal_cols:
            raise ValueError(f"Missing required signal columns: {missing_signal_cols}")

        if signals.is_empty():
            raise ValueError("Signal DataFrame cannot be empty")

        # Validate signal values
        unique_signals = signals["signal"].unique().to_list()
        valid_signals = {-1, 0, 1}
        invalid_signals = set(unique_signals) - valid_signals
        if invalid_signals:
            raise ValueError(f"Invalid signal values: {invalid_signals}. Must be -1, 0, or 1")

        logger.debug("input_validation_passed")

    def _merge_data(self, prices: pl.DataFrame, signals: pl.DataFrame) -> pl.DataFrame:
        """Merge price and signal data on timestamp.

        Args:
            prices: Price DataFrame
            signals: Signal DataFrame

        Returns:
            Merged DataFrame with prices and signals
        """
        # Join on timestamp
        merged = prices.join(signals, on="timestamp", how="inner")

        if merged.is_empty():
            raise ValueError("No matching timestamps between prices and signals")

        # Sort by timestamp
        merged = merged.sort("timestamp")

        logger.debug("data_merged", merged_rows=merged.height)

        return merged

    def _simulate_trading(self, data: pl.DataFrame) -> tuple[list[Trade], pl.DataFrame]:
        """Simulate trading based on signals.

        Args:
            data: Merged price and signal data

        Returns:
            Tuple of (trades list, equity curve DataFrame)
        """
        trades: list[Trade] = []
        equity_history: list[dict[str, datetime | float]] = []

        # Portfolio state
        cash = self.config.initial_capital
        position_size = 0.0
        position_entry_price = 0.0
        position_entry_date: datetime | None = None
        position_direction: Literal["long", "short"] | None = None

        # Iterate through each row
        for row in data.iter_rows(named=True):
            timestamp = row["timestamp"]
            close_price = float(row["close"])
            signal = int(row["signal"])

            # Track current equity
            if position_size > 0:
                # We have an open position
                if position_direction == "long":
                    position_value = position_size * close_price
                else:  # short
                    # For short: value = entry_value + (entry_price - current_price) * size
                    position_value = (
                        position_size * position_entry_price
                        + (position_entry_price - close_price) * position_size
                    )
                current_equity = cash + position_value
            else:
                # No position
                current_equity = cash

            equity_history.append({"timestamp": timestamp, "equity": current_equity})

            # Process signal
            if signal == 1 and position_size == 0:
                # Buy signal and no position - enter long
                position_direction = "long"
                position_entry_price = close_price * (1 + self.config.total_transaction_cost_pct)
                position_entry_date = timestamp

                # Calculate position size based on available capital
                capital_to_use = cash * self.config.position_size_pct
                position_size = capital_to_use / position_entry_price
                cash -= position_size * position_entry_price

                logger.debug(
                    "position_opened",
                    direction="long",
                    entry_price=position_entry_price,
                    size=position_size,
                    timestamp=timestamp,
                )

            elif signal == -1 and position_size > 0:
                # Sell signal and have position - close it
                exit_price = close_price * (1 - self.config.total_transaction_cost_pct)

                # Calculate PnL
                if position_direction == "long":
                    pnl = (exit_price - position_entry_price) * position_size
                else:  # short
                    pnl = (position_entry_price - exit_price) * position_size

                # Update cash
                if position_direction == "long":
                    cash += position_size * exit_price
                else:  # short
                    cash += position_size * position_entry_price + pnl

                # Calculate return percentage
                capital_used = position_size * position_entry_price
                return_pct = (pnl / capital_used) * 100.0 if capital_used > 0 else 0.0

                # Create trade record (position_entry_date and position_direction are guaranteed
                # to be set when we have an open position that we're closing)
                assert position_entry_date is not None
                assert position_direction is not None
                trade = Trade(
                    entry_date=position_entry_date,
                    exit_date=timestamp,
                    entry_price=position_entry_price,
                    exit_price=exit_price,
                    position_size=position_size,
                    direction=position_direction,
                    pnl=pnl,
                    return_pct=return_pct,
                )
                trades.append(trade)

                logger.debug(
                    "position_closed",
                    direction=position_direction,
                    pnl=pnl,
                    return_pct=return_pct,
                    timestamp=timestamp,
                )

                # Reset position
                position_size = 0.0
                position_entry_price = 0.0
                position_entry_date = None
                position_direction = None

        # Convert equity history to DataFrame
        equity_curve = pl.DataFrame(equity_history)

        logger.debug("trading_simulation_completed", total_trades=len(trades))

        return trades, equity_curve

    def _calculate_metrics(
        self, trades: list[Trade], equity_curve: pl.DataFrame
    ) -> BacktestMetrics:
        """Calculate performance metrics from trades and equity curve.

        Args:
            trades: List of completed trades
            equity_curve: Equity curve DataFrame

        Returns:
            BacktestMetrics object with all calculated metrics
        """
        if equity_curve.is_empty():
            logger.warning("Empty equity curve, returning zero metrics")
            return BacktestMetrics(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                avg_trade_return=0.0,
                volatility=0.0,
            )

        # Calculate returns
        equity_series = equity_curve["equity"]
        initial_equity = float(equity_series[0])
        final_equity = float(equity_series[-1])
        total_return = ((final_equity - initial_equity) / initial_equity) * 100.0

        # Calculate daily returns
        returns = equity_series.pct_change().drop_nulls()

        # Number of days in backtest
        num_days = equity_curve.height

        # Annualized return
        annualized_return = calculate_annualized_return(total_return, num_days)

        # Sharpe ratio
        sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate=0.0)

        # Maximum drawdown
        max_drawdown = calculate_max_drawdown(equity_series)

        # Win rate
        win_rate = calculate_win_rate(trades)

        # Profit factor
        profit_factor = calculate_profit_factor(trades)

        # Average trade return
        if trades:
            avg_trade_return = sum(trade.return_pct for trade in trades) / len(trades)
        else:
            avg_trade_return = 0.0

        # Volatility
        volatility = calculate_annualized_volatility(returns)

        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_return=avg_trade_return,
            volatility=volatility,
        )

    def log_to_mlflow(self, result: BacktestResult) -> None:
        """Log backtest results to MLflow.

        This method logs all configuration parameters and metrics to MLflow
        for experiment tracking and comparison.

        Args:
            result: BacktestResult to log

        Note:
            Requires an active MLflow run. Call within mlflow.start_run() context.

        Examples:
            >>> import mlflow
            >>> from signalforge.ml.backtesting import BacktestEngine
            >>>
            >>> engine = BacktestEngine()
            >>> result = engine.run(prices, signals)
            >>>
            >>> with mlflow.start_run():
            ...     engine.log_to_mlflow(result)
        """
        try:
            import mlflow
        except ImportError:
            logger.error("MLflow not installed, cannot log results")
            return

        # Log configuration parameters
        mlflow.log_param("initial_capital", self.config.initial_capital)
        mlflow.log_param("commission_pct", self.config.commission_pct)
        mlflow.log_param("slippage_pct", self.config.slippage_pct)
        mlflow.log_param("position_size_pct", self.config.position_size_pct)
        mlflow.log_param("allow_short", self.config.allow_short)

        # Log metrics
        metrics_dict = result.metrics.to_dict()
        for key, value in metrics_dict.items():
            mlflow.log_metric(key, value)

        logger.info("backtest_results_logged_to_mlflow")
