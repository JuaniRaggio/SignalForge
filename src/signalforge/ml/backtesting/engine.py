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
    calculate_sortino_ratio,
    calculate_win_rate,
)

if TYPE_CHECKING:
    from signalforge.ml.backtesting.strategies import TradingStrategy
    from signalforge.ml.models.base import BasePredictor

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents an open position in the portfolio.

    Attributes:
        size: Number of shares held
        entry_price: Price at which position was entered
        entry_date: Date when position was opened
    """

    size: float
    entry_price: float
    entry_date: datetime


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
        total_return: Total return over backtest period (%)
        annualized_return: Annualized return (%)
        sharpe_ratio: Risk-adjusted return metric
        sortino_ratio: Downside risk-adjusted return metric
        max_drawdown: Maximum peak-to-trough decline (%)
        calmar_ratio: Annualized return / max drawdown
        win_rate: Percentage of winning trades
        profit_factor: Ratio of gross profit to gross loss
        total_trades: Number of completed trades
        avg_trade_return: Average return per trade (%)
        avg_holding_period: Average days holding a position
        equity_curve: DataFrame with timestamp and portfolio value
        trade_log: DataFrame with all trade details
        monthly_returns: DataFrame with monthly return breakdown
        config: Configuration used for this backtest (deprecated, for compatibility)

    Examples:
        Access backtest results:

        >>> result = engine.run(prices, signals, strategy)
        >>> print(f"Total Return: {result.total_return:.2f}%")
        >>> print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        >>> print(f"Total Trades: {result.total_trades}")
        >>>
        >>> # Plot equity curve
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(result.equity_curve["timestamp"], result.equity_curve["equity"])
    """

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    avg_holding_period: float
    equity_curve: pl.DataFrame
    trade_log: pl.DataFrame
    monthly_returns: pl.DataFrame
    config: BacktestConfig | None = None  # For backward compatibility


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

        # Create trade log
        if trades:
            trade_log = pl.DataFrame(
                {
                    "entry_date": [t.entry_date for t in trades],
                    "exit_date": [t.exit_date for t in trades],
                    "entry_price": [t.entry_price for t in trades],
                    "exit_price": [t.exit_price for t in trades],
                    "position_size": [t.position_size for t in trades],
                    "direction": [t.direction for t in trades],
                    "pnl": [t.pnl for t in trades],
                    "return_pct": [t.return_pct for t in trades],
                }
            )
        else:
            trade_log = pl.DataFrame()

        # Calculate average holding period
        if trades:
            holding_periods = [(t.exit_date - t.entry_date).days for t in trades]
            avg_holding_period = sum(holding_periods) / len(holding_periods)
        else:
            avg_holding_period = 0.0

        # Calculate monthly returns
        monthly_returns = self._calculate_monthly_returns(equity_curve)

        # Calculate Sortino and Calmar ratios
        returns = equity_curve["equity"].pct_change().drop_nulls()
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
        calmar = (
            metrics.annualized_return / metrics.max_drawdown
            if metrics.max_drawdown > 0
            else 0.0
        )

        return BacktestResult(
            total_return=metrics.total_return,
            annualized_return=metrics.annualized_return,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=sortino,
            max_drawdown=metrics.max_drawdown,
            calmar_ratio=calmar,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            total_trades=metrics.total_trades,
            avg_trade_return=metrics.avg_trade_return,
            avg_holding_period=avg_holding_period,
            equity_curve=equity_curve,
            trade_log=trade_log,
            monthly_returns=monthly_returns,
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

        # Log metrics directly from result
        mlflow.log_metric("total_return", result.total_return)
        mlflow.log_metric("annualized_return", result.annualized_return)
        mlflow.log_metric("sharpe_ratio", result.sharpe_ratio)
        mlflow.log_metric("sortino_ratio", result.sortino_ratio)
        mlflow.log_metric("max_drawdown", result.max_drawdown)
        mlflow.log_metric("calmar_ratio", result.calmar_ratio)
        mlflow.log_metric("win_rate", result.win_rate)
        mlflow.log_metric("profit_factor", result.profit_factor)
        mlflow.log_metric("total_trades", result.total_trades)
        mlflow.log_metric("avg_trade_return", result.avg_trade_return)
        mlflow.log_metric("avg_holding_period", result.avg_holding_period)

        logger.info("backtest_results_logged_to_mlflow")

    def run_with_strategy(
        self,
        predictions: pl.DataFrame,
        prices: pl.DataFrame,
        strategy: TradingStrategy,
    ) -> BacktestResult:
        """Run backtest using ML predictions and a trading strategy.

        This is the main method for strategy-based backtesting. It:
        1. Generates signals from predictions using the strategy
        2. Executes trades with commission/slippage
        3. Tracks portfolio value over time
        4. Calculates comprehensive performance metrics

        Args:
            predictions: DataFrame with [symbol, timestamp, predicted_return, confidence]
                        predicted_return should be decimal (e.g., 0.02 for 2%)
            prices: DataFrame with [symbol, timestamp, open, high, low, close, volume]
            strategy: Trading strategy instance to generate signals

        Returns:
            BacktestResult with all metrics and trade history

        Raises:
            ValueError: If input data is invalid or missing required columns

        Examples:
            >>> from signalforge.ml.backtesting import BacktestEngine
            >>> from signalforge.ml.backtesting.strategies import ThresholdStrategy
            >>>
            >>> strategy = ThresholdStrategy(buy_threshold=0.02)
            >>> result = engine.run_with_strategy(predictions, prices, strategy)
        """
        logger.info(
            "starting_strategy_backtest",
            prediction_rows=predictions.height,
            price_rows=prices.height,
            strategy=strategy.__class__.__name__,
        )

        # Validate inputs
        self._validate_strategy_inputs(predictions, prices)

        # Initialize portfolio state
        cash = self.config.initial_capital
        positions: dict[str, Position] = {}  # symbol -> Position
        trades: list[Trade] = []
        equity_history: list[dict[str, datetime | float]] = []

        # Get unique timestamps and sort
        timestamps = sorted(predictions["timestamp"].unique().to_list())

        # Iterate through each timestamp
        for timestamp in timestamps:
            # Get predictions for this timestamp
            current_preds = predictions.filter(pl.col("timestamp") == timestamp)

            # Calculate current portfolio value
            current_positions = {
                symbol: pos.size for symbol, pos in positions.items() if pos.size > 0
            }

            # Generate signals from strategy
            signals = strategy.generate_signals(current_preds, current_positions)

            # Execute signals
            for signal in signals:
                symbol = signal.symbol

                # Get current price for this symbol
                symbol_prices = prices.filter(
                    (pl.col("symbol") == symbol) & (pl.col("timestamp") == timestamp)
                )

                if symbol_prices.is_empty():
                    logger.warning(f"No price data for {symbol} at {timestamp}")
                    continue

                close_price = float(symbol_prices["close"][0])

                # Process signal
                if signal.action == "buy":
                    # Enter long position
                    if symbol not in positions or positions[symbol].size == 0:
                        # Apply transaction costs
                        entry_price = close_price * (1 + self.config.total_transaction_cost_pct)
                        capital_to_use = cash * signal.size
                        position_size = capital_to_use / entry_price

                        if capital_to_use <= cash:
                            cash -= capital_to_use
                            positions[symbol] = Position(
                                size=position_size,
                                entry_price=entry_price,
                                entry_date=timestamp,
                            )
                            logger.debug(
                                "position_opened",
                                symbol=symbol,
                                size=position_size,
                                entry_price=entry_price,
                            )

                elif signal.action == "sell" and symbol in positions and positions[symbol].size > 0:
                    # Exit position
                        pos = positions[symbol]
                        position_size = pos.size
                        entry_price = pos.entry_price
                        entry_date = pos.entry_date

                        # Apply transaction costs
                        exit_price = close_price * (1 - self.config.total_transaction_cost_pct)

                        # Calculate PnL
                        pnl = (exit_price - entry_price) * position_size

                        # Update cash
                        cash += position_size * exit_price

                        # Calculate return
                        capital_used = position_size * entry_price
                        return_pct = (pnl / capital_used) * 100.0 if capital_used > 0 else 0.0

                        # Record trade
                        trade = Trade(
                            entry_date=entry_date,
                            exit_date=timestamp,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            position_size=position_size,
                            direction="long",
                            pnl=pnl,
                            return_pct=return_pct,
                        )
                        trades.append(trade)

                        # Clear position
                        positions[symbol] = Position(size=0.0, entry_price=0.0, entry_date=timestamp)

                        logger.debug(
                            "position_closed",
                            symbol=symbol,
                            pnl=pnl,
                            return_pct=return_pct,
                        )

            # Calculate current equity
            positions_value = 0.0
            for symbol, pos in positions.items():
                if pos.size > 0:
                    # Get current price
                    symbol_prices = prices.filter(
                        (pl.col("symbol") == symbol) & (pl.col("timestamp") == timestamp)
                    )
                    if not symbol_prices.is_empty():
                        current_price = float(symbol_prices["close"][0])
                        positions_value += pos.size * current_price

            current_equity = cash + positions_value
            equity_history.append({"timestamp": timestamp, "equity": current_equity})

        # Create equity curve DataFrame
        equity_curve = pl.DataFrame(equity_history)

        # Calculate metrics
        result = self._calculate_strategy_metrics(trades, equity_curve)

        logger.info(
            "strategy_backtest_completed",
            total_trades=result.total_trades,
            total_return=result.total_return,
            sharpe_ratio=result.sharpe_ratio,
        )

        return result

    def run_walk_forward(
        self,
        model: BasePredictor,
        data: pl.DataFrame,
        strategy: TradingStrategy,
        n_splits: int = 5,
        train_size: int = 252,
        test_size: int = 63,
    ) -> list[BacktestResult]:
        """Run walk-forward backtest with model retraining.

        Walk-forward analysis is a robust backtesting methodology that:
        1. Splits data into sequential train/test periods
        2. Trains model on train period
        3. Generates predictions on test period
        4. Runs backtest on test period
        5. Moves window forward and repeats

        This prevents look-ahead bias and simulates real trading conditions.

        Args:
            model: Model instance to train and predict with
            data: DataFrame with [symbol, timestamp, features, target]
            strategy: Trading strategy to use
            n_splits: Number of walk-forward windows
            train_size: Days in training window
            test_size: Days in test window

        Returns:
            List of BacktestResult, one per split

        Examples:
            >>> from signalforge.ml.models import LSTMPredictor
            >>> from signalforge.ml.backtesting.strategies import ThresholdStrategy
            >>>
            >>> model = LSTMPredictor()
            >>> strategy = ThresholdStrategy()
            >>> results = engine.run_walk_forward(model, data, strategy, n_splits=5)
            >>> avg_sharpe = sum(r.sharpe_ratio for r in results) / len(results)
        """
        logger.info(
            "starting_walk_forward_backtest",
            n_splits=n_splits,
            train_size=train_size,
            test_size=test_size,
        )

        # Validate data
        required_cols = ["symbol", "timestamp"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Sort data by timestamp
        data = data.sort("timestamp")

        # Get unique timestamps
        timestamps = sorted(data["timestamp"].unique().to_list())

        if len(timestamps) < train_size + test_size:
            raise ValueError(
                f"Insufficient data: need at least {train_size + test_size} days, "
                f"got {len(timestamps)}"
            )

        results: list[BacktestResult] = []

        # Calculate split points
        total_window = train_size + test_size
        max_start = len(timestamps) - total_window

        if n_splits > max_start:
            logger.warning(
                f"Reducing n_splits from {n_splits} to {max_start} due to insufficient data"
            )
            n_splits = max_start

        step_size = max(1, max_start // n_splits)

        # Run walk-forward splits
        for i in range(n_splits):
            start_idx = i * step_size
            train_end_idx = start_idx + train_size
            test_end_idx = train_end_idx + test_size

            if test_end_idx > len(timestamps):
                break

            train_start = timestamps[start_idx]
            train_end = timestamps[train_end_idx - 1]
            test_start = timestamps[train_end_idx]
            test_end = timestamps[test_end_idx - 1]

            logger.info(
                "walk_forward_split",
                split=i + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

            # Split data
            train_data = data.filter(
                (pl.col("timestamp") >= train_start) & (pl.col("timestamp") <= train_end)
            )
            test_data = data.filter(
                (pl.col("timestamp") >= test_start) & (pl.col("timestamp") <= test_end)
            )

            # Train model
            # Note: This assumes data has features and target columns
            # Implementation may need to be adjusted based on actual data structure
            try:
                # Extract features (all columns except symbol, timestamp, target)
                feature_cols = [
                    col
                    for col in train_data.columns
                    if col not in ["symbol", "timestamp", "target"]
                ]
                X_train = train_data.select(feature_cols)
                y_train = train_data["target"] if "target" in train_data.columns else pl.Series([])

                model.fit(X_train, y_train)

                # Generate predictions on test data
                X_test = test_data.select(feature_cols)
                predictions_list = model.predict(X_test)

                # Convert predictions to DataFrame
                predictions = pl.DataFrame(
                    {
                        "symbol": [p.symbol for p in predictions_list],
                        "timestamp": [p.timestamp for p in predictions_list],
                        "predicted_return": [p.prediction for p in predictions_list],
                        "confidence": [p.confidence for p in predictions_list],
                    }
                )

                # Extract price data from test_data
                prices = test_data.select(
                    ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
                )

                # Run backtest on this split
                result = self.run_with_strategy(predictions, prices, strategy)
                results.append(result)

                logger.info(
                    "walk_forward_split_completed",
                    split=i + 1,
                    total_return=result.total_return,
                    sharpe_ratio=result.sharpe_ratio,
                )

            except Exception as e:
                logger.error(f"Walk-forward split {i + 1} failed: {e}")
                continue

        logger.info(
            "walk_forward_backtest_completed",
            total_splits=len(results),
            avg_return=sum(r.total_return for r in results) / len(results) if results else 0,
        )

        return results

    def _validate_strategy_inputs(
        self, predictions: pl.DataFrame, prices: pl.DataFrame
    ) -> None:
        """Validate inputs for strategy-based backtesting."""
        # Check predictions
        required_pred_cols = ["symbol", "timestamp", "predicted_return", "confidence"]
        missing_pred_cols = [col for col in required_pred_cols if col not in predictions.columns]
        if missing_pred_cols:
            raise ValueError(f"Missing required prediction columns: {missing_pred_cols}")

        # Check prices
        required_price_cols = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        missing_price_cols = [col for col in required_price_cols if col not in prices.columns]
        if missing_price_cols:
            raise ValueError(f"Missing required price columns: {missing_price_cols}")

        if predictions.is_empty():
            raise ValueError("Predictions DataFrame cannot be empty")

        if prices.is_empty():
            raise ValueError("Prices DataFrame cannot be empty")

    def _calculate_strategy_metrics(
        self, trades: list[Trade], equity_curve: pl.DataFrame
    ) -> BacktestResult:
        """Calculate comprehensive metrics for strategy-based backtest."""
        if equity_curve.is_empty():
            logger.warning("Empty equity curve, returning zero metrics")
            return BacktestResult(
                total_return=0.0,
                annualized_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                avg_trade_return=0.0,
                avg_holding_period=0.0,
                equity_curve=equity_curve,
                trade_log=pl.DataFrame(),
                monthly_returns=pl.DataFrame(),
            )

        # Calculate returns
        equity_series = equity_curve["equity"]
        initial_equity = float(equity_series[0])
        final_equity = float(equity_series[-1])
        total_return = ((final_equity - initial_equity) / initial_equity) * 100.0

        # Daily returns
        returns = equity_series.pct_change().drop_nulls()
        num_days = equity_curve.height

        # Annualized return
        annualized_return = calculate_annualized_return(total_return, num_days)

        # Sharpe ratio
        sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate=0.0)

        # Sortino ratio
        sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate=0.0)

        # Maximum drawdown
        max_drawdown = calculate_max_drawdown(equity_series)

        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

        # Win rate
        win_rate = calculate_win_rate(trades)

        # Profit factor
        profit_factor = calculate_profit_factor(trades)

        # Average trade return
        avg_trade_return = sum(t.return_pct for t in trades) / len(trades) if trades else 0.0

        # Average holding period
        if trades:
            holding_periods = [(t.exit_date - t.entry_date).days for t in trades]
            avg_holding_period = sum(holding_periods) / len(holding_periods)
        else:
            avg_holding_period = 0.0

        # Create trade log
        if trades:
            trade_log = pl.DataFrame(
                {
                    "entry_date": [t.entry_date for t in trades],
                    "exit_date": [t.exit_date for t in trades],
                    "entry_price": [t.entry_price for t in trades],
                    "exit_price": [t.exit_price for t in trades],
                    "position_size": [t.position_size for t in trades],
                    "direction": [t.direction for t in trades],
                    "pnl": [t.pnl for t in trades],
                    "return_pct": [t.return_pct for t in trades],
                }
            )
        else:
            trade_log = pl.DataFrame()

        # Calculate monthly returns
        monthly_returns = self._calculate_monthly_returns(equity_curve)

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_return=avg_trade_return,
            avg_holding_period=avg_holding_period,
            equity_curve=equity_curve,
            trade_log=trade_log,
            monthly_returns=monthly_returns,
        )

    def _calculate_monthly_returns(self, equity_curve: pl.DataFrame) -> pl.DataFrame:
        """Calculate monthly returns from equity curve."""
        if equity_curve.is_empty():
            return pl.DataFrame()

        # Add year-month column
        equity_with_month = equity_curve.with_columns(
            [
                pl.col("timestamp").dt.year().alias("year"),
                pl.col("timestamp").dt.month().alias("month"),
            ]
        )

        # Group by year-month and get first and last equity
        monthly = (
            equity_with_month.group_by(["year", "month"])
            .agg(
                [
                    pl.col("equity").first().alias("start_equity"),
                    pl.col("equity").last().alias("end_equity"),
                ]
            )
            .sort(["year", "month"])
        )

        # Calculate monthly return
        monthly = monthly.with_columns(
            [
                ((pl.col("end_equity") - pl.col("start_equity")) / pl.col("start_equity") * 100.0).alias(
                    "return_pct"
                )
            ]
        )

        return monthly
