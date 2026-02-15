"""Trading strategies for backtesting ML predictions.

This module provides various trading strategies that convert ML predictions
into actionable trading signals. Each strategy implements different logic
for position sizing, entry/exit criteria, and risk management.

The strategies are designed to work with prediction DataFrames containing:
- symbol: Stock ticker symbol
- timestamp: Prediction timestamp
- predicted_return: Predicted return (as decimal or percentage)
- confidence: Model confidence score (0.0-1.0)

Examples:
    Simple threshold strategy:

    >>> import polars as pl
    >>> from signalforge.ml.backtesting.strategies import ThresholdStrategy
    >>>
    >>> predictions = pl.DataFrame({
    ...     "symbol": ["AAPL", "GOOGL", "MSFT"],
    ...     "timestamp": [...],
    ...     "predicted_return": [0.025, -0.015, 0.032],
    ...     "confidence": [0.75, 0.82, 0.68],
    ... })
    >>> strategy = ThresholdStrategy(buy_threshold=0.02, confidence_threshold=0.7)
    >>> signals = strategy.generate_signals(predictions, {})

    Long-short strategy:

    >>> from signalforge.ml.backtesting.strategies import LongShortStrategy
    >>>
    >>> strategy = LongShortStrategy(long_n=5, short_n=5)
    >>> signals = strategy.generate_signals(predictions, {})
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import polars as pl

from signalforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TradeSignal:
    """Represents a trading signal generated from predictions.

    A trade signal encapsulates all information needed to execute a trade,
    including direction, size, and confidence level.

    Attributes:
        symbol: Stock ticker symbol
        action: Trade action ("buy", "sell", "hold")
        size: Position size as fraction of capital (0.0-1.0)
        confidence: Confidence score for this signal (0.0-1.0)
        timestamp: When this signal was generated
        predicted_return: The predicted return that generated this signal (optional)

    Note:
        - "buy" means enter or increase long position
        - "sell" means exit or reduce position
        - "hold" means maintain current position
        - size=0.0 with action="sell" means close entire position
    """

    symbol: str
    action: Literal["buy", "sell", "hold"]
    size: float
    confidence: float
    timestamp: datetime
    predicted_return: float | None = None

    def __post_init__(self) -> None:
        """Validate signal data after initialization."""
        if self.action not in ("buy", "sell", "hold"):
            raise ValueError(f"Invalid action: {self.action}")
        if not 0.0 <= self.size <= 1.0:
            raise ValueError(f"Size must be between 0 and 1, got {self.size}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")


class TradingStrategy(ABC):
    """Abstract base class for trading strategies.

    All trading strategies must inherit from this class and implement
    the generate_signals method. This ensures a consistent interface
    across different strategy implementations.

    Subclasses should implement specific logic for converting ML predictions
    into trading signals based on their strategy rules.
    """

    @abstractmethod
    def generate_signals(
        self,
        predictions: pl.DataFrame,
        current_positions: dict[str, float],
    ) -> list[TradeSignal]:
        """Generate trade signals from predictions.

        This method analyzes predictions and current portfolio state to
        generate trading signals according to the strategy's rules.

        Args:
            predictions: DataFrame with columns [symbol, timestamp, predicted_return, confidence]
                         predicted_return should be decimal (e.g., 0.02 for 2%)
            current_positions: Dictionary mapping symbol to current position size
                              (as fraction of capital)

        Returns:
            List of TradeSignal objects indicating what trades to make

        Raises:
            ValueError: If predictions DataFrame is missing required columns
        """
        ...


class ThresholdStrategy(TradingStrategy):
    """Simple threshold-based trading strategy.

    This strategy generates buy signals when predicted returns exceed a threshold
    and sell signals when they fall below another threshold. It also considers
    model confidence to filter out low-confidence predictions.

    The strategy is long-only (no short positions) unless explicitly configured.

    Attributes:
        buy_threshold: Minimum predicted return to generate buy signal (decimal)
        sell_threshold: Maximum predicted return to generate sell signal (decimal)
        confidence_threshold: Minimum confidence score to act on prediction
        max_positions: Maximum number of concurrent positions
        position_size: Size of each position as fraction of capital (0.0-1.0)

    Examples:
        >>> strategy = ThresholdStrategy(
        ...     buy_threshold=0.02,  # Buy if predicted return > 2%
        ...     sell_threshold=-0.02,  # Sell if predicted return < -2%
        ...     confidence_threshold=0.6,
        ...     max_positions=10,
        ... )
    """

    def __init__(
        self,
        buy_threshold: float = 0.02,
        sell_threshold: float = -0.02,
        confidence_threshold: float = 0.6,
        max_positions: int = 10,
        position_size: float = 0.1,
    ) -> None:
        """Initialize threshold strategy.

        Args:
            buy_threshold: Buy if predicted return > this value (decimal)
            sell_threshold: Sell if predicted return < this value (decimal)
            confidence_threshold: Minimum confidence to act on signal
            max_positions: Maximum concurrent positions
            position_size: Position size as fraction of capital
        """
        if buy_threshold <= sell_threshold:
            raise ValueError("buy_threshold must be greater than sell_threshold")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if not 0.0 < position_size <= 1.0:
            raise ValueError("position_size must be between 0 and 1")

        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.confidence_threshold = confidence_threshold
        self.max_positions = max_positions
        self.position_size = position_size

        logger.info(
            "threshold_strategy_initialized",
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            confidence_threshold=confidence_threshold,
        )

    def generate_signals(
        self,
        predictions: pl.DataFrame,
        current_positions: dict[str, float],
    ) -> list[TradeSignal]:
        """Generate signals based on prediction thresholds.

        Args:
            predictions: Prediction DataFrame
            current_positions: Current portfolio positions

        Returns:
            List of trade signals
        """
        self._validate_predictions(predictions)

        signals: list[TradeSignal] = []

        # Filter predictions by confidence
        high_conf = predictions.filter(pl.col("confidence") >= self.confidence_threshold)

        if high_conf.is_empty():
            logger.debug("No predictions meet confidence threshold")
            return signals

        # Process each prediction
        for row in high_conf.iter_rows(named=True):
            symbol = str(row["symbol"])
            timestamp = row["timestamp"]
            pred_return = float(row["predicted_return"])
            confidence = float(row["confidence"])

            current_size = current_positions.get(symbol, 0.0)

            # Generate buy signal
            if pred_return > self.buy_threshold and current_size == 0.0:
                # Check if we haven't exceeded max positions
                num_positions = sum(1 for pos in current_positions.values() if pos > 0)
                if num_positions < self.max_positions:
                    signals.append(
                        TradeSignal(
                            symbol=symbol,
                            action="buy",
                            size=self.position_size,
                            confidence=confidence,
                            timestamp=timestamp,
                            predicted_return=pred_return,
                        )
                    )

            # Generate sell signal
            elif pred_return < self.sell_threshold and current_size > 0.0:
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action="sell",
                        size=0.0,  # Close entire position
                        confidence=confidence,
                        timestamp=timestamp,
                        predicted_return=pred_return,
                    )
                )

        logger.debug("signals_generated", num_signals=len(signals))
        return signals

    def _validate_predictions(self, predictions: pl.DataFrame) -> None:
        """Validate prediction DataFrame has required columns."""
        required_cols = ["symbol", "timestamp", "predicted_return", "confidence"]
        missing_cols = [col for col in required_cols if col not in predictions.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")


class RankingStrategy(TradingStrategy):
    """Rank stocks by predicted return and go long top N.

    This strategy ranks all predictions by predicted return and enters
    long positions in the top N stocks. It rebalances periodically,
    closing positions that fall out of the top N and opening new ones.

    Attributes:
        top_n: Number of top stocks to hold
        rebalance_frequency: Days between rebalances
        confidence_threshold: Minimum confidence to consider prediction
        equal_weight: If True, equal-weight positions; else weight by predicted return

    Examples:
        >>> strategy = RankingStrategy(
        ...     top_n=10,
        ...     rebalance_frequency=5,  # Rebalance weekly
        ...     equal_weight=True,
        ... )
    """

    def __init__(
        self,
        top_n: int = 5,
        rebalance_frequency: int = 5,
        confidence_threshold: float = 0.5,
        equal_weight: bool = True,
    ) -> None:
        """Initialize ranking strategy.

        Args:
            top_n: Number of top stocks to hold
            rebalance_frequency: Days between rebalances
            confidence_threshold: Minimum confidence threshold
            equal_weight: Whether to use equal weighting
        """
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        if rebalance_frequency <= 0:
            raise ValueError("rebalance_frequency must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        self.top_n = top_n
        self.rebalance_frequency = rebalance_frequency
        self.confidence_threshold = confidence_threshold
        self.equal_weight = equal_weight
        self._last_rebalance: datetime | None = None

        logger.info(
            "ranking_strategy_initialized",
            top_n=top_n,
            rebalance_frequency=rebalance_frequency,
        )

    def generate_signals(
        self,
        predictions: pl.DataFrame,
        current_positions: dict[str, float],
    ) -> list[TradeSignal]:
        """Generate signals by ranking predictions.

        Args:
            predictions: Prediction DataFrame
            current_positions: Current portfolio positions

        Returns:
            List of trade signals
        """
        self._validate_predictions(predictions)

        signals: list[TradeSignal] = []

        # Check if we need to rebalance
        if predictions.is_empty():
            return signals

        current_time = predictions["timestamp"][0]
        if self._last_rebalance is not None:
            days_since_rebalance = (current_time - self._last_rebalance).days
            if days_since_rebalance < self.rebalance_frequency:
                return signals  # Not time to rebalance yet

        # Filter by confidence
        high_conf = predictions.filter(pl.col("confidence") >= self.confidence_threshold)

        if high_conf.is_empty():
            logger.debug("No predictions meet confidence threshold")
            return signals

        # Rank by predicted return (descending)
        ranked = high_conf.sort("predicted_return", descending=True)

        # Get top N symbols
        top_symbols = set(ranked["symbol"].head(self.top_n).to_list())

        # Calculate position sizes
        if self.equal_weight:
            position_size = 1.0 / self.top_n
            weights = dict.fromkeys(top_symbols, position_size)
        else:
            # Weight by predicted return (normalized)
            top_ranked = ranked.head(self.top_n)
            pred_returns = top_ranked["predicted_return"].to_numpy()
            # Normalize to sum to 1
            total = float(pred_returns.sum())
            if total > 0:
                weights = {
                    symbol: float(pred_return) / total
                    for symbol, pred_return in zip(
                        top_ranked["symbol"].to_list(),
                        pred_returns,
                        strict=True,
                    )
                }
            else:
                # Fallback to equal weight if all predictions are negative
                position_size = 1.0 / self.top_n
                weights = dict.fromkeys(top_symbols, position_size)

        # Generate sell signals for positions not in top N
        for symbol, size in current_positions.items():
            if size > 0 and symbol not in top_symbols:
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action="sell",
                        size=0.0,
                        confidence=1.0,
                        timestamp=current_time,
                    )
                )

        # Generate buy signals for top N not currently held
        for row in ranked.head(self.top_n).iter_rows(named=True):
            symbol = str(row["symbol"])
            if symbol not in current_positions or current_positions[symbol] == 0.0:
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action="buy",
                        size=weights[symbol],
                        confidence=float(row["confidence"]),
                        timestamp=current_time,
                        predicted_return=float(row["predicted_return"]),
                    )
                )

        self._last_rebalance = current_time
        logger.debug("rebalance_signals_generated", num_signals=len(signals))
        return signals

    def _validate_predictions(self, predictions: pl.DataFrame) -> None:
        """Validate prediction DataFrame has required columns."""
        required_cols = ["symbol", "timestamp", "predicted_return", "confidence"]
        missing_cols = [col for col in required_cols if col not in predictions.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")


class LongShortStrategy(TradingStrategy):
    """Long top N stocks, short bottom N stocks.

    This strategy goes long the top N stocks by predicted return and
    short the bottom N stocks. It's market-neutral, betting on relative
    performance rather than absolute market direction.

    Attributes:
        long_n: Number of stocks to go long
        short_n: Number of stocks to go short
        rebalance_frequency: Days between rebalances
        confidence_threshold: Minimum confidence threshold
        equal_weight: If True, equal-weight positions

    Examples:
        >>> strategy = LongShortStrategy(
        ...     long_n=5,
        ...     short_n=5,
        ...     rebalance_frequency=5,
        ... )

    Note:
        This strategy requires short selling to be enabled in the backtest configuration.
    """

    def __init__(
        self,
        long_n: int = 5,
        short_n: int = 5,
        rebalance_frequency: int = 5,
        confidence_threshold: float = 0.5,
        equal_weight: bool = True,
    ) -> None:
        """Initialize long-short strategy.

        Args:
            long_n: Number of stocks to long
            short_n: Number of stocks to short
            rebalance_frequency: Days between rebalances
            confidence_threshold: Minimum confidence threshold
            equal_weight: Whether to use equal weighting
        """
        if long_n <= 0:
            raise ValueError("long_n must be positive")
        if short_n <= 0:
            raise ValueError("short_n must be positive")
        if rebalance_frequency <= 0:
            raise ValueError("rebalance_frequency must be positive")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        self.long_n = long_n
        self.short_n = short_n
        self.rebalance_frequency = rebalance_frequency
        self.confidence_threshold = confidence_threshold
        self.equal_weight = equal_weight
        self._last_rebalance: datetime | None = None

        logger.info(
            "longshort_strategy_initialized",
            long_n=long_n,
            short_n=short_n,
            rebalance_frequency=rebalance_frequency,
        )

    def generate_signals(
        self,
        predictions: pl.DataFrame,
        current_positions: dict[str, float],
    ) -> list[TradeSignal]:
        """Generate long and short signals.

        Args:
            predictions: Prediction DataFrame
            current_positions: Current portfolio positions (positive=long, negative=short)

        Returns:
            List of trade signals
        """
        self._validate_predictions(predictions)

        signals: list[TradeSignal] = []

        # Check if we need to rebalance
        if predictions.is_empty():
            return signals

        current_time = predictions["timestamp"][0]
        if self._last_rebalance is not None:
            days_since_rebalance = (current_time - self._last_rebalance).days
            if days_since_rebalance < self.rebalance_frequency:
                return signals

        # Filter by confidence
        high_conf = predictions.filter(pl.col("confidence") >= self.confidence_threshold)

        if high_conf.is_empty():
            logger.debug("No predictions meet confidence threshold")
            return signals

        # Need at least long_n + short_n predictions
        if high_conf.height < self.long_n + self.short_n:
            logger.warning(
                "Insufficient predictions for long-short strategy",
                available=high_conf.height,
                required=self.long_n + self.short_n,
            )
            return signals

        # Rank by predicted return
        ranked = high_conf.sort("predicted_return", descending=True)

        # Get top N (long) and bottom N (short)
        long_symbols = set(ranked["symbol"].head(self.long_n).to_list())
        short_symbols = set(ranked["symbol"].tail(self.short_n).to_list())

        # Calculate position sizes
        if self.equal_weight:
            long_size = 0.5 / self.long_n  # 50% in longs
            short_size = 0.5 / self.short_n  # 50% in shorts
            long_weights = dict.fromkeys(long_symbols, long_size)
            short_weights = dict.fromkeys(short_symbols, short_size)
        else:
            # Weight by absolute predicted return
            top_ranked = ranked.head(self.long_n)
            bottom_ranked = ranked.tail(self.short_n)

            long_returns = top_ranked["predicted_return"].abs().to_numpy()
            short_returns = bottom_ranked["predicted_return"].abs().to_numpy()

            long_total = float(long_returns.sum())
            short_total = float(short_returns.sum())

            if long_total > 0:
                long_weights = {
                    symbol: 0.5 * float(pred_return) / long_total
                    for symbol, pred_return in zip(
                        top_ranked["symbol"].to_list(),
                        long_returns,
                        strict=True,
                    )
                }
            else:
                long_size = 0.5 / self.long_n
                long_weights = dict.fromkeys(long_symbols, long_size)

            if short_total > 0:
                short_weights = {
                    symbol: 0.5 * float(pred_return) / short_total
                    for symbol, pred_return in zip(
                        bottom_ranked["symbol"].to_list(),
                        short_returns,
                        strict=True,
                    )
                }
            else:
                short_size = 0.5 / self.short_n
                short_weights = dict.fromkeys(short_symbols, short_size)

        # Close positions not in long or short lists
        for symbol, size in current_positions.items():
            if size != 0 and symbol not in long_symbols and symbol not in short_symbols:
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action="sell",
                        size=0.0,
                        confidence=1.0,
                        timestamp=current_time,
                    )
                )

        # Generate long signals
        for row in ranked.head(self.long_n).iter_rows(named=True):
            symbol = str(row["symbol"])
            current_size = current_positions.get(symbol, 0.0)
            # Only signal if not currently long
            if current_size <= 0:
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action="buy",
                        size=long_weights[symbol],
                        confidence=float(row["confidence"]),
                        timestamp=current_time,
                        predicted_return=float(row["predicted_return"]),
                    )
                )

        # Generate short signals
        # Note: In the current implementation, short positions would need
        # to be handled by the backtesting engine with negative position sizes
        for row in ranked.tail(self.short_n).iter_rows(named=True):
            symbol = str(row["symbol"])
            current_size = current_positions.get(symbol, 0.0)
            # Only signal if not currently short
            if current_size >= 0:
                signals.append(
                    TradeSignal(
                        symbol=symbol,
                        action="sell",  # "sell" to enter short
                        size=short_weights[symbol],
                        confidence=float(row["confidence"]),
                        timestamp=current_time,
                        predicted_return=float(row["predicted_return"]),
                    )
                )

        self._last_rebalance = current_time
        logger.debug("longshort_signals_generated", num_signals=len(signals))
        return signals

    def _validate_predictions(self, predictions: pl.DataFrame) -> None:
        """Validate prediction DataFrame has required columns."""
        required_cols = ["symbol", "timestamp", "predicted_return", "confidence"]
        missing_cols = [col for col in required_cols if col not in predictions.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
