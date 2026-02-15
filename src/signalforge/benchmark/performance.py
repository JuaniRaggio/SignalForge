"""Performance metrics calculation for trading strategies.

This module provides comprehensive performance analysis tools for backtesting results,
including risk-adjusted metrics, drawdown analysis, and trade statistics.

All metrics follow standard financial industry formulas:
- Sharpe Ratio: (mean_return - risk_free_rate) / std_return * sqrt(periods_per_year)
- Sortino Ratio: (mean_return - risk_free_rate) / downside_deviation * sqrt(periods_per_year)
- Calmar Ratio: annualized_return / max_drawdown
- CAGR: (final_value / initial_value) ^ (365 / days) - 1
- Maximum Drawdown: max((peak - trough) / peak)

Examples:
    Calculate comprehensive performance metrics:

    >>> from decimal import Decimal
    >>> from signalforge.benchmark import PerformanceCalculator
    >>> from signalforge.ml.backtesting.engine import Trade
    >>>
    >>> equity_curve = [Decimal("100000"), Decimal("105000"), Decimal("103000")]
    >>> trades = [...]  # List of Trade objects
    >>>
    >>> calculator = PerformanceCalculator(risk_free_rate=0.02, periods_per_year=252)
    >>> metrics = calculator.calculate_all(equity_curve, trades)
    >>> print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    >>> print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import polars as pl

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    from signalforge.ml.backtesting.engine import Trade

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for a trading strategy.

    All percentage metrics are expressed as decimals (e.g., 0.155 for 15.5%),
    following industry conventions for ratio calculations.

    Attributes:
        sharpe_ratio: Risk-adjusted return (annualized)
        sortino_ratio: Downside risk-adjusted return (annualized)
        calmar_ratio: Return to max drawdown ratio (annualized)
        max_drawdown: Maximum peak-to-trough decline (as decimal)
        max_drawdown_duration: Duration of maximum drawdown in periods
        cagr: Compound Annual Growth Rate (as decimal)
        volatility: Annualized volatility of returns (as decimal)
        win_rate: Percentage of winning trades (as decimal, 0-1)
        profit_factor: Ratio of gross profit to gross loss
        avg_win: Average winning trade amount
        avg_loss: Average losing trade amount (positive number)
        expectancy: Expected value per trade
        total_trades: Total number of trades
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades

    Note:
        Sharpe > 1.0 is good, > 2.0 is excellent.
        Sortino > 2.0 is good, > 3.0 is excellent.
        Calmar > 0.5 is acceptable, > 1.0 is good.
        Max drawdown should ideally be < 0.20 (20%).
        Profit factor > 1.5 indicates a profitable strategy.
    """

    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    cagr: float
    volatility: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy: float
    total_trades: int
    winning_trades: int
    losing_trades: int

    def __post_init__(self) -> None:
        """Validate metrics after initialization."""
        if self.total_trades < 0:
            raise ValueError("total_trades cannot be negative")
        if self.winning_trades < 0:
            raise ValueError("winning_trades cannot be negative")
        if self.losing_trades < 0:
            raise ValueError("losing_trades cannot be negative")
        if not 0 <= self.win_rate <= 1:
            raise ValueError(f"win_rate must be between 0 and 1, got {self.win_rate}")
        if self.profit_factor < 0:
            raise ValueError("profit_factor cannot be negative")
        if self.max_drawdown < 0:
            raise ValueError("max_drawdown cannot be negative")
        if self.max_drawdown_duration < 0:
            raise ValueError("max_drawdown_duration cannot be negative")
        if self.avg_loss < 0:
            raise ValueError("avg_loss must be positive (represents absolute value)")


def annualize_returns(returns: float, periods: int) -> float:
    """Annualize a return rate based on the number of periods.

    Args:
        returns: Total return as decimal (e.g., 0.15 for 15%)
        periods: Number of periods per year (e.g., 252 for daily, 12 for monthly)

    Returns:
        Annualized return as decimal

    Examples:
        >>> # 15% return over 126 trading days (half year)
        >>> annualize_returns(0.15, 252)
        0.3225...
    """
    if periods <= 0:
        raise ValueError("periods must be positive")
    return float(returns * periods)


def annualize_volatility(vol: float, periods: int) -> float:
    """Annualize volatility based on the number of periods.

    Uses square root of time rule for volatility scaling.

    Args:
        vol: Volatility as decimal (e.g., 0.02 for 2%)
        periods: Number of periods per year (e.g., 252 for daily, 12 for monthly)

    Returns:
        Annualized volatility as decimal

    Examples:
        >>> # 2% daily volatility
        >>> annualize_volatility(0.02, 252)
        0.317...
    """
    if periods <= 0:
        raise ValueError("periods must be positive")
    return float(vol * (periods**0.5))


class PerformanceCalculator:
    """Calculator for comprehensive trading performance metrics.

    This class provides methods to calculate various performance metrics
    from equity curves and trade lists, following standard financial
    industry formulas.

    Attributes:
        risk_free_rate: Annual risk-free rate as decimal (e.g., 0.02 for 2%)
        periods_per_year: Number of trading periods per year (252 for daily)

    Examples:
        Initialize and calculate metrics:

        >>> from decimal import Decimal
        >>> calculator = PerformanceCalculator(risk_free_rate=0.02, periods_per_year=252)
        >>> equity = [Decimal("100000"), Decimal("105000"), Decimal("110000")]
        >>> trades = [...]  # List of Trade objects
        >>> metrics = calculator.calculate_all(equity, trades)
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> None:
        """Initialize the performance calculator.

        Args:
            risk_free_rate: Annual risk-free rate as decimal (default: 0.0)
            periods_per_year: Number of trading periods per year (default: 252)

        Raises:
            ValueError: If risk_free_rate is negative or periods_per_year is not positive
        """
        if risk_free_rate < 0:
            raise ValueError("risk_free_rate cannot be negative")
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")

        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        logger.info(
            "performance_calculator_initialized",
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

    def calculate_returns(self, equity_curve: list[Decimal]) -> pl.Series:
        """Calculate period-over-period returns from equity curve.

        Args:
            equity_curve: List of equity values over time

        Returns:
            Polars Series of returns (as decimals), excluding the first null value

        Raises:
            ValueError: If equity_curve is empty or contains non-positive values

        Examples:
            >>> from decimal import Decimal
            >>> equity = [Decimal("100"), Decimal("105"), Decimal("110")]
            >>> calc = PerformanceCalculator()
            >>> returns = calc.calculate_returns(equity)
            >>> list(returns)  # [0.05, 0.047619...]
        """
        if not equity_curve:
            raise ValueError("equity_curve cannot be empty")

        if any(e <= 0 for e in equity_curve):
            raise ValueError("equity_curve values must be positive")

        # Convert to float for Polars
        equity_floats = [float(e) for e in equity_curve]
        equity_series = pl.Series("equity", equity_floats)

        # Calculate percentage change
        returns = equity_series.pct_change().drop_nulls()

        logger.debug("returns_calculated", num_returns=returns.len())
        return returns

    def calculate_sharpe_ratio(self, returns: pl.Series) -> float:
        """Calculate annualized Sharpe ratio.

        The Sharpe ratio measures risk-adjusted returns by comparing the mean
        excess return to the standard deviation of returns.

        Formula: (mean_return - risk_free_rate) / std_return * sqrt(periods_per_year)

        Args:
            returns: Series of period returns (as decimals)

        Returns:
            Annualized Sharpe ratio. Higher is better (>1 is good, >2 is excellent).
            Returns 0.0 if volatility is zero or insufficient data.

        Examples:
            >>> import polars as pl
            >>> returns = pl.Series([0.01, -0.005, 0.02, 0.015])
            >>> calc = PerformanceCalculator(risk_free_rate=0.02)
            >>> sharpe = calc.calculate_sharpe_ratio(returns)
        """
        if returns.is_empty() or returns.len() < 2:
            logger.warning("calculate_sharpe_ratio: insufficient data")
            return 0.0

        valid_returns = returns.drop_nulls()
        if valid_returns.len() < 2:
            logger.warning("calculate_sharpe_ratio: insufficient non-null data")
            return 0.0

        mean_val = valid_returns.mean()
        std_val = valid_returns.std()

        if mean_val is None or std_val is None:
            return 0.0

        mean_return = float(mean_val)  # type: ignore[arg-type]
        std_return = float(std_val)  # type: ignore[arg-type]

        if std_return == 0.0:
            logger.debug("calculate_sharpe_ratio: zero volatility")
            return 0.0

        # Convert annual risk-free rate to period rate
        period_rf_rate = self.risk_free_rate / self.periods_per_year

        # Calculate Sharpe ratio and annualize
        sharpe = (mean_return - period_rf_rate) / std_return * (self.periods_per_year**0.5)

        return float(sharpe)

    def calculate_sortino_ratio(self, returns: pl.Series) -> float:
        """Calculate annualized Sortino ratio.

        The Sortino ratio is similar to Sharpe but only penalizes downside volatility,
        making it more appropriate for strategies with asymmetric returns.

        Formula: (mean_return - risk_free_rate) / downside_deviation * sqrt(periods_per_year)

        Args:
            returns: Series of period returns (as decimals)

        Returns:
            Annualized Sortino ratio. Higher is better (>2 is good, >3 is excellent).
            Returns 0.0 if downside deviation is zero or insufficient data.

        Examples:
            >>> import polars as pl
            >>> returns = pl.Series([0.01, -0.005, 0.02, 0.015])
            >>> calc = PerformanceCalculator(risk_free_rate=0.02)
            >>> sortino = calc.calculate_sortino_ratio(returns)
        """
        if returns.is_empty() or returns.len() < 2:
            logger.warning("calculate_sortino_ratio: insufficient data")
            return 0.0

        valid_returns = returns.drop_nulls()
        if valid_returns.len() < 2:
            logger.warning("calculate_sortino_ratio: insufficient non-null data")
            return 0.0

        mean_val = valid_returns.mean()
        if mean_val is None:
            return 0.0

        mean_return = float(mean_val)  # type: ignore[arg-type]

        # Calculate downside deviation (only negative returns)
        period_rf_rate = self.risk_free_rate / self.periods_per_year
        downside_returns = valid_returns.filter(valid_returns < period_rf_rate)

        if downside_returns.is_empty():
            logger.debug("calculate_sortino_ratio: no downside returns")
            # If no downside, return a high value (but not infinity)
            return 100.0

        downside_std_val = downside_returns.std()
        if downside_std_val is None or downside_std_val == 0.0:
            return 100.0

        downside_std = float(downside_std_val)  # type: ignore[arg-type]

        # Calculate Sortino ratio and annualize
        sortino = (mean_return - period_rf_rate) / downside_std * (self.periods_per_year**0.5)

        return float(sortino)

    def calculate_calmar_ratio(self, returns: pl.Series, max_dd: float) -> float:
        """Calculate annualized Calmar ratio.

        The Calmar ratio measures return relative to maximum drawdown, indicating
        how much return is generated per unit of drawdown risk.

        Formula: annualized_return / max_drawdown

        Args:
            returns: Series of period returns (as decimals)
            max_dd: Maximum drawdown (as decimal, positive number)

        Returns:
            Calmar ratio. Higher is better (>0.5 is acceptable, >1.0 is good).
            Returns 0.0 if max drawdown is zero or insufficient data.

        Examples:
            >>> import polars as pl
            >>> returns = pl.Series([0.01, -0.005, 0.02, 0.015])
            >>> calc = PerformanceCalculator()
            >>> calmar = calc.calculate_calmar_ratio(returns, 0.15)
        """
        if returns.is_empty() or max_dd == 0.0:
            logger.warning("calculate_calmar_ratio: insufficient data or zero drawdown")
            return 0.0

        valid_returns = returns.drop_nulls()
        if valid_returns.is_empty():
            return 0.0

        mean_val = valid_returns.mean()
        if mean_val is None:
            return 0.0

        mean_return = float(mean_val)  # type: ignore[arg-type]
        annualized_return = mean_return * self.periods_per_year

        calmar = annualized_return / max_dd

        return float(calmar)

    def calculate_max_drawdown(self, equity_curve: list[Decimal]) -> tuple[float, int]:
        """Calculate maximum drawdown and its duration.

        Maximum drawdown is the largest peak-to-trough decline in the equity curve.
        Duration is measured in number of periods.

        Args:
            equity_curve: List of equity values over time

        Returns:
            Tuple of (max_drawdown as decimal, duration in periods)
            Returns (0.0, 0) if insufficient data.

        Raises:
            ValueError: If equity_curve is empty

        Examples:
            >>> from decimal import Decimal
            >>> equity = [Decimal("100"), Decimal("110"), Decimal("90"), Decimal("95")]
            >>> calc = PerformanceCalculator()
            >>> max_dd, duration = calc.calculate_max_drawdown(equity)
            >>> print(f"Max DD: {max_dd:.2%}, Duration: {duration}")
        """
        if not equity_curve:
            raise ValueError("equity_curve cannot be empty")

        if len(equity_curve) < 2:
            logger.warning("calculate_max_drawdown: insufficient data")
            return 0.0, 0

        # Convert to float for calculations
        equity_floats = [float(e) for e in equity_curve]
        equity_series = pl.Series("equity", equity_floats)

        # Calculate running maximum (peak)
        running_max = equity_series.cum_max()

        # Calculate drawdown at each point (as decimal)
        drawdown = (equity_series - running_max) / running_max

        # Find maximum drawdown (most negative value)
        min_val = drawdown.min()
        if min_val is None:
            return 0.0, 0

        max_dd = abs(float(min_val))  # type: ignore[arg-type]

        # Calculate drawdown duration
        # Find the index where max drawdown occurred
        drawdown_list = drawdown.to_list()
        max_dd_idx = drawdown_list.index(min_val)

        # Count periods from previous peak to recovery (or end)
        duration = 0
        if max_dd > 0:
            # Find the peak before max drawdown
            peak_idx = 0
            peak_value = equity_floats[0]
            for i in range(max_dd_idx + 1):
                if equity_floats[i] >= peak_value:
                    peak_value = equity_floats[i]
                    peak_idx = i

            # Find recovery point (when equity exceeds previous peak)
            recovery_idx = len(equity_floats)
            for i in range(max_dd_idx + 1, len(equity_floats)):
                if equity_floats[i] >= peak_value:
                    recovery_idx = i
                    break

            duration = recovery_idx - peak_idx

        logger.debug(
            "max_drawdown_calculated",
            max_drawdown=max_dd,
            duration=duration,
        )

        return max_dd, duration

    def calculate_cagr(self, initial: Decimal, final: Decimal, days: int) -> float:
        """Calculate Compound Annual Growth Rate.

        CAGR represents the annual growth rate assuming constant compounding.

        Formula: (final_value / initial_value) ^ (365 / days) - 1

        Args:
            initial: Initial portfolio value
            final: Final portfolio value
            days: Number of days in the period

        Returns:
            CAGR as decimal (e.g., 0.15 for 15% annual growth)
            Returns 0.0 if initial value is zero or days is zero.

        Raises:
            ValueError: If initial or final values are negative or days is negative

        Examples:
            >>> from decimal import Decimal
            >>> calc = PerformanceCalculator()
            >>> cagr = calc.calculate_cagr(Decimal("100000"), Decimal("150000"), 365)
            >>> print(f"CAGR: {cagr:.2%}")
        """
        if initial < 0 or final < 0:
            raise ValueError("initial and final values must be non-negative")
        if days < 0:
            raise ValueError("days cannot be negative")

        if initial == 0 or days == 0:
            logger.warning("calculate_cagr: zero initial value or zero days")
            return 0.0

        initial_float = float(initial)
        final_float = float(final)

        # CAGR formula
        years = days / 365.0
        cagr = (final_float / initial_float) ** (1.0 / years) - 1.0

        return float(cagr)

    def calculate_win_rate(self, trades: list[Trade]) -> float:
        """Calculate percentage of winning trades.

        Win rate is the percentage of trades that resulted in a profit.

        Args:
            trades: List of completed Trade objects

        Returns:
            Win rate as decimal (0-1). Returns 0.0 if no trades.

        Examples:
            >>> from signalforge.ml.backtesting.engine import Trade
            >>> trades = [...]  # List of trades
            >>> calc = PerformanceCalculator()
            >>> win_rate = calc.calculate_win_rate(trades)
            >>> print(f"Win Rate: {win_rate:.1%}")
        """
        if not trades:
            logger.warning("calculate_win_rate: no trades")
            return 0.0

        winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        total_trades = len(trades)

        return winning_trades / total_trades

    def calculate_profit_factor(self, trades: list[Trade]) -> float:
        """Calculate profit factor (gross profit / gross loss).

        Profit factor measures the ratio of total profits to total losses.
        A value above 1.0 indicates profitability.

        Args:
            trades: List of completed Trade objects

        Returns:
            Profit factor. Values > 1.0 indicate profitable strategy.
            Returns 0.0 if no trades or no losses (but has profits returns inf).

        Examples:
            >>> from signalforge.ml.backtesting.engine import Trade
            >>> trades = [...]  # List of trades
            >>> calc = PerformanceCalculator()
            >>> pf = calc.calculate_profit_factor(trades)
            >>> print(f"Profit Factor: {pf:.2f}")
        """
        if not trades:
            logger.warning("calculate_profit_factor: no trades")
            return 0.0

        gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in trades if trade.pnl < 0))

        if gross_loss == 0.0:
            # No losses
            if gross_profit > 0.0:
                return float("inf")
            return 0.0

        return gross_profit / gross_loss

    def calculate_expectancy(self, trades: list[Trade]) -> float:
        """Calculate expected value per trade.

        Expectancy represents the average profit/loss per trade, accounting
        for win rate and average win/loss sizes.

        Formula: (win_rate * avg_win) - (loss_rate * avg_loss)

        Args:
            trades: List of completed Trade objects

        Returns:
            Expected value per trade in currency units. Positive is profitable.
            Returns 0.0 if no trades.

        Examples:
            >>> from signalforge.ml.backtesting.engine import Trade
            >>> trades = [...]  # List of trades
            >>> calc = PerformanceCalculator()
            >>> expectancy = calc.calculate_expectancy(trades)
            >>> print(f"Expectancy: ${expectancy:.2f}")
        """
        if not trades:
            logger.warning("calculate_expectancy: no trades")
            return 0.0

        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]

        if not winning_trades and not losing_trades:
            return 0.0

        win_rate = len(winning_trades) / len(trades)
        loss_rate = len(losing_trades) / len(trades)

        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
        avg_loss = (
            abs(sum(t.pnl for t in losing_trades) / len(losing_trades))
            if losing_trades
            else 0.0
        )

        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        return expectancy

    def calculate_all(
        self,
        equity_curve: list[Decimal],
        trades: list[Trade],
    ) -> PerformanceMetrics:
        """Calculate all performance metrics.

        This is the main entry point that calculates comprehensive performance
        metrics from an equity curve and trade list.

        Args:
            equity_curve: List of equity values over time
            trades: List of completed Trade objects

        Returns:
            PerformanceMetrics object with all calculated metrics

        Raises:
            ValueError: If equity_curve is empty or contains invalid values

        Examples:
            >>> from decimal import Decimal
            >>> from signalforge.benchmark import PerformanceCalculator
            >>> equity = [Decimal("100000"), Decimal("105000"), Decimal("110000")]
            >>> trades = [...]  # List of trades
            >>> calc = PerformanceCalculator(risk_free_rate=0.02)
            >>> metrics = calc.calculate_all(equity, trades)
            >>> print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
        """
        if not equity_curve:
            raise ValueError("equity_curve cannot be empty")

        logger.info("calculating_all_metrics", num_periods=len(equity_curve), num_trades=len(trades))

        # Calculate returns
        returns = self.calculate_returns(equity_curve)

        # Calculate risk-adjusted metrics
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)

        # Calculate drawdown metrics
        max_drawdown, max_drawdown_duration = self.calculate_max_drawdown(equity_curve)

        # Calculate Calmar ratio
        calmar_ratio = self.calculate_calmar_ratio(returns, max_drawdown)

        # Calculate CAGR
        initial_equity = equity_curve[0]
        final_equity = equity_curve[-1]

        # Estimate days from number of periods (assuming daily data by default)
        # In practice, you might want to use actual timestamps
        days = len(equity_curve) - 1 if len(equity_curve) > 1 else 1
        cagr = self.calculate_cagr(initial_equity, final_equity, days)

        # Calculate volatility
        std_val = returns.std()
        if std_val is None:
            volatility = 0.0
        else:
            volatility = annualize_volatility(float(std_val), self.periods_per_year)  # type: ignore[arg-type]

        # Calculate trade statistics
        win_rate = self.calculate_win_rate(trades)
        profit_factor = self.calculate_profit_factor(trades)
        expectancy = self.calculate_expectancy(trades)

        # Calculate average win/loss
        winning_trades_list = [t for t in trades if t.pnl > 0]
        losing_trades_list = [t for t in trades if t.pnl < 0]

        avg_win = (
            sum(t.pnl for t in winning_trades_list) / len(winning_trades_list)
            if winning_trades_list
            else 0.0
        )
        avg_loss = (
            abs(sum(t.pnl for t in losing_trades_list) / len(losing_trades_list))
            if losing_trades_list
            else 0.0
        )

        total_trades = len(trades)
        winning_trades = len(winning_trades_list)
        losing_trades = len(losing_trades_list)

        logger.info(
            "metrics_calculated",
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            cagr=cagr,
            total_trades=total_trades,
        )

        return PerformanceMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            cagr=cagr,
            volatility=volatility,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=expectancy,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
        )
