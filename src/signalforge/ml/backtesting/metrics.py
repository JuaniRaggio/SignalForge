"""Performance metrics calculation for backtesting results.

This module provides functions for calculating comprehensive performance metrics
from backtesting results. It includes both risk-adjusted and absolute performance
measures commonly used in quantitative trading.

All metrics handle edge cases appropriately (empty data, single values, etc.)
and are designed to work efficiently with Polars Series.

Examples:
    Calculate Sharpe ratio from returns:

    >>> import polars as pl
    >>> from signalforge.ml.backtesting.metrics import calculate_sharpe_ratio
    >>>
    >>> returns = pl.Series([0.01, -0.005, 0.02, 0.015, -0.01])
    >>> sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
    >>> print(f"Sharpe Ratio: {sharpe:.2f}")

    Calculate maximum drawdown:

    >>> from signalforge.ml.backtesting.metrics import calculate_max_drawdown
    >>>
    >>> equity = pl.Series([100000, 105000, 103000, 108000, 102000, 110000])
    >>> max_dd = calculate_max_drawdown(equity)
    >>> print(f"Max Drawdown: {max_dd:.2%}")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    from signalforge.ml.backtesting.engine import Trade

logger = get_logger(__name__)


@dataclass
class BacktestMetrics:
    """Metrics from a backtest run.

    This dataclass encapsulates all performance metrics calculated from
    a backtesting session. All percentage metrics are expressed as
    percentages (e.g., 15.5 means 15.5%), not decimals.

    Attributes:
        total_return: Total return over the backtest period (%)
        annualized_return: Annualized return assuming 252 trading days (%)
        sharpe_ratio: Risk-adjusted return (annualized Sharpe ratio)
        max_drawdown: Maximum peak-to-trough decline (%)
        win_rate: Percentage of profitable trades (%)
        profit_factor: Ratio of gross profit to gross loss
        total_trades: Number of completed trades
        avg_trade_return: Average return per trade (%)
        volatility: Annualized volatility of returns (%)

    Note:
        A Sharpe ratio above 1.0 is considered good, above 2.0 is excellent.
        Maximum drawdown should ideally be kept below 20-30% for most strategies.
        A profit factor above 1.5 indicates a profitable strategy.
    """

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    volatility: float

    def __post_init__(self) -> None:
        """Validate metrics after initialization."""
        if self.total_trades < 0:
            raise ValueError("total_trades cannot be negative")
        if not 0 <= self.win_rate <= 100:
            raise ValueError(f"win_rate must be between 0 and 100, got {self.win_rate}")
        if self.profit_factor < 0:
            raise ValueError("profit_factor cannot be negative")

    def to_dict(self) -> dict[str, float | int]:
        """Convert metrics to dictionary for MLflow logging.

        Returns:
            Dictionary with metric names as keys and values.
        """
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "avg_trade_return": self.avg_trade_return,
            "volatility": self.volatility,
        }


def calculate_sharpe_ratio(returns: pl.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from returns series.

    The Sharpe ratio measures risk-adjusted returns by comparing the mean
    excess return to the standard deviation of returns. It's annualized
    assuming 252 trading days per year.

    Args:
        returns: Series of period returns (as decimals, e.g., 0.01 for 1%)
        risk_free_rate: Annual risk-free rate (as decimal, e.g., 0.02 for 2%)

    Returns:
        Annualized Sharpe ratio. Higher is better, with > 1.0 considered good.
        Returns 0.0 if volatility is zero or insufficient data.

    Examples:
        >>> import polars as pl
        >>> returns = pl.Series([0.01, -0.005, 0.02, 0.015, -0.01])
        >>> sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)
        >>> print(f"{sharpe:.2f}")
    """
    if returns.is_empty() or returns.len() < 2:
        logger.warning("calculate_sharpe_ratio called with insufficient data")
        return 0.0

    # Filter out null values
    valid_returns = returns.drop_nulls()
    if valid_returns.len() < 2:
        logger.warning("calculate_sharpe_ratio: insufficient non-null data")
        return 0.0

    # Calculate mean and std of returns
    mean_val = valid_returns.mean()
    std_val = valid_returns.std()

    # Handle None values
    if mean_val is None or std_val is None:
        return 0.0

    mean_return = float(mean_val)  # type: ignore[arg-type]
    std_return = float(std_val)  # type: ignore[arg-type]

    # If no volatility, Sharpe is undefined
    if std_return == 0.0:
        return 0.0

    # Convert risk-free rate from annual to daily
    daily_rf_rate = risk_free_rate / 252.0

    # Calculate Sharpe ratio and annualize it (sqrt(252) for daily returns)
    sharpe = (mean_return - daily_rf_rate) / std_return * (252.0**0.5)

    return float(sharpe)


def calculate_max_drawdown(equity_curve: pl.Series) -> float:
    """Calculate maximum drawdown from equity curve.

    Maximum drawdown is the largest peak-to-trough decline in the equity curve,
    expressed as a percentage. It represents the worst possible loss an investor
    could have experienced during the backtest period.

    Args:
        equity_curve: Series of portfolio equity values over time

    Returns:
        Maximum drawdown as a percentage (positive number, e.g., 15.5 for -15.5%).
        Returns 0.0 if equity curve is empty or has insufficient data.

    Examples:
        >>> import polars as pl
        >>> equity = pl.Series([100000, 105000, 103000, 108000, 102000, 110000])
        >>> max_dd = calculate_max_drawdown(equity)
        >>> print(f"Max Drawdown: {max_dd:.2%}")
    """
    if equity_curve.is_empty() or equity_curve.len() < 2:
        logger.warning("calculate_max_drawdown called with insufficient data")
        return 0.0

    # Filter out null values
    valid_equity = equity_curve.drop_nulls()
    if valid_equity.len() < 2:
        logger.warning("calculate_max_drawdown: insufficient non-null data")
        return 0.0

    # Calculate running maximum (peak)
    running_max = valid_equity.cum_max()

    # Calculate drawdown at each point
    drawdown = (valid_equity - running_max) / running_max * 100.0

    # Maximum drawdown is the most negative value
    min_val = drawdown.min()
    if min_val is None:
        return 0.0
    max_dd = float(min_val)  # type: ignore[arg-type]

    # Return as positive number
    return abs(max_dd)


def calculate_win_rate(trades: list[Trade]) -> float:
    """Calculate percentage of winning trades.

    Win rate is the percentage of trades that resulted in a profit,
    excluding break-even trades.

    Args:
        trades: List of completed Trade objects

    Returns:
        Win rate as a percentage (0-100). Returns 0.0 if no trades.

    Examples:
        >>> from signalforge.ml.backtesting.engine import Trade
        >>> from datetime import datetime
        >>>
        >>> trades = [
        ...     Trade(..., pnl=100.0, ...),
        ...     Trade(..., pnl=-50.0, ...),
        ...     Trade(..., pnl=75.0, ...),
        ... ]
        >>> win_rate = calculate_win_rate(trades)
        >>> print(f"Win Rate: {win_rate:.1f}%")
    """
    if not trades:
        logger.warning("calculate_win_rate called with no trades")
        return 0.0

    winning_trades = sum(1 for trade in trades if trade.pnl > 0)
    total_trades = len(trades)

    if total_trades == 0:
        return 0.0

    return (winning_trades / total_trades) * 100.0


def calculate_profit_factor(trades: list[Trade]) -> float:
    """Calculate profit factor (gross profit / gross loss).

    Profit factor measures the ratio of total profits to total losses.
    A value above 1.0 indicates profitability, with higher values being better.

    Args:
        trades: List of completed Trade objects

    Returns:
        Profit factor. Values > 1.0 indicate profitable strategy.
        Returns 0.0 if no trades or no losses.

    Note:
        A profit factor of 0.0 when there are trades means all trades were losses.
        A very high profit factor (e.g., > 10) with losses means exceptional performance.
    """
    if not trades:
        logger.warning("calculate_profit_factor called with no trades")
        return 0.0

    gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
    gross_loss = abs(sum(trade.pnl for trade in trades if trade.pnl < 0))

    if gross_loss == 0.0:
        # No losses: return 0.0 if no profits either, else undefined (use large number)
        return 0.0 if gross_profit == 0.0 else float("inf")

    return gross_profit / gross_loss


def calculate_annualized_return(total_return: float, num_days: int) -> float:
    """Calculate annualized return from total return and time period.

    Uses compound annual growth rate (CAGR) formula assuming 252 trading days per year.

    Args:
        total_return: Total return as percentage (e.g., 15.5 for 15.5%)
        num_days: Number of days in the period

    Returns:
        Annualized return as percentage. Returns total_return if period < 1 year.
    """
    if num_days <= 0:
        logger.warning("calculate_annualized_return: invalid num_days")
        return 0.0

    if num_days < 252:
        # For periods less than a year, return the total return
        logger.debug("Period less than 1 year, returning total return", num_days=num_days)
        return total_return

    # Convert percentage to decimal
    total_return_decimal = total_return / 100.0

    # Calculate number of years
    years = num_days / 252.0

    # CAGR formula: ((1 + total_return) ^ (1/years)) - 1
    annualized = ((1.0 + total_return_decimal) ** (1.0 / years) - 1.0) * 100.0

    return float(annualized)


def calculate_annualized_volatility(returns: pl.Series) -> float:
    """Calculate annualized volatility from returns series.

    Volatility is the standard deviation of returns, annualized assuming
    252 trading days per year.

    Args:
        returns: Series of period returns (as decimals)

    Returns:
        Annualized volatility as percentage. Returns 0.0 if insufficient data.
    """
    if returns.is_empty() or returns.len() < 2:
        logger.warning("calculate_annualized_volatility called with insufficient data")
        return 0.0

    valid_returns = returns.drop_nulls()
    if valid_returns.len() < 2:
        return 0.0

    std_val = valid_returns.std()
    if std_val is None:
        return 0.0
    std_return = float(std_val)  # type: ignore[arg-type]

    # Annualize volatility
    annualized_vol = std_return * (252.0**0.5) * 100.0

    return float(annualized_vol)
