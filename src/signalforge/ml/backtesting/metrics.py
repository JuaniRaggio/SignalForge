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


def calculate_sortino_ratio(returns: pl.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino ratio (downside deviation).

    The Sortino ratio is similar to the Sharpe ratio but uses downside deviation
    instead of total volatility. It only considers negative returns when calculating
    risk, making it more appropriate for strategies with asymmetric return distributions.

    Args:
        returns: Series of period returns (as decimals, e.g., 0.01 for 1%)
        risk_free_rate: Annual risk-free rate (as decimal, e.g., 0.02 for 2%)

    Returns:
        Annualized Sortino ratio. Higher is better, with > 1.0 considered good.
        Returns 0.0 if downside deviation is zero or insufficient data.

    Examples:
        >>> import polars as pl
        >>> returns = pl.Series([0.01, -0.005, 0.02, 0.015, -0.01])
        >>> sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)
        >>> print(f"{sortino:.2f}")
    """
    if returns.is_empty() or returns.len() < 2:
        logger.warning("calculate_sortino_ratio called with insufficient data")
        return 0.0

    # Filter out null values
    valid_returns = returns.drop_nulls()
    if valid_returns.len() < 2:
        logger.warning("calculate_sortino_ratio: insufficient non-null data")
        return 0.0

    # Calculate mean return
    mean_val = valid_returns.mean()
    if mean_val is None:
        return 0.0
    mean_return = float(mean_val)  # type: ignore[arg-type]

    # Calculate downside deviation (only negative returns)
    downside_returns = valid_returns.filter(valid_returns < 0.0)

    if downside_returns.is_empty() or downside_returns.len() < 2:
        # No downside risk - return a high value or 0 depending on convention
        # If mean return is positive and no downside, return high value
        return float("inf") if mean_return > 0 else 0.0

    downside_std_val = downside_returns.std()
    if downside_std_val is None or float(downside_std_val) == 0.0:  # type: ignore[arg-type]
        return float("inf") if mean_return > 0 else 0.0

    downside_deviation = float(downside_std_val)  # type: ignore[arg-type]

    # Convert risk-free rate from annual to daily
    daily_rf_rate = risk_free_rate / 252.0

    # Calculate Sortino ratio and annualize it
    sortino = (mean_return - daily_rf_rate) / downside_deviation * (252.0**0.5)

    return float(sortino)


def calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """Calculate Calmar ratio.

    The Calmar ratio is the annualized return divided by the maximum drawdown.
    It measures return per unit of downside risk.

    Args:
        annualized_return: Annualized return as percentage (e.g., 15.5 for 15.5%)
        max_drawdown: Maximum drawdown as percentage (positive number, e.g., 10.0 for -10%)

    Returns:
        Calmar ratio. Higher is better. Returns 0.0 if max_drawdown is zero.

    Examples:
        >>> calmar = calculate_calmar_ratio(annualized_return=20.0, max_drawdown=10.0)
        >>> print(f"Calmar Ratio: {calmar:.2f}")
    """
    if max_drawdown == 0.0:
        logger.warning("calculate_calmar_ratio: max_drawdown is zero")
        return 0.0

    return annualized_return / max_drawdown


def calculate_information_ratio(
    returns: pl.Series,
    benchmark_returns: pl.Series,
) -> float:
    """Calculate information ratio vs benchmark.

    The information ratio measures the risk-adjusted excess return relative to a benchmark.
    It's calculated as the mean of excess returns divided by the tracking error
    (standard deviation of excess returns).

    Args:
        returns: Series of portfolio returns (as decimals)
        benchmark_returns: Series of benchmark returns (as decimals)

    Returns:
        Annualized information ratio. Higher is better.
        Returns 0.0 if tracking error is zero or insufficient data.

    Examples:
        >>> import polars as pl
        >>> portfolio_returns = pl.Series([0.01, 0.02, -0.01, 0.015])
        >>> benchmark_returns = pl.Series([0.008, 0.015, -0.005, 0.012])
        >>> ir = calculate_information_ratio(portfolio_returns, benchmark_returns)
        >>> print(f"Information Ratio: {ir:.2f}")
    """
    if returns.is_empty() or benchmark_returns.is_empty():
        logger.warning("calculate_information_ratio called with empty data")
        return 0.0

    if returns.len() != benchmark_returns.len():
        logger.warning(
            "calculate_information_ratio: returns and benchmark have different lengths"
        )
        return 0.0

    # Calculate excess returns
    excess_returns = returns - benchmark_returns
    valid_excess = excess_returns.drop_nulls()

    if valid_excess.len() < 2:
        logger.warning("calculate_information_ratio: insufficient valid data")
        return 0.0

    # Mean excess return
    mean_val = valid_excess.mean()
    if mean_val is None:
        return 0.0
    mean_excess = float(mean_val)  # type: ignore[arg-type]

    # Tracking error (std of excess returns)
    std_val = valid_excess.std()
    if std_val is None or float(std_val) == 0.0:  # type: ignore[arg-type]
        return 0.0
    tracking_error = float(std_val)  # type: ignore[arg-type]

    # Annualize
    ir = mean_excess / tracking_error * (252.0**0.5)

    return float(ir)


def calculate_all(
    equity_curve: pl.Series,
    returns: pl.Series,
    trade_log: pl.DataFrame,
) -> dict[str, float]:
    """Calculate all metrics at once.

    This convenience function calculates all available performance metrics
    from the provided data.

    Args:
        equity_curve: Series of portfolio equity values over time
        returns: Series of period returns (as decimals)
        trade_log: DataFrame with trade details

    Returns:
        Dictionary mapping metric name to value

    Examples:
        >>> import polars as pl
        >>> equity = pl.Series([100000, 105000, 103000, 108000])
        >>> returns = equity.pct_change().drop_nulls()
        >>> trades_df = pl.DataFrame({...})
        >>> metrics = calculate_all(equity, returns, trades_df)
        >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """
    if equity_curve.is_empty() or returns.is_empty():
        logger.warning("calculate_all called with empty data")
        return {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "information_ratio": 0.0,
            "volatility": 0.0,
        }

    # Convert trade_log to list of Trade objects if needed
    # This is a simplified version - actual implementation may vary
    from signalforge.ml.backtesting.engine import Trade

    trades: list[Trade] = []
    if not trade_log.is_empty() and "pnl" in trade_log.columns:
        for row in trade_log.iter_rows(named=True):
            try:
                trade = Trade(
                    entry_date=row["entry_date"],
                    exit_date=row["exit_date"],
                    entry_price=float(row["entry_price"]),
                    exit_price=float(row["exit_price"]),
                    position_size=float(row["position_size"]),
                    direction=row["direction"],
                    pnl=float(row["pnl"]),
                    return_pct=float(row["return_pct"]),
                )
                trades.append(trade)
            except Exception as e:
                logger.warning(f"Failed to parse trade: {e}")
                continue

    # Calculate all metrics
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd = calculate_max_drawdown(equity_curve)
    win_rate = calculate_win_rate(trades)
    pf = calculate_profit_factor(trades)
    volatility = calculate_annualized_volatility(returns)

    # Calculate annualized return for Calmar ratio
    initial = float(equity_curve[0])
    final = float(equity_curve[-1])
    total_return = ((final - initial) / initial) * 100.0
    num_days = equity_curve.len()
    ann_return = calculate_annualized_return(total_return, num_days)

    calmar = calculate_calmar_ratio(ann_return, max_dd)

    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "profit_factor": pf,
        "information_ratio": 0.0,  # Requires benchmark data
        "volatility": volatility,
    }
