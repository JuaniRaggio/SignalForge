"""Performance metrics calculation for paper trading portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID

import polars as pl
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.logging import get_logger
from signalforge.models.paper_trading import (
    PaperPortfolioSnapshot,
    PaperTrade,
)

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""

    portfolio_id: UUID
    total_return_pct: Decimal
    annualized_return_pct: Decimal
    sharpe_ratio: Decimal
    sortino_ratio: Decimal
    max_drawdown_pct: Decimal
    max_drawdown_duration_days: int
    win_rate: Decimal
    profit_factor: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win_pct: Decimal
    avg_loss_pct: Decimal
    best_trade_pct: Decimal
    worst_trade_pct: Decimal
    avg_holding_period_days: Decimal
    volatility_annualized: Decimal
    calmar_ratio: Decimal


class PerformanceService:
    """Calculate performance metrics for portfolios."""

    def __init__(self, risk_free_rate: float = 0.05) -> None:
        """Initialize with risk-free rate for Sharpe calculation.

        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate

    async def calculate_metrics(
        self,
        portfolio_id: UUID,
        session: AsyncSession,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics.

        Args:
            portfolio_id: UUID of portfolio to analyze
            session: Database session
            start_date: Optional start date for analysis period
            end_date: Optional end date for analysis period

        Returns:
            PerformanceMetrics dataclass with all calculated metrics
        """
        logger.info(
            "calculating_performance_metrics",
            portfolio_id=str(portfolio_id),
            start_date=start_date,
            end_date=end_date,
        )

        # Fetch snapshots for equity curve and returns
        snapshots_query = (
            select(PaperPortfolioSnapshot)
            .where(PaperPortfolioSnapshot.portfolio_id == portfolio_id)
            .order_by(PaperPortfolioSnapshot.snapshot_date)
        )

        if start_date:
            snapshots_query = snapshots_query.where(
                PaperPortfolioSnapshot.snapshot_date >= start_date
            )
        if end_date:
            snapshots_query = snapshots_query.where(
                PaperPortfolioSnapshot.snapshot_date <= end_date
            )

        result = await session.execute(snapshots_query)
        snapshots = result.scalars().all()

        if not snapshots:
            logger.warning(
                "no_snapshots_found",
                portfolio_id=str(portfolio_id),
            )
            # Return zero metrics if no data
            return self._zero_metrics(portfolio_id)

        # Convert to Polars DataFrame for efficient calculations
        snapshot_data = [
            {
                "date": s.snapshot_date,
                "equity": float(s.equity_value),
                "daily_return_pct": float(s.daily_return_pct) if s.daily_return_pct else None,
            }
            for s in snapshots
        ]

        df = pl.DataFrame(snapshot_data)

        # Calculate returns series (filter out None values)
        returns_series = df.filter(pl.col("daily_return_pct").is_not_null()).select(
            "daily_return_pct"
        )["daily_return_pct"]

        # Calculate equity curve metrics
        equity_series = df["equity"]
        initial_equity = equity_series[0]
        final_equity = equity_series[-1]

        # Total return
        total_return_pct = Decimal(str(((final_equity - initial_equity) / initial_equity) * 100))

        # Calculate number of trading days
        days_elapsed = (snapshots[-1].snapshot_date - snapshots[0].snapshot_date).days
        years_elapsed = max(days_elapsed / 252, 1 / 252)  # Use trading days (252 per year)

        # Annualized return
        annualized_return_pct = Decimal(
            str((pow(final_equity / initial_equity, 1 / years_elapsed) - 1) * 100)
        )

        # Risk metrics
        if len(returns_series) > 1:
            sharpe_ratio = self.calculate_sharpe_ratio(returns_series, self.risk_free_rate)
            sortino_ratio = self.calculate_sortino_ratio(returns_series, self.risk_free_rate)
            volatility_annualized = Decimal(
                str(returns_series.std() * (252**0.5))
            )  # Annualized volatility
        else:
            sharpe_ratio = Decimal("0")
            sortino_ratio = Decimal("0")
            volatility_annualized = Decimal("0")

        # Drawdown metrics
        max_drawdown_pct, max_drawdown_duration_days = self.calculate_max_drawdown(equity_series)

        # Calmar ratio
        calmar_ratio = self.calculate_calmar_ratio(annualized_return_pct, max_drawdown_pct)

        # Trade statistics
        trade_stats = await self.get_trade_statistics(portfolio_id, session, start_date, end_date)

        return PerformanceMetrics(
            portfolio_id=portfolio_id,
            total_return_pct=total_return_pct,
            annualized_return_pct=annualized_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown_pct=max_drawdown_pct,
            max_drawdown_duration_days=max_drawdown_duration_days,
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            total_trades=trade_stats["total_trades"],
            winning_trades=trade_stats["winning_trades"],
            losing_trades=trade_stats["losing_trades"],
            avg_win_pct=trade_stats["avg_win_pct"],
            avg_loss_pct=trade_stats["avg_loss_pct"],
            best_trade_pct=trade_stats["best_trade_pct"],
            worst_trade_pct=trade_stats["worst_trade_pct"],
            avg_holding_period_days=trade_stats["avg_holding_period_days"],
            volatility_annualized=volatility_annualized,
            calmar_ratio=calmar_ratio,
        )

    def calculate_sharpe_ratio(
        self, returns: pl.Series, risk_free_rate: float = 0.05
    ) -> Decimal:
        """Calculate annualized Sharpe ratio.

        Args:
            returns: Series of daily returns (as percentages)
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio as Decimal
        """
        if len(returns) == 0 or returns.std() == 0:
            return Decimal("0")

        # Convert daily returns from percentages to decimals
        returns_decimal = returns / 100.0

        # Daily risk-free rate
        daily_rf_rate = risk_free_rate / 252

        # Calculate excess returns
        mean_return = returns_decimal.mean()
        std_return = returns_decimal.std()

        if std_return == 0:
            return Decimal("0")

        # Annualized Sharpe ratio
        sharpe = ((mean_return - daily_rf_rate) / std_return) * (252**0.5)  # type: ignore[operator]

        return Decimal(str(sharpe))

    def calculate_sortino_ratio(
        self, returns: pl.Series, risk_free_rate: float = 0.05
    ) -> Decimal:
        """Calculate Sortino ratio using downside deviation.

        Args:
            returns: Series of daily returns (as percentages)
            risk_free_rate: Annual risk-free rate

        Returns:
            Sortino ratio as Decimal
        """
        if len(returns) == 0:
            return Decimal("0")

        # Convert daily returns from percentages to decimals
        returns_decimal = returns / 100.0

        # Daily risk-free rate
        daily_rf_rate = risk_free_rate / 252

        mean_return = returns_decimal.mean()

        # Calculate downside deviation (only negative returns)
        downside_returns = returns_decimal.filter(returns_decimal < 0)

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return Decimal("0")

        downside_std = downside_returns.std()

        # Annualized Sortino ratio
        sortino = ((mean_return - daily_rf_rate) / downside_std) * (252**0.5)  # type: ignore[operator]

        return Decimal(str(sortino))

    def calculate_max_drawdown(self, equity_curve: pl.Series) -> tuple[Decimal, int]:
        """Calculate maximum drawdown and duration.

        Args:
            equity_curve: Series of portfolio equity values

        Returns:
            Tuple of (max_drawdown_pct, max_drawdown_duration_days)
        """
        if len(equity_curve) == 0:
            return Decimal("0"), 0

        # Calculate running maximum
        running_max = equity_curve.cum_max()

        # Calculate drawdown at each point
        drawdown = (equity_curve - running_max) / running_max * 100

        # Maximum drawdown (most negative value)
        min_dd = drawdown.min()
        max_dd = abs(min_dd) if min_dd is not None and min_dd < 0 else 0.0  # type: ignore[operator]

        # Calculate drawdown duration
        # Find periods where we are in drawdown
        in_drawdown = drawdown < 0
        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return Decimal(str(max_dd)), max_duration

    def calculate_calmar_ratio(
        self, annualized_return: Decimal, max_drawdown: Decimal
    ) -> Decimal:
        """Calculate Calmar ratio (return / max drawdown).

        Args:
            annualized_return: Annualized return percentage
            max_drawdown: Maximum drawdown percentage

        Returns:
            Calmar ratio as Decimal
        """
        if max_drawdown == 0:
            return Decimal("0")

        calmar = annualized_return / max_drawdown
        return calmar

    async def get_trade_statistics(
        self,
        portfolio_id: UUID,
        session: AsyncSession,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict:
        """Calculate trade-based statistics.

        Args:
            portfolio_id: UUID of portfolio
            session: Database session
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Dictionary with trade statistics
        """
        # Fetch all closed trades
        trades_query = (
            select(PaperTrade)
            .where(PaperTrade.portfolio_id == portfolio_id)
            .order_by(PaperTrade.executed_at)
        )

        if start_date:
            trades_query = trades_query.where(PaperTrade.executed_at >= start_date)
        if end_date:
            trades_query = trades_query.where(PaperTrade.executed_at <= end_date)

        result = await session.execute(trades_query)
        trades = result.scalars().all()

        if not trades:
            return self._zero_trade_stats()

        # Group trades by symbol to calculate P&L
        # We need matching buy/sell pairs to calculate trade statistics
        trades_data = [
            {
                "symbol": t.symbol,
                "side": t.side.value,
                "quantity": t.quantity,
                "price": float(t.price),
                "executed_at": t.executed_at,
            }
            for t in trades
        ]

        df = pl.DataFrame(trades_data)

        # For simplicity, calculate based on round-trip trades
        # Group by symbol and calculate P&L for completed positions
        wins = []
        losses = []
        holding_periods = []

        # Process trades by symbol
        for symbol in df["symbol"].unique():
            symbol_trades = df.filter(pl.col("symbol") == symbol).sort("executed_at")

            position = 0
            avg_entry_price = 0.0
            entry_time = None

            for row in symbol_trades.iter_rows(named=True):
                if row["side"] == "buy":
                    # Opening or adding to position
                    total_cost = position * avg_entry_price + row["quantity"] * row["price"]
                    position += row["quantity"]
                    avg_entry_price = total_cost / position if position > 0 else 0.0
                    if entry_time is None:
                        entry_time = row["executed_at"]
                else:  # sell
                    # Closing or reducing position
                    if position > 0:
                        exit_price = row["price"]
                        pnl_pct = ((exit_price - avg_entry_price) / avg_entry_price) * 100

                        if pnl_pct > 0:
                            wins.append(pnl_pct)
                        else:
                            losses.append(pnl_pct)

                        # Calculate holding period
                        if entry_time:
                            holding_period = (row["executed_at"] - entry_time).days
                            holding_periods.append(holding_period)

                        # Reduce position
                        position -= row["quantity"]
                        if position <= 0:
                            position = 0
                            avg_entry_price = 0.0
                            entry_time = None

        total_trades = len(wins) + len(losses)
        winning_trades = len(wins)
        losing_trades = len(losses)

        win_rate = Decimal(str((winning_trades / total_trades * 100) if total_trades > 0 else 0))

        avg_win_pct = Decimal(str(sum(wins) / len(wins))) if wins else Decimal("0")
        avg_loss_pct = Decimal(str(sum(losses) / len(losses))) if losses else Decimal("0")

        best_trade_pct = Decimal(str(max(wins))) if wins else Decimal("0")
        worst_trade_pct = Decimal(str(min(losses))) if losses else Decimal("0")

        # Profit factor: sum of wins / abs(sum of losses)
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = (
            Decimal(str(total_wins / total_losses)) if total_losses > 0 else Decimal("0")
        )

        avg_holding_period_days = (
            Decimal(str(sum(holding_periods) / len(holding_periods)))
            if holding_periods
            else Decimal("0")
        )

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win_pct": avg_win_pct,
            "avg_loss_pct": avg_loss_pct,
            "best_trade_pct": best_trade_pct,
            "worst_trade_pct": worst_trade_pct,
            "avg_holding_period_days": avg_holding_period_days,
        }

    def _zero_metrics(self, portfolio_id: UUID) -> PerformanceMetrics:
        """Return zero metrics when no data available."""
        return PerformanceMetrics(
            portfolio_id=portfolio_id,
            total_return_pct=Decimal("0"),
            annualized_return_pct=Decimal("0"),
            sharpe_ratio=Decimal("0"),
            sortino_ratio=Decimal("0"),
            max_drawdown_pct=Decimal("0"),
            max_drawdown_duration_days=0,
            win_rate=Decimal("0"),
            profit_factor=Decimal("0"),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            avg_win_pct=Decimal("0"),
            avg_loss_pct=Decimal("0"),
            best_trade_pct=Decimal("0"),
            worst_trade_pct=Decimal("0"),
            avg_holding_period_days=Decimal("0"),
            volatility_annualized=Decimal("0"),
            calmar_ratio=Decimal("0"),
        )

    def _zero_trade_stats(self) -> dict:
        """Return zero trade statistics."""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": Decimal("0"),
            "profit_factor": Decimal("0"),
            "avg_win_pct": Decimal("0"),
            "avg_loss_pct": Decimal("0"),
            "best_trade_pct": Decimal("0"),
            "worst_trade_pct": Decimal("0"),
            "avg_holding_period_days": Decimal("0"),
        }
