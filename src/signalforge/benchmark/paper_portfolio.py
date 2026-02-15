"""Paper portfolio engine for backtesting and simulation.

This module implements a paper trading portfolio that tracks positions, P&L,
and enforces risk management rules without executing real trades.

Key Features:
- Position lifecycle management (open, update, close)
- Automatic stop-loss and take-profit execution
- Portfolio constraints validation (max positions, position sizing)
- Real-time equity and cash tracking
- Snapshot creation for persistence

Examples:
    Basic portfolio usage:

    >>> from decimal import Decimal
    >>> from signalforge.benchmark.paper_portfolio import PaperPortfolio, PortfolioConfig
    >>>
    >>> config = PortfolioConfig(
    ...     initial_capital=Decimal("100000"),
    ...     max_positions=5,
    ...     max_position_size_pct=Decimal("20"),
    ... )
    >>> portfolio = PaperPortfolio(config)
    >>>
    >>> # Open a position
    >>> position = portfolio.open_position(
    ...     symbol="AAPL",
    ...     quantity=100,
    ...     price=Decimal("150.00"),
    ...     stop_loss=Decimal("145.00"),
    ...     take_profit=Decimal("160.00"),
    ... )
    >>> print(f"Opened position: {position.symbol} at ${position.entry_price}")
    >>>
    >>> # Update prices
    >>> portfolio.update_prices({"AAPL": Decimal("155.00")})
    >>>
    >>> # Check stop-loss/take-profit
    >>> closed = portfolio.check_stop_loss_take_profit()
    >>> print(f"Positions closed: {closed}")
    >>>
    >>> # Close position manually
    >>> pnl = portfolio.close_position("AAPL", Decimal("155.00"))
    >>> print(f"Realized P&L: ${pnl}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from signalforge.core.logging import get_logger
from signalforge.models.paper_trade import PortfolioSnapshot

logger = get_logger(__name__)

# Precision for Decimal calculations
DECIMAL_PLACES = 4


@dataclass
class PortfolioConfig:
    """Configuration for paper portfolio.

    Attributes:
        initial_capital: Starting capital in dollars.
        max_positions: Maximum number of concurrent open positions.
        max_position_size_pct: Maximum position size as percentage of equity.
        stop_loss_pct: Default stop-loss percentage (optional).
        take_profit_pct: Default take-profit percentage (optional).
    """

    initial_capital: Decimal
    max_positions: int
    max_position_size_pct: Decimal
    stop_loss_pct: Decimal | None = None
    take_profit_pct: Decimal | None = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")
        if self.max_position_size_pct <= 0 or self.max_position_size_pct > 100:
            raise ValueError("max_position_size_pct must be between 0 and 100")
        if self.stop_loss_pct is not None and (
            self.stop_loss_pct <= 0 or self.stop_loss_pct >= 100
        ):
            raise ValueError("stop_loss_pct must be between 0 and 100")
        if self.take_profit_pct is not None and self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be positive")


@dataclass
class PositionState:
    """Current state of a paper trading position.

    Attributes:
        symbol: Asset symbol.
        quantity: Number of shares/units.
        entry_price: Price at which position was opened.
        entry_date: Timestamp when position was opened.
        current_price: Current market price.
        stop_loss: Stop-loss price level.
        take_profit: Take-profit price level.
        unrealized_pnl: Current unrealized profit/loss in dollars.
        unrealized_pnl_pct: Current unrealized profit/loss as percentage.
        realized_pnl: Realized profit/loss in dollars (zero until closed).
    """

    symbol: str
    quantity: int
    entry_price: Decimal
    entry_date: datetime
    current_price: Decimal
    stop_loss: Decimal | None
    take_profit: Decimal | None
    unrealized_pnl: Decimal = field(default=Decimal("0"))
    unrealized_pnl_pct: Decimal = field(default=Decimal("0"))
    realized_pnl: Decimal = field(default=Decimal("0"))

    def __post_init__(self) -> None:
        """Validate position state and calculate initial P&L."""
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.entry_price <= 0:
            raise ValueError("entry_price must be positive")
        if self.current_price < 0:
            raise ValueError("current_price cannot be negative")
        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("stop_loss must be positive")
        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError("take_profit must be positive")

        # Calculate initial unrealized P&L
        self._update_pnl()

    def _update_pnl(self) -> None:
        """Update unrealized P&L based on current price."""
        price_diff = self.current_price - self.entry_price
        self.unrealized_pnl = (price_diff * self.quantity).quantize(
            Decimal(f"0.{'0' * DECIMAL_PLACES}")
        )
        if self.entry_price > 0:
            self.unrealized_pnl_pct = ((price_diff / self.entry_price) * 100).quantize(
                Decimal(f"0.{'0' * DECIMAL_PLACES}")
            )

    def update_price(self, new_price: Decimal) -> None:
        """Update current price and recalculate P&L.

        Args:
            new_price: New market price.

        Raises:
            ValueError: If new_price is negative.
        """
        if new_price < 0:
            raise ValueError("new_price cannot be negative")

        self.current_price = new_price
        self._update_pnl()

        logger.debug(
            "position_price_updated",
            symbol=self.symbol,
            new_price=float(new_price),
            unrealized_pnl=float(self.unrealized_pnl),
            unrealized_pnl_pct=float(self.unrealized_pnl_pct),
        )

    def check_stop_loss(self) -> bool:
        """Check if stop-loss has been triggered.

        Returns:
            True if current price is at or below stop-loss level.
        """
        if self.stop_loss is None:
            return False
        return self.current_price <= self.stop_loss

    def check_take_profit(self) -> bool:
        """Check if take-profit has been triggered.

        Returns:
            True if current price is at or above take-profit level.
        """
        if self.take_profit is None:
            return False
        return self.current_price >= self.take_profit

    def get_market_value(self) -> Decimal:
        """Calculate current market value of the position.

        Returns:
            Market value in dollars.
        """
        return (self.current_price * self.quantity).quantize(
            Decimal(f"0.{'0' * DECIMAL_PLACES}")
        )


class PaperPortfolio:
    """Paper trading portfolio simulator.

    Manages a portfolio of paper positions with risk management rules
    and automatic stop-loss/take-profit execution.
    """

    def __init__(self, config: PortfolioConfig) -> None:
        """Initialize paper portfolio.

        Args:
            config: Portfolio configuration.
        """
        self.config = config
        self._cash = config.initial_capital
        self._positions: dict[str, PositionState] = {}
        self._closed_positions: list[tuple[PositionState, Decimal, str]] = []

        logger.info(
            "paper_portfolio_initialized",
            initial_capital=float(config.initial_capital),
            max_positions=config.max_positions,
            max_position_size_pct=float(config.max_position_size_pct),
        )

    def open_position(
        self,
        symbol: str,
        quantity: int,
        price: Decimal,
        stop_loss: Decimal | None = None,
        take_profit: Decimal | None = None,
    ) -> PositionState:
        """Open a new position in the portfolio.

        Args:
            symbol: Asset symbol.
            quantity: Number of shares/units to purchase.
            price: Entry price per share/unit.
            stop_loss: Optional stop-loss price level.
            take_profit: Optional take-profit price level.

        Returns:
            Newly created position state.

        Raises:
            ValueError: If position cannot be opened due to constraints.
        """
        # Validate symbol is not empty
        if not symbol:
            raise ValueError("symbol cannot be empty")

        # Check if position already exists
        if symbol in self._positions:
            raise ValueError(f"Position already exists for symbol: {symbol}")

        # Check max positions constraint
        if len(self._positions) >= self.config.max_positions:
            raise ValueError(
                f"Cannot open position: max positions ({self.config.max_positions}) reached"
            )

        # Calculate position cost
        position_cost = (price * quantity).quantize(Decimal(f"0.{'0' * DECIMAL_PLACES}"))

        # Check if we have enough cash
        if position_cost > self._cash:
            raise ValueError(
                f"Insufficient cash: need {position_cost}, have {self._cash}"
            )

        # Check max position size constraint
        equity = self.get_equity()
        max_position_size = (equity * self.config.max_position_size_pct / 100).quantize(
            Decimal(f"0.{'0' * DECIMAL_PLACES}")
        )

        if position_cost > max_position_size:
            raise ValueError(
                f"Position size {position_cost} exceeds maximum "
                f"{max_position_size} ({self.config.max_position_size_pct}% of equity)"
            )

        # Create position
        position = PositionState(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_date=datetime.now(UTC),
            current_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        # Deduct cash
        self._cash = (self._cash - position_cost).quantize(
            Decimal(f"0.{'0' * DECIMAL_PLACES}")
        )

        # Add to positions
        self._positions[symbol] = position

        logger.info(
            "position_opened",
            symbol=symbol,
            quantity=quantity,
            price=float(price),
            cost=float(position_cost),
            remaining_cash=float(self._cash),
            stop_loss=float(stop_loss) if stop_loss else None,
            take_profit=float(take_profit) if take_profit else None,
        )

        return position

    def close_position(
        self, symbol: str, price: Decimal, reason: str = "manual"
    ) -> Decimal:
        """Close an existing position.

        Args:
            symbol: Symbol of position to close.
            price: Exit price per share/unit.
            reason: Reason for closing (manual, stop_loss, take_profit).

        Returns:
            Realized profit/loss in dollars.

        Raises:
            ValueError: If position does not exist.
        """
        if symbol not in self._positions:
            raise ValueError(f"No open position found for symbol: {symbol}")

        position = self._positions[symbol]

        # Calculate realized P&L
        price_diff = price - position.entry_price
        realized_pnl = (price_diff * position.quantity).quantize(
            Decimal(f"0.{'0' * DECIMAL_PLACES}")
        )

        # Add proceeds back to cash
        proceeds = (price * position.quantity).quantize(
            Decimal(f"0.{'0' * DECIMAL_PLACES}")
        )
        self._cash = (self._cash + proceeds).quantize(
            Decimal(f"0.{'0' * DECIMAL_PLACES}")
        )

        # Update position with realized P&L
        position.realized_pnl = realized_pnl

        # Store closed position
        self._closed_positions.append((position, price, reason))

        # Remove from active positions
        del self._positions[symbol]

        logger.info(
            "position_closed",
            symbol=symbol,
            exit_price=float(price),
            realized_pnl=float(realized_pnl),
            reason=reason,
            new_cash_balance=float(self._cash),
        )

        return realized_pnl

    def update_prices(self, prices: dict[str, Decimal]) -> None:
        """Update market prices for positions.

        Args:
            prices: Dictionary mapping symbols to current prices.
        """
        for symbol, price in prices.items():
            if symbol in self._positions:
                self._positions[symbol].update_price(price)

        logger.debug(
            "portfolio_prices_updated",
            symbols_updated=list(prices.keys()),
            positions_count=len(self._positions),
        )

    def check_stop_loss_take_profit(self) -> list[str]:
        """Check and execute stop-loss/take-profit orders.

        Returns:
            List of symbols that were closed.
        """
        closed_symbols: list[str] = []

        for symbol, position in list(self._positions.items()):
            reason: Literal["stop_loss", "take_profit"] | None = None

            if position.check_stop_loss():
                reason = "stop_loss"
                exit_price = position.stop_loss
            elif position.check_take_profit():
                reason = "take_profit"
                exit_price = position.take_profit

            if reason and exit_price:
                self.close_position(symbol, exit_price, reason)
                closed_symbols.append(symbol)

        if closed_symbols:
            logger.info(
                "automatic_positions_closed",
                closed_symbols=closed_symbols,
                count=len(closed_symbols),
            )

        return closed_symbols

    def get_equity(self) -> Decimal:
        """Calculate total portfolio equity.

        Returns:
            Total equity (cash + positions value) in dollars.
        """
        positions_value = sum(
            (position.get_market_value() for position in self._positions.values()),
            start=Decimal("0"),
        )
        equity = (self._cash + positions_value).quantize(
            Decimal(f"0.{'0' * DECIMAL_PLACES}")
        )
        return equity

    def get_cash(self) -> Decimal:
        """Get current cash balance.

        Returns:
            Cash balance in dollars.
        """
        return self._cash

    def get_positions(self) -> list[PositionState]:
        """Get all open positions.

        Returns:
            List of current position states.
        """
        return list(self._positions.values())

    def get_positions_value(self) -> Decimal:
        """Calculate total value of open positions.

        Returns:
            Total positions value in dollars.
        """
        return sum(
            (position.get_market_value() for position in self._positions.values()),
            start=Decimal("0"),
        ).quantize(Decimal(f"0.{'0' * DECIMAL_PLACES}"))

    def get_total_pnl(self) -> Decimal:
        """Calculate total profit/loss (realized + unrealized).

        Returns:
            Total P&L in dollars.
        """
        unrealized_pnl = sum(
            (position.unrealized_pnl for position in self._positions.values()),
            start=Decimal("0"),
        )
        realized_pnl = sum(
            (closed[0].realized_pnl for closed in self._closed_positions),
            start=Decimal("0"),
        )
        return (unrealized_pnl + realized_pnl).quantize(
            Decimal(f"0.{'0' * DECIMAL_PLACES}")
        )

    def get_position(self, symbol: str) -> PositionState | None:
        """Get position state for a specific symbol.

        Args:
            symbol: Asset symbol.

        Returns:
            Position state if exists, None otherwise.
        """
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if portfolio has an open position for symbol.

        Args:
            symbol: Asset symbol.

        Returns:
            True if position exists, False otherwise.
        """
        return symbol in self._positions

    def to_snapshot(self, snapshot_date: datetime | None = None) -> PortfolioSnapshot:
        """Create a portfolio snapshot for persistence.

        Args:
            snapshot_date: Timestamp for snapshot (defaults to now).

        Returns:
            PortfolioSnapshot model instance.
        """
        if snapshot_date is None:
            snapshot_date = datetime.now(UTC)

        equity = self.get_equity()
        positions_value = self.get_positions_value()

        snapshot = PortfolioSnapshot(
            snapshot_date=snapshot_date,
            equity_value=equity.quantize(Decimal("0.01")),
            cash_balance=self._cash.quantize(Decimal("0.01")),
            positions_value=positions_value.quantize(Decimal("0.01")),
            positions_count=len(self._positions),
            daily_return_pct=None,  # Calculated externally by comparing snapshots
        )

        logger.debug(
            "portfolio_snapshot_created",
            snapshot_date=snapshot_date.isoformat(),
            equity=float(equity),
            cash=float(self._cash),
            positions_value=float(positions_value),
            positions_count=len(self._positions),
        )

        return snapshot
