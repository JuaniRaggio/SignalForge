"""Execution engine for paper trading order fulfillment."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.exceptions import ValidationError
from signalforge.core.logging import get_logger
from signalforge.models.paper_trading import (
    OrderSide,
    OrderStatus,
    OrderType,
    PaperOrder,
    PaperPortfolio,
    PaperPosition,
    PaperTrade,
)

logger = get_logger(__name__)

# Constants for slippage calculation
BASE_SLIPPAGE_BPS = 1  # 0.01% base slippage
LIQUIDITY_IMPACT_FACTOR = Decimal("0.00001")  # Per share impact


class ExecutionEngine:
    """Engine for executing paper trading orders."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize execution engine.

        Args:
            session: Database session.
        """
        self.session = session

    async def execute_market_order(
        self,
        order: PaperOrder,
        current_price: Decimal,
    ) -> PaperTrade:
        """Execute a market order immediately.

        Args:
            order: Order to execute.
            current_price: Current market price for the symbol.

        Returns:
            Executed trade.

        Raises:
            ValidationError: If order cannot be executed.
        """
        if order.order_type != OrderType.MARKET:
            raise ValidationError("Only market orders can be executed as market orders")

        if order.status != OrderStatus.PENDING:
            raise ValidationError(
                f"Cannot execute order with status {order.status.value}"
            )

        # Calculate slippage
        slippage = self.apply_slippage(current_price, order.quantity, order.side)
        execution_price = current_price + slippage

        # Validate sufficient cash for buy orders
        if order.side == OrderSide.BUY:
            await self._validate_sufficient_cash(order, execution_price)

        # Create trade
        trade = await self._create_trade(
            order=order,
            execution_price=execution_price,
            slippage=abs(slippage),
        )

        # Update portfolio (cash and positions)
        await self._update_portfolio_after_execution(order, trade)

        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = execution_price
        order.filled_at = datetime.now(UTC)

        await self.session.flush()

        logger.info(
            "market_order_executed",
            order_id=str(order.id),
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            price=float(execution_price),
            slippage=float(slippage),
        )

        return trade

    async def check_limit_orders(
        self,
        portfolio_id: UUID,
        current_prices: dict[str, Decimal],
    ) -> list[PaperTrade]:
        """Check and execute limit orders that have hit their price.

        Args:
            portfolio_id: Portfolio ID.
            current_prices: Map of symbol to current price.

        Returns:
            List of executed trades.
        """
        # Get pending limit orders for this portfolio
        result = await self.session.execute(
            select(PaperOrder)
            .where(PaperOrder.portfolio_id == portfolio_id)
            .where(PaperOrder.status == OrderStatus.PENDING)
            .where(PaperOrder.order_type == OrderType.LIMIT)
        )
        orders = result.scalars().all()

        executed_trades: list[PaperTrade] = []

        for order in orders:
            if order.symbol not in current_prices:
                continue

            current_price = current_prices[order.symbol]
            should_execute = False

            # Check if limit price is hit (skip if no limit price)
            if order.limit_price is None:
                continue
            if (
                (order.side == OrderSide.BUY and current_price <= order.limit_price)
                or (order.side == OrderSide.SELL and current_price >= order.limit_price)
            ):
                should_execute = True

            if should_execute:
                try:
                    # Execute at limit price (not current price for limit orders)
                    execution_price = order.limit_price

                    # Validate sufficient cash for buy orders
                    if order.side == OrderSide.BUY:
                        await self._validate_sufficient_cash(order, execution_price)

                    # Create trade
                    trade = await self._create_trade(
                        order=order,
                        execution_price=execution_price,
                        slippage=Decimal("0"),  # No slippage on limit orders
                    )

                    # Update portfolio
                    await self._update_portfolio_after_execution(order, trade)

                    # Update order
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.filled_price = execution_price
                    order.filled_at = datetime.now(UTC)

                    executed_trades.append(trade)

                    logger.info(
                        "limit_order_executed",
                        order_id=str(order.id),
                        symbol=order.symbol,
                        side=order.side.value,
                        quantity=order.quantity,
                        limit_price=float(order.limit_price),
                        current_price=float(current_price),
                    )

                except ValidationError as e:
                    logger.warning(
                        "limit_order_execution_failed",
                        order_id=str(order.id),
                        symbol=order.symbol,
                        error=str(e),
                    )
                    continue

        if executed_trades:
            await self.session.flush()

        return executed_trades

    async def check_stop_orders(
        self,
        portfolio_id: UUID,
        current_prices: dict[str, Decimal],
    ) -> list[PaperTrade]:
        """Check and execute stop orders that have been triggered.

        Args:
            portfolio_id: Portfolio ID.
            current_prices: Map of symbol to current price.

        Returns:
            List of executed trades.
        """
        # Get pending stop orders for this portfolio
        result = await self.session.execute(
            select(PaperOrder)
            .where(PaperOrder.portfolio_id == portfolio_id)
            .where(PaperOrder.status == OrderStatus.PENDING)
            .where(PaperOrder.order_type == OrderType.STOP)
        )
        orders = result.scalars().all()

        executed_trades: list[PaperTrade] = []

        for order in orders:
            if order.symbol not in current_prices:
                continue

            current_price = current_prices[order.symbol]
            should_execute = False

            # Check if stop price is hit (skip if no stop price)
            if order.stop_price is None:
                continue
            if (
                (order.side == OrderSide.BUY and current_price >= order.stop_price)
                or (order.side == OrderSide.SELL and current_price <= order.stop_price)
            ):
                should_execute = True

            if should_execute:
                try:
                    # Execute at market price with slippage
                    slippage = self.apply_slippage(current_price, order.quantity, order.side)
                    execution_price = current_price + slippage

                    # Validate sufficient cash for buy orders
                    if order.side == OrderSide.BUY:
                        await self._validate_sufficient_cash(order, execution_price)

                    # Create trade
                    trade = await self._create_trade(
                        order=order,
                        execution_price=execution_price,
                        slippage=abs(slippage),
                    )

                    # Update portfolio
                    await self._update_portfolio_after_execution(order, trade)

                    # Update order
                    order.status = OrderStatus.FILLED
                    order.filled_quantity = order.quantity
                    order.filled_price = execution_price
                    order.filled_at = datetime.now(UTC)

                    executed_trades.append(trade)

                    logger.info(
                        "stop_order_executed",
                        order_id=str(order.id),
                        symbol=order.symbol,
                        side=order.side.value,
                        quantity=order.quantity,
                        stop_price=float(order.stop_price),
                        execution_price=float(execution_price),
                    )

                except ValidationError as e:
                    logger.warning(
                        "stop_order_execution_failed",
                        order_id=str(order.id),
                        symbol=order.symbol,
                        error=str(e),
                    )
                    continue

        if executed_trades:
            await self.session.flush()

        return executed_trades

    def apply_slippage(
        self,
        price: Decimal,
        quantity: int,
        side: OrderSide,
    ) -> Decimal:
        """Calculate realistic slippage for an order.

        Slippage model:
        - Base slippage: 0.01% (1 bps)
        - Liquidity impact: increases with quantity
        - Buy orders: positive slippage (price increases)
        - Sell orders: negative slippage (price decreases)

        Args:
            price: Base price.
            quantity: Order quantity.
            side: Order side (buy/sell).

        Returns:
            Slippage amount (positive for buy, negative for sell).
        """
        # Base slippage in basis points
        base_slippage = price * Decimal(BASE_SLIPPAGE_BPS) / Decimal("10000")

        # Liquidity impact based on quantity
        liquidity_impact = price * LIQUIDITY_IMPACT_FACTOR * Decimal(quantity)

        # Total slippage
        total_slippage = base_slippage + liquidity_impact

        # Apply direction (positive for buy, negative for sell)
        if side == OrderSide.BUY:
            return total_slippage
        else:
            return -total_slippage

    async def _validate_sufficient_cash(
        self,
        order: PaperOrder,
        execution_price: Decimal,
    ) -> None:
        """Validate portfolio has sufficient cash for buy order.

        Args:
            order: Order to validate.
            execution_price: Price at which order will execute.

        Raises:
            ValidationError: If insufficient cash.
        """
        result = await self.session.execute(
            select(PaperPortfolio).where(PaperPortfolio.id == order.portfolio_id)
        )
        portfolio = result.scalar_one()

        required_cash = execution_price * order.quantity

        if portfolio.current_cash < required_cash:
            raise ValidationError(
                f"Insufficient cash: need {required_cash}, have {portfolio.current_cash}"
            )

    async def _create_trade(
        self,
        order: PaperOrder,
        execution_price: Decimal,
        slippage: Decimal,
    ) -> PaperTrade:
        """Create a trade record for an executed order.

        Args:
            order: Order being executed.
            execution_price: Execution price.
            slippage: Slippage amount.

        Returns:
            Created trade.
        """
        # Calculate commission (for now, set to 0 for paper trading)
        commission = Decimal("0")

        trade = PaperTrade(
            portfolio_id=order.portfolio_id,
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=execution_price,
            commission=commission,
            slippage=slippage,
            executed_at=datetime.now(UTC),
        )

        self.session.add(trade)
        await self.session.flush()
        await self.session.refresh(trade)

        return trade

    async def _update_portfolio_after_execution(
        self,
        order: PaperOrder,
        trade: PaperTrade,
    ) -> None:
        """Update portfolio cash and positions after trade execution.

        Args:
            order: Executed order.
            trade: Trade record.
        """
        # Get portfolio
        result = await self.session.execute(
            select(PaperPortfolio).where(PaperPortfolio.id == order.portfolio_id)
        )
        portfolio = result.scalar_one()

        # Calculate total cost/proceeds
        total_value = trade.price * trade.quantity + trade.commission

        if order.side == OrderSide.BUY:
            # Deduct cash
            portfolio.current_cash -= total_value

            # Update or create position
            await self._update_position_after_buy(
                portfolio_id=order.portfolio_id,
                symbol=order.symbol,
                quantity=order.quantity,
                price=trade.price,
            )

        else:  # SELL
            # Add cash
            portfolio.current_cash += total_value

            # Update or close position
            await self._update_position_after_sell(
                portfolio_id=order.portfolio_id,
                symbol=order.symbol,
                quantity=order.quantity,
                price=trade.price,
            )

    async def _update_position_after_buy(
        self,
        portfolio_id: UUID,
        symbol: str,
        quantity: int,
        price: Decimal,
    ) -> None:
        """Update position after buy order execution.

        Args:
            portfolio_id: Portfolio ID.
            symbol: Symbol.
            quantity: Quantity bought.
            price: Execution price.
        """
        # Get existing position if any
        result = await self.session.execute(
            select(PaperPosition)
            .where(PaperPosition.portfolio_id == portfolio_id)
            .where(PaperPosition.symbol == symbol)
        )
        position = result.scalar_one_or_none()

        if position is None:
            # Create new position
            position = PaperPosition(
                portfolio_id=portfolio_id,
                symbol=symbol,
                quantity=quantity,
                avg_entry_price=price,
                current_price=price,
                unrealized_pnl=Decimal("0"),
                unrealized_pnl_pct=Decimal("0"),
            )
            self.session.add(position)
        else:
            # Update existing position (average price)
            total_cost = (position.avg_entry_price * position.quantity) + (price * quantity)
            total_quantity = position.quantity + quantity
            position.avg_entry_price = total_cost / total_quantity
            position.quantity = total_quantity
            position.current_price = price

            # Recalculate unrealized P&L
            price_diff = position.current_price - position.avg_entry_price
            position.unrealized_pnl = price_diff * position.quantity
            if position.avg_entry_price > 0:
                position.unrealized_pnl_pct = (price_diff / position.avg_entry_price) * 100

    async def _update_position_after_sell(
        self,
        portfolio_id: UUID,
        symbol: str,
        quantity: int,
        price: Decimal,
    ) -> None:
        """Update position after sell order execution.

        Args:
            portfolio_id: Portfolio ID.
            symbol: Symbol.
            quantity: Quantity sold.
            price: Execution price.

        Raises:
            ValidationError: If insufficient shares to sell.
        """
        # Get existing position
        result = await self.session.execute(
            select(PaperPosition)
            .where(PaperPosition.portfolio_id == portfolio_id)
            .where(PaperPosition.symbol == symbol)
        )
        position = result.scalar_one_or_none()

        if position is None:
            raise ValidationError(f"No position found for {symbol} to sell")

        if position.quantity < quantity:
            raise ValidationError(
                f"Insufficient shares: trying to sell {quantity}, "
                f"but only have {position.quantity}"
            )

        # Update position
        position.quantity -= quantity
        position.current_price = price

        # If position is closed, remove it
        if position.quantity == 0:
            await self.session.delete(position)
        else:
            # Recalculate unrealized P&L
            price_diff = position.current_price - position.avg_entry_price
            position.unrealized_pnl = price_diff * position.quantity
            if position.avg_entry_price > 0:
                position.unrealized_pnl_pct = (price_diff / position.avg_entry_price) * 100
