"""Order service for managing paper trading orders."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.exceptions import NotFoundError, ValidationError
from signalforge.core.logging import get_logger
from signalforge.models.paper_trading import (
    OrderStatus,
    PaperOrder,
    PaperPortfolio,
    PortfolioStatus,
)
from signalforge.paper_trading.schemas import OrderCreate

logger = get_logger(__name__)


class OrderService:
    """Service for managing paper trading orders."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize order service.

        Args:
            session: Database session.
        """
        self.session = session

    async def create_order(
        self,
        portfolio_id: UUID,
        order_data: OrderCreate,
    ) -> PaperOrder:
        """Create a new paper trading order.

        Args:
            portfolio_id: Portfolio ID.
            order_data: Order creation data.

        Returns:
            Newly created order.

        Raises:
            NotFoundError: If portfolio not found.
            ValidationError: If order validation fails.
        """
        # Get portfolio and validate
        result = await self.session.execute(
            select(PaperPortfolio).where(PaperPortfolio.id == portfolio_id)
        )
        portfolio = result.scalar_one_or_none()

        if portfolio is None:
            raise NotFoundError(f"Portfolio {portfolio_id} not found")

        if portfolio.status != PortfolioStatus.ACTIVE:
            raise ValidationError(
                f"Cannot place orders on portfolio with status {portfolio.status.value}"
            )

        # Validate order parameters
        self._validate_order_params(order_data)

        # Create order
        order = PaperOrder(
            portfolio_id=portfolio_id,
            symbol=order_data.symbol.upper(),
            order_type=order_data.order_type,
            side=order_data.side,
            quantity=order_data.quantity,
            limit_price=order_data.limit_price,
            stop_price=order_data.stop_price,
            status=OrderStatus.PENDING,
            filled_quantity=0,
            expires_at=order_data.expires_at,
        )

        self.session.add(order)
        await self.session.flush()
        await self.session.refresh(order)

        logger.info(
            "order_created",
            order_id=str(order.id),
            portfolio_id=str(portfolio_id),
            symbol=order.symbol,
            order_type=order.order_type.value,
            side=order.side.value,
            quantity=order.quantity,
        )

        return order

    def _validate_order_params(self, order_data: OrderCreate) -> None:
        """Validate order parameters based on order type.

        Args:
            order_data: Order data to validate.

        Raises:
            ValidationError: If validation fails.
        """
        from signalforge.models.paper_trading import OrderType

        if order_data.order_type == OrderType.LIMIT and order_data.limit_price is None:
            raise ValidationError("Limit price is required for limit orders")

        if order_data.order_type == OrderType.STOP and order_data.stop_price is None:
            raise ValidationError("Stop price is required for stop orders")

        if (
            order_data.order_type == OrderType.STOP_LIMIT
            and (order_data.limit_price is None or order_data.stop_price is None)
        ):
            raise ValidationError(
                "Both limit price and stop price are required for stop-limit orders"
            )

        if (
            order_data.order_type == OrderType.MARKET
            and (order_data.limit_price is not None or order_data.stop_price is not None)
        ):
            raise ValidationError(
                "Market orders should not have limit or stop prices"
            )

    async def cancel_order(self, order_id: UUID) -> PaperOrder:
        """Cancel a pending order.

        Args:
            order_id: Order ID.

        Returns:
            Cancelled order.

        Raises:
            NotFoundError: If order not found.
            ValidationError: If order cannot be cancelled.
        """
        result = await self.session.execute(
            select(PaperOrder).where(PaperOrder.id == order_id)
        )
        order = result.scalar_one_or_none()

        if order is None:
            raise NotFoundError(f"Order {order_id} not found")

        if order.status != OrderStatus.PENDING:
            raise ValidationError(
                f"Cannot cancel order with status {order.status.value}"
            )

        order.status = OrderStatus.CANCELLED
        await self.session.flush()

        logger.info("order_cancelled", order_id=str(order_id), symbol=order.symbol)

        return order

    async def get_orders(
        self,
        portfolio_id: UUID,
        status: OrderStatus | None = None,
        symbol: str | None = None,
    ) -> list[PaperOrder]:
        """Get orders for a portfolio.

        Args:
            portfolio_id: Portfolio ID.
            status: Optional filter by order status.
            symbol: Optional filter by symbol.

        Returns:
            List of orders.
        """
        query = select(PaperOrder).where(PaperOrder.portfolio_id == portfolio_id)

        if status is not None:
            query = query.where(PaperOrder.status == status)

        if symbol is not None:
            query = query.where(PaperOrder.symbol == symbol.upper())

        query = query.order_by(PaperOrder.created_at.desc())

        result = await self.session.execute(query)
        orders = result.scalars().all()

        logger.debug(
            "orders_retrieved",
            portfolio_id=str(portfolio_id),
            count=len(orders),
            status=status.value if status else None,
            symbol=symbol,
        )

        return list(orders)

    async def get_pending_orders(self, portfolio_id: UUID) -> list[PaperOrder]:
        """Get all pending orders for a portfolio.

        Args:
            portfolio_id: Portfolio ID.

        Returns:
            List of pending orders.
        """
        return await self.get_orders(portfolio_id, status=OrderStatus.PENDING)

    async def get_order(self, order_id: UUID) -> PaperOrder:
        """Get a single order by ID.

        Args:
            order_id: Order ID.

        Returns:
            Order instance.

        Raises:
            NotFoundError: If order not found.
        """
        result = await self.session.execute(
            select(PaperOrder).where(PaperOrder.id == order_id)
        )
        order = result.scalar_one_or_none()

        if order is None:
            raise NotFoundError(f"Order {order_id} not found")

        return order

    async def reject_order(self, order_id: UUID, reason: str) -> PaperOrder:
        """Reject an order.

        Args:
            order_id: Order ID.
            reason: Rejection reason.

        Returns:
            Rejected order.

        Raises:
            NotFoundError: If order not found.
        """
        order = await self.get_order(order_id)

        order.status = OrderStatus.REJECTED
        order.rejection_reason = reason
        await self.session.flush()

        logger.info(
            "order_rejected",
            order_id=str(order_id),
            symbol=order.symbol,
            reason=reason,
        )

        return order

    async def expire_old_orders(self) -> list[PaperOrder]:
        """Expire orders that have passed their expiration time.

        Returns:
            List of expired orders.
        """
        now = datetime.now(UTC)

        result = await self.session.execute(
            select(PaperOrder)
            .where(PaperOrder.status == OrderStatus.PENDING)
            .where(PaperOrder.expires_at.isnot(None))
            .where(PaperOrder.expires_at <= now)
        )
        orders = result.scalars().all()

        expired_orders = []
        for order in orders:
            order.status = OrderStatus.EXPIRED
            expired_orders.append(order)
            logger.info(
                "order_expired",
                order_id=str(order.id),
                symbol=order.symbol,
                expires_at=order.expires_at.isoformat() if order.expires_at else None,
            )

        if expired_orders:
            await self.session.flush()
            logger.info("orders_expired", count=len(expired_orders))

        return expired_orders
