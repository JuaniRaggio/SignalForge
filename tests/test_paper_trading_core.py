"""Tests for paper trading core functionality."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.exceptions import NotFoundError, ValidationError
from signalforge.models.paper_trading import (
    OrderSide,
    OrderStatus,
    OrderType,
    PortfolioStatus,
)
from signalforge.models.user import User, UserType
from signalforge.paper_trading.execution_engine import ExecutionEngine
from signalforge.paper_trading.order_service import OrderService
from signalforge.paper_trading.portfolio_service import PortfolioService
from signalforge.paper_trading.schemas import OrderCreate, PortfolioCreate


@pytest.fixture
async def test_user(db_session: AsyncSession) -> User:
    """Create a test user."""
    user = User(
        email="test@example.com",
        username="testuser",
        hashed_password="hashed_password",
        user_type=UserType.ACTIVE_TRADER,
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)
    return user


@pytest.fixture
async def portfolio_service(db_session: AsyncSession) -> PortfolioService:
    """Create portfolio service."""
    return PortfolioService(db_session)


@pytest.fixture
async def order_service(db_session: AsyncSession) -> OrderService:
    """Create order service."""
    return OrderService(db_session)


@pytest.fixture
async def execution_engine(db_session: AsyncSession) -> ExecutionEngine:
    """Create execution engine."""
    return ExecutionEngine(db_session)


class TestPortfolioService:
    """Tests for PortfolioService."""

    @pytest.mark.asyncio
    async def test_create_portfolio(
        self,
        portfolio_service: PortfolioService,
        test_user: User,
    ) -> None:
        """Test creating a portfolio."""
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
        )

        portfolio = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        assert portfolio.name == "Test Portfolio"
        assert portfolio.initial_capital == Decimal("100000.00")
        assert portfolio.current_cash == Decimal("100000.00")
        assert portfolio.status == PortfolioStatus.ACTIVE
        assert portfolio.user_id == test_user.id

    @pytest.mark.asyncio
    async def test_get_portfolio(
        self,
        portfolio_service: PortfolioService,
        test_user: User,
    ) -> None:
        """Test retrieving a portfolio."""
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
        )
        created = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        retrieved = await portfolio_service.get_portfolio(created.id)

        assert retrieved.id == created.id
        assert retrieved.name == "Test Portfolio"

    @pytest.mark.asyncio
    async def test_get_portfolio_not_found(
        self,
        portfolio_service: PortfolioService,
    ) -> None:
        """Test retrieving non-existent portfolio."""
        with pytest.raises(NotFoundError):
            await portfolio_service.get_portfolio(uuid4())

    @pytest.mark.asyncio
    async def test_get_user_portfolios(
        self,
        portfolio_service: PortfolioService,
        test_user: User,
    ) -> None:
        """Test retrieving user's portfolios."""
        # Create multiple portfolios
        for i in range(3):
            portfolio_data = PortfolioCreate(
                name=f"Portfolio {i}",
                initial_capital=Decimal("100000.00"),
            )
            await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        portfolios = await portfolio_service.get_user_portfolios(test_user.id)

        assert len(portfolios) == 3

    @pytest.mark.asyncio
    async def test_get_portfolio_summary(
        self,
        portfolio_service: PortfolioService,
        test_user: User,
    ) -> None:
        """Test getting portfolio summary."""
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
        )
        portfolio = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        summary = await portfolio_service.get_portfolio_summary(portfolio.id)

        assert summary.id == portfolio.id
        assert summary.equity_value == Decimal("100000.00")
        assert summary.cash_balance == Decimal("100000.00")
        assert summary.positions_value == Decimal("0")
        assert summary.positions_count == 0
        assert summary.total_pnl == Decimal("0")
        assert summary.total_pnl_pct == Decimal("0")

    @pytest.mark.asyncio
    async def test_update_position_prices(
        self,
        portfolio_service: PortfolioService,
        order_service: OrderService,
        execution_engine: ExecutionEngine,
        test_user: User,
        db_session: AsyncSession,
    ) -> None:
        """Test updating position prices."""
        # Create portfolio
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
        )
        portfolio = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        # Place and execute a buy order
        order_data = OrderCreate(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=10,
        )
        order = await order_service.create_order(portfolio.id, order_data)
        await execution_engine.execute_market_order(order, Decimal("150.00"))
        await db_session.commit()

        # Update prices
        updated_positions = await portfolio_service.update_position_prices(
            portfolio.id,
            {"AAPL": Decimal("160.00")},
        )

        assert len(updated_positions) == 1
        assert updated_positions[0].symbol == "AAPL"
        assert updated_positions[0].current_price == Decimal("160.00")
        assert updated_positions[0].unrealized_pnl > Decimal("0")  # Profit

    @pytest.mark.asyncio
    async def test_close_portfolio(
        self,
        portfolio_service: PortfolioService,
        test_user: User,
    ) -> None:
        """Test closing a portfolio."""
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
        )
        portfolio = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        closed = await portfolio_service.close_portfolio(portfolio.id)

        assert closed.status == PortfolioStatus.CLOSED


class TestOrderService:
    """Tests for OrderService."""

    @pytest.mark.asyncio
    async def test_create_market_order(
        self,
        portfolio_service: PortfolioService,
        order_service: OrderService,
        test_user: User,
    ) -> None:
        """Test creating a market order."""
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
        )
        portfolio = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        order_data = OrderCreate(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=10,
        )

        order = await order_service.create_order(portfolio.id, order_data)

        assert order.symbol == "AAPL"
        assert order.order_type == OrderType.MARKET
        assert order.side == OrderSide.BUY
        assert order.quantity == 10
        assert order.status == OrderStatus.PENDING

    @pytest.mark.asyncio
    async def test_create_limit_order(
        self,
        portfolio_service: PortfolioService,
        order_service: OrderService,
        test_user: User,
    ) -> None:
        """Test creating a limit order."""
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
        )
        portfolio = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        order_data = OrderCreate(
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=10,
            limit_price=Decimal("145.00"),
        )

        order = await order_service.create_order(portfolio.id, order_data)

        assert order.order_type == OrderType.LIMIT
        assert order.limit_price == Decimal("145.00")

    @pytest.mark.asyncio
    async def test_create_limit_order_without_price_fails(
        self,
        portfolio_service: PortfolioService,
        order_service: OrderService,
        test_user: User,
    ) -> None:
        """Test that limit order without limit price fails."""
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
        )
        portfolio = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        order_data = OrderCreate(
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=10,
        )

        with pytest.raises(ValidationError, match="Limit price is required"):
            await order_service.create_order(portfolio.id, order_data)

    @pytest.mark.asyncio
    async def test_cancel_order(
        self,
        portfolio_service: PortfolioService,
        order_service: OrderService,
        test_user: User,
    ) -> None:
        """Test cancelling an order."""
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
        )
        portfolio = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        order_data = OrderCreate(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=10,
        )
        order = await order_service.create_order(portfolio.id, order_data)

        cancelled = await order_service.cancel_order(order.id)

        assert cancelled.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_get_orders(
        self,
        portfolio_service: PortfolioService,
        order_service: OrderService,
        test_user: User,
    ) -> None:
        """Test getting orders for a portfolio."""
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
        )
        portfolio = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        # Create multiple orders
        for i in range(3):
            order_data = OrderCreate(
                symbol=f"SYMB{i}",
                order_type=OrderType.MARKET,
                side=OrderSide.BUY,
                quantity=10,
            )
            await order_service.create_order(portfolio.id, order_data)

        orders = await order_service.get_orders(portfolio.id)

        assert len(orders) == 3


class TestExecutionEngine:
    """Tests for ExecutionEngine."""

    @pytest.mark.asyncio
    async def test_execute_market_order_buy(
        self,
        portfolio_service: PortfolioService,
        order_service: OrderService,
        execution_engine: ExecutionEngine,
        test_user: User,
        db_session: AsyncSession,
    ) -> None:
        """Test executing a market buy order."""
        # Create portfolio
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
        )
        portfolio = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        # Create order
        order_data = OrderCreate(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=10,
        )
        order = await order_service.create_order(portfolio.id, order_data)

        # Execute order
        trade = await execution_engine.execute_market_order(order, Decimal("150.00"))
        await db_session.commit()

        # Verify trade
        assert trade.symbol == "AAPL"
        assert trade.side == OrderSide.BUY
        assert trade.quantity == 10
        assert trade.price > Decimal("150.00")  # Should include slippage

        # Verify order updated
        await db_session.refresh(order)
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == 10

        # Verify portfolio updated
        await db_session.refresh(portfolio)
        assert portfolio.current_cash < Decimal("100000.00")

        # Verify position created - use explicit query to avoid lazy loading issues
        from sqlalchemy import select
        from signalforge.models.paper_trading import PaperPosition
        result = await db_session.execute(
            select(PaperPosition).where(PaperPosition.portfolio_id == portfolio.id)
        )
        positions = result.scalars().all()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity == 10

    @pytest.mark.asyncio
    async def test_execute_market_order_sell(
        self,
        portfolio_service: PortfolioService,
        order_service: OrderService,
        execution_engine: ExecutionEngine,
        test_user: User,
        db_session: AsyncSession,
    ) -> None:
        """Test executing a market sell order."""
        # Create portfolio
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
        )
        portfolio = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        # Buy first
        buy_order_data = OrderCreate(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=10,
        )
        buy_order = await order_service.create_order(portfolio.id, buy_order_data)
        await execution_engine.execute_market_order(buy_order, Decimal("150.00"))
        await db_session.commit()

        # Get current cash after buy
        await db_session.refresh(portfolio)
        cash_after_buy = portfolio.current_cash

        # Sell
        sell_order_data = OrderCreate(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=5,
        )
        sell_order = await order_service.create_order(portfolio.id, sell_order_data)
        await execution_engine.execute_market_order(sell_order, Decimal("155.00"))
        await db_session.commit()

        # Verify trade
        await db_session.refresh(sell_order)
        assert sell_order.status == OrderStatus.FILLED

        # Verify portfolio updated
        await db_session.refresh(portfolio)
        assert portfolio.current_cash > cash_after_buy

        # Verify position updated - use explicit query to avoid lazy loading issues
        from sqlalchemy import select
        from signalforge.models.paper_trading import PaperPosition
        result = await db_session.execute(
            select(PaperPosition).where(PaperPosition.portfolio_id == portfolio.id)
        )
        positions = result.scalars().all()
        assert len(positions) == 1
        assert positions[0].quantity == 5  # 10 - 5

    @pytest.mark.asyncio
    async def test_slippage_calculation(
        self,
        execution_engine: ExecutionEngine,
    ) -> None:
        """Test slippage calculation."""
        price = Decimal("100.00")
        quantity = 100

        # Buy order should have positive slippage
        buy_slippage = execution_engine.apply_slippage(price, quantity, OrderSide.BUY)
        assert buy_slippage > Decimal("0")

        # Sell order should have negative slippage
        sell_slippage = execution_engine.apply_slippage(price, quantity, OrderSide.SELL)
        assert sell_slippage < Decimal("0")

        # Larger quantity should have more slippage impact
        large_buy_slippage = execution_engine.apply_slippage(price, 1000, OrderSide.BUY)
        assert large_buy_slippage > buy_slippage

    @pytest.mark.asyncio
    async def test_check_limit_orders_buy(
        self,
        portfolio_service: PortfolioService,
        order_service: OrderService,
        execution_engine: ExecutionEngine,
        test_user: User,
        db_session: AsyncSession,
    ) -> None:
        """Test checking and executing limit buy orders."""
        # Create portfolio
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
        )
        portfolio = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        # Create limit buy order
        order_data = OrderCreate(
            symbol="AAPL",
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=10,
            limit_price=Decimal("150.00"),
        )
        order = await order_service.create_order(portfolio.id, order_data)
        await db_session.commit()

        # Check with price above limit (should not execute)
        trades = await execution_engine.check_limit_orders(
            portfolio.id,
            {"AAPL": Decimal("155.00")},
        )
        assert len(trades) == 0

        # Check with price at limit (should execute)
        trades = await execution_engine.check_limit_orders(
            portfolio.id,
            {"AAPL": Decimal("150.00")},
        )
        await db_session.commit()
        assert len(trades) == 1

        # Verify order filled
        await db_session.refresh(order)
        assert order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_check_stop_orders_sell(
        self,
        portfolio_service: PortfolioService,
        order_service: OrderService,
        execution_engine: ExecutionEngine,
        test_user: User,
        db_session: AsyncSession,
    ) -> None:
        """Test checking and executing stop sell orders."""
        # Create portfolio
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
        )
        portfolio = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        # Buy first
        buy_order_data = OrderCreate(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=10,
        )
        buy_order = await order_service.create_order(portfolio.id, buy_order_data)
        await execution_engine.execute_market_order(buy_order, Decimal("150.00"))
        await db_session.commit()

        # Create stop sell order
        stop_order_data = OrderCreate(
            symbol="AAPL",
            order_type=OrderType.STOP,
            side=OrderSide.SELL,
            quantity=10,
            stop_price=Decimal("145.00"),
        )
        stop_order = await order_service.create_order(portfolio.id, stop_order_data)
        await db_session.commit()

        # Check with price above stop (should not execute)
        trades = await execution_engine.check_stop_orders(
            portfolio.id,
            {"AAPL": Decimal("148.00")},
        )
        assert len(trades) == 0

        # Check with price at stop (should execute)
        trades = await execution_engine.check_stop_orders(
            portfolio.id,
            {"AAPL": Decimal("145.00")},
        )
        await db_session.commit()
        assert len(trades) == 1

        # Verify order filled
        await db_session.refresh(stop_order)
        assert stop_order.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_insufficient_cash_rejected(
        self,
        portfolio_service: PortfolioService,
        order_service: OrderService,
        execution_engine: ExecutionEngine,
        test_user: User,
    ) -> None:
        """Test that orders with insufficient cash are rejected."""
        # Create portfolio with low capital
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100.00"),
        )
        portfolio = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        # Create order that exceeds capital
        order_data = OrderCreate(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=100,  # 100 shares at ~$150 = $15,000
        )
        order = await order_service.create_order(portfolio.id, order_data)

        # Attempt to execute should fail
        with pytest.raises(ValidationError, match="Insufficient cash"):
            await execution_engine.execute_market_order(order, Decimal("150.00"))

    @pytest.mark.asyncio
    async def test_insufficient_shares_rejected(
        self,
        portfolio_service: PortfolioService,
        order_service: OrderService,
        execution_engine: ExecutionEngine,
        test_user: User,
        db_session: AsyncSession,
    ) -> None:
        """Test that sell orders with insufficient shares are rejected."""
        # Create portfolio
        portfolio_data = PortfolioCreate(
            name="Test Portfolio",
            initial_capital=Decimal("100000.00"),
        )
        portfolio = await portfolio_service.create_portfolio(test_user.id, portfolio_data)

        # Buy 10 shares
        buy_order_data = OrderCreate(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=10,
        )
        buy_order = await order_service.create_order(portfolio.id, buy_order_data)
        await execution_engine.execute_market_order(buy_order, Decimal("150.00"))
        await db_session.commit()

        # Try to sell 20 shares (should fail)
        sell_order_data = OrderCreate(
            symbol="AAPL",
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=20,
        )
        sell_order = await order_service.create_order(portfolio.id, sell_order_data)

        with pytest.raises(ValidationError, match="Insufficient shares"):
            await execution_engine.execute_market_order(sell_order, Decimal("155.00"))
