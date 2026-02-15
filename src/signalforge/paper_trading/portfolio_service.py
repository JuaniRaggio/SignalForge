"""Portfolio service for managing paper trading portfolios."""

from __future__ import annotations

from decimal import Decimal
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from signalforge.core.exceptions import NotFoundError, ValidationError
from signalforge.core.logging import get_logger
from signalforge.models.paper_trading import (
    PaperPortfolio,
    PaperPosition,
    PortfolioStatus,
)
from signalforge.paper_trading.schemas import (
    PortfolioCreate,
    PortfolioSummary,
)

logger = get_logger(__name__)


class PortfolioService:
    """Service for managing paper trading portfolios."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize portfolio service.

        Args:
            session: Database session.
        """
        self.session = session

    async def create_portfolio(
        self,
        user_id: UUID,
        portfolio_data: PortfolioCreate,
    ) -> PaperPortfolio:
        """Create a new paper trading portfolio.

        Args:
            user_id: User ID who owns the portfolio.
            portfolio_data: Portfolio creation data.

        Returns:
            Newly created portfolio.

        Raises:
            ValidationError: If portfolio creation fails validation.
        """
        try:
            portfolio = PaperPortfolio(
                user_id=user_id,
                name=portfolio_data.name,
                initial_capital=portfolio_data.initial_capital,
                current_cash=portfolio_data.initial_capital,
                status=PortfolioStatus.ACTIVE,
                is_competition_portfolio=portfolio_data.is_competition_portfolio,
                competition_id=portfolio_data.competition_id,
            )

            self.session.add(portfolio)
            await self.session.flush()
            await self.session.refresh(portfolio)

            logger.info(
                "portfolio_created",
                portfolio_id=str(portfolio.id),
                user_id=str(user_id),
                name=portfolio_data.name,
                initial_capital=float(portfolio_data.initial_capital),
            )

            return portfolio

        except IntegrityError as e:
            await self.session.rollback()
            logger.error("portfolio_creation_failed", error=str(e), user_id=str(user_id))
            raise ValidationError("Failed to create portfolio") from e

    async def get_portfolio(self, portfolio_id: UUID) -> PaperPortfolio:
        """Get portfolio by ID.

        Args:
            portfolio_id: Portfolio ID.

        Returns:
            Portfolio instance.

        Raises:
            NotFoundError: If portfolio not found.
        """
        result = await self.session.execute(
            select(PaperPortfolio)
            .where(PaperPortfolio.id == portfolio_id)
            .options(
                selectinload(PaperPortfolio.positions),
                selectinload(PaperPortfolio.orders),
            )
        )
        portfolio = result.scalar_one_or_none()

        if portfolio is None:
            logger.warning("portfolio_not_found", portfolio_id=str(portfolio_id))
            raise NotFoundError(f"Portfolio {portfolio_id} not found")

        return portfolio

    async def get_user_portfolios(
        self,
        user_id: UUID,
        status: PortfolioStatus | None = None,
    ) -> list[PaperPortfolio]:
        """Get all portfolios for a user.

        Args:
            user_id: User ID.
            status: Optional filter by portfolio status.

        Returns:
            List of portfolios.
        """
        query = select(PaperPortfolio).where(PaperPortfolio.user_id == user_id)

        if status is not None:
            query = query.where(PaperPortfolio.status == status)

        query = query.order_by(PaperPortfolio.created_at.desc())

        result = await self.session.execute(query)
        portfolios = result.scalars().all()

        logger.debug(
            "user_portfolios_retrieved",
            user_id=str(user_id),
            count=len(portfolios),
            status=status.value if status else None,
        )

        return list(portfolios)

    async def get_portfolio_summary(self, portfolio_id: UUID) -> PortfolioSummary:
        """Get portfolio summary with calculated metrics.

        Args:
            portfolio_id: Portfolio ID.

        Returns:
            Portfolio summary with metrics.

        Raises:
            NotFoundError: If portfolio not found.
        """
        portfolio = await self.get_portfolio(portfolio_id)

        # Calculate positions value
        positions_value = Decimal("0")
        for position in portfolio.positions:
            position_value = position.current_price * position.quantity
            positions_value += position_value

        # Calculate total equity
        equity_value = portfolio.current_cash + positions_value

        # Calculate total P&L
        total_pnl = equity_value - portfolio.initial_capital

        # Calculate P&L percentage
        if portfolio.initial_capital > 0:
            total_pnl_pct = (total_pnl / portfolio.initial_capital) * 100
        else:
            total_pnl_pct = Decimal("0")

        summary = PortfolioSummary(
            id=portfolio.id,
            name=portfolio.name,
            equity_value=equity_value,
            cash_balance=portfolio.current_cash,
            positions_value=positions_value,
            positions_count=len(portfolio.positions),
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            status=portfolio.status,
        )

        logger.debug(
            "portfolio_summary_calculated",
            portfolio_id=str(portfolio_id),
            equity=float(equity_value),
            pnl=float(total_pnl),
            pnl_pct=float(total_pnl_pct),
        )

        return summary

    async def update_position_prices(
        self,
        portfolio_id: UUID,
        prices: dict[str, Decimal],
    ) -> list[PaperPosition]:
        """Update current prices for portfolio positions.

        Args:
            portfolio_id: Portfolio ID.
            prices: Dictionary mapping symbol to current price.

        Returns:
            List of updated positions.

        Raises:
            NotFoundError: If portfolio not found.
        """
        portfolio = await self.get_portfolio(portfolio_id)

        updated_positions: list[PaperPosition] = []

        for position in portfolio.positions:
            if position.symbol in prices:
                new_price = prices[position.symbol]
                position.current_price = new_price

                # Recalculate unrealized P&L
                price_diff = new_price - position.avg_entry_price
                position.unrealized_pnl = price_diff * position.quantity

                # Recalculate unrealized P&L percentage
                if position.avg_entry_price > 0:
                    position.unrealized_pnl_pct = (price_diff / position.avg_entry_price) * 100
                else:
                    position.unrealized_pnl_pct = Decimal("0")

                updated_positions.append(position)

                logger.debug(
                    "position_price_updated",
                    portfolio_id=str(portfolio_id),
                    symbol=position.symbol,
                    new_price=float(new_price),
                    unrealized_pnl=float(position.unrealized_pnl),
                    unrealized_pnl_pct=float(position.unrealized_pnl_pct),
                )

        await self.session.flush()

        logger.info(
            "portfolio_prices_updated",
            portfolio_id=str(portfolio_id),
            positions_updated=len(updated_positions),
        )

        return updated_positions

    async def close_portfolio(self, portfolio_id: UUID) -> PaperPortfolio:
        """Close a portfolio (mark as closed).

        Args:
            portfolio_id: Portfolio ID.

        Returns:
            Updated portfolio.

        Raises:
            NotFoundError: If portfolio not found.
            ValidationError: If portfolio has open positions.
        """
        portfolio = await self.get_portfolio(portfolio_id)

        if len(portfolio.positions) > 0:
            raise ValidationError(
                "Cannot close portfolio with open positions. "
                "Please close all positions first."
            )

        portfolio.status = PortfolioStatus.CLOSED
        await self.session.flush()

        logger.info("portfolio_closed", portfolio_id=str(portfolio_id))

        return portfolio

    async def suspend_portfolio(self, portfolio_id: UUID) -> PaperPortfolio:
        """Suspend a portfolio (temporarily disable trading).

        Args:
            portfolio_id: Portfolio ID.

        Returns:
            Updated portfolio.

        Raises:
            NotFoundError: If portfolio not found.
        """
        portfolio = await self.get_portfolio(portfolio_id)
        portfolio.status = PortfolioStatus.SUSPENDED
        await self.session.flush()

        logger.info("portfolio_suspended", portfolio_id=str(portfolio_id))

        return portfolio

    async def reactivate_portfolio(self, portfolio_id: UUID) -> PaperPortfolio:
        """Reactivate a suspended portfolio.

        Args:
            portfolio_id: Portfolio ID.

        Returns:
            Updated portfolio.

        Raises:
            NotFoundError: If portfolio not found.
            ValidationError: If portfolio is closed.
        """
        portfolio = await self.get_portfolio(portfolio_id)

        if portfolio.status == PortfolioStatus.CLOSED:
            raise ValidationError("Cannot reactivate a closed portfolio")

        portfolio.status = PortfolioStatus.ACTIVE
        await self.session.flush()

        logger.info("portfolio_reactivated", portfolio_id=str(portfolio_id))

        return portfolio
