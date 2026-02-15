"""Paper trading API routes."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.api.dependencies.auth import get_current_active_user
from signalforge.api.dependencies.database import get_db
from signalforge.core.exceptions import NotFoundError, ValidationError
from signalforge.core.logging import get_logger
from signalforge.models.paper_trading import OrderStatus, PortfolioStatus
from signalforge.models.user import User
from signalforge.paper_trading.order_service import OrderService
from signalforge.paper_trading.performance_schemas import (
    EquityCurveResponse,
    PerformanceMetricsResponse,
    ReturnsSeriesResponse,
)
from signalforge.paper_trading.performance_service import PerformanceService
from signalforge.paper_trading.portfolio_service import PortfolioService
from signalforge.paper_trading.schemas import (
    OrderCreate,
    OrderResponse,
    PortfolioCreate,
    PortfolioResponse,
    PortfolioSummary,
    PositionResponse,
    PriceUpdate,
    TradeResponse,
)
from signalforge.paper_trading.snapshot_service import SnapshotService

router = APIRouter()
logger = get_logger(__name__)


# Portfolio Endpoints
@router.post(
    "/portfolios",
    response_model=PortfolioResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_portfolio(
    portfolio_data: PortfolioCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PortfolioResponse:
    """Create a new paper trading portfolio."""
    service = PortfolioService(db)
    portfolio = await service.create_portfolio(current_user.id, portfolio_data)
    return PortfolioResponse.model_validate(portfolio)


@router.get("/portfolios", response_model=list[PortfolioResponse])
async def list_portfolios(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    status_filter: PortfolioStatus | None = Query(default=None, alias="status"),
) -> list[PortfolioResponse]:
    """List all portfolios for the current user."""
    service = PortfolioService(db)
    portfolios = await service.get_user_portfolios(current_user.id, status_filter)
    return [PortfolioResponse.model_validate(p) for p in portfolios]


@router.get("/portfolios/{portfolio_id}", response_model=PortfolioResponse)
async def get_portfolio(
    portfolio_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PortfolioResponse:
    """Get portfolio details by ID."""
    service = PortfolioService(db)
    try:
        portfolio = await service.get_portfolio(portfolio_id)

        # Verify ownership
        if portfolio.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this portfolio",
            )

        return PortfolioResponse.model_validate(portfolio)

    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.get("/portfolios/{portfolio_id}/summary", response_model=PortfolioSummary)
async def get_portfolio_summary(
    portfolio_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PortfolioSummary:
    """Get portfolio summary with calculated metrics."""
    service = PortfolioService(db)
    try:
        portfolio = await service.get_portfolio(portfolio_id)

        # Verify ownership
        if portfolio.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this portfolio",
            )

        summary = await service.get_portfolio_summary(portfolio_id)
        return summary

    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.post("/portfolios/{portfolio_id}/close", response_model=PortfolioResponse)
async def close_portfolio(
    portfolio_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PortfolioResponse:
    """Close a portfolio (must have no open positions)."""
    service = PortfolioService(db)
    try:
        portfolio = await service.get_portfolio(portfolio_id)

        # Verify ownership
        if portfolio.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this portfolio",
            )

        portfolio = await service.close_portfolio(portfolio_id)
        return PortfolioResponse.model_validate(portfolio)

    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


# Order Endpoints
@router.post("/portfolios/{portfolio_id}/orders", response_model=OrderResponse, status_code=status.HTTP_201_CREATED)
async def place_order(
    portfolio_id: UUID,
    order_data: OrderCreate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> OrderResponse:
    """Place a new order in a portfolio."""
    # Verify portfolio ownership
    portfolio_service = PortfolioService(db)
    try:
        portfolio = await portfolio_service.get_portfolio(portfolio_id)

        if portfolio.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this portfolio",
            )

        # Create order
        order_service = OrderService(db)
        order = await order_service.create_order(portfolio_id, order_data)
        return OrderResponse.model_validate(order)

    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


@router.get("/portfolios/{portfolio_id}/orders", response_model=list[OrderResponse])
async def list_orders(
    portfolio_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    status_filter: OrderStatus | None = Query(default=None, alias="status"),
    symbol: str | None = Query(default=None),
) -> list[OrderResponse]:
    """List orders for a portfolio."""
    # Verify portfolio ownership
    portfolio_service = PortfolioService(db)
    try:
        portfolio = await portfolio_service.get_portfolio(portfolio_id)

        if portfolio.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this portfolio",
            )

        # Get orders
        order_service = OrderService(db)
        orders = await order_service.get_orders(portfolio_id, status_filter, symbol)
        return [OrderResponse.model_validate(o) for o in orders]

    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.delete("/orders/{order_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_order(
    order_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> None:
    """Cancel a pending order."""
    order_service = OrderService(db)
    try:
        order = await order_service.get_order(order_id)

        # Verify ownership through portfolio
        portfolio_service = PortfolioService(db)
        portfolio = await portfolio_service.get_portfolio(order.portfolio_id)

        if portfolio.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this order",
            )

        await order_service.cancel_order(order_id)

    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e


# Position Endpoints
@router.get("/portfolios/{portfolio_id}/positions", response_model=list[PositionResponse])
async def list_positions(
    portfolio_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[PositionResponse]:
    """List all positions in a portfolio."""
    portfolio_service = PortfolioService(db)
    try:
        portfolio = await portfolio_service.get_portfolio(portfolio_id)

        if portfolio.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this portfolio",
            )

        # Convert positions to response format
        positions = []
        for position in portfolio.positions:
            market_value = position.current_price * position.quantity
            positions.append(
                PositionResponse(
                    id=position.id,
                    portfolio_id=position.portfolio_id,
                    symbol=position.symbol,
                    quantity=position.quantity,
                    avg_entry_price=position.avg_entry_price,
                    current_price=position.current_price,
                    unrealized_pnl=position.unrealized_pnl,
                    unrealized_pnl_pct=position.unrealized_pnl_pct,
                    market_value=market_value,
                    created_at=position.created_at,
                    updated_at=position.updated_at,
                )
            )

        return positions

    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


# Trade Endpoints
@router.get("/portfolios/{portfolio_id}/trades", response_model=list[TradeResponse])
async def list_trades(
    portfolio_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=100, ge=1, le=500),
) -> list[TradeResponse]:
    """List trade history for a portfolio."""
    from sqlalchemy import select

    from signalforge.models.paper_trading import PaperTrade

    portfolio_service = PortfolioService(db)
    try:
        portfolio = await portfolio_service.get_portfolio(portfolio_id)

        if portfolio.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this portfolio",
            )

        # Get trades
        result = await db.execute(
            select(PaperTrade)
            .where(PaperTrade.portfolio_id == portfolio_id)
            .order_by(PaperTrade.executed_at.desc())
            .limit(limit)
        )
        trades = result.scalars().all()

        return [TradeResponse.model_validate(t) for t in trades]

    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


# Price Update Endpoint (for testing/admin)
@router.post("/portfolios/{portfolio_id}/update-prices", response_model=list[PositionResponse])
async def update_position_prices(
    portfolio_id: UUID,
    price_data: PriceUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[PositionResponse]:
    """Update current prices for portfolio positions."""
    portfolio_service = PortfolioService(db)
    try:
        portfolio = await portfolio_service.get_portfolio(portfolio_id)

        if portfolio.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this portfolio",
            )

        updated_positions = await portfolio_service.update_position_prices(
            portfolio_id, price_data.prices
        )

        positions = []
        for position in updated_positions:
            market_value = position.current_price * position.quantity
            positions.append(
                PositionResponse(
                    id=position.id,
                    portfolio_id=position.portfolio_id,
                    symbol=position.symbol,
                    quantity=position.quantity,
                    avg_entry_price=position.avg_entry_price,
                    current_price=position.current_price,
                    unrealized_pnl=position.unrealized_pnl,
                    unrealized_pnl_pct=position.unrealized_pnl_pct,
                    market_value=market_value,
                    created_at=position.created_at,
                    updated_at=position.updated_at,
                )
            )

        return positions

    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


# Performance Metrics Endpoints
@router.get("/portfolios/{portfolio_id}/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    portfolio_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    start_date: str | None = Query(default=None, description="Start date (ISO format)"),
    end_date: str | None = Query(default=None, description="End date (ISO format)"),
) -> PerformanceMetricsResponse:
    """Get comprehensive performance metrics for a portfolio."""
    from datetime import datetime

    portfolio_service = PortfolioService(db)
    try:
        portfolio = await portfolio_service.get_portfolio(portfolio_id)

        if portfolio.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this portfolio",
            )

        # Parse dates if provided
        start_dt = datetime.fromisoformat(start_date) if start_date else None
        end_dt = datetime.fromisoformat(end_date) if end_date else None

        # Calculate metrics
        perf_service = PerformanceService()
        metrics = await perf_service.calculate_metrics(portfolio_id, db, start_dt, end_dt)

        return PerformanceMetricsResponse(
            portfolio_id=metrics.portfolio_id,
            total_return_pct=metrics.total_return_pct,
            annualized_return_pct=metrics.annualized_return_pct,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            max_drawdown_pct=metrics.max_drawdown_pct,
            max_drawdown_duration_days=metrics.max_drawdown_duration_days,
            win_rate=metrics.win_rate,
            profit_factor=metrics.profit_factor,
            total_trades=metrics.total_trades,
            winning_trades=metrics.winning_trades,
            losing_trades=metrics.losing_trades,
            avg_win_pct=metrics.avg_win_pct,
            avg_loss_pct=metrics.avg_loss_pct,
            best_trade_pct=metrics.best_trade_pct,
            worst_trade_pct=metrics.worst_trade_pct,
            avg_holding_period_days=metrics.avg_holding_period_days,
            volatility_annualized=metrics.volatility_annualized,
            calmar_ratio=metrics.calmar_ratio,
        )

    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.get("/portfolios/{portfolio_id}/equity-curve", response_model=EquityCurveResponse)
async def get_equity_curve(
    portfolio_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    days: int = Query(default=30, ge=1, le=365, description="Number of days to retrieve"),
) -> EquityCurveResponse:
    """Get equity curve data for a portfolio."""
    from decimal import Decimal

    from signalforge.paper_trading.performance_schemas import EquityCurvePoint

    portfolio_service = PortfolioService(db)
    try:
        portfolio = await portfolio_service.get_portfolio(portfolio_id)

        if portfolio.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this portfolio",
            )

        # Get equity curve data
        snapshot_service = SnapshotService()
        equity_df = await snapshot_service.get_equity_curve(portfolio_id, db, days)

        # Convert Polars DataFrame to response format
        data_points = []
        if not equity_df.is_empty():
            for row in equity_df.iter_rows(named=True):
                data_points.append(
                    EquityCurvePoint(
                        date=row["date"],
                        equity=Decimal(str(row["equity"])),
                        cash_balance=Decimal(str(row["cash_balance"])),
                        positions_value=Decimal(str(row["positions_value"])),
                        positions_count=row["positions_count"],
                        daily_return_pct=(
                            Decimal(str(row["daily_return_pct"]))
                            if row["daily_return_pct"] is not None
                            else None
                        ),
                        cumulative_return_pct=(
                            Decimal(str(row["cumulative_return_pct"]))
                            if row["cumulative_return_pct"] is not None
                            else None
                        ),
                    )
                )

        start_date = data_points[0].date if data_points else None
        end_date = data_points[-1].date if data_points else None

        return EquityCurveResponse(
            portfolio_id=portfolio_id,
            start_date=start_date,
            end_date=end_date,
            data=data_points,
        )

    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


@router.get("/portfolios/{portfolio_id}/returns", response_model=ReturnsSeriesResponse)
async def get_returns_series(
    portfolio_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    days: int = Query(default=30, ge=1, le=365, description="Number of days to retrieve"),
) -> ReturnsSeriesResponse:
    """Get daily returns series for a portfolio."""
    from decimal import Decimal

    portfolio_service = PortfolioService(db)
    try:
        portfolio = await portfolio_service.get_portfolio(portfolio_id)

        if portfolio.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this portfolio",
            )

        # Get returns data
        snapshot_service = SnapshotService()
        returns_series = await snapshot_service.get_returns_series(portfolio_id, db, days)

        # Convert Polars Series to list
        returns_list = [Decimal(str(r)) for r in returns_series.to_list()]

        # Calculate statistics
        mean_return = Decimal(str(returns_series.mean())) if len(returns_series) > 0 else Decimal("0")
        std_return = Decimal(str(returns_series.std())) if len(returns_series) > 1 else Decimal("0")

        # Get date range from snapshots
        equity_df = await snapshot_service.get_equity_curve(portfolio_id, db, days)
        start_date = None
        end_date = None

        if not equity_df.is_empty():
            dates = equity_df["date"].to_list()
            start_date = dates[0] if dates else None
            end_date = dates[-1] if dates else None

        return ReturnsSeriesResponse(
            portfolio_id=portfolio_id,
            start_date=start_date,
            end_date=end_date,
            returns=returns_list,
            mean_return=mean_return,
            std_return=std_return,
        )

    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e
