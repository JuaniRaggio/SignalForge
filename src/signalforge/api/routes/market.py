"""Market data routes."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.api.dependencies.auth import get_current_active_user
from signalforge.api.dependencies.database import get_db
from signalforge.models.price import Price
from signalforge.models.user import User
from signalforge.schemas.market import (
    PriceData,
    PriceHistoryResponse,
    SymbolInfo,
    SymbolListResponse,
)

router = APIRouter()


@router.get("/prices/{symbol}", response_model=PriceHistoryResponse)
async def get_price_history(
    symbol: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
    start_date: datetime | None = Query(None, description="Start date for price data"),
    end_date: datetime | None = Query(None, description="End date for price data"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
) -> PriceHistoryResponse:
    """Get historical price data for a symbol."""
    symbol = symbol.upper()

    query = select(Price).where(Price.symbol == symbol)

    if start_date:
        query = query.where(Price.timestamp >= start_date)
    if end_date:
        query = query.where(Price.timestamp <= end_date)

    query = query.order_by(Price.timestamp.desc()).limit(limit)

    result = await db.execute(query)
    prices = result.scalars().all()

    if not prices:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No price data found for symbol: {symbol}",
        )

    price_data = [
        PriceData(
            symbol=p.symbol,
            timestamp=p.timestamp,
            open=p.open,
            high=p.high,
            low=p.low,
            close=p.close,
            volume=p.volume,
            adj_close=p.adj_close,
        )
        for p in reversed(prices)
    ]

    return PriceHistoryResponse(
        symbol=symbol,
        data=price_data,
        count=len(price_data),
    )


@router.get("/symbols", response_model=SymbolListResponse)
async def get_available_symbols(
    db: Annotated[AsyncSession, Depends(get_db)],
    _current_user: Annotated[User, Depends(get_current_active_user)],
) -> SymbolListResponse:
    """Get list of available symbols with data."""
    query = select(
        Price.symbol,
        func.min(Price.timestamp).label("first_date"),
        func.max(Price.timestamp).label("last_date"),
    ).group_by(Price.symbol)

    result = await db.execute(query)
    rows = result.all()

    symbols = [
        SymbolInfo(
            symbol=row.symbol,
            data_available=True,
            first_date=row.first_date,
            last_date=row.last_date,
        )
        for row in rows
    ]

    return SymbolListResponse(
        symbols=symbols,
        count=len(symbols),
    )
