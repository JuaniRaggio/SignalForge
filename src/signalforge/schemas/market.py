"""Market data schemas."""

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel


class PriceData(BaseModel):
    """Schema for OHLCV price data."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    adj_close: Decimal | None = None

    model_config = {"from_attributes": True}


class PriceHistoryRequest(BaseModel):
    """Schema for price history request parameters."""

    period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"] = "1mo"
    interval: Literal["1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"] = "1d"


class PriceHistoryResponse(BaseModel):
    """Schema for price history response."""

    symbol: str
    data: list[PriceData]
    count: int


class SymbolInfo(BaseModel):
    """Schema for symbol information."""

    symbol: str
    name: str | None = None
    exchange: str | None = None
    currency: str | None = None
    data_available: bool = False
    first_date: datetime | None = None
    last_date: datetime | None = None


class SymbolListResponse(BaseModel):
    """Schema for symbol list response."""

    symbols: list[SymbolInfo]
    count: int
