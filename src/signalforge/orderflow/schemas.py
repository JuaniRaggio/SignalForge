"""Pydantic schemas for order flow module."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from signalforge.models.orderflow import FlowDirection, FlowType


class DarkPoolPrint(BaseModel):
    """Individual dark pool trade record."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str = Field(..., max_length=20)
    timestamp: datetime
    shares: int = Field(..., gt=0)
    price: float = Field(..., gt=0)
    value: float = Field(..., gt=0)
    venue: str = Field(..., max_length=100)
    is_large: bool = False
    z_score: float | None = None


class DarkPoolSummary(BaseModel):
    """Aggregated dark pool statistics for a symbol."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str = Field(..., max_length=20)
    period_days: int = Field(..., gt=0)
    total_volume: int = Field(..., ge=0)
    total_value: float = Field(..., ge=0)
    trade_count: int = Field(..., ge=0)
    avg_trade_size: float = Field(..., ge=0)
    dark_pool_ratio: float = Field(..., ge=0, le=1)
    largest_print: DarkPoolPrint | None = None
    institutional_bias: FlowDirection


class OptionsActivityRecord(BaseModel):
    """Options activity record."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str = Field(..., max_length=20)
    timestamp: datetime
    option_type: str = Field(..., pattern="^(call|put)$")
    strike: float = Field(..., gt=0)
    expiry: datetime
    volume: int = Field(..., ge=0)
    open_interest: int = Field(..., ge=0)
    premium: float = Field(..., ge=0)
    implied_volatility: float | None = Field(None, ge=0, le=5)
    delta: float | None = Field(None, ge=-1, le=1)
    is_unusual: bool = False


class UnusualOptionsActivity(BaseModel):
    """Unusual options activity detection."""

    model_config = ConfigDict(from_attributes=True)

    record: OptionsActivityRecord
    volume_ratio: float = Field(..., gt=0)
    oi_ratio: float = Field(..., gt=0)
    premium_percentile: float = Field(..., ge=0, le=100)
    z_score: float
    reason: str = Field(..., min_length=1)


class ShortInterestRecord(BaseModel):
    """Short interest data for a symbol."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str = Field(..., max_length=20)
    report_date: datetime
    short_interest: int = Field(..., ge=0)
    shares_outstanding: int = Field(..., gt=0)
    short_percent: float = Field(..., ge=0, le=100)
    days_to_cover: float | None = Field(None, ge=0)
    change_percent: float | None = None


class ShortInterestChange(BaseModel):
    """Short interest change analysis."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str = Field(..., max_length=20)
    current: ShortInterestRecord
    previous: ShortInterestRecord | None = None
    change_percent: float
    change_shares: int
    is_increasing: bool
    is_significant: bool


class FlowAggregation(BaseModel):
    """Aggregated flow data for a symbol."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str = Field(..., max_length=20)
    period_days: int = Field(..., gt=0)
    net_flow: float
    bullish_flow: float = Field(..., ge=0)
    bearish_flow: float = Field(..., ge=0)
    bias: FlowDirection
    z_score: float
    flow_momentum: float
    dark_pool_volume: int = Field(..., ge=0)
    options_premium: float = Field(..., ge=0)
    short_interest_change: float


class AnomalySeverity(str, Enum):
    """Severity level of detected anomaly."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FlowAnomaly(BaseModel):
    """Detected flow anomaly."""

    model_config = ConfigDict(from_attributes=True)

    symbol: str = Field(..., max_length=20)
    timestamp: datetime
    anomaly_type: str = Field(..., min_length=1)
    severity: AnomalySeverity
    description: str = Field(..., min_length=1)
    z_score: float
    metadata: dict[str, float | str | int] | None = None


class OrderFlowQuery(BaseModel):
    """Query filters for order flow data."""

    model_config = ConfigDict(from_attributes=True)

    symbols: list[str] | None = None
    flow_types: list[FlowType] | None = None
    direction: FlowDirection | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None
    min_value: float | None = Field(None, gt=0)
    min_z_score: float | None = None
    only_unusual: bool = False
    limit: int = Field(100, gt=0, le=10000)
    offset: int = Field(0, ge=0)
