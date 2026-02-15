"""Execution Quality API schemas."""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class LiquidityMetrics(BaseModel):
    """Liquidity metrics breakdown."""

    avg_volume_1d: int
    avg_volume_5d: int
    avg_volume_20d: int
    avg_dollar_volume_20d: Decimal
    relative_volume: Decimal
    bid_ask_spread_pct: Decimal

    model_config = {"from_attributes": True}


class LiquidityScoreResponse(BaseModel):
    """Response schema for liquidity score."""

    symbol: str
    timestamp: datetime
    liquidity_score: float = Field(..., ge=0.0, le=100.0, description="Score from 0-100")
    liquidity_tier: str  # 'high', 'medium', 'low', 'very_low'
    metrics: LiquidityMetrics
    recommendation: str
    warnings: list[str] | None = None

    model_config = {"from_attributes": True}


class SlippageRequest(BaseModel):
    """Request schema for slippage estimation."""

    symbol: str
    order_size: int = Field(..., gt=0)
    side: str = Field(..., pattern="^(buy|sell)$")
    urgency: str = Field(
        default="normal",
        pattern="^(low|normal|high)$",
        description="Trade urgency level",
    )

    model_config = {"from_attributes": True}


class SlippageComponents(BaseModel):
    """Breakdown of slippage components."""

    market_impact_bps: Decimal
    spread_cost_bps: Decimal
    timing_risk_bps: Decimal
    total_slippage_bps: Decimal

    model_config = {"from_attributes": True}


class SlippageResponse(BaseModel):
    """Response schema for slippage estimation."""

    symbol: str
    order_size: int
    side: str
    current_price: Decimal
    estimated_execution_price: Decimal
    slippage: SlippageComponents
    estimated_cost: Decimal
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime

    model_config = {"from_attributes": True}


class SpreadMetrics(BaseModel):
    """Spread metrics for a symbol."""

    current_bid: Decimal
    current_ask: Decimal
    spread_absolute: Decimal
    spread_bps: Decimal
    avg_spread_1h: Decimal
    avg_spread_1d: Decimal
    spread_percentile: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Current spread vs historical (0-100)",
    )

    model_config = {"from_attributes": True}


class SpreadMetricsResponse(BaseModel):
    """Response schema for spread metrics."""

    symbol: str
    timestamp: datetime
    metrics: SpreadMetrics
    is_favorable: bool
    recommendation: str

    model_config = {"from_attributes": True}


class VolumeFilterRequest(BaseModel):
    """Request schema for volume filter check."""

    symbol: str
    order_size: int = Field(..., gt=0)
    max_volume_participation: Decimal = Field(
        default=Decimal("0.05"),
        gt=0,
        le=1.0,
        description="Maximum participation rate (0-1)",
    )

    model_config = {"from_attributes": True}


class VolumeAnalysis(BaseModel):
    """Volume analysis details."""

    current_volume: int
    avg_volume_20d: int
    order_as_pct_avg_volume: Decimal
    estimated_time_to_fill_minutes: int
    volume_profile: str  # 'high', 'normal', 'low'

    model_config = {"from_attributes": True}


class VolumeFilterResponse(BaseModel):
    """Response schema for volume filter check."""

    symbol: str
    order_size: int
    passes_filter: bool
    analysis: VolumeAnalysis
    warnings: list[str]
    recommendation: str
    timestamp: datetime

    model_config = {"from_attributes": True}


class ExecutionQualityMetrics(BaseModel):
    """Comprehensive execution quality metrics."""

    liquidity_score: float
    estimated_slippage_bps: Decimal
    spread_bps: Decimal
    volume_participation_pct: Decimal
    execution_difficulty: str  # 'easy', 'moderate', 'difficult', 'very_difficult'
    overall_score: float = Field(..., ge=0.0, le=100.0)


class ExecutionWarning(BaseModel):
    """Execution warning detail."""

    severity: str  # 'low', 'medium', 'high'
    category: str  # 'liquidity', 'slippage', 'spread', 'volume'
    message: str


class ExecutionQualityResponse(BaseModel):
    """Response schema for comprehensive execution quality assessment."""

    symbol: str
    order_size: int
    side: str
    timestamp: datetime
    current_price: Decimal
    metrics: ExecutionQualityMetrics
    liquidity: LiquidityScoreResponse
    slippage: SlippageResponse
    spread: SpreadMetricsResponse
    volume: VolumeFilterResponse
    overall_recommendation: str
    warnings: list[ExecutionWarning]
    is_tradeable: bool

    model_config = {"from_attributes": True}
