"""Pydantic schemas for execution quality assessment.

This module defines the data models used throughout the execution quality
module for liquidity scoring, slippage estimation, spread metrics, and
volume filtering.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LiquidityScore(BaseModel):
    """Liquidity score for a trading symbol.

    Attributes:
        symbol: Trading symbol being evaluated.
        timestamp: Time of assessment.
        score: Overall liquidity score (0-100).
        volume_score: Score based on average daily volume.
        spread_score: Score based on bid-ask spread.
        market_cap_score: Score based on market capitalization.
        adv_20: Average Daily Volume over 20 days.
        rating: Categorical rating of liquidity quality.
    """

    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Assessment timestamp")
    score: float = Field(..., ge=0, le=100, description="Overall liquidity score (0-100)")
    volume_score: float = Field(..., ge=0, le=100, description="Volume-based score")
    spread_score: float = Field(..., ge=0, le=100, description="Spread-based score")
    market_cap_score: float = Field(..., ge=0, le=100, description="Market cap score")
    adv_20: float = Field(..., ge=0, description="Average Daily Volume (20-day)")
    rating: Literal["excellent", "good", "fair", "poor", "illiquid"] = Field(
        ..., description="Liquidity rating"
    )

    @field_validator("score", "volume_score", "spread_score", "market_cap_score")
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure scores are within valid range."""
        if not 0 <= v <= 100:
            raise ValueError("Score must be between 0 and 100")
        return v

    @field_validator("adv_20")
    @classmethod
    def validate_adv(cls, v: float) -> float:
        """Ensure ADV is non-negative."""
        if v < 0:
            raise ValueError("Average Daily Volume cannot be negative")
        return v


class SlippageEstimate(BaseModel):
    """Estimated slippage for a trade order.

    Attributes:
        symbol: Trading symbol.
        order_size: Order size in dollars.
        estimated_slippage_bps: Estimated slippage in basis points.
        estimated_slippage_pct: Estimated slippage as percentage.
        confidence: Confidence level of estimate (0-1).
        market_impact_cost: Estimated market impact cost in dollars.
    """

    symbol: str = Field(..., description="Trading symbol")
    order_size: float = Field(..., ge=0, description="Order size in dollars")
    estimated_slippage_bps: float = Field(..., ge=0, description="Slippage in basis points")
    estimated_slippage_pct: float = Field(..., ge=0, description="Slippage percentage")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level (0-1)")
    market_impact_cost: float = Field(..., ge=0, description="Market impact cost in dollars")

    @field_validator("estimated_slippage_bps", "estimated_slippage_pct", "market_impact_cost")
    @classmethod
    def validate_non_negative(cls, v: float) -> float:
        """Ensure values are non-negative."""
        if v < 0:
            raise ValueError("Value cannot be negative")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        return v


class SpreadMetrics(BaseModel):
    """Bid-ask spread metrics for a symbol.

    Attributes:
        symbol: Trading symbol.
        current_spread_bps: Current spread in basis points.
        avg_spread_20d_bps: 20-day average spread in basis points.
        spread_volatility: Standard deviation of spread.
        percentile_rank: Historical percentile rank (0-100).
    """

    symbol: str = Field(..., description="Trading symbol")
    current_spread_bps: float = Field(..., ge=0, description="Current spread (bps)")
    avg_spread_20d_bps: float = Field(..., ge=0, description="20-day average spread (bps)")
    spread_volatility: float = Field(..., ge=0, description="Spread standard deviation")
    percentile_rank: float = Field(..., ge=0, le=100, description="Historical percentile (0-100)")

    @field_validator("current_spread_bps", "avg_spread_20d_bps", "spread_volatility")
    @classmethod
    def validate_non_negative(cls, v: float) -> float:
        """Ensure spread values are non-negative."""
        if v < 0:
            raise ValueError("Spread values cannot be negative")
        return v

    @field_validator("percentile_rank")
    @classmethod
    def validate_percentile(cls, v: float) -> float:
        """Ensure percentile is between 0 and 100."""
        if not 0 <= v <= 100:
            raise ValueError("Percentile rank must be between 0 and 100")
        return v


class VolumeFilterResult(BaseModel):
    """Result of volume filtering assessment.

    Attributes:
        symbol: Trading symbol.
        passes_filter: Whether the symbol passes volume filters.
        position_pct_of_adv: Position size as percentage of ADV.
        max_position_size: Maximum recommended position size in dollars.
        reason: Reason for failure (None if passes).
    """

    symbol: str = Field(..., description="Trading symbol")
    passes_filter: bool = Field(..., description="Whether filter passed")
    position_pct_of_adv: float = Field(..., ge=0, description="Position as % of ADV")
    max_position_size: float = Field(..., ge=0, description="Max position size (dollars)")
    reason: str | None = Field(None, description="Failure reason if applicable")

    @field_validator("position_pct_of_adv", "max_position_size")
    @classmethod
    def validate_non_negative(cls, v: float) -> float:
        """Ensure values are non-negative."""
        if v < 0:
            raise ValueError("Value cannot be negative")
        return v
