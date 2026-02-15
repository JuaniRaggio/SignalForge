"""Integration layer schemas for signal aggregation."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class MarketRegime(str, Enum):
    """Market regime classifications."""

    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRISIS = "crisis"


class SignalDirection(str, Enum):
    """Trading signal directions."""

    STRONG_LONG = "strong_long"
    LONG = "long"
    NEUTRAL = "neutral"
    SHORT = "short"
    STRONG_SHORT = "strong_short"


class IntegratedSignal(BaseModel):
    """Integrated signal combining all data sources."""

    symbol: str
    direction: SignalDirection
    strength: float = Field(..., ge=-1.0, le=1.0, description="Signal strength from -1 to 1")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")

    # Component contributions
    ml_contribution: float = Field(..., description="ML model contribution to signal")
    nlp_contribution: float = Field(..., description="NLP analysis contribution to signal")
    execution_contribution: float = Field(
        ..., description="Execution quality contribution to signal"
    )
    regime_adjustment: float = Field(..., description="Market regime adjustment factor")

    # Component details
    ml_prediction: float | None = Field(
        default=None, description="Raw ML prediction if available"
    )
    nlp_sentiment: float | None = Field(default=None, description="NLP sentiment if available")
    execution_score: float | None = Field(
        default=None, description="Execution quality score if available"
    )
    current_regime: MarketRegime

    # Final recommendation
    recommendation: str = Field(
        ...,
        description="Final recommendation: strong_buy, buy, hold, sell, strong_sell",
    )
    position_size_pct: float = Field(
        ..., ge=0.0, le=1.0, description="Suggested position size as % of portfolio"
    )
    stop_loss_pct: float | None = Field(
        default=None, ge=0.0, description="Suggested stop loss percentage"
    )
    take_profit_pct: float | None = Field(
        default=None, ge=0.0, description="Suggested take profit percentage"
    )

    # Metadata
    explanation: str = Field(..., description="Human-readable explanation of the signal")
    generated_at: datetime
    valid_until: datetime
    warnings: list[str] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class AggregationConfig(BaseModel):
    """Configuration for signal aggregation."""

    ml_weight: float = Field(
        default=0.4, ge=0.0, le=1.0, description="Weight for ML predictions"
    )
    nlp_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for NLP signals"
    )
    execution_weight: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Weight for execution quality"
    )
    regime_weight: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Weight for regime adjustment"
    )
    min_confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for actionable signals",
    )
    regime_multipliers: dict[str, float] = Field(
        default={
            "bull": 1.2,
            "bear": 0.8,
            "sideways": 1.0,
            "volatile": 0.7,
            "crisis": 0.5,
        },
        description="Multipliers applied based on market regime",
    )

    model_config = {"from_attributes": True}
