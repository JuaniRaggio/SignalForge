"""Schema definitions for sector-specific signals."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SignalStrength(str, Enum):
    """Signal strength enumeration."""

    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class SectorSignal(BaseModel):
    """Base sector signal model."""

    sector: str = Field(..., description="Sector name")
    signal_type: str = Field(..., description="Type of signal extracted")
    strength: SignalStrength = Field(..., description="Signal strength")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    description: str = Field(..., description="Human-readable description")
    affected_symbols: list[str] = Field(default_factory=list, description="Affected stock symbols")
    source_text: str = Field(..., description="Source text of the signal")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Signal timestamp")


class TechnologySignal(SectorSignal):
    """Technology sector-specific signal."""

    product_cycle_phase: str | None = Field(None, description="Product cycle phase")
    ai_relevance: float | None = Field(None, ge=0.0, le=1.0, description="AI relevance score")
    chip_demand_indicator: str | None = Field(None, description="Chip demand indicator")


class HealthcareSignal(SectorSignal):
    """Healthcare sector-specific signal."""

    drug_phase: str | None = Field(None, description="Drug development phase")
    fda_action: str | None = Field(None, description="FDA action type")
    patent_status: str | None = Field(None, description="Patent status")


class EnergySignal(SectorSignal):
    """Energy sector-specific signal."""

    commodity: str | None = Field(None, description="Commodity type")
    price_sensitivity: float | None = Field(None, ge=0.0, le=1.0, description="Price sensitivity")
    esg_score_impact: float | None = Field(None, ge=-1.0, le=1.0, description="ESG score impact")


class FinancialSignal(SectorSignal):
    """Financial sector-specific signal."""

    interest_rate_sensitivity: float | None = Field(
        None, ge=0.0, le=1.0, description="Interest rate sensitivity"
    )
    credit_quality_indicator: str | None = Field(None, description="Credit quality indicator")
    regulatory_impact: str | None = Field(None, description="Regulatory impact")


class ConsumerSignal(SectorSignal):
    """Consumer sector-specific signal."""

    consumer_sentiment: float | None = Field(
        None, ge=-1.0, le=1.0, description="Consumer sentiment score"
    )
    seasonal_factor: str | None = Field(None, description="Seasonal factor")
    brand_momentum: str | None = Field(None, description="Brand momentum indicator")
