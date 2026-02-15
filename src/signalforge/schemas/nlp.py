"""NLP API schemas."""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class PriceTargetSchema(BaseModel):
    """Price target schema."""

    symbol: str
    target_price: Decimal
    current_price: Decimal | None = None
    upside_percent: Decimal | None = None
    action: str  # upgrade, downgrade, maintain, initiate, reiterate
    rating: str | None = None  # strong_buy, buy, hold, sell, strong_sell
    analyst: str | None = None
    firm: str | None = None
    date: datetime
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_text: str

    model_config = {"from_attributes": True}


class SentimentSchema(BaseModel):
    """Sentiment analysis schema."""

    score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score from -1 to 1")
    label: str  # bullish, bearish, neutral
    delta_vs_baseline: float = Field(..., description="Change vs historical baseline")
    is_new_information: bool

    model_config = {"from_attributes": True}


class AnalystConsensusSchema(BaseModel):
    """Analyst consensus schema."""

    rating: str  # strong_buy, buy, hold, sell, strong_sell
    confidence: float = Field(..., ge=0.0, le=1.0)
    bullish_count: int = Field(..., ge=0)
    bearish_count: int = Field(..., ge=0)
    neutral_count: int = Field(..., ge=0)

    model_config = {"from_attributes": True}


class SectorSignalSchema(BaseModel):
    """Sector-specific signal schema."""

    sector: str
    signal: str  # buy, sell, hold
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

    model_config = {"from_attributes": True}


class AnalyzeDocumentRequest(BaseModel):
    """Request schema for document analysis."""

    document_id: str | None = Field(
        default=None,
        description="ID of existing document to analyze",
    )
    text: str | None = Field(
        default=None,
        min_length=10,
        description="Raw text to analyze (if no document_id)",
    )
    title: str | None = Field(default=None, description="Document title")
    symbols: list[str] | None = Field(
        default=None,
        max_length=10,
        description="Related stock symbols",
    )
    include_price_targets: bool = Field(
        default=True,
        description="Whether to extract price targets",
    )
    include_sector_signals: bool = Field(
        default=True,
        description="Whether to generate sector-specific signals",
    )

    model_config = {"from_attributes": True}


class AnalyzeDocumentResponse(BaseModel):
    """Response schema for document analysis."""

    ticker: str
    sentiment: SentimentSchema
    price_targets: list[PriceTargetSchema]
    analyst_consensus: AnalystConsensusSchema
    sector_signals: list[SectorSignalSchema]
    urgency: str  # low, medium, high, critical
    generated_at: datetime
    processing_time_ms: int

    model_config = {"from_attributes": True}


class NLPSignalsResponse(BaseModel):
    """Response schema for NLP signals query."""

    symbol: str
    signals: list[AnalyzeDocumentResponse]
    count: int
    period_days: int
    summary: dict[str, float | int | str]

    model_config = {"from_attributes": True}


class SectorReportResponse(BaseModel):
    """Response schema for sector intelligence report."""

    sector: str
    timestamp: datetime
    overall_sentiment: SentimentSchema
    top_symbols: list[dict[str, str | float]]  # symbol, sentiment_score, signal
    analyst_consensus: AnalystConsensusSchema
    key_themes: list[str]
    average_price_target_upside: Decimal | None = None
    document_count: int
    high_urgency_count: int

    model_config = {"from_attributes": True}


class AnalystConsensusResponse(BaseModel):
    """Response schema for analyst consensus query."""

    symbol: str
    timestamp: datetime
    consensus: AnalystConsensusSchema
    price_targets: list[PriceTargetSchema]
    mean_target: Decimal | None = None
    median_target: Decimal | None = None
    high_target: Decimal | None = None
    low_target: Decimal | None = None
    current_price: Decimal | None = None
    implied_upside: Decimal | None = None
    num_analysts: int

    model_config = {"from_attributes": True}


class AggregateSignalsRequest(BaseModel):
    """Request schema for signal aggregation."""

    symbol: str
    include_ml_prediction: bool = Field(
        default=True,
        description="Include ML model prediction",
    )
    include_nlp_signals: bool = Field(
        default=True,
        description="Include NLP signals",
    )
    include_execution_quality: bool = Field(
        default=True,
        description="Include execution quality assessment",
    )
    ml_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for ML predictions",
    )
    nlp_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for NLP signals",
    )
    market_regime: str | None = Field(
        default=None,
        pattern="^(bull|bear|sideways|volatile)$",
        description="Current market regime for conditioning",
    )

    model_config = {"from_attributes": True}


class AggregatedSignalSchema(BaseModel):
    """Aggregated signal schema."""

    symbol: str
    direction: str  # long, short, neutral
    strength: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    ml_contribution: float
    nlp_contribution: float
    execution_feasibility: float = Field(..., ge=0.0, le=1.0)
    recommendation: str
    explanation: str

    model_config = {"from_attributes": True}


class AggregatedSignalResponse(BaseModel):
    """Response schema for aggregated signals."""

    timestamp: datetime
    signal: AggregatedSignalSchema
    ml_prediction: dict[str, str | float | int] | None = None
    nlp_signals: AnalyzeDocumentResponse | None = None
    execution_quality: dict[str, str | float | int] | None = None
    market_regime: str | None = None

    model_config = {"from_attributes": True}
