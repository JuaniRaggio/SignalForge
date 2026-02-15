"""Machine Learning API schemas."""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for single prediction."""

    symbol: str = Field(..., description="Stock symbol to predict")
    horizon: int = Field(default=1, ge=1, le=30, description="Prediction horizon in days")
    model_id: str | None = Field(
        default=None,
        description="Specific model to use (defaults to best model)",
    )
    include_features: bool = Field(
        default=False,
        description="Include feature values in response",
    )

    model_config = {"from_attributes": True}


class FeatureValues(BaseModel):
    """Feature values used for prediction."""

    feature_name: str
    value: float
    importance: float | None = None

    model_config = {"from_attributes": True}


class PredictionResult(BaseModel):
    """Individual prediction result."""

    horizon: int
    predicted_price: Decimal
    predicted_direction: str  # 'up', 'down', 'neutral'
    confidence: float = Field(..., ge=0.0, le=1.0)
    lower_bound: Decimal | None = None
    upper_bound: Decimal | None = None


class PredictResponse(BaseModel):
    """Response schema for single prediction."""

    symbol: str
    current_price: Decimal
    timestamp: datetime
    predictions: list[PredictionResult]
    model_id: str
    model_name: str
    features: list[FeatureValues] | None = None

    model_config = {"from_attributes": True}


class BatchPredictRequest(BaseModel):
    """Request schema for batch predictions."""

    symbols: list[str] = Field(..., min_length=1, max_length=100)
    horizon: int = Field(default=1, ge=1, le=30)
    model_id: str | None = None

    model_config = {"from_attributes": True}


class ModelInfo(BaseModel):
    """Model information."""

    model_id: str
    name: str
    model_type: str  # 'lstm', 'transformer', 'ensemble', etc.
    version: str
    created_at: datetime
    metrics: dict[str, float]
    is_active: bool
    description: str | None = None

    model_config = {"from_attributes": True}


class ModelListResponse(BaseModel):
    """Response for model listing."""

    models: list[ModelInfo]
    count: int
    active_model_id: str | None = None

    model_config = {"from_attributes": True}


class ModelInfoResponse(BaseModel):
    """Detailed model information response."""

    model_id: str
    name: str
    model_type: str
    version: str
    created_at: datetime
    updated_at: datetime | None = None
    metrics: dict[str, float]
    hyperparameters: dict[str, float | str | int | bool]
    is_active: bool
    training_period: dict[str, datetime | None]
    feature_count: int
    description: str | None = None

    model_config = {"from_attributes": True}


class BacktestRequest(BaseModel):
    """Request schema for backtesting."""

    symbol: str
    model_id: str
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal = Field(default=Decimal("100000.0"), gt=0)
    position_size: Decimal = Field(default=Decimal("0.1"), gt=0, le=1.0)
    transaction_cost: Decimal = Field(
        default=Decimal("0.001"),
        ge=0,
        description="Transaction cost as percentage",
    )

    model_config = {"from_attributes": True}


class BacktestMetrics(BaseModel):
    """Backtest performance metrics."""

    total_return: Decimal
    annualized_return: Decimal
    sharpe_ratio: Decimal
    max_drawdown: Decimal
    win_rate: Decimal
    profit_factor: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int


class BacktestTrade(BaseModel):
    """Individual backtest trade."""

    entry_date: datetime
    exit_date: datetime
    entry_price: Decimal
    exit_price: Decimal
    position_size: int
    pnl: Decimal
    return_pct: Decimal


class BacktestResponse(BaseModel):
    """Response schema for backtesting results."""

    symbol: str
    model_id: str
    model_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    metrics: BacktestMetrics
    trades: list[BacktestTrade]
    equity_curve: list[dict[str, datetime | Decimal]]

    model_config = {"from_attributes": True}
