"""Explainability API schemas for model interpretability."""

from datetime import datetime

from pydantic import BaseModel, Field


class FeatureImpact(BaseModel):
    """Feature impact for a specific prediction."""

    feature_name: str = Field(..., description="Name of the feature")
    impact_value: float = Field(..., description="Numerical impact value")
    direction: str = Field(
        ...,
        description="Direction of impact (positive/negative)",
        pattern="^(positive|negative)$",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable explanation of feature impact",
    )

    model_config = {"from_attributes": True}


class PredictionExplanationResponse(BaseModel):
    """Explanation for a specific prediction."""

    prediction_id: str = Field(..., description="Unique prediction identifier")
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    top_features: list[FeatureImpact] = Field(
        ...,
        description="Top features impacting this prediction",
    )
    explanation_text: str = Field(
        ...,
        description="Natural language explanation",
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score",
    )
    shap_base_value: float | None = Field(
        default=None,
        description="SHAP base value for this prediction",
    )

    model_config = {"from_attributes": True}


class FeatureImportance(BaseModel):
    """Global feature importance metrics."""

    feature_name: str = Field(..., description="Name of the feature")
    importance_value: float = Field(
        ...,
        ge=0.0,
        description="Normalized importance value",
    )
    rank: int = Field(..., ge=1, description="Importance rank (1 is highest)")

    model_config = {"from_attributes": True}


class GlobalFeatureImportanceResponse(BaseModel):
    """Global feature importance for a model."""

    model_id: str = Field(..., description="Model identifier")
    features: list[FeatureImportance] = Field(
        ...,
        description="List of feature importance values",
    )
    calculated_at: datetime = Field(..., description="Calculation timestamp")

    model_config = {"from_attributes": True}


class FeatureImportanceHistoryPoint(BaseModel):
    """Single point in feature importance history."""

    timestamp: datetime = Field(..., description="Time of measurement")
    importance_value: float = Field(..., ge=0.0, description="Importance value")

    model_config = {"from_attributes": True}


class FeatureImportanceHistoryResponse(BaseModel):
    """Historical importance data for a feature."""

    model_id: str = Field(..., description="Model identifier")
    feature_name: str = Field(..., description="Feature name")
    history: list[FeatureImportanceHistoryPoint] = Field(
        ...,
        description="Historical importance values",
    )

    model_config = {"from_attributes": True}


class ModelContribution(BaseModel):
    """Individual model contribution in ensemble."""

    model_name: str = Field(..., description="Name of the model")
    prediction: float = Field(..., description="Model's prediction value")
    weight: float = Field(..., ge=0.0, le=1.0, description="Model weight in ensemble")
    contribution_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage contribution to final prediction",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model's confidence in prediction",
    )

    model_config = {"from_attributes": True}


class ModelComparisonResponse(BaseModel):
    """Ensemble model comparison for a prediction."""

    ensemble_id: str = Field(..., description="Ensemble identifier")
    prediction_id: str = Field(..., description="Prediction identifier")
    models: list[ModelContribution] = Field(
        ...,
        description="Individual model contributions",
    )

    model_config = {"from_attributes": True}


class FeatureAccuracy(BaseModel):
    """Feature-accuracy correlation analysis."""

    feature_name: str = Field(..., description="Name of the feature")
    accuracy_when_high: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Accuracy when feature value is high",
    )
    accuracy_when_low: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Accuracy when feature value is low",
    )
    correlation: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Correlation coefficient with accuracy",
    )

    model_config = {"from_attributes": True}


class AccuracyByFeatureResponse(BaseModel):
    """Feature-accuracy correlation analysis for a model."""

    model_id: str = Field(..., description="Model identifier")
    features: list[FeatureAccuracy] = Field(
        ...,
        description="Feature accuracy correlations",
    )

    model_config = {"from_attributes": True}


class ExplanationSummary(BaseModel):
    """Condensed explanation summary for dashboard widget."""

    symbol: str = Field(..., description="Trading symbol")
    model_id: str | None = Field(default=None, description="Model identifier")
    timestamp: datetime = Field(..., description="Latest prediction timestamp")
    top_feature: str = Field(..., description="Most important feature")
    top_feature_impact: float = Field(..., description="Top feature impact value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    direction: str = Field(
        ...,
        description="Predicted direction",
        pattern="^(up|down|neutral)$",
    )
    explanation: str = Field(
        ...,
        max_length=200,
        description="Brief explanation text",
    )

    model_config = {"from_attributes": True}


class BatchPredictionExplanationRequest(BaseModel):
    """Request for batch prediction explanations."""

    prediction_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of prediction IDs",
    )

    model_config = {"from_attributes": True}
