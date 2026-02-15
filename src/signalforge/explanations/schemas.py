"""Schemas for SHAP-based prediction explanations."""

from datetime import datetime

from pydantic import BaseModel, Field


class FeatureContribution(BaseModel):
    """Individual feature contribution to a prediction."""

    feature_name: str = Field(..., description="Name of the feature")
    feature_value: float = Field(..., description="Actual value of the feature")
    contribution: float = Field(..., description="SHAP value (contribution to prediction)")
    direction: str = Field(..., description="Direction of contribution: 'positive' or 'negative'")
    importance_rank: int = Field(..., description="Rank by absolute contribution (1 = most important)")
    human_readable: str = Field(..., description="Human-readable description of contribution")


class PredictionExplanation(BaseModel):
    """Complete explanation for a single prediction."""

    symbol: str = Field(..., description="Stock symbol")
    prediction: float = Field(..., description="Model prediction value")
    base_value: float = Field(..., description="Expected prediction without features (baseline)")
    top_features: list[FeatureContribution] = Field(..., description="Top contributing features")
    total_positive_contribution: float = Field(..., description="Sum of positive SHAP values")
    total_negative_contribution: float = Field(..., description="Sum of negative SHAP values")
    narrative: str = Field(..., description="Natural language explanation")
    confidence_factors: list[str] = Field(..., description="Factors affecting prediction confidence")
    generated_at: datetime = Field(..., description="Timestamp of explanation generation")


class ExplanationConfig(BaseModel):
    """Configuration for explanation generation."""

    top_k_features: int = Field(5, description="Number of top features to include", ge=1)
    include_negative: bool = Field(True, description="Include negatively contributing features")
    generate_narrative: bool = Field(True, description="Generate natural language narrative")
    include_visualization: bool = Field(False, description="Include visualization data")
