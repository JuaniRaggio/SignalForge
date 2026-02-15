"""Schemas for recommendation algorithms.

This module provides data structures for recommendation requests, responses,
and recommendation items used across all recommendation algorithms.

The schemas support:
- Flexible recommendation requests with filtering options
- Rich recommendation items with metadata and explanations
- Timestamp tracking for recommendation generation
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class RecommendationItem(BaseModel):
    """A single recommendation item with metadata.

    Attributes:
        item_id: Unique identifier for the item (symbol, signal_id, report_id).
        item_type: Type of item being recommended (signal, stock, report, alert).
        score: Relevance score between 0 and 1, higher is more relevant.
        source: Name of the algorithm that produced this recommendation.
        explanation: Human-readable explanation of why this item is recommended.
        metadata: Additional metadata about the recommendation.
    """

    item_id: str
    item_type: str
    score: float = Field(ge=0.0, le=1.0)
    source: str
    explanation: str
    metadata: dict[str, str | int | float] = Field(default_factory=dict)

    @field_validator("item_id", "item_type", "source", "explanation")
    @classmethod
    def validate_non_empty_string(cls, v: str) -> str:
        """Ensure string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


class RecommendationRequest(BaseModel):
    """Request for generating recommendations.

    Attributes:
        user_id: Unique identifier for the user requesting recommendations.
        item_types: Optional list of item types to filter recommendations.
        limit: Maximum number of recommendations to return.
        exclude_seen: Whether to exclude items the user has already seen.
        context: Optional context information (market conditions, time of day, etc.).
    """

    user_id: str
    item_types: list[str] | None = None
    limit: int = Field(default=10, ge=1, le=100)
    exclude_seen: bool = True
    context: dict[str, str | int | float | bool] | None = None

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        """Ensure user_id is not empty."""
        if not v or not v.strip():
            raise ValueError("user_id cannot be empty")
        return v.strip()


class RecommendationResponse(BaseModel):
    """Response containing generated recommendations.

    Attributes:
        user_id: User ID for whom recommendations were generated.
        recommendations: List of recommendation items.
        generated_at: Timestamp when recommendations were generated.
        algorithm_used: Name of the algorithm used to generate recommendations.
    """

    user_id: str
    recommendations: list[RecommendationItem]
    generated_at: datetime
    algorithm_used: str

    @field_validator("user_id", "algorithm_used")
    @classmethod
    def validate_non_empty_string(cls, v: str) -> str:
        """Ensure string fields are not empty."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()
