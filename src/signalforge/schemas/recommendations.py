"""Pydantic schemas for recommendation API endpoints."""

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class FeedItemResponse(BaseModel):
    """Response schema for a single feed item."""

    item_id: str = Field(..., description="Unique identifier for the item")
    item_type: str = Field(..., description="Type of content (signal, news, alert, report)")
    title: str = Field(..., description="Display title for the item")
    summary: str = Field(..., description="Brief summary or description")
    symbol: str | None = Field(None, description="Associated stock symbol")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    explanation: str = Field(..., description="Human-readable explanation")
    timestamp: datetime = Field(..., description="When the item was created")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("item_type")
    @classmethod
    def validate_item_type(cls, v: str) -> str:
        """Validate item type."""
        allowed_types = {"signal", "news", "alert", "report"}
        if v not in allowed_types:
            raise ValueError(f"item_type must be one of {allowed_types}")
        return v


class FeedResponse(BaseModel):
    """Response schema for personalized feed."""

    user_id: str = Field(..., description="User identifier")
    items: list[FeedItemResponse] = Field(..., description="List of feed items")
    generated_at: datetime = Field(..., description="When feed was generated")
    next_refresh_at: datetime = Field(..., description="Suggested next refresh time")


class DailyDigestResponse(BaseModel):
    """Response schema for daily digest."""

    user_id: str = Field(..., description="User identifier")
    digest_date: date = Field(..., description="Date for this digest")
    watchlist_signals: list[FeedItemResponse] = Field(
        default_factory=list, description="Signals for watchlist symbols"
    )
    portfolio_alerts: list[FeedItemResponse] = Field(
        default_factory=list, description="Portfolio-related alerts"
    )
    sector_highlights: list[FeedItemResponse] = Field(
        default_factory=list, description="Notable signals by sector"
    )
    upcoming_events: list[FeedItemResponse] = Field(
        default_factory=list, description="Upcoming events"
    )
    summary_text: str = Field(..., description="Overall summary")


class WeeklySummaryResponse(BaseModel):
    """Response schema for weekly summary."""

    user_id: str = Field(..., description="User identifier")
    week_start: date = Field(..., description="Start date of the week")
    week_end: date = Field(..., description="End date of the week")
    portfolio_performance: float = Field(..., description="Portfolio return percentage")
    top_signals_accuracy: float = Field(
        ..., ge=0.0, le=1.0, description="Accuracy of top signals"
    )
    engagement_stats: dict[str, Any] = Field(..., description="User engagement statistics")
    recommendations_for_next_week: list[str] = Field(
        ..., description="Actionable recommendations"
    )


class FeedbackRequest(BaseModel):
    """Request schema for submitting recommendation feedback."""

    item_id: str = Field(..., description="ID of the recommended item")
    feedback_type: str = Field(..., description="Type of feedback (like, dislike, hide, save)")
    rating: int | None = Field(None, ge=1, le=5, description="Optional rating (1-5)")
    comment: str | None = Field(None, max_length=500, description="Optional comment")

    @field_validator("feedback_type")
    @classmethod
    def validate_feedback_type(cls, v: str) -> str:
        """Validate feedback type."""
        allowed_types = {"like", "dislike", "hide", "save", "report"}
        if v not in allowed_types:
            raise ValueError(f"feedback_type must be one of {allowed_types}")
        return v


class FeedbackResponse(BaseModel):
    """Response schema for feedback submission."""

    success: bool = Field(..., description="Whether feedback was recorded")
    message: str = Field(..., description="Response message")


class ProfileResponse(BaseModel):
    """Response schema for user recommendation profile."""

    user_id: str = Field(..., description="User identifier")
    risk_tolerance: str = Field(..., description="Risk tolerance level")
    investment_horizon: int = Field(..., description="Investment horizon in days")
    preferred_sectors: list[str] = Field(..., description="Preferred sectors")
    watchlist: list[str] = Field(..., description="Watchlist symbols")
    notification_preferences: dict[str, Any] = Field(
        default_factory=dict, description="Notification preferences"
    )


class UpdateProfileRequest(BaseModel):
    """Request schema for updating recommendation profile."""

    risk_tolerance: str | None = Field(None, description="Risk tolerance level")
    investment_horizon: int | None = Field(None, gt=0, description="Investment horizon in days")
    preferred_sectors: list[str] | None = Field(None, description="Preferred sectors")
    watchlist: list[str] | None = Field(None, description="Watchlist symbols")
    notification_preferences: dict[str, Any] | None = Field(
        None, description="Notification preferences"
    )

    @field_validator("risk_tolerance")
    @classmethod
    def validate_risk_tolerance(cls, v: str | None) -> str | None:
        """Validate risk tolerance."""
        if v is not None:
            allowed_values = {"low", "medium", "high"}
            if v not in allowed_values:
                raise ValueError(f"risk_tolerance must be one of {allowed_values}")
        return v


class OnboardingAnswer(BaseModel):
    """Schema for a single onboarding question answer."""

    question_id: str = Field(..., description="Question identifier")
    answer: str | int | list[str] = Field(..., description="User's answer")


class OnboardingRequest(BaseModel):
    """Request schema for onboarding quiz submission."""

    answers: list[OnboardingAnswer] = Field(..., description="List of answers")


class OnboardingResponse(BaseModel):
    """Response schema for onboarding quiz submission."""

    success: bool = Field(..., description="Whether onboarding was successful")
    profile_created: bool = Field(..., description="Whether profile was created")
    message: str = Field(..., description="Response message")


class OnboardingQuestion(BaseModel):
    """Schema for a single onboarding question."""

    question_id: str = Field(..., description="Question identifier")
    question_text: str = Field(..., description="Question text")
    question_type: str = Field(..., description="Question type (single, multiple, text)")
    options: list[str] | None = Field(None, description="Answer options for choice questions")
    required: bool = Field(True, description="Whether question is required")

    @field_validator("question_type")
    @classmethod
    def validate_question_type(cls, v: str) -> str:
        """Validate question type."""
        allowed_types = {"single", "multiple", "text", "number"}
        if v not in allowed_types:
            raise ValueError(f"question_type must be one of {allowed_types}")
        return v


class OnboardingQuestionsResponse(BaseModel):
    """Response schema for onboarding questions."""

    questions: list[OnboardingQuestion] = Field(..., description="List of onboarding questions")


class RecommendationItem(BaseModel):
    """Schema for a recommendation item (used in explanations)."""

    item_id: str = Field(..., description="Unique identifier")
    symbol: str | None = Field(None, description="Stock symbol")
    sector: str | None = Field(None, description="Market sector")
    score: float = Field(..., ge=0.0, le=1.0, description="Recommendation score")
    source: str = Field(..., description="Recommendation source")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
