"""Recommendation API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.api.dependencies.auth import get_current_active_user
from signalforge.api.dependencies.database import get_db
from signalforge.core.logging import get_logger
from signalforge.models.user import User
from signalforge.schemas.recommendations import (
    DailyDigestResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedResponse,
    OnboardingQuestion,
    OnboardingQuestionsResponse,
    OnboardingRequest,
    OnboardingResponse,
    ProfileResponse,
    UpdateProfileRequest,
    WeeklySummaryResponse,
)

logger = get_logger(__name__)

router = APIRouter()


@router.get("/feed", response_model=FeedResponse)
async def get_personalized_feed(
    feed_type: str = Query("default", description="Feed type (default, signals, news, reports)"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of items"),
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
    current_user: Annotated[User, Depends(get_current_active_user)] = None,  # type: ignore[assignment]
) -> FeedResponse:
    """Get personalized feed for current user.

    This endpoint generates a personalized content feed based on the user's
    preferences, past activity, and similar user behavior.

    Args:
        feed_type: Type of feed to generate (default, signals, news, reports).
        limit: Maximum number of items to return (1-100).
        db: Database session.
        current_user: Authenticated user.

    Returns:
        FeedResponse with personalized items.

    Raises:
        HTTPException: If feed generation fails.
    """
    logger.info(
        "get_personalized_feed_request",
        user_id=str(current_user.id),
        feed_type=feed_type,
        limit=limit,
    )

    try:
        # Placeholder: Initialize feed generator
        # feed_generator = FeedGenerator(recommender, signal_service, ml_service)
        # feed = await feed_generator.generate_feed(
        #     user_id=str(current_user.id),
        #     feed_type=feed_type,
        #     limit=limit
        # )

        # Mock response for now
        from datetime import datetime, timedelta

        feed = FeedResponse(
            user_id=str(current_user.id),
            items=[],
            generated_at=datetime.utcnow(),
            next_refresh_at=datetime.utcnow() + timedelta(hours=1),
        )

        logger.info(
            "personalized_feed_generated",
            user_id=str(current_user.id),
            num_items=len(feed.items),
        )

        return feed

    except Exception as e:
        logger.error(
            "feed_generation_failed",
            user_id=str(current_user.id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate personalized feed",
        )


@router.get("/digest/daily", response_model=DailyDigestResponse)
async def get_daily_digest(
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
    current_user: Annotated[User, Depends(get_current_active_user)] = None,  # type: ignore[assignment]
) -> DailyDigestResponse:
    """Get daily digest for current user.

    This endpoint generates a daily digest containing:
    - Top signals for watchlist symbols
    - Portfolio alerts
    - Sector highlights
    - Upcoming events

    Args:
        db: Database session.
        current_user: Authenticated user.

    Returns:
        DailyDigestResponse with categorized content.

    Raises:
        HTTPException: If digest generation fails.
    """
    logger.info("get_daily_digest_request", user_id=str(current_user.id))

    try:
        # Placeholder: Generate daily digest

        from datetime import date as date_type

        digest = DailyDigestResponse(
            user_id=str(current_user.id),
            digest_date=date_type.today(),
            watchlist_signals=[],
            portfolio_alerts=[],
            sector_highlights=[],
            upcoming_events=[],
            summary_text="Your daily market digest",
        )

        logger.info("daily_digest_generated", user_id=str(current_user.id))

        return digest

    except Exception as e:
        logger.error(
            "digest_generation_failed",
            user_id=str(current_user.id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate daily digest",
        )


@router.get("/digest/weekly", response_model=WeeklySummaryResponse)
async def get_weekly_summary(
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
    current_user: Annotated[User, Depends(get_current_active_user)] = None,  # type: ignore[assignment]
) -> WeeklySummaryResponse:
    """Get weekly summary for current user.

    This endpoint generates a weekly performance summary containing:
    - Portfolio performance metrics
    - Signal accuracy statistics
    - User engagement data
    - Recommendations for next week

    Args:
        db: Database session.
        current_user: Authenticated user.

    Returns:
        WeeklySummaryResponse with performance data.

    Raises:
        HTTPException: If summary generation fails.
    """
    logger.info("get_weekly_summary_request", user_id=str(current_user.id))

    try:
        # Placeholder: Generate weekly summary
        from datetime import date, timedelta

        today = date.today()
        week_start = today - timedelta(days=today.weekday() + 7)
        week_end = week_start + timedelta(days=6)

        summary = WeeklySummaryResponse(
            user_id=str(current_user.id),
            week_start=week_start,
            week_end=week_end,
            portfolio_performance=0.0,
            top_signals_accuracy=0.0,
            engagement_stats={},
            recommendations_for_next_week=[],
        )

        logger.info("weekly_summary_generated", user_id=str(current_user.id))

        return summary

    except Exception as e:
        logger.error(
            "summary_generation_failed",
            user_id=str(current_user.id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate weekly summary",
        )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
    current_user: Annotated[User, Depends(get_current_active_user)] = None,  # type: ignore[assignment]
) -> FeedbackResponse:
    """Submit feedback on a recommendation.

    This endpoint allows users to provide feedback on recommended items,
    which is used to improve future recommendations.

    Args:
        request: Feedback data (item_id, feedback_type, rating, comment).
        db: Database session.
        current_user: Authenticated user.

    Returns:
        FeedbackResponse with success status.

    Raises:
        HTTPException: If feedback submission fails.
    """
    logger.info(
        "submit_feedback_request",
        user_id=str(current_user.id),
        item_id=request.item_id,
        feedback_type=request.feedback_type,
    )

    try:
        # Placeholder: Store feedback in database
        # This would typically:
        # 1. Create a UserActivity record
        # 2. Update implicit user profile
        # 3. Trigger model retraining if needed

        logger.info(
            "feedback_submitted",
            user_id=str(current_user.id),
            item_id=request.item_id,
            feedback_type=request.feedback_type,
        )

        return FeedbackResponse(
            success=True,
            message="Feedback recorded successfully",
        )

    except Exception as e:
        logger.error(
            "feedback_submission_failed",
            user_id=str(current_user.id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback",
        )


@router.get("/profile", response_model=ProfileResponse)
async def get_recommendation_profile(
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
    current_user: Annotated[User, Depends(get_current_active_user)] = None,  # type: ignore[assignment]
) -> ProfileResponse:
    """Get user's recommendation profile.

    This endpoint returns the user's current recommendation preferences
    and settings.

    Args:
        db: Database session.
        current_user: Authenticated user.

    Returns:
        ProfileResponse with user's recommendation settings.
    """
    logger.info("get_recommendation_profile_request", user_id=str(current_user.id))

    # Map database values to response
    investment_horizon_map = {
        "short": 7,
        "medium": 30,
        "long": 90,
    }

    profile = ProfileResponse(
        user_id=str(current_user.id),
        risk_tolerance=current_user.risk_tolerance.value,
        investment_horizon=investment_horizon_map.get(
            current_user.investment_horizon.value, 30
        ),
        preferred_sectors=current_user.preferred_sectors or [],
        watchlist=current_user.watchlist or [],
        notification_preferences=current_user.notification_preferences or {},
    )

    logger.info("recommendation_profile_retrieved", user_id=str(current_user.id))

    return profile


@router.put("/profile", response_model=ProfileResponse)
async def update_recommendation_profile(
    request: UpdateProfileRequest,
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
    current_user: Annotated[User, Depends(get_current_active_user)] = None,  # type: ignore[assignment]
) -> ProfileResponse:
    """Update user's recommendation profile.

    This endpoint allows users to update their recommendation preferences,
    which will be used to personalize future recommendations.

    Args:
        request: Profile update data.
        db: Database session.
        current_user: Authenticated user.

    Returns:
        ProfileResponse with updated settings.

    Raises:
        HTTPException: If profile update fails.
    """
    logger.info("update_recommendation_profile_request", user_id=str(current_user.id))

    try:
        # Update user profile fields
        if request.risk_tolerance is not None:
            from signalforge.models.user import RiskTolerance

            current_user.risk_tolerance = RiskTolerance(request.risk_tolerance)

        if request.investment_horizon is not None:
            from signalforge.models.user import InvestmentHorizon

            # Map days to horizon enum
            if request.investment_horizon <= 7:
                current_user.investment_horizon = InvestmentHorizon.SHORT
            elif request.investment_horizon <= 90:
                current_user.investment_horizon = InvestmentHorizon.MEDIUM
            else:
                current_user.investment_horizon = InvestmentHorizon.LONG

        if request.preferred_sectors is not None:
            current_user.preferred_sectors = request.preferred_sectors

        if request.watchlist is not None:
            current_user.watchlist = request.watchlist

        if request.notification_preferences is not None:
            current_user.notification_preferences = request.notification_preferences

        await db.commit()
        await db.refresh(current_user)

        logger.info("recommendation_profile_updated", user_id=str(current_user.id))

        # Return updated profile
        return await get_recommendation_profile(db, current_user)

    except Exception as e:
        await db.rollback()
        logger.error(
            "profile_update_failed",
            user_id=str(current_user.id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update recommendation profile",
        )


@router.post("/onboarding", response_model=OnboardingResponse)
async def submit_onboarding_quiz(
    request: OnboardingRequest,
    db: Annotated[AsyncSession, Depends(get_db)] = None,  # type: ignore[assignment]
    current_user: Annotated[User, Depends(get_current_active_user)] = None,  # type: ignore[assignment]
) -> OnboardingResponse:
    """Submit onboarding quiz answers.

    This endpoint processes onboarding quiz answers to build the initial
    user recommendation profile.

    Args:
        request: Onboarding quiz answers.
        db: Database session.
        current_user: Authenticated user.

    Returns:
        OnboardingResponse with success status.

    Raises:
        HTTPException: If onboarding submission fails.
    """
    logger.info(
        "submit_onboarding_request",
        user_id=str(current_user.id),
        num_answers=len(request.answers),
    )

    try:
        # Process onboarding answers to build profile
        # This would typically:
        # 1. Parse answers to determine preferences
        # 2. Update user profile
        # 3. Create initial recommendation profile

        for answer in request.answers:
            logger.debug(
                "processing_onboarding_answer",
                user_id=str(current_user.id),
                question_id=answer.question_id,
            )

        logger.info("onboarding_completed", user_id=str(current_user.id))

        return OnboardingResponse(
            success=True,
            profile_created=True,
            message="Onboarding completed successfully",
        )

    except Exception as e:
        logger.error(
            "onboarding_submission_failed",
            user_id=str(current_user.id),
            error=str(e),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process onboarding quiz",
        )


@router.get("/onboarding/questions", response_model=OnboardingQuestionsResponse)
async def get_onboarding_questions() -> OnboardingQuestionsResponse:
    """Get onboarding quiz questions.

    This endpoint returns the set of questions for the onboarding quiz
    used to build initial user recommendation profiles.

    Returns:
        OnboardingQuestionsResponse with quiz questions.
    """
    logger.info("get_onboarding_questions_request")

    questions = [
        OnboardingQuestion(
            question_id="risk_tolerance",
            question_text="What is your risk tolerance for investments?",
            question_type="single",
            options=["Low - I prefer safe, stable investments", "Medium - Balanced approach", "High - I'm comfortable with volatility"],
            required=True,
        ),
        OnboardingQuestion(
            question_id="investment_horizon",
            question_text="What is your typical investment time horizon?",
            question_type="single",
            options=["Short-term (< 1 week)", "Medium-term (1 week - 3 months)", "Long-term (> 3 months)"],
            required=True,
        ),
        OnboardingQuestion(
            question_id="preferred_sectors",
            question_text="Which market sectors are you most interested in?",
            question_type="multiple",
            options=["Technology", "Healthcare", "Finance", "Energy", "Consumer", "Industrial", "Materials", "Utilities", "Real Estate"],
            required=True,
        ),
        OnboardingQuestion(
            question_id="experience_level",
            question_text="What is your experience level with trading?",
            question_type="single",
            options=["Beginner - New to trading", "Intermediate - Some experience", "Advanced - Experienced trader", "Professional - Trading is my profession"],
            required=True,
        ),
        OnboardingQuestion(
            question_id="initial_watchlist",
            question_text="Enter stock symbols you're interested in tracking (comma-separated)",
            question_type="text",
            options=None,
            required=False,
        ),
    ]

    logger.info("onboarding_questions_retrieved", num_questions=len(questions))

    return OnboardingQuestionsResponse(questions=questions)
