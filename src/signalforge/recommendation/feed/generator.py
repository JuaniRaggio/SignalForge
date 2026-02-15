"""Feed generation for personalized content recommendations.

This module provides the core feed generation logic for creating personalized
content feeds for users. It combines:
- User profile data (explicit and implicit preferences)
- Recommendation algorithms (hybrid recommender)
- Signal and content services
- Explanation generation

Key Features:
- Personalized feed generation with multiple feed types
- Daily digest generation for email delivery
- Weekly summary reports
- Multi-factor ranking with diversity
- Content explanation generation

Examples:
    Generating a personalized feed:

    >>> from signalforge.recommendation.feed.generator import FeedGenerator
    >>> generator = FeedGenerator(recommender, signal_service, ml_service)
    >>> feed = await generator.generate_feed(
    ...     user_id="user_123",
    ...     feed_type="default",
    ...     limit=20
    ... )
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from signalforge.core.logging import get_logger
from signalforge.recommendation.feed.explainer import RecommendationExplainer
from signalforge.recommendation.user_model import ExplicitProfile, UserProfile

logger = get_logger(__name__)


@dataclass
class FeedItem:
    """A single item in a personalized feed.

    Attributes:
        item_id: Unique identifier for the item.
        item_type: Type of content (signal, news, alert, report).
        title: Display title for the item.
        summary: Brief summary or description.
        symbol: Associated stock symbol (if applicable).
        relevance_score: Relevance score for ranking (0.0 to 1.0).
        explanation: Human-readable explanation for recommendation.
        timestamp: When the item was created/published.
        metadata: Additional item-specific metadata.
    """

    item_id: str
    item_type: str
    title: str
    summary: str
    symbol: str | None
    relevance_score: float
    explanation: str
    timestamp: datetime
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate feed item."""
        if not self.item_id:
            raise ValueError("Item ID cannot be empty")

        if self.item_type not in ("signal", "news", "alert", "report"):
            raise ValueError(f"Invalid item type: {self.item_type}")

        if not 0.0 <= self.relevance_score <= 1.0:
            raise ValueError(
                f"Relevance score must be between 0.0 and 1.0, got {self.relevance_score}"
            )


@dataclass
class PersonalizedFeed:
    """A complete personalized feed for a user.

    Attributes:
        user_id: Unique identifier for the user.
        items: List of feed items sorted by relevance.
        generated_at: Timestamp when feed was generated.
        next_refresh_at: Suggested time for next feed refresh.
    """

    user_id: str
    items: list[FeedItem]
    generated_at: datetime
    next_refresh_at: datetime

    def __post_init__(self) -> None:
        """Validate personalized feed."""
        if not self.user_id:
            raise ValueError("User ID cannot be empty")

        if self.next_refresh_at <= self.generated_at:
            raise ValueError("Next refresh time must be after generation time")


@dataclass
class DailyDigest:
    """Daily digest email content for a user.

    Attributes:
        user_id: Unique identifier for the user.
        date: Date for this digest.
        watchlist_signals: Signals for symbols on user's watchlist.
        portfolio_alerts: Alerts related to portfolio holdings.
        sector_highlights: Notable signals by sector of interest.
        upcoming_events: Events that may impact watched securities.
        summary_text: Overall summary of market conditions.
    """

    user_id: str
    date: date
    watchlist_signals: list[FeedItem]
    portfolio_alerts: list[FeedItem]
    sector_highlights: list[FeedItem]
    upcoming_events: list[FeedItem]
    summary_text: str

    def __post_init__(self) -> None:
        """Validate daily digest."""
        if not self.user_id:
            raise ValueError("User ID cannot be empty")


@dataclass
class WeeklySummary:
    """Weekly performance summary for a user.

    Attributes:
        user_id: Unique identifier for the user.
        week_start: Start date of the week.
        week_end: End date of the week.
        portfolio_performance: Portfolio return percentage.
        top_signals_accuracy: Accuracy of top signals (0.0 to 1.0).
        engagement_stats: User engagement statistics.
        recommendations_for_next_week: Actionable recommendations.
    """

    user_id: str
    week_start: date
    week_end: date
    portfolio_performance: float
    top_signals_accuracy: float
    engagement_stats: dict[str, Any]
    recommendations_for_next_week: list[str]

    def __post_init__(self) -> None:
        """Validate weekly summary."""
        if not self.user_id:
            raise ValueError("User ID cannot be empty")

        if self.week_end < self.week_start:
            raise ValueError("Week end must be after week start")

        if not 0.0 <= self.top_signals_accuracy <= 1.0:
            raise ValueError(
                f"Accuracy must be between 0.0 and 1.0, got {self.top_signals_accuracy}"
            )


class FeedGenerator:
    """Generate personalized content feeds for users.

    This class orchestrates the feed generation process by combining
    user profiles, recommendations, and content sources to create
    personalized feeds, digests, and summaries.

    Examples:
        >>> generator = FeedGenerator(recommender, signal_service, ml_service)
        >>> feed = await generator.generate_feed("user_123", limit=20)
    """

    def __init__(
        self,
        recommender: Any,
        signal_service: Any,
        ml_service: Any,
    ) -> None:
        """Initialize the feed generator.

        Args:
            recommender: Hybrid recommender instance for generating recommendations.
            signal_service: NLP signal service for fetching signal data.
            ml_service: ML prediction service for confidence scores.
        """
        self._recommender = recommender
        self._signal_service = signal_service
        self._ml_service = ml_service
        self._explainer = RecommendationExplainer()

        logger.info("feed_generator_initialized")

    async def generate_feed(
        self,
        user_id: str,
        feed_type: str = "default",
        limit: int = 20,
    ) -> PersonalizedFeed:
        """Generate personalized feed for a user.

        The generation process:
        1. Fetch user profile (explicit and implicit preferences)
        2. Get recommendations from hybrid recommender
        3. Fetch content for recommended items
        4. Generate explanations for each recommendation
        5. Apply final ranking with diversity
        6. Return top-k items

        Args:
            user_id: Unique identifier for the user.
            feed_type: Type of feed to generate (default, signals, news, reports).
            limit: Maximum number of items to return (1 to 100).

        Returns:
            PersonalizedFeed with ranked and explained items.

        Raises:
            ValueError: If feed_type is invalid or limit is out of range.

        Examples:
            >>> feed = await generator.generate_feed(
            ...     user_id="user_123",
            ...     feed_type="signals",
            ...     limit=10
            ... )
        """
        if feed_type not in ("default", "signals", "news", "reports"):
            raise ValueError(f"Invalid feed type: {feed_type}")

        if not 1 <= limit <= 100:
            raise ValueError(f"Limit must be between 1 and 100, got {limit}")

        logger.info(
            "generating_feed",
            user_id=user_id,
            feed_type=feed_type,
            limit=limit,
        )

        # Step 1: Get user profile
        user_profile = await self._get_user_profile(user_id)

        # Step 2: Get recommendations
        recommended_items = await self._get_recommendations(
            user_profile, feed_type, limit * 2
        )

        # Step 3: Fetch content for recommended items
        feed_items = await self._fetch_content(recommended_items, user_profile)

        # Step 4: Generate explanations
        for item in feed_items:
            item.explanation = self._explainer.generate_explanation(
                item, user_profile.explicit, feed_type
            )

        # Step 5: Apply final ranking with diversity
        ranked_items = self._rank_feed_items(feed_items, user_profile.explicit)

        # Step 6: Take top-k
        final_items = ranked_items[:limit]

        # Calculate next refresh time based on user activity
        generated_at = datetime.utcnow()
        next_refresh_at = generated_at + timedelta(hours=1)

        feed = PersonalizedFeed(
            user_id=user_id,
            items=final_items,
            generated_at=generated_at,
            next_refresh_at=next_refresh_at,
        )

        logger.info(
            "feed_generated",
            user_id=user_id,
            num_items=len(final_items),
            feed_type=feed_type,
        )

        return feed

    async def generate_daily_digest(
        self,
        user_id: str,
    ) -> DailyDigest:
        """Generate daily digest email content.

        The digest includes:
        - Top signals for watchlist symbols
        - Portfolio-related alerts
        - Sector highlights based on preferences
        - Upcoming events that may impact investments

        Args:
            user_id: Unique identifier for the user.

        Returns:
            DailyDigest with categorized content.

        Examples:
            >>> digest = await generator.generate_daily_digest("user_123")
        """
        logger.info("generating_daily_digest", user_id=user_id)

        # Get user profile
        user_profile = await self._get_user_profile(user_id)

        # Get watchlist signals
        watchlist_signals = await self._get_watchlist_signals(user_profile)

        # Get portfolio alerts
        portfolio_alerts = await self._get_portfolio_alerts(user_profile)

        # Get sector highlights
        sector_highlights = await self._get_sector_highlights(user_profile)

        # Get upcoming events
        upcoming_events = await self._get_upcoming_events(user_profile)

        # Generate summary text
        summary_text = self._generate_digest_summary(
            user_profile,
            len(watchlist_signals),
            len(portfolio_alerts),
            len(sector_highlights),
        )

        digest = DailyDigest(
            user_id=user_id,
            date=datetime.utcnow().date(),
            watchlist_signals=watchlist_signals,
            portfolio_alerts=portfolio_alerts,
            sector_highlights=sector_highlights,
            upcoming_events=upcoming_events,
            summary_text=summary_text,
        )

        logger.info(
            "daily_digest_generated",
            user_id=user_id,
            total_items=len(watchlist_signals)
            + len(portfolio_alerts)
            + len(sector_highlights)
            + len(upcoming_events),
        )

        return digest

    async def generate_weekly_summary(
        self,
        user_id: str,
    ) -> WeeklySummary:
        """Generate weekly performance summary.

        The summary includes:
        - Portfolio performance over the week
        - Accuracy of top recommended signals
        - User engagement statistics
        - Actionable recommendations for next week

        Args:
            user_id: Unique identifier for the user.

        Returns:
            WeeklySummary with performance metrics and recommendations.

        Examples:
            >>> summary = await generator.generate_weekly_summary("user_123")
        """
        logger.info("generating_weekly_summary", user_id=user_id)

        # Calculate week boundaries
        today = date.today()
        week_start = today - timedelta(days=today.weekday() + 7)
        week_end = week_start + timedelta(days=6)

        # Get user profile
        user_profile = await self._get_user_profile(user_id)

        # Calculate portfolio performance
        portfolio_performance = await self._calculate_portfolio_performance(
            user_id, week_start, week_end
        )

        # Calculate signal accuracy
        top_signals_accuracy = await self._calculate_signal_accuracy(
            user_id, week_start, week_end
        )

        # Get engagement stats
        engagement_stats = await self._get_engagement_stats(user_id, week_start, week_end)

        # Generate recommendations for next week
        recommendations = await self._generate_next_week_recommendations(user_profile)

        summary = WeeklySummary(
            user_id=user_id,
            week_start=week_start,
            week_end=week_end,
            portfolio_performance=portfolio_performance,
            top_signals_accuracy=top_signals_accuracy,
            engagement_stats=engagement_stats,
            recommendations_for_next_week=recommendations,
        )

        logger.info(
            "weekly_summary_generated",
            user_id=user_id,
            performance=portfolio_performance,
            accuracy=top_signals_accuracy,
        )

        return summary

    def _rank_feed_items(
        self,
        items: list[FeedItem],
        user_profile: ExplicitProfile,
    ) -> list[FeedItem]:
        """Apply final ranking considering recency, relevance, and diversity.

        This method re-ranks items to ensure:
        - High relevance items are prioritized
        - Recent items are favored
        - Diversity across sectors and symbols

        Args:
            items: List of feed items to rank.
            user_profile: User's explicit profile for ranking context.

        Returns:
            Sorted list of feed items.
        """
        if not items:
            return items

        logger.debug("ranking_feed_items", num_items=len(items))

        # Track sector and symbol counts for diversity
        sector_counts: dict[str, int] = defaultdict(int)
        symbol_counts: dict[str, int] = defaultdict(int)

        # Adjust scores based on diversity
        for item in items:
            # Apply diversity penalty for over-represented sectors
            if item.symbol:
                symbol_penalty = symbol_counts.get(item.symbol, 0) * 0.05
                sector_penalty = 0.0

                # Extract sector from metadata if available
                if "sector" in item.metadata:
                    sector = item.metadata["sector"]
                    sector_penalty = sector_counts.get(sector, 0) * 0.03
                    sector_counts[sector] += 1

                symbol_counts[item.symbol] += 1

                # Reduce score by diversity penalty
                item.relevance_score = max(
                    0.0, item.relevance_score - symbol_penalty - sector_penalty
                )

            # Apply recency boost
            age_hours = (datetime.utcnow() - item.timestamp).total_seconds() / 3600.0
            if age_hours < 24:
                recency_boost = 0.1 * (1.0 - age_hours / 24.0)
                item.relevance_score = min(1.0, item.relevance_score + recency_boost)

        # Sort by adjusted relevance score
        ranked = sorted(items, key=lambda x: x.relevance_score, reverse=True)

        logger.debug("feed_items_ranked", num_items=len(ranked))

        return ranked

    async def _get_user_profile(self, user_id: str) -> UserProfile:
        """Fetch user profile from database or cache."""
        # Placeholder: In real implementation, fetch from database
        import numpy as np

        from signalforge.recommendation.user_model import (
            ExplicitProfile,
            ImplicitProfile,
            UserProfile,
        )

        explicit = ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology", "Healthcare"],
            watchlist=["AAPL", "GOOGL", "MSFT"],
        )

        implicit = ImplicitProfile(
            viewed_sectors={"Technology": 10, "Healthcare": 5},
            viewed_symbols={"AAPL": 15, "GOOGL": 8},
            avg_holding_period=25.0,
            preferred_volatility=0.3,
        )

        profile = UserProfile(
            user_id=user_id,
            explicit=explicit,
            implicit=implicit,
            combined_embedding=np.random.rand(32),
        )

        return profile

    async def _get_recommendations(
        self, user_profile: UserProfile, feed_type: str, limit: int
    ) -> list[Any]:
        """Get recommendations from hybrid recommender."""
        # Placeholder: Call actual recommender
        logger.debug(
            "getting_recommendations",
            user_id=user_profile.user_id,
            feed_type=feed_type,
            limit=limit,
        )
        return []

    async def _fetch_content(
        self, recommended_items: list[Any], user_profile: UserProfile
    ) -> list[FeedItem]:
        """Fetch content for recommended items."""
        # Placeholder: Fetch actual content
        logger.debug(
            "fetching_content",
            user_id=user_profile.user_id,
            num_items=len(recommended_items),
        )
        return []

    async def _get_watchlist_signals(self, user_profile: UserProfile) -> list[FeedItem]:
        """Get signals for watchlist symbols."""
        # Placeholder implementation
        return []

    async def _get_portfolio_alerts(self, user_profile: UserProfile) -> list[FeedItem]:
        """Get portfolio-related alerts."""
        # Placeholder implementation
        return []

    async def _get_sector_highlights(self, user_profile: UserProfile) -> list[FeedItem]:
        """Get sector highlights."""
        # Placeholder implementation
        return []

    async def _get_upcoming_events(self, user_profile: UserProfile) -> list[FeedItem]:
        """Get upcoming events."""
        # Placeholder implementation
        return []

    def _generate_digest_summary(
        self,
        user_profile: UserProfile,
        num_watchlist: int,
        num_alerts: int,
        num_highlights: int,
    ) -> str:
        """Generate summary text for digest."""
        return (
            f"Your daily market digest: {num_watchlist} watchlist signals, "
            f"{num_alerts} portfolio alerts, and {num_highlights} sector highlights."
        )

    async def _calculate_portfolio_performance(
        self, user_id: str, start: date, end: date
    ) -> float:
        """Calculate portfolio performance over period."""
        # Placeholder: Calculate actual performance
        return 0.0

    async def _calculate_signal_accuracy(
        self, user_id: str, start: date, end: date
    ) -> float:
        """Calculate accuracy of top signals."""
        # Placeholder: Calculate actual accuracy
        return 0.0

    async def _get_engagement_stats(
        self, user_id: str, start: date, end: date
    ) -> dict[str, Any]:
        """Get user engagement statistics."""
        # Placeholder: Get actual stats
        return {
            "signals_viewed": 0,
            "signals_saved": 0,
            "avg_session_duration": 0.0,
        }

    async def _generate_next_week_recommendations(
        self, user_profile: UserProfile
    ) -> list[str]:
        """Generate actionable recommendations for next week."""
        # Placeholder: Generate actual recommendations
        return [
            "Consider diversifying into Energy sector",
            "Review your Technology holdings for rebalancing",
        ]
