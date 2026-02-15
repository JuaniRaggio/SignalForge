"""Tests for feed generation and personalization."""

from datetime import date, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from signalforge.recommendation.feed.explainer import RecommendationExplainer
from signalforge.recommendation.feed.generator import (
    DailyDigest,
    FeedGenerator,
    FeedItem,
    PersonalizedFeed,
    WeeklySummary,
)
from signalforge.recommendation.user_model import (
    ExplicitProfile,
    ImplicitProfile,
    UserProfile,
)


class TestFeedItem:
    """Tests for FeedItem dataclass."""

    def test_feed_item_creation(self) -> None:
        """Test creating a valid feed item."""
        item = FeedItem(
            item_id="signal_001",
            item_type="signal",
            title="AAPL Buy Signal",
            summary="Strong buy signal for AAPL",
            symbol="AAPL",
            relevance_score=0.85,
            explanation="Based on your interest in Technology stocks",
            timestamp=datetime.utcnow(),
            metadata={"sector": "Technology"},
        )

        assert item.item_id == "signal_001"
        assert item.item_type == "signal"
        assert item.relevance_score == 0.85

    def test_feed_item_invalid_type(self) -> None:
        """Test feed item with invalid type raises error."""
        with pytest.raises(ValueError, match="Invalid item type"):
            FeedItem(
                item_id="item_001",
                item_type="invalid_type",
                title="Test",
                summary="Test",
                symbol=None,
                relevance_score=0.5,
                explanation="Test",
                timestamp=datetime.utcnow(),
                metadata={},
            )

    def test_feed_item_invalid_score(self) -> None:
        """Test feed item with invalid relevance score raises error."""
        with pytest.raises(ValueError, match="Relevance score must be between"):
            FeedItem(
                item_id="item_001",
                item_type="signal",
                title="Test",
                summary="Test",
                symbol=None,
                relevance_score=1.5,
                explanation="Test",
                timestamp=datetime.utcnow(),
                metadata={},
            )

    def test_feed_item_empty_id(self) -> None:
        """Test feed item with empty ID raises error."""
        with pytest.raises(ValueError, match="Item ID cannot be empty"):
            FeedItem(
                item_id="",
                item_type="signal",
                title="Test",
                summary="Test",
                symbol=None,
                relevance_score=0.5,
                explanation="Test",
                timestamp=datetime.utcnow(),
                metadata={},
            )


class TestPersonalizedFeed:
    """Tests for PersonalizedFeed dataclass."""

    def test_personalized_feed_creation(self) -> None:
        """Test creating a valid personalized feed."""
        now = datetime.utcnow()
        feed = PersonalizedFeed(
            user_id="user_123",
            items=[],
            generated_at=now,
            next_refresh_at=now + timedelta(hours=1),
        )

        assert feed.user_id == "user_123"
        assert len(feed.items) == 0
        assert feed.next_refresh_at > feed.generated_at

    def test_personalized_feed_empty_user_id(self) -> None:
        """Test feed with empty user ID raises error."""
        now = datetime.utcnow()
        with pytest.raises(ValueError, match="User ID cannot be empty"):
            PersonalizedFeed(
                user_id="",
                items=[],
                generated_at=now,
                next_refresh_at=now + timedelta(hours=1),
            )

    def test_personalized_feed_invalid_refresh_time(self) -> None:
        """Test feed with invalid refresh time raises error."""
        now = datetime.utcnow()
        with pytest.raises(ValueError, match="Next refresh time must be after"):
            PersonalizedFeed(
                user_id="user_123",
                items=[],
                generated_at=now,
                next_refresh_at=now - timedelta(hours=1),
            )


class TestDailyDigest:
    """Tests for DailyDigest dataclass."""

    def test_daily_digest_creation(self) -> None:
        """Test creating a valid daily digest."""
        digest = DailyDigest(
            user_id="user_123",
            date=date.today(),
            watchlist_signals=[],
            portfolio_alerts=[],
            sector_highlights=[],
            upcoming_events=[],
            summary_text="Your daily digest",
        )

        assert digest.user_id == "user_123"
        assert digest.date == date.today()
        assert len(digest.watchlist_signals) == 0

    def test_daily_digest_empty_user_id(self) -> None:
        """Test digest with empty user ID raises error."""
        with pytest.raises(ValueError, match="User ID cannot be empty"):
            DailyDigest(
                user_id="",
                date=date.today(),
                watchlist_signals=[],
                portfolio_alerts=[],
                sector_highlights=[],
                upcoming_events=[],
                summary_text="Test",
            )


class TestWeeklySummary:
    """Tests for WeeklySummary dataclass."""

    def test_weekly_summary_creation(self) -> None:
        """Test creating a valid weekly summary."""
        today = date.today()
        week_start = today - timedelta(days=7)
        week_end = today

        summary = WeeklySummary(
            user_id="user_123",
            week_start=week_start,
            week_end=week_end,
            portfolio_performance=0.05,
            top_signals_accuracy=0.75,
            engagement_stats={"views": 10},
            recommendations_for_next_week=["Diversify into Energy"],
        )

        assert summary.user_id == "user_123"
        assert summary.portfolio_performance == 0.05
        assert summary.top_signals_accuracy == 0.75

    def test_weekly_summary_invalid_date_range(self) -> None:
        """Test summary with invalid date range raises error."""
        today = date.today()
        with pytest.raises(ValueError, match="Week end must be after"):
            WeeklySummary(
                user_id="user_123",
                week_start=today,
                week_end=today - timedelta(days=7),
                portfolio_performance=0.0,
                top_signals_accuracy=0.5,
                engagement_stats={},
                recommendations_for_next_week=[],
            )

    def test_weekly_summary_invalid_accuracy(self) -> None:
        """Test summary with invalid accuracy raises error."""
        today = date.today()
        with pytest.raises(ValueError, match="Accuracy must be between"):
            WeeklySummary(
                user_id="user_123",
                week_start=today - timedelta(days=7),
                week_end=today,
                portfolio_performance=0.0,
                top_signals_accuracy=1.5,
                engagement_stats={},
                recommendations_for_next_week=[],
            )


class TestFeedGenerator:
    """Tests for FeedGenerator class."""

    @pytest.fixture
    def mock_recommender(self) -> MagicMock:
        """Create mock recommender."""
        return MagicMock()

    @pytest.fixture
    def mock_signal_service(self) -> MagicMock:
        """Create mock signal service."""
        return MagicMock()

    @pytest.fixture
    def mock_ml_service(self) -> MagicMock:
        """Create mock ML service."""
        return MagicMock()

    @pytest.fixture
    def feed_generator(
        self,
        mock_recommender: MagicMock,
        mock_signal_service: MagicMock,
        mock_ml_service: MagicMock,
    ) -> FeedGenerator:
        """Create feed generator instance."""
        return FeedGenerator(mock_recommender, mock_signal_service, mock_ml_service)

    @pytest.mark.asyncio
    async def test_feed_generator_initialization(self, feed_generator: FeedGenerator) -> None:
        """Test feed generator initializes correctly."""
        assert feed_generator._recommender is not None
        assert feed_generator._signal_service is not None
        assert feed_generator._ml_service is not None
        assert isinstance(feed_generator._explainer, RecommendationExplainer)

    @pytest.mark.asyncio
    async def test_generate_feed_invalid_type(self, feed_generator: FeedGenerator) -> None:
        """Test generate_feed with invalid feed type raises error."""
        with pytest.raises(ValueError, match="Invalid feed type"):
            await feed_generator.generate_feed("user_123", feed_type="invalid", limit=10)

    @pytest.mark.asyncio
    async def test_generate_feed_invalid_limit(self, feed_generator: FeedGenerator) -> None:
        """Test generate_feed with invalid limit raises error."""
        with pytest.raises(ValueError, match="Limit must be between"):
            await feed_generator.generate_feed("user_123", feed_type="default", limit=150)

    @pytest.mark.asyncio
    async def test_generate_feed_success(self, feed_generator: FeedGenerator) -> None:
        """Test successful feed generation."""
        feed = await feed_generator.generate_feed("user_123", feed_type="default", limit=20)

        assert isinstance(feed, PersonalizedFeed)
        assert feed.user_id == "user_123"
        assert feed.generated_at <= datetime.utcnow()
        assert feed.next_refresh_at > feed.generated_at

    @pytest.mark.asyncio
    async def test_generate_daily_digest_success(self, feed_generator: FeedGenerator) -> None:
        """Test successful daily digest generation."""
        digest = await feed_generator.generate_daily_digest("user_123")

        assert isinstance(digest, DailyDigest)
        assert digest.user_id == "user_123"
        assert digest.date == date.today()
        assert isinstance(digest.summary_text, str)

    @pytest.mark.asyncio
    async def test_generate_weekly_summary_success(
        self, feed_generator: FeedGenerator
    ) -> None:
        """Test successful weekly summary generation."""
        summary = await feed_generator.generate_weekly_summary("user_123")

        assert isinstance(summary, WeeklySummary)
        assert summary.user_id == "user_123"
        assert summary.week_start < summary.week_end
        assert 0.0 <= summary.top_signals_accuracy <= 1.0

    def test_rank_feed_items_empty_list(self, feed_generator: FeedGenerator) -> None:
        """Test ranking empty list returns empty list."""
        explicit_profile = ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology"],
            watchlist=["AAPL"],
        )

        ranked = feed_generator._rank_feed_items([], explicit_profile)
        assert ranked == []

    def test_rank_feed_items_sorts_by_relevance(self, feed_generator: FeedGenerator) -> None:
        """Test ranking sorts items by relevance score."""
        explicit_profile = ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology"],
            watchlist=["AAPL"],
        )

        now = datetime.utcnow()
        items = [
            FeedItem(
                item_id="item_1",
                item_type="signal",
                title="Signal 1",
                summary="Test",
                symbol="AAPL",
                relevance_score=0.5,
                explanation="Test",
                timestamp=now,
                metadata={},
            ),
            FeedItem(
                item_id="item_2",
                item_type="signal",
                title="Signal 2",
                summary="Test",
                symbol="GOOGL",
                relevance_score=0.8,
                explanation="Test",
                timestamp=now,
                metadata={},
            ),
        ]

        ranked = feed_generator._rank_feed_items(items, explicit_profile)

        assert len(ranked) == 2
        assert ranked[0].item_id == "item_2"
        assert ranked[1].item_id == "item_1"

    def test_rank_feed_items_applies_recency_boost(
        self, feed_generator: FeedGenerator
    ) -> None:
        """Test ranking applies recency boost to recent items."""
        explicit_profile = ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology"],
            watchlist=["AAPL"],
        )

        now = datetime.utcnow()
        old_time = now - timedelta(hours=48)

        items = [
            FeedItem(
                item_id="item_old",
                item_type="signal",
                title="Old Signal",
                summary="Test",
                symbol="AAPL",
                relevance_score=0.7,
                explanation="Test",
                timestamp=old_time,
                metadata={},
            ),
            FeedItem(
                item_id="item_new",
                item_type="signal",
                title="New Signal",
                summary="Test",
                symbol="GOOGL",
                relevance_score=0.65,
                explanation="Test",
                timestamp=now,
                metadata={},
            ),
        ]

        ranked = feed_generator._rank_feed_items(items, explicit_profile)

        # New item should be ranked higher due to recency boost
        assert ranked[0].item_id == "item_new"

    @pytest.mark.asyncio
    async def test_get_user_profile_returns_profile(
        self, feed_generator: FeedGenerator
    ) -> None:
        """Test _get_user_profile returns a UserProfile."""
        profile = await feed_generator._get_user_profile("user_123")

        assert isinstance(profile, UserProfile)
        assert profile.user_id == "user_123"
        assert isinstance(profile.explicit, ExplicitProfile)
        assert isinstance(profile.implicit, ImplicitProfile)

    def test_generate_digest_summary(self, feed_generator: FeedGenerator) -> None:
        """Test digest summary generation."""
        profile = UserProfile(
            user_id="user_123",
            explicit=ExplicitProfile(
                risk_tolerance="medium",
                investment_horizon=30,
                preferred_sectors=["Technology"],
                watchlist=["AAPL"],
            ),
            implicit=ImplicitProfile(),
            combined_embedding=np.random.rand(32),
        )

        summary = feed_generator._generate_digest_summary(profile, 5, 2, 3)

        assert isinstance(summary, str)
        assert "5" in summary
        assert "2" in summary
        assert "3" in summary


class TestRecommendationExplainer:
    """Tests for RecommendationExplainer class."""

    @pytest.fixture
    def explainer(self) -> RecommendationExplainer:
        """Create explainer instance."""
        return RecommendationExplainer()

    @pytest.fixture
    def user_profile(self) -> ExplicitProfile:
        """Create user profile for testing."""
        return ExplicitProfile(
            risk_tolerance="medium",
            investment_horizon=30,
            preferred_sectors=["Technology", "Healthcare"],
            watchlist=["AAPL", "GOOGL"],
        )

    def test_explainer_initialization(self, explainer: RecommendationExplainer) -> None:
        """Test explainer initializes correctly."""
        assert len(explainer.EXPLANATION_TEMPLATES) > 0
        assert "watchlist_match" in explainer.EXPLANATION_TEMPLATES
        assert "sector_interest" in explainer.EXPLANATION_TEMPLATES

    def test_fill_template_success(self, explainer: RecommendationExplainer) -> None:
        """Test template filling with valid context."""
        template = "Based on your interest in {sector} stocks"
        context = {"sector": "Technology"}

        result = explainer._fill_template(template, context)

        assert result == "Based on your interest in Technology stocks"

    def test_fill_template_missing_key(self, explainer: RecommendationExplainer) -> None:
        """Test template filling with missing context key."""
        template = "Based on your interest in {sector} and {missing}"
        context = {"sector": "Technology"}

        result = explainer._fill_template(template, context)

        # Should still fill available keys
        assert "Technology" in result

    def test_select_template_watchlist_match(
        self, explainer: RecommendationExplainer, user_profile: ExplicitProfile
    ) -> None:
        """Test template selection for watchlist match."""
        item = MagicMock()
        item.symbol = "AAPL"
        item.metadata = {}

        template_key = explainer._select_template(item, user_profile, "content_based")

        assert template_key == "watchlist_match"

    def test_select_template_sector_interest(
        self, explainer: RecommendationExplainer, user_profile: ExplicitProfile
    ) -> None:
        """Test template selection for sector interest."""
        item = MagicMock()
        item.symbol = "TSLA"
        item.metadata = {"sector": "Technology"}

        template_key = explainer._select_template(item, user_profile, "content_based")

        assert template_key == "sector_interest"

    def test_build_context(
        self, explainer: RecommendationExplainer, user_profile: ExplicitProfile
    ) -> None:
        """Test context building for template filling."""
        item = MagicMock()
        item.symbol = "AAPL"
        item.metadata = {"sector": "Technology", "confidence": 0.85}

        context = explainer._build_context(item, user_profile, "content_based")

        assert "symbol" in context
        assert context["symbol"] == "AAPL"
        assert "sector" in context
        assert context["sector"] == "Technology"
        assert "confidence" in context
        assert context["risk_level"] == "medium"
