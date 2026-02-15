"""Comprehensive tests for user profiles system."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from signalforge.recommendation.profiles import (
    EngagementPattern,
    ExplicitProfileManager,
    ImplicitProfileTracker,
    InvestmentHorizon,
    OnboardingAnswers,
    OnboardingQuiz,
    PortfolioPosition,
    RiskTolerance,
    UserEmbedding,
    UserEmbeddingGenerator,
    UserType,
)
from signalforge.recommendation.profiles import (
    ExplicitProfile as ProfileExplicitProfile,
)
from signalforge.recommendation.profiles import (
    ImplicitProfile as ProfileImplicitProfile,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


# Fixtures


@pytest.fixture
def mock_session() -> AsyncSession:
    """Create mock async session."""
    return AsyncMock(spec=["execute", "commit", "rollback", "close"])


@pytest.fixture
def explicit_manager(mock_session: AsyncSession) -> ExplicitProfileManager:
    """Create explicit profile manager."""
    return ExplicitProfileManager(mock_session)


@pytest.fixture
def implicit_tracker(mock_session: AsyncSession) -> ImplicitProfileTracker:
    """Create implicit profile tracker."""
    return ImplicitProfileTracker(mock_session)


@pytest.fixture
def embedding_generator() -> UserEmbeddingGenerator:
    """Create embedding generator."""
    return UserEmbeddingGenerator(embedding_dim=64)


@pytest.fixture
def onboarding_quiz() -> OnboardingQuiz:
    """Create onboarding quiz."""
    return OnboardingQuiz()


@pytest.fixture
def sample_portfolio() -> list[PortfolioPosition]:
    """Create sample portfolio."""
    return [
        PortfolioPosition(
            symbol="AAPL",
            shares=100.0,
            cost_basis=150.0,
            acquired_at=datetime.utcnow() - timedelta(days=30),
        ),
        PortfolioPosition(
            symbol="GOOGL",
            shares=50.0,
            cost_basis=2800.0,
            acquired_at=datetime.utcnow() - timedelta(days=60),
        ),
        PortfolioPosition(
            symbol="MSFT",
            shares=75.0,
            cost_basis=300.0,
            acquired_at=datetime.utcnow() - timedelta(days=15),
        ),
    ]


@pytest.fixture
def sample_explicit_profile(
    sample_portfolio: list[PortfolioPosition],
) -> ProfileExplicitProfile:
    """Create sample explicit profile."""
    return ProfileExplicitProfile(
        user_id="user_123",
        portfolio=sample_portfolio,
        watchlist=["TSLA", "NVDA", "AMD"],
        preferred_sectors=["Technology", "Healthcare"],
        risk_tolerance=RiskTolerance.MODERATE,
        investment_horizon=InvestmentHorizon.POSITION,
    )


@pytest.fixture
def sample_implicit_profile() -> ProfileImplicitProfile:
    """Create sample implicit profile."""
    now = datetime.utcnow()
    return ProfileImplicitProfile(
        user_id="user_123",
        viewed_symbols=[
            ("AAPL", now - timedelta(hours=2), 45.0),
            ("GOOGL", now - timedelta(hours=1), 30.0),
            ("TSLA", now - timedelta(minutes=30), 60.0),
        ],
        clicked_signals=[
            ("sig_001", now - timedelta(hours=3)),
            ("sig_002", now - timedelta(hours=1)),
        ],
        search_history=[
            ("technology stocks", now - timedelta(hours=4)),
            ("AAPL analysis", now - timedelta(hours=2)),
        ],
        sector_affinity={"Technology": 0.8, "Healthcare": 0.5},
    )


# Explicit Profile Manager Tests


class TestExplicitProfileManager:
    """Test suite for ExplicitProfileManager."""

    @pytest.mark.asyncio
    async def test_create_profile(
        self,
        explicit_manager: ExplicitProfileManager,
        sample_explicit_profile: ProfileExplicitProfile,
    ) -> None:
        """Test creating a new profile."""
        profile = await explicit_manager.create_profile(
            "user_123", sample_explicit_profile
        )

        assert profile.user_id == "user_123"
        assert len(profile.portfolio) == 3
        assert len(profile.watchlist) == 3
        assert profile.risk_tolerance == RiskTolerance.MODERATE

    @pytest.mark.asyncio
    async def test_create_duplicate_profile(
        self,
        explicit_manager: ExplicitProfileManager,
        sample_explicit_profile: ProfileExplicitProfile,
    ) -> None:
        """Test creating duplicate profile raises error."""
        await explicit_manager.create_profile("user_123", sample_explicit_profile)

        with pytest.raises(ValueError, match="Profile already exists"):
            await explicit_manager.create_profile("user_123", sample_explicit_profile)

    @pytest.mark.asyncio
    async def test_get_profile(
        self,
        explicit_manager: ExplicitProfileManager,
        sample_explicit_profile: ProfileExplicitProfile,
    ) -> None:
        """Test retrieving a profile."""
        await explicit_manager.create_profile("user_123", sample_explicit_profile)
        profile = await explicit_manager.get_profile("user_123")

        assert profile is not None
        assert profile.user_id == "user_123"

    @pytest.mark.asyncio
    async def test_get_nonexistent_profile(
        self, explicit_manager: ExplicitProfileManager
    ) -> None:
        """Test getting non-existent profile returns None."""
        profile = await explicit_manager.get_profile("nonexistent")
        assert profile is None

    @pytest.mark.asyncio
    async def test_update_profile(
        self,
        explicit_manager: ExplicitProfileManager,
        sample_explicit_profile: ProfileExplicitProfile,
    ) -> None:
        """Test updating profile fields."""
        await explicit_manager.create_profile("user_123", sample_explicit_profile)

        updates: dict[str, object] = {
            "risk_tolerance": RiskTolerance.AGGRESSIVE,
            "investment_horizon": InvestmentHorizon.DAY_TRADE,
        }
        updated = await explicit_manager.update_profile("user_123", updates)

        assert updated.risk_tolerance == RiskTolerance.AGGRESSIVE
        assert updated.investment_horizon == InvestmentHorizon.DAY_TRADE

    @pytest.mark.asyncio
    async def test_update_nonexistent_profile(
        self, explicit_manager: ExplicitProfileManager
    ) -> None:
        """Test updating non-existent profile raises error."""
        with pytest.raises(ValueError, match="Profile not found"):
            await explicit_manager.update_profile("nonexistent", {})

    @pytest.mark.asyncio
    async def test_add_to_portfolio(
        self,
        explicit_manager: ExplicitProfileManager,
        sample_explicit_profile: ProfileExplicitProfile,
    ) -> None:
        """Test adding position to portfolio."""
        await explicit_manager.create_profile("user_123", sample_explicit_profile)

        new_position = PortfolioPosition(
            symbol="AMZN",
            shares=25.0,
            cost_basis=3200.0,
            acquired_at=datetime.utcnow(),
        )
        await explicit_manager.add_to_portfolio("user_123", new_position)

        profile = await explicit_manager.get_profile("user_123")
        assert profile is not None
        assert len(profile.portfolio) == 4
        assert any(p.symbol == "AMZN" for p in profile.portfolio)

    @pytest.mark.asyncio
    async def test_update_existing_portfolio_position(
        self,
        explicit_manager: ExplicitProfileManager,
        sample_explicit_profile: ProfileExplicitProfile,
    ) -> None:
        """Test updating existing portfolio position."""
        await explicit_manager.create_profile("user_123", sample_explicit_profile)

        updated_position = PortfolioPosition(
            symbol="AAPL",
            shares=150.0,
            cost_basis=160.0,
            acquired_at=datetime.utcnow(),
        )
        await explicit_manager.add_to_portfolio("user_123", updated_position)

        profile = await explicit_manager.get_profile("user_123")
        assert profile is not None
        assert len(profile.portfolio) == 3  # No new position added
        aapl_position = next(p for p in profile.portfolio if p.symbol == "AAPL")
        assert aapl_position.shares == 150.0
        assert aapl_position.cost_basis == 160.0

    @pytest.mark.asyncio
    async def test_remove_from_portfolio(
        self,
        explicit_manager: ExplicitProfileManager,
        sample_explicit_profile: ProfileExplicitProfile,
    ) -> None:
        """Test removing position from portfolio."""
        await explicit_manager.create_profile("user_123", sample_explicit_profile)
        await explicit_manager.remove_from_portfolio("user_123", "AAPL")

        profile = await explicit_manager.get_profile("user_123")
        assert profile is not None
        assert len(profile.portfolio) == 2
        assert not any(p.symbol == "AAPL" for p in profile.portfolio)

    @pytest.mark.asyncio
    async def test_update_watchlist(
        self,
        explicit_manager: ExplicitProfileManager,
        sample_explicit_profile: ProfileExplicitProfile,
    ) -> None:
        """Test updating watchlist."""
        await explicit_manager.create_profile("user_123", sample_explicit_profile)
        new_watchlist = ["AAPL", "GOOGL", "MSFT", "AMZN"]
        await explicit_manager.update_watchlist("user_123", new_watchlist)

        profile = await explicit_manager.get_profile("user_123")
        assert profile is not None
        assert profile.watchlist == new_watchlist

    @pytest.mark.asyncio
    async def test_get_portfolio_sectors(
        self,
        explicit_manager: ExplicitProfileManager,
        sample_explicit_profile: ProfileExplicitProfile,
    ) -> None:
        """Test calculating portfolio sector allocation."""
        await explicit_manager.create_profile("user_123", sample_explicit_profile)
        sectors = await explicit_manager.get_portfolio_sectors("user_123")

        assert isinstance(sectors, dict)
        assert len(sectors) > 0

    @pytest.mark.asyncio
    async def test_get_portfolio_sectors_empty_portfolio(
        self, explicit_manager: ExplicitProfileManager
    ) -> None:
        """Test sector allocation with empty portfolio."""
        empty_profile = ProfileExplicitProfile(user_id="user_456")
        await explicit_manager.create_profile("user_456", empty_profile)

        sectors = await explicit_manager.get_portfolio_sectors("user_456")
        assert sectors == {}


# Implicit Profile Tracker Tests


class TestImplicitProfileTracker:
    """Test suite for ImplicitProfileTracker."""

    @pytest.mark.asyncio
    async def test_track_view(
        self, implicit_tracker: ImplicitProfileTracker
    ) -> None:
        """Test tracking symbol view."""
        await implicit_tracker.track_view("user_123", "AAPL", 45.0)

        profile = await implicit_tracker.get_implicit_profile("user_123")
        assert len(profile.viewed_symbols) == 1
        assert profile.viewed_symbols[0][0] == "AAPL"
        assert profile.viewed_symbols[0][2] == 45.0

    @pytest.mark.asyncio
    async def test_track_multiple_views(
        self, implicit_tracker: ImplicitProfileTracker
    ) -> None:
        """Test tracking multiple views."""
        await implicit_tracker.track_view("user_123", "AAPL", 30.0)
        await implicit_tracker.track_view("user_123", "GOOGL", 45.0)
        await implicit_tracker.track_view("user_123", "MSFT", 60.0)

        profile = await implicit_tracker.get_implicit_profile("user_123")
        assert len(profile.viewed_symbols) == 3

    @pytest.mark.asyncio
    async def test_track_click(
        self, implicit_tracker: ImplicitProfileTracker
    ) -> None:
        """Test tracking signal click."""
        await implicit_tracker.track_click("user_123", "sig_001")

        profile = await implicit_tracker.get_implicit_profile("user_123")
        assert len(profile.clicked_signals) == 1
        assert profile.clicked_signals[0][0] == "sig_001"

    @pytest.mark.asyncio
    async def test_track_search(
        self, implicit_tracker: ImplicitProfileTracker
    ) -> None:
        """Test tracking search query."""
        await implicit_tracker.track_search("user_123", "technology stocks")

        profile = await implicit_tracker.get_implicit_profile("user_123")
        assert len(profile.search_history) == 1
        assert profile.search_history[0][0] == "technology stocks"

    @pytest.mark.asyncio
    async def test_calculate_sector_affinity(
        self, implicit_tracker: ImplicitProfileTracker
    ) -> None:
        """Test calculating sector affinity."""
        await implicit_tracker.track_view("user_123", "AAPL", 60.0)
        await implicit_tracker.track_view("user_123", "GOOGL", 45.0)
        await implicit_tracker.track_view("user_123", "MSFT", 30.0)

        affinity = await implicit_tracker.calculate_sector_affinity("user_123")
        assert isinstance(affinity, dict)
        assert "Technology" in affinity
        assert 0.0 <= affinity["Technology"] <= 1.0

    @pytest.mark.asyncio
    async def test_calculate_sector_affinity_empty(
        self, implicit_tracker: ImplicitProfileTracker
    ) -> None:
        """Test sector affinity with no data."""
        affinity = await implicit_tracker.calculate_sector_affinity("user_456")
        assert affinity == {}

    @pytest.mark.asyncio
    async def test_calculate_engagement_pattern(
        self, implicit_tracker: ImplicitProfileTracker
    ) -> None:
        """Test calculating engagement pattern."""
        # Add various interactions
        await implicit_tracker.track_view("user_123", "AAPL", 45.0)
        await implicit_tracker.track_click("user_123", "sig_001")
        await implicit_tracker.track_search("user_123", "tech stocks")

        pattern = await implicit_tracker.calculate_engagement_pattern("user_123")
        assert isinstance(pattern, EngagementPattern)
        assert pattern.sessions_per_week >= 0
        assert isinstance(pattern.preferred_hours, list)
        assert isinstance(pattern.preferred_days, list)

    @pytest.mark.asyncio
    async def test_calculate_engagement_pattern_empty(
        self, implicit_tracker: ImplicitProfileTracker
    ) -> None:
        """Test engagement pattern with no data."""
        pattern = await implicit_tracker.calculate_engagement_pattern("user_456")
        assert pattern.avg_session_duration == 0.0
        assert pattern.sessions_per_week == 0.0

    @pytest.mark.asyncio
    async def test_get_implicit_profile_new_user(
        self, implicit_tracker: ImplicitProfileTracker
    ) -> None:
        """Test getting profile for new user."""
        profile = await implicit_tracker.get_implicit_profile("new_user")
        assert profile.user_id == "new_user"
        assert len(profile.viewed_symbols) == 0
        assert len(profile.clicked_signals) == 0


# User Embedding Generator Tests


class TestUserEmbeddingGenerator:
    """Test suite for UserEmbeddingGenerator."""

    def test_initialize_generator(
        self, embedding_generator: UserEmbeddingGenerator
    ) -> None:
        """Test initializing embedding generator."""
        assert embedding_generator._embedding_dim == 64

    def test_generate_embedding(
        self,
        embedding_generator: UserEmbeddingGenerator,
        sample_explicit_profile: ProfileExplicitProfile,
        sample_implicit_profile: ProfileImplicitProfile,
    ) -> None:
        """Test generating user embedding."""
        embedding = embedding_generator.generate_embedding(
            sample_explicit_profile, sample_implicit_profile
        )

        assert isinstance(embedding, UserEmbedding)
        assert embedding.user_id == "user_123"
        assert len(embedding.embedding) == 64
        assert embedding.version == 1

    def test_embedding_normalized(
        self,
        embedding_generator: UserEmbeddingGenerator,
        sample_explicit_profile: ProfileExplicitProfile,
        sample_implicit_profile: ProfileImplicitProfile,
    ) -> None:
        """Test that embedding is normalized."""
        embedding = embedding_generator.generate_embedding(
            sample_explicit_profile, sample_implicit_profile
        )

        magnitude = math.sqrt(sum(x * x for x in embedding.embedding))
        assert abs(magnitude - 1.0) < 0.0001  # Should be unit vector

    def test_compute_similarity_identical(
        self,
        embedding_generator: UserEmbeddingGenerator,
        sample_explicit_profile: ProfileExplicitProfile,
        sample_implicit_profile: ProfileImplicitProfile,
    ) -> None:
        """Test similarity between identical embeddings."""
        emb1 = embedding_generator.generate_embedding(
            sample_explicit_profile, sample_implicit_profile
        )
        emb2 = embedding_generator.generate_embedding(
            sample_explicit_profile, sample_implicit_profile
        )

        similarity = embedding_generator.compute_similarity(emb1, emb2)
        assert abs(similarity - 1.0) < 0.0001  # Should be 1.0

    def test_compute_similarity_different(
        self,
        embedding_generator: UserEmbeddingGenerator,
        sample_explicit_profile: ProfileExplicitProfile,
        sample_implicit_profile: ProfileImplicitProfile,
    ) -> None:
        """Test similarity between different embeddings."""
        emb1 = embedding_generator.generate_embedding(
            sample_explicit_profile, sample_implicit_profile
        )

        # Create different profile
        different_profile = ProfileExplicitProfile(
            user_id="user_456",
            risk_tolerance=RiskTolerance.AGGRESSIVE,
            investment_horizon=InvestmentHorizon.DAY_TRADE,
        )
        emb2 = embedding_generator.generate_embedding(
            different_profile, sample_implicit_profile
        )

        similarity = embedding_generator.compute_similarity(emb1, emb2)
        assert 0.0 <= similarity <= 1.0

    def test_compute_similarity_mismatched_dimensions(
        self, embedding_generator: UserEmbeddingGenerator
    ) -> None:
        """Test similarity with mismatched dimensions."""
        emb1 = UserEmbedding(user_id="user1", embedding=[0.5] * 64)
        emb2 = UserEmbedding(user_id="user2", embedding=[0.5] * 32)

        with pytest.raises(ValueError, match="same dimension"):
            embedding_generator.compute_similarity(emb1, emb2)

    def test_find_similar_users(
        self,
        embedding_generator: UserEmbeddingGenerator,
        sample_explicit_profile: ProfileExplicitProfile,
        sample_implicit_profile: ProfileImplicitProfile,
    ) -> None:
        """Test finding similar users."""
        target = embedding_generator.generate_embedding(
            sample_explicit_profile, sample_implicit_profile
        )

        # Create candidate profiles
        candidates = []
        for i in range(5):
            profile = ProfileExplicitProfile(
                user_id=f"user_{i}",
                risk_tolerance=RiskTolerance.MODERATE,
            )
            emb = embedding_generator.generate_embedding(
                profile, sample_implicit_profile
            )
            candidates.append(emb)

        similar = embedding_generator.find_similar_users(target, candidates, top_k=3)

        assert len(similar) <= 3
        assert all(isinstance(user_id, str) for user_id, _ in similar)
        assert all(isinstance(score, float) for _, score in similar)

    def test_find_similar_users_excludes_self(
        self,
        embedding_generator: UserEmbeddingGenerator,
        sample_explicit_profile: ProfileExplicitProfile,
        sample_implicit_profile: ProfileImplicitProfile,
    ) -> None:
        """Test that find_similar_users excludes target user."""
        target = embedding_generator.generate_embedding(
            sample_explicit_profile, sample_implicit_profile
        )

        # Include target in candidates
        candidates = [target]
        for i in range(3):
            profile = ProfileExplicitProfile(user_id=f"user_{i}")
            emb = embedding_generator.generate_embedding(
                profile, sample_implicit_profile
            )
            candidates.append(emb)

        similar = embedding_generator.find_similar_users(target, candidates, top_k=5)

        # Should not include target
        assert target.user_id not in [user_id for user_id, _ in similar]


# Onboarding Quiz Tests


class TestOnboardingQuiz:
    """Test suite for OnboardingQuiz."""

    def test_get_questions(self, onboarding_quiz: OnboardingQuiz) -> None:
        """Test getting quiz questions."""
        questions = onboarding_quiz.get_questions()
        assert len(questions) == 5
        assert all("id" in q for q in questions)
        assert all("question" in q for q in questions)
        assert all("options" in q for q in questions)

    def test_process_answers_casual(self, onboarding_quiz: OnboardingQuiz) -> None:
        """Test classifying casual user."""
        answers = OnboardingAnswers(
            time_following_markets="rarely",
            recent_trading_activity="none",
            financial_knowledge="beginner",
            preference_style="guidance",
            api_interest=False,
        )

        user_type = onboarding_quiz.process_answers(answers)
        assert user_type == UserType.CASUAL

    def test_process_answers_active(self, onboarding_quiz: OnboardingQuiz) -> None:
        """Test classifying active user."""
        # Score: constantly(3) + occasional(2) + intermediate(2) = 7 >= 7 -> ACTIVE
        answers = OnboardingAnswers(
            time_following_markets="constantly",
            recent_trading_activity="occasional",
            financial_knowledge="intermediate",
            preference_style="both",
            api_interest=False,
        )

        user_type = onboarding_quiz.process_answers(answers)
        assert user_type == UserType.ACTIVE

    def test_process_answers_professional(
        self, onboarding_quiz: OnboardingQuiz
    ) -> None:
        """Test classifying professional user."""
        answers = OnboardingAnswers(
            time_following_markets="constantly",
            recent_trading_activity="frequent",
            financial_knowledge="advanced",
            preference_style="raw_data",
            api_interest=True,
        )

        user_type = onboarding_quiz.process_answers(answers)
        assert user_type == UserType.PROFESSIONAL

    def test_process_answers_api_interest(
        self, onboarding_quiz: OnboardingQuiz
    ) -> None:
        """Test API interest leads to professional classification."""
        answers = OnboardingAnswers(
            time_following_markets="daily",
            recent_trading_activity="occasional",
            financial_knowledge="advanced",
            preference_style="raw_data",
            api_interest=True,
        )

        user_type = onboarding_quiz.process_answers(answers)
        assert user_type == UserType.PROFESSIONAL

    def test_generate_initial_profile(
        self, onboarding_quiz: OnboardingQuiz
    ) -> None:
        """Test generating initial profile."""
        answers = OnboardingAnswers(
            time_following_markets="daily",
            recent_trading_activity="occasional",
            financial_knowledge="intermediate",
            preference_style="both",
            api_interest=False,
        )

        profile = onboarding_quiz.generate_initial_profile("user_123", answers)

        assert profile.user_id == "user_123"
        assert isinstance(profile.risk_tolerance, RiskTolerance)
        assert isinstance(profile.investment_horizon, InvestmentHorizon)
        assert len(profile.preferred_sectors) > 0

    def test_suggest_sectors_casual(self, onboarding_quiz: OnboardingQuiz) -> None:
        """Test sector suggestions for casual user."""
        sectors = onboarding_quiz.suggest_sectors(UserType.CASUAL)
        assert len(sectors) == 3
        assert "Technology" in sectors

    def test_suggest_sectors_professional(
        self, onboarding_quiz: OnboardingQuiz
    ) -> None:
        """Test sector suggestions for professional user."""
        sectors = onboarding_quiz.suggest_sectors(UserType.PROFESSIONAL)
        assert len(sectors) >= 6
        assert "Technology" in sectors

    def test_suggest_risk_tolerance_beginner(
        self, onboarding_quiz: OnboardingQuiz
    ) -> None:
        """Test risk tolerance for beginner."""
        answers = OnboardingAnswers(
            time_following_markets="rarely",
            recent_trading_activity="none",
            financial_knowledge="beginner",
            preference_style="guidance",
            api_interest=False,
        )

        risk = onboarding_quiz.suggest_risk_tolerance(answers)
        assert risk == RiskTolerance.CONSERVATIVE

    def test_suggest_risk_tolerance_aggressive(
        self, onboarding_quiz: OnboardingQuiz
    ) -> None:
        """Test risk tolerance for aggressive trader."""
        answers = OnboardingAnswers(
            time_following_markets="constantly",
            recent_trading_activity="frequent",
            financial_knowledge="advanced",
            preference_style="raw_data",
            api_interest=True,
        )

        risk = onboarding_quiz.suggest_risk_tolerance(answers)
        assert risk == RiskTolerance.AGGRESSIVE

    def test_suggest_risk_tolerance_moderate(
        self, onboarding_quiz: OnboardingQuiz
    ) -> None:
        """Test risk tolerance for moderate trader."""
        answers = OnboardingAnswers(
            time_following_markets="daily",
            recent_trading_activity="occasional",
            financial_knowledge="intermediate",
            preference_style="both",
            api_interest=False,
        )

        risk = onboarding_quiz.suggest_risk_tolerance(answers)
        assert risk == RiskTolerance.MODERATE


# Integration Tests


class TestUserProfilesIntegration:
    """Integration tests for user profiles system."""

    @pytest.mark.asyncio
    async def test_complete_user_journey(
        self,
        mock_session: AsyncSession,
    ) -> None:
        """Test complete user journey from onboarding to embedding."""
        # Step 1: Onboarding
        quiz = OnboardingQuiz()
        answers = OnboardingAnswers(
            time_following_markets="daily",
            recent_trading_activity="occasional",
            financial_knowledge="intermediate",
            preference_style="both",
            api_interest=False,
        )

        explicit_profile = quiz.generate_initial_profile("user_123", answers)
        assert explicit_profile.user_id == "user_123"

        # Step 2: Create profile
        manager = ExplicitProfileManager(mock_session)
        created = await manager.create_profile("user_123", explicit_profile)
        assert created.user_id == "user_123"

        # Step 3: Track behavior
        tracker = ImplicitProfileTracker(mock_session)
        await tracker.track_view("user_123", "AAPL", 45.0)
        await tracker.track_click("user_123", "sig_001")

        implicit_profile = await tracker.get_implicit_profile("user_123")
        assert len(implicit_profile.viewed_symbols) > 0

        # Step 4: Generate embedding
        generator = UserEmbeddingGenerator()
        embedding = generator.generate_embedding(created, implicit_profile)
        assert len(embedding.embedding) == 64

    @pytest.mark.asyncio
    async def test_portfolio_management_workflow(
        self,
        explicit_manager: ExplicitProfileManager,
        sample_explicit_profile: ProfileExplicitProfile,
    ) -> None:
        """Test complete portfolio management workflow."""
        # Create profile
        await explicit_manager.create_profile("user_123", sample_explicit_profile)

        # Add position
        new_pos = PortfolioPosition(
            symbol="TSLA",
            shares=50.0,
            cost_basis=700.0,
            acquired_at=datetime.utcnow(),
        )
        await explicit_manager.add_to_portfolio("user_123", new_pos)

        # Update watchlist
        await explicit_manager.update_watchlist("user_123", ["AAPL", "GOOGL"])

        # Get portfolio sectors
        sectors = await explicit_manager.get_portfolio_sectors("user_123")

        profile = await explicit_manager.get_profile("user_123")
        assert profile is not None
        assert len(profile.portfolio) == 4
        assert len(profile.watchlist) == 2
        assert isinstance(sectors, dict)

    @pytest.mark.asyncio
    async def test_user_similarity_matching(
        self,
        embedding_generator: UserEmbeddingGenerator,
        sample_implicit_profile: ProfileImplicitProfile,
    ) -> None:
        """Test finding similar users based on embeddings."""
        # Create multiple user profiles
        users = []
        for i in range(10):
            profile = ProfileExplicitProfile(
                user_id=f"user_{i}",
                risk_tolerance=(
                    RiskTolerance.AGGRESSIVE
                    if i % 3 == 0
                    else RiskTolerance.MODERATE
                ),
                investment_horizon=(
                    InvestmentHorizon.DAY_TRADE
                    if i % 2 == 0
                    else InvestmentHorizon.POSITION
                ),
            )
            emb = embedding_generator.generate_embedding(
                profile, sample_implicit_profile
            )
            users.append(emb)

        # Find similar users to first user
        similar = embedding_generator.find_similar_users(users[0], users[1:], top_k=5)

        assert len(similar) <= 5
        assert all(0.0 <= score <= 1.0 for _, score in similar)
        # Scores should be in descending order
        scores = [score for _, score in similar]
        assert scores == sorted(scores, reverse=True)
