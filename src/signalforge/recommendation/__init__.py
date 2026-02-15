"""Recommendation engine for personalized signal recommendations.

This package provides a complete recommendation system for trading signals,
including item modeling, user profiling, ranking, and feed generation.

Key Components:
- ItemModel: Signal representation and similarity computation
- UserModel: User profile creation and management
- RankingEngine: Multi-factor signal ranking
- FeedGenerator: Personalized feed generation

Examples:
    Complete recommendation pipeline:

    >>> from signalforge.recommendation import (
    ...     ItemModel,
    ...     UserModel,
    ...     RankingEngine,
    ...     FeedGenerator,
    ...     FeedConfig,
    ... )
    >>>
    >>> # Initialize components
    >>> item_model = ItemModel()
    >>> user_model = UserModel()
    >>> ranking_engine = RankingEngine()
    >>> feed_gen = FeedGenerator(FeedConfig(), ranking_engine)
    >>>
    >>> # Create user profile
    >>> preferences = {
    ...     "risk_tolerance": "medium",
    ...     "investment_horizon": 30,
    ...     "preferred_sectors": ["Technology"],
    ...     "watchlist": ["AAPL", "GOOGL"]
    ... }
    >>> explicit = user_model.from_preferences(preferences)
    >>> implicit = user_model.from_activity([])
    >>> user_profile = user_model.create_user_profile(
    ...     "user_123",
    ...     explicit,
    ...     implicit
    ... )
    >>>
    >>> # Create signal items
    >>> signal_data = {
    ...     "symbol": "AAPL",
    ...     "sector": "Technology",
    ...     "volatility": 0.25,
    ...     "expected_return": 0.05,
    ...     "holding_period": 5,
    ...     "risk_level": "medium",
    ...     "sentiment_score": 0.7,
    ...     "regime": "bull"
    ... }
    >>> features = item_model.extract_features(signal_data)
    >>> signal_item = item_model.create_signal_item("sig_001", features)
    >>>
    >>> # Generate personalized feed
    >>> feed = feed_gen.generate_feed(user_profile, [signal_item])
"""

from signalforge.recommendation.feed_generator import FeedConfig, FeedGenerator, FeedItem
from signalforge.recommendation.item_model import (
    ItemModel,
    RegimeType,
    RiskLevel,
    SignalFeatures,
    SignalItem,
)
from signalforge.recommendation.profiles import (
    EngagementPattern,
    ExplicitProfileManager,
    ImplicitProfileTracker,
    InvestmentHorizon,
    NotificationPrefs,
    OnboardingAnswers,
    OnboardingQuiz,
    PortfolioPosition,
    UserEmbedding,
    UserEmbeddingGenerator,
    UserType,
)
from signalforge.recommendation.profiles import ExplicitProfile as ProfileExplicitProfile
from signalforge.recommendation.profiles import ImplicitProfile as ProfileImplicitProfile
from signalforge.recommendation.profiles import RiskTolerance as ProfileRiskTolerance
from signalforge.recommendation.ranking import RankedSignal, RankingConfig, RankingEngine
from signalforge.recommendation.user_model import (
    ExplicitProfile,
    ImplicitProfile,
    RiskTolerance,
    UserModel,
    UserProfile,
)

__all__ = [
    # Item Model
    "ItemModel",
    "SignalFeatures",
    "SignalItem",
    "RiskLevel",
    "RegimeType",
    # User Model
    "UserModel",
    "ExplicitProfile",
    "ImplicitProfile",
    "UserProfile",
    "RiskTolerance",
    # Ranking
    "RankingEngine",
    "RankingConfig",
    "RankedSignal",
    # Feed Generation
    "FeedGenerator",
    "FeedConfig",
    "FeedItem",
    # User Profiles
    "ExplicitProfileManager",
    "ImplicitProfileTracker",
    "UserEmbeddingGenerator",
    "OnboardingQuiz",
    "ProfileExplicitProfile",
    "ProfileImplicitProfile",
    "ProfileRiskTolerance",
    "PortfolioPosition",
    "NotificationPrefs",
    "EngagementPattern",
    "UserType",
    "InvestmentHorizon",
    "UserEmbedding",
    "OnboardingAnswers",
]
