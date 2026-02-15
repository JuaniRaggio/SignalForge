"""User profile management for recommendation engine."""

from signalforge.recommendation.profiles.embedding import UserEmbeddingGenerator
from signalforge.recommendation.profiles.explicit import ExplicitProfileManager
from signalforge.recommendation.profiles.implicit import ImplicitProfileTracker
from signalforge.recommendation.profiles.onboarding import OnboardingQuiz
from signalforge.recommendation.profiles.schemas import (
    EngagementPattern,
    ExplicitProfile,
    ImplicitProfile,
    InvestmentHorizon,
    NotificationPrefs,
    OnboardingAnswers,
    PortfolioPosition,
    RiskTolerance,
    UserEmbedding,
    UserType,
)

__all__ = [
    "ExplicitProfileManager",
    "ImplicitProfileTracker",
    "UserEmbeddingGenerator",
    "OnboardingQuiz",
    "ExplicitProfile",
    "ImplicitProfile",
    "UserEmbedding",
    "OnboardingAnswers",
    "PortfolioPosition",
    "NotificationPrefs",
    "EngagementPattern",
    "UserType",
    "RiskTolerance",
    "InvestmentHorizon",
]
