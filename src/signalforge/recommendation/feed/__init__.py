"""Feed generation and personalization module."""

from signalforge.recommendation.feed.explainer import RecommendationExplainer
from signalforge.recommendation.feed.generator import (
    DailyDigest,
    FeedGenerator,
    FeedItem,
    PersonalizedFeed,
    WeeklySummary,
)

__all__ = [
    "FeedGenerator",
    "FeedItem",
    "PersonalizedFeed",
    "DailyDigest",
    "WeeklySummary",
    "RecommendationExplainer",
]
