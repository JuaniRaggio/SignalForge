"""Recommendation algorithms for SignalForge.

This module provides various recommendation algorithms for generating
personalized trading signal and stock recommendations.

Available algorithms:
- CollaborativeFilteringRecommender: User-based and item-based collaborative filtering
- ContentBasedRecommender: Content-based filtering using item features
- KnowledgeBasedRecommender: Rule-based recommendations using domain knowledge
- HybridRecommender: Ensemble combining multiple algorithms with adaptive weighting

Bandit algorithms:
- ContextualBandit: UCB-based adaptive algorithm selection
- ThompsonSamplingBandit: Bayesian algorithm selection

Feedback and filtering:
- FeedbackProcessor: Track and process user feedback
- AntiHerdingFilter: Prevent herding behavior in recommendations

Each algorithm implements the BaseRecommender interface for consistency.

Examples:
    Using collaborative filtering:

    >>> from signalforge.recommendation.algorithms import CollaborativeFilteringRecommender
    >>> recommender = CollaborativeFilteringRecommender(method="user_based")
    >>> await recommender.train(interaction_data)
    >>> recommendations = await recommender.recommend(request, user_profile)

    Using content-based filtering:

    >>> from signalforge.recommendation.algorithms import ContentBasedRecommender
    >>> recommender = ContentBasedRecommender()
    >>> await recommender.train(item_features)
    >>> recommendations = await recommender.recommend(request, user_profile)

    Using knowledge-based recommendations:

    >>> from signalforge.recommendation.algorithms import KnowledgeBasedRecommender
    >>> recommender = KnowledgeBasedRecommender()
    >>> recommendations = await recommender.recommend(request, user_profile)

    Using hybrid recommender:

    >>> from signalforge.recommendation.algorithms import HybridRecommender
    >>> recommenders = [
    ...     CollaborativeFilteringRecommender(),
    ...     ContentBasedRecommender(),
    ...     KnowledgeBasedRecommender(),
    ... ]
    >>> hybrid = HybridRecommender(recommenders, use_bandit=True)
    >>> recommendations = await hybrid.recommend(request, user_profile)
"""

from signalforge.recommendation.algorithms.bandit import (
    ContextualBandit,
    ThompsonSamplingBandit,
)
from signalforge.recommendation.algorithms.base import BaseRecommender
from signalforge.recommendation.algorithms.collaborative import (
    CollaborativeFilteringRecommender,
)
from signalforge.recommendation.algorithms.content_based import ContentBasedRecommender
from signalforge.recommendation.algorithms.feedback import (
    AntiHerdingFilter,
    FeedbackProcessor,
)
from signalforge.recommendation.algorithms.hybrid import HybridRecommender
from signalforge.recommendation.algorithms.knowledge_based import (
    KnowledgeBasedRecommender,
    RecommendationRule,
)
from signalforge.recommendation.algorithms.schemas import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)

__all__ = [
    "BaseRecommender",
    "CollaborativeFilteringRecommender",
    "ContentBasedRecommender",
    "KnowledgeBasedRecommender",
    "HybridRecommender",
    "RecommendationItem",
    "RecommendationRequest",
    "RecommendationResponse",
    "RecommendationRule",
    "ContextualBandit",
    "ThompsonSamplingBandit",
    "FeedbackProcessor",
    "AntiHerdingFilter",
]
