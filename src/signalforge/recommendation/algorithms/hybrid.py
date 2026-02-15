"""Hybrid recommendation algorithm combining multiple strategies.

This module implements a hybrid recommender that combines multiple recommendation
algorithms using ensemble techniques. It supports:
- Weighted combination of different algorithms
- Adaptive weight selection using contextual bandits
- Dynamic algorithm selection based on user context
- Deduplication and re-ranking of merged recommendations

The hybrid approach provides:
- Better coverage than individual algorithms
- Adaptive learning from user feedback
- Balance between exploration and exploitation
- Robustness to algorithm-specific biases

Examples:
    Creating a hybrid recommender:

    >>> from signalforge.recommendation.algorithms import (
    ...     CollaborativeFilteringRecommender,
    ...     ContentBasedRecommender,
    ...     KnowledgeBasedRecommender,
    ... )
    >>> from signalforge.recommendation.algorithms.hybrid import HybridRecommender
    >>>
    >>> recommenders = [
    ...     CollaborativeFilteringRecommender(),
    ...     ContentBasedRecommender(),
    ...     KnowledgeBasedRecommender(),
    ... ]
    >>> hybrid = HybridRecommender(recommenders, use_bandit=True)
    >>>
    >>> # Generate recommendations
    >>> recommendations = await hybrid.recommend(request, user_profile)
    >>>
    >>> # Update weights based on feedback
    >>> hybrid.update_weights_from_feedback(
    ...     user_id="user_123",
    ...     item_id="AAPL",
    ...     reward=1.0,
    ...     context={"time_of_day": "morning", "market_regime": "bull"},
    ... )
"""

from __future__ import annotations

import polars as pl

from signalforge.core.logging import get_logger
from signalforge.recommendation.algorithms.bandit import ContextualBandit
from signalforge.recommendation.algorithms.base import BaseRecommender
from signalforge.recommendation.algorithms.schemas import (
    RecommendationItem,
    RecommendationRequest,
)
from signalforge.recommendation.user_model import ExplicitProfile, ImplicitProfile

logger = get_logger(__name__)


class HybridRecommender(BaseRecommender):
    """Hybrid ensemble combining multiple recommendation strategies.

    This recommender merges recommendations from multiple algorithms using
    weighted aggregation. Weights can be static or dynamically adjusted
    using contextual bandits based on user feedback.

    Attributes:
        algorithm_name: Name of the algorithm.
        recommenders: List of component recommenders.
        weights: Weights for each recommender.
        use_bandit: Whether to use contextual bandit for adaptive weighting.
        bandit: Contextual bandit instance if enabled.
    """

    algorithm_name = "hybrid"

    def __init__(
        self,
        recommenders: list[BaseRecommender],
        weights: list[float] | None = None,
        use_bandit: bool = True,
    ):
        """Initialize the hybrid recommender.

        Args:
            recommenders: List of recommender instances to combine.
            weights: Optional static weights for each recommender.
                If None, equal weights are assigned.
            use_bandit: Whether to use contextual bandit for adaptive weighting.

        Raises:
            ValueError: If recommenders list is empty or weights are invalid.
        """
        if not recommenders:
            raise ValueError("At least one recommender is required")

        if weights is not None:
            if len(weights) != len(recommenders):
                raise ValueError(
                    f"Number of weights ({len(weights)}) must match "
                    f"number of recommenders ({len(recommenders)})"
                )
            if not all(w >= 0.0 for w in weights):
                raise ValueError("All weights must be non-negative")

            # Normalize weights
            total = sum(weights)
            if total == 0.0:
                raise ValueError("At least one weight must be positive")
            self.weights = [w / total for w in weights]
        else:
            # Equal weights by default
            self.weights = [1.0 / len(recommenders)] * len(recommenders)

        self.recommenders = recommenders
        self.use_bandit = use_bandit
        self.bandit = ContextualBandit(n_arms=len(recommenders)) if use_bandit else None

        # Track which item came from which algorithm for feedback
        self._item_to_algorithm: dict[str, int] = {}

        logger.info(
            "hybrid_recommender_initialized",
            num_recommenders=len(recommenders),
            use_bandit=use_bandit,
            weights=self.weights,
        )

    async def recommend(
        self,
        request: RecommendationRequest,
        user_profile: ExplicitProfile,
        implicit_profile: ImplicitProfile | None = None,
    ) -> list[RecommendationItem]:
        """Combine recommendations from all strategies.

        The method:
        1. Gets recommendations from each component algorithm
        2. Uses bandit to select weights if enabled, otherwise uses static weights
        3. Merges and deduplicates recommendations
        4. Re-ranks by combined weighted score
        5. Returns top-k items

        Args:
            request: Recommendation request.
            user_profile: Explicit user preferences.
            implicit_profile: Optional implicit user behavior.

        Returns:
            List of recommendation items sorted by combined score.

        Raises:
            ValueError: If request or profiles are invalid.
        """
        logger.debug(
            "generating_hybrid_recommendations",
            user_id=request.user_id,
            num_recommenders=len(self.recommenders),
            use_bandit=self.use_bandit,
        )

        # Prepare context for bandit
        context = self._extract_context(request, user_profile, implicit_profile)

        # Get recommendations from each algorithm
        all_recommendations: list[list[RecommendationItem]] = []

        for _idx, recommender in enumerate(self.recommenders):
            try:
                recs = await recommender.recommend(request, user_profile, implicit_profile)
                all_recommendations.append(recs)
                logger.debug(
                    "algorithm_recommendations_received",
                    algorithm=recommender.algorithm_name,
                    num_items=len(recs),
                )
            except Exception as e:
                logger.warning(
                    "algorithm_recommendation_failed",
                    algorithm=recommender.algorithm_name,
                    error=str(e),
                )
                # Continue with empty recommendations for this algorithm
                all_recommendations.append([])

        # Determine weights to use
        if self.use_bandit and self.bandit is not None:
            weights = self.bandit.get_weights(context)
            logger.debug("using_bandit_weights", weights=weights)
        else:
            weights = self.weights
            logger.debug("using_static_weights", weights=weights)

        # Merge recommendations with weighted scores
        merged = self._merge_recommendations(all_recommendations, weights)

        # Deduplicate keeping highest score
        deduplicated = self._deduplicate(merged)

        # Apply any additional filtering from request
        filtered = deduplicated
        if request.exclude_seen and hasattr(request, "seen_items"):
            seen_ids: set[str] = getattr(request, "seen_items", set())
            if isinstance(seen_ids, set):
                filtered = self.filter_seen(filtered, seen_ids)

        # Apply diversity if needed
        if len(filtered) > request.limit:
            # Apply some diversity to avoid all items from one algorithm
            filtered = self.apply_diversity(filtered, diversity_factor=0.2)

        # Return top-k
        result = filtered[: request.limit]

        logger.info(
            "hybrid_recommendations_generated",
            user_id=request.user_id,
            num_recommendations=len(result),
            avg_score=sum(item.score for item in result) / len(result) if result else 0.0,
        )

        return result

    async def train(self, interaction_data: pl.DataFrame) -> None:
        """Train all component recommenders.

        Args:
            interaction_data: DataFrame containing user interaction history.
                Expected columns: user_id, item_id, rating, timestamp

        Raises:
            ValueError: If interaction_data has invalid schema.
        """
        logger.info("training_hybrid_recommender", num_rows=len(interaction_data))

        # Train each component recommender
        for recommender in self.recommenders:
            try:
                await recommender.train(interaction_data)
                logger.debug(
                    "component_recommender_trained",
                    algorithm=recommender.algorithm_name,
                )
            except Exception as e:
                logger.warning(
                    "component_recommender_training_failed",
                    algorithm=recommender.algorithm_name,
                    error=str(e),
                )

        logger.info("hybrid_recommender_training_completed")

    def _merge_recommendations(
        self,
        recommendations: list[list[RecommendationItem]],
        weights: list[float],
    ) -> list[RecommendationItem]:
        """Merge recommendations with weighted scores.

        Args:
            recommendations: List of recommendation lists from each algorithm.
            weights: Weight for each algorithm.

        Returns:
            Merged list of recommendation items with weighted scores.
        """
        # Build a map of item_id to list of (score, weight, algorithm_idx)
        item_scores: dict[str, list[tuple[float, float, int]]] = {}

        for algorithm_idx, (recs, weight) in enumerate(zip(recommendations, weights, strict=True)):
            for item in recs:
                if item.item_id not in item_scores:
                    item_scores[item.item_id] = []
                item_scores[item.item_id].append((item.score, weight, algorithm_idx))

        # Create merged items with weighted average scores
        merged_items: list[RecommendationItem] = []

        for item_id, scores in item_scores.items():
            # Weighted average of scores
            weighted_score = sum(score * weight for score, weight, _ in scores)
            total_weight = sum(weight for _, weight, _ in scores)

            final_score = weighted_score / total_weight if total_weight > 0 else 0.0

            # Find the item with highest individual score to use as template
            best_idx = max(range(len(scores)), key=lambda i: scores[i][0])
            algorithm_idx = scores[best_idx][2]

            # Find the original item
            original_item = None
            for item in recommendations[algorithm_idx]:
                if item.item_id == item_id:
                    original_item = item
                    break

            if original_item is not None:
                # Track which algorithm produced this item
                self._item_to_algorithm[item_id] = algorithm_idx

                # Create merged item
                merged_item = RecommendationItem(
                    item_id=item_id,
                    item_type=original_item.item_type,
                    score=min(final_score, 1.0),  # Ensure score <= 1.0
                    source="hybrid",
                    explanation=(
                        f"Combined recommendation from {len(scores)} algorithm(s): "
                        f"{original_item.explanation}"
                    ),
                    metadata={
                        **original_item.metadata,
                        "hybrid_num_sources": len(scores),
                        "hybrid_base_algorithm": self.recommenders[algorithm_idx].algorithm_name,
                    },
                )
                merged_items.append(merged_item)

        # Sort by score descending
        merged_items.sort(key=lambda x: x.score, reverse=True)

        logger.debug(
            "recommendations_merged",
            total_unique_items=len(merged_items),
            avg_score=sum(item.score for item in merged_items) / len(merged_items)
            if merged_items
            else 0.0,
        )

        return merged_items

    def _deduplicate(
        self,
        items: list[RecommendationItem],
    ) -> list[RecommendationItem]:
        """Remove duplicates, keeping highest score.

        Args:
            items: List of recommendation items potentially containing duplicates.

        Returns:
            Deduplicated list of recommendation items.
        """
        # Already deduplicated in merge step, but this ensures it
        seen_ids: set[str] = set()
        deduplicated: list[RecommendationItem] = []

        for item in items:
            if item.item_id not in seen_ids:
                seen_ids.add(item.item_id)
                deduplicated.append(item)

        if len(deduplicated) < len(items):
            logger.debug(
                "duplicates_removed",
                original_count=len(items),
                deduplicated_count=len(deduplicated),
            )

        return deduplicated

    def update_weights_from_feedback(
        self,
        user_id: str,
        item_id: str,
        reward: float,
        context: dict[str, str | int | float | bool],
    ) -> None:
        """Update bandit weights based on feedback.

        Args:
            user_id: User who interacted with the item.
            item_id: Item that received feedback.
            reward: Reward value (typically 0.0 to 1.0).
                - 0.0: Item ignored
                - 0.3: Item clicked/viewed
                - 0.5: Item added to watchlist
                - 1.0: Item resulted in profitable trade
            context: Context information for the recommendation.

        Raises:
            ValueError: If reward is negative.
        """
        if reward < 0.0:
            raise ValueError(f"Reward must be non-negative, got {reward}")

        if not self.use_bandit or self.bandit is None:
            logger.debug("bandit_not_enabled_skipping_update")
            return

        # Find which algorithm produced this item
        algorithm_idx = self._item_to_algorithm.get(item_id)

        if algorithm_idx is None:
            logger.warning(
                "item_not_found_in_tracking",
                item_id=item_id,
                user_id=user_id,
            )
            return

        # Update bandit
        self.bandit.update(context, algorithm_idx, reward)

        logger.debug(
            "bandit_weights_updated",
            user_id=user_id,
            item_id=item_id,
            algorithm_idx=algorithm_idx,
            reward=reward,
        )

    def _extract_context(
        self,
        request: RecommendationRequest,
        user_profile: ExplicitProfile,
        implicit_profile: ImplicitProfile | None,
    ) -> dict[str, str | int | float | bool]:
        """Extract context information for bandit.

        Args:
            request: Recommendation request.
            user_profile: User's explicit profile.
            implicit_profile: User's implicit profile if available.

        Returns:
            Context dictionary with relevant features.
        """
        context: dict[str, str | int | float | bool] = {}

        # Add request context if available
        if request.context:
            context.update(request.context)

        # Add user profile features
        context["risk_tolerance"] = user_profile.risk_tolerance
        context["investment_horizon"] = user_profile.investment_horizon
        context["num_watchlist"] = len(user_profile.watchlist)
        context["num_preferred_sectors"] = len(user_profile.preferred_sectors)

        # Add implicit profile features if available
        if implicit_profile is not None:
            context["num_viewed_sectors"] = len(implicit_profile.viewed_sectors)
            context["num_viewed_symbols"] = len(implicit_profile.viewed_symbols)
            context["avg_holding_period"] = implicit_profile.avg_holding_period
            context["preferred_volatility"] = implicit_profile.preferred_volatility

        return context

    def _get_arm_index(self, item_id: str) -> int:
        """Get the algorithm index (arm) that produced this item.

        Args:
            item_id: Item identifier.

        Returns:
            Algorithm index, or 0 if not found.
        """
        return self._item_to_algorithm.get(item_id, 0)

    def get_algorithm_weights(self) -> dict[str, float]:
        """Get current weights for each algorithm.

        Returns:
            Dictionary mapping algorithm names to weights.
        """
        if self.use_bandit and self.bandit is not None:
            # Get current bandit weights
            weights = self.bandit.get_weights(None)
        else:
            weights = self.weights

        return {
            self.recommenders[i].algorithm_name: weights[i]
            for i in range(len(self.recommenders))
        }
