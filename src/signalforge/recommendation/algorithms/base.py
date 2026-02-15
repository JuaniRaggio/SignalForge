"""Base classes for recommendation algorithms.

This module provides the abstract base class that all recommendation algorithms
must implement, along with common utility methods for filtering and diversity.

Key features:
- Abstract interface for recommendation algorithms
- Filtering of previously seen items
- Diversity enforcement to avoid monotonous recommendations
- Training interface for model-based algorithms
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import polars as pl

from signalforge.core.logging import get_logger
from signalforge.recommendation.algorithms.schemas import RecommendationItem
from signalforge.recommendation.user_model import ExplicitProfile, ImplicitProfile

if TYPE_CHECKING:
    from signalforge.recommendation.algorithms.schemas import RecommendationRequest

logger = get_logger(__name__)


class BaseRecommender(ABC):
    """Base class for recommendation algorithms.

    All recommendation algorithms must inherit from this class and implement
    the abstract methods. This ensures a consistent interface across all
    recommendation strategies.
    """

    algorithm_name: str = "base"

    @abstractmethod
    async def recommend(
        self,
        request: RecommendationRequest,
        user_profile: ExplicitProfile,
        implicit_profile: ImplicitProfile | None = None,
    ) -> list[RecommendationItem]:
        """Generate recommendations for a user.

        Args:
            request: Recommendation request with user preferences.
            user_profile: Explicit user preferences.
            implicit_profile: Optional implicit user behavior profile.

        Returns:
            List of recommendation items sorted by relevance.

        Raises:
            ValueError: If request or profiles are invalid.
        """
        ...

    @abstractmethod
    async def train(self, interaction_data: pl.DataFrame) -> None:
        """Train or update the recommendation model.

        Args:
            interaction_data: DataFrame containing user interaction history.
                Expected columns: user_id, item_id, rating, timestamp

        Raises:
            ValueError: If interaction_data has invalid schema.
        """
        ...

    def filter_seen(
        self,
        items: list[RecommendationItem],
        seen_ids: set[str],
    ) -> list[RecommendationItem]:
        """Filter out items that the user has already seen.

        Args:
            items: List of recommendation items to filter.
            seen_ids: Set of item IDs that have been seen by the user.

        Returns:
            Filtered list containing only unseen items.
        """
        if not seen_ids:
            logger.debug("no_seen_items_to_filter", num_items=len(items))
            return items

        filtered = [item for item in items if item.item_id not in seen_ids]

        logger.debug(
            "filtered_seen_items",
            original_count=len(items),
            filtered_count=len(filtered),
            removed_count=len(items) - len(filtered),
        )

        return filtered

    def apply_diversity(
        self,
        items: list[RecommendationItem],
        diversity_factor: float = 0.3,
    ) -> list[RecommendationItem]:
        """Apply diversity to avoid too similar recommendations.

        This method uses a greedy algorithm to select diverse items:
        1. Select the highest-scored item
        2. For each subsequent item, balance score with diversity from selected items
        3. Penalize items that are too similar to already selected items

        Args:
            items: List of recommendation items to diversify.
            diversity_factor: Weight for diversity (0.0 = no diversity, 1.0 = maximum diversity).

        Returns:
            Diversified list of recommendation items.

        Raises:
            ValueError: If diversity_factor is not in [0.0, 1.0].
        """
        if not 0.0 <= diversity_factor <= 1.0:
            raise ValueError(f"diversity_factor must be in [0.0, 1.0], got {diversity_factor}")

        if diversity_factor == 0.0 or len(items) <= 1:
            # No diversity needed
            return items

        logger.debug("applying_diversity", num_items=len(items), diversity_factor=diversity_factor)

        # Start with the highest-scored item
        selected: list[RecommendationItem] = []
        remaining = items.copy()

        while remaining:
            if not selected:
                # First item: select highest score
                best_item = max(remaining, key=lambda x: x.score)
                selected.append(best_item)
                remaining.remove(best_item)
            else:
                # Subsequent items: balance score and diversity
                best_item = None
                best_combined_score = -1.0

                for candidate in remaining:
                    # Calculate diversity as minimum similarity to selected items
                    diversity_score = self._calculate_diversity_score(candidate, selected)

                    # Combine score and diversity
                    combined_score = (
                        1.0 - diversity_factor
                    ) * candidate.score + diversity_factor * diversity_score

                    if combined_score > best_combined_score:
                        best_combined_score = combined_score
                        best_item = candidate

                if best_item is not None:
                    selected.append(best_item)
                    remaining.remove(best_item)
                else:
                    break

        logger.debug(
            "diversity_applied",
            original_count=len(items),
            diversified_count=len(selected),
        )

        return selected

    def _calculate_diversity_score(
        self,
        candidate: RecommendationItem,
        selected: list[RecommendationItem],
    ) -> float:
        """Calculate diversity score for a candidate item.

        The diversity score measures how different the candidate is from
        already selected items. Higher scores indicate more diversity.

        Args:
            candidate: Candidate item to evaluate.
            selected: List of already selected items.

        Returns:
            Diversity score between 0.0 and 1.0.
        """
        if not selected:
            return 1.0

        # Calculate minimum similarity across all selected items
        min_similarity = 1.0

        for selected_item in selected:
            similarity = self._item_similarity(candidate, selected_item)
            min_similarity = min(min_similarity, similarity)

        # Diversity is inverse of similarity
        diversity = 1.0 - min_similarity
        return diversity

    def _item_similarity(
        self,
        item1: RecommendationItem,
        item2: RecommendationItem,
    ) -> float:
        """Calculate similarity between two recommendation items.

        This is a simple implementation based on item type and metadata.
        Subclasses can override this method for more sophisticated similarity.

        Args:
            item1: First item.
            item2: Second item.

        Returns:
            Similarity score between 0.0 (completely different) and 1.0 (identical).
        """
        # Same item
        if item1.item_id == item2.item_id:
            return 1.0

        similarity = 0.0

        # Same type increases similarity
        if item1.item_type == item2.item_type:
            similarity += 0.5

        # Check metadata similarity
        common_keys = set(item1.metadata.keys()) & set(item2.metadata.keys())
        if common_keys:
            matches = sum(
                1 for key in common_keys if item1.metadata.get(key) == item2.metadata.get(key)
            )
            metadata_similarity = matches / len(common_keys) if common_keys else 0.0
            similarity += 0.5 * metadata_similarity

        return min(similarity, 1.0)
