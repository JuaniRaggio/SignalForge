"""Collaborative filtering recommendation algorithms.

This module implements user-based and item-based collaborative filtering.
These algorithms make recommendations based on patterns in user behavior:

- User-based: Find similar users and recommend items they liked
- Item-based: Find similar items to what the user liked and recommend them

The algorithms use cosine similarity for computing user/item similarities.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import polars as pl

from signalforge.core.logging import get_logger
from signalforge.recommendation.algorithms.base import BaseRecommender
from signalforge.recommendation.algorithms.schemas import (
    RecommendationItem,
    RecommendationRequest,
)
from signalforge.recommendation.user_model import ExplicitProfile, ImplicitProfile

logger = get_logger(__name__)


class CollaborativeFilteringRecommender(BaseRecommender):
    """User-based and item-based collaborative filtering.

    This recommender builds similarity matrices from historical user interactions
    and uses them to predict ratings and generate recommendations.

    Attributes:
        algorithm_name: Name of the algorithm.
        method: Either "user_based" or "item_based".
        n_neighbors: Number of similar users/items to consider.
        min_common_items: Minimum number of common items required for similarity.
    """

    algorithm_name = "collaborative"

    def __init__(
        self,
        method: Literal["user_based", "item_based"] = "user_based",
        n_neighbors: int = 20,
        min_common_items: int = 3,
    ):
        """Initialize the collaborative filtering recommender.

        Args:
            method: Collaborative filtering method to use.
            n_neighbors: Number of neighbors to consider for recommendations.
            min_common_items: Minimum overlap required to compute similarity.

        Raises:
            ValueError: If parameters are invalid.
        """
        if method not in ("user_based", "item_based"):
            raise ValueError(f"Invalid method: {method}. Must be 'user_based' or 'item_based'")

        if n_neighbors <= 0:
            raise ValueError(f"n_neighbors must be positive, got {n_neighbors}")

        if min_common_items <= 0:
            raise ValueError(f"min_common_items must be positive, got {min_common_items}")

        self.method = method
        self.n_neighbors = n_neighbors
        self.min_common_items = min_common_items

        # User-item interaction matrix
        self.user_items: dict[str, dict[str, float]] = defaultdict(dict)
        self.item_users: dict[str, dict[str, float]] = defaultdict(dict)

        # Similarity caches
        self.user_similarity_cache: dict[tuple[str, str], float] = {}
        self.item_similarity_cache: dict[tuple[str, str], float] = {}

        logger.info(
            "collaborative_recommender_initialized",
            method=method,
            n_neighbors=n_neighbors,
            min_common_items=min_common_items,
        )

    async def recommend(
        self,
        request: RecommendationRequest,
        user_profile: ExplicitProfile,  # noqa: ARG002
        implicit_profile: ImplicitProfile | None = None,  # noqa: ARG002
    ) -> list[RecommendationItem]:
        """Generate recommendations using collaborative filtering.

        Args:
            request: Recommendation request.
            user_profile: Explicit user preferences.
            implicit_profile: Optional implicit user behavior.

        Returns:
            List of recommended items sorted by predicted rating.
        """
        user_id = request.user_id

        # Check if user exists in our interaction data
        if user_id not in self.user_items:
            logger.warning("user_not_in_training_data", user_id=user_id)
            return []

        user_items = self.user_items[user_id]

        # Get candidate items (items user hasn't seen)
        all_items = set(self.item_users.keys())
        seen_items = set(user_items.keys())
        candidate_items = all_items - seen_items

        if not candidate_items:
            logger.debug("no_candidate_items", user_id=user_id)
            return []

        # Filter by item type if specified
        if request.item_types:
            candidate_items = {
                item_id
                for item_id in candidate_items
                if self._get_item_type(item_id) in request.item_types
            }

        # Predict ratings for candidate items
        predictions: list[tuple[str, float]] = []
        for item_id in candidate_items:
            predicted_rating = self._predict_rating(user_id, item_id)
            if predicted_rating > 0.0:
                predictions.append((item_id, predicted_rating))

        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Convert to recommendation items
        recommendations = []
        for item_id, score in predictions[: request.limit]:
            explanation = self._generate_explanation(user_id, item_id, score)
            recommendations.append(
                RecommendationItem(
                    item_id=item_id,
                    item_type=self._get_item_type(item_id),
                    score=min(score, 1.0),
                    source=f"collaborative_{self.method}",
                    explanation=explanation,
                    metadata={"predicted_rating": score},
                )
            )

        logger.info(
            "collaborative_recommendations_generated",
            user_id=user_id,
            num_recommendations=len(recommendations),
            method=self.method,
        )

        return recommendations

    async def train(self, interaction_data: pl.DataFrame) -> None:
        """Build similarity matrices from interaction data.

        Args:
            interaction_data: DataFrame with columns: user_id, item_id, rating, timestamp

        Raises:
            ValueError: If interaction_data has invalid schema.
        """
        required_columns = {"user_id", "item_id", "rating"}
        actual_columns = set(interaction_data.columns)

        if not required_columns.issubset(actual_columns):
            missing = required_columns - actual_columns
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("training_collaborative_filter", num_interactions=len(interaction_data))

        # Build user-item and item-user matrices
        self.user_items.clear()
        self.item_users.clear()

        for row in interaction_data.iter_rows(named=True):
            user_id = str(row["user_id"])
            item_id = str(row["item_id"])
            rating = float(row["rating"])

            self.user_items[user_id][item_id] = rating
            self.item_users[item_id][user_id] = rating

        # Clear similarity caches
        self.user_similarity_cache.clear()
        self.item_similarity_cache.clear()

        logger.info(
            "collaborative_filter_trained",
            num_users=len(self.user_items),
            num_items=len(self.item_users),
        )

    def _compute_user_similarity(self, user1_items: dict[str, float], user2_items: dict[str, float]) -> float:
        """Compute cosine similarity between two users.

        Args:
            user1_items: First user's item ratings.
            user2_items: Second user's item ratings.

        Returns:
            Cosine similarity between 0.0 and 1.0.
        """
        # Find common items
        common_items = set(user1_items.keys()) & set(user2_items.keys())

        if len(common_items) < self.min_common_items:
            return 0.0

        # Build vectors
        user1_vec = np.array([user1_items[item] for item in common_items])
        user2_vec = np.array([user2_items[item] for item in common_items])

        # Compute cosine similarity
        norm1 = np.linalg.norm(user1_vec)
        norm2 = np.linalg.norm(user2_vec)

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        similarity = float(np.dot(user1_vec, user2_vec) / (norm1 * norm2))
        return max(0.0, min(similarity, 1.0))

    def _compute_item_similarity(self, item1_users: dict[str, float], item2_users: dict[str, float]) -> float:
        """Compute cosine similarity between two items.

        Args:
            item1_users: First item's user ratings.
            item2_users: Second item's user ratings.

        Returns:
            Cosine similarity between 0.0 and 1.0.
        """
        # Find common users
        common_users = set(item1_users.keys()) & set(item2_users.keys())

        if len(common_users) < self.min_common_items:
            return 0.0

        # Build vectors
        item1_vec = np.array([item1_users[user] for user in common_users])
        item2_vec = np.array([item2_users[user] for user in common_users])

        # Compute cosine similarity
        norm1 = np.linalg.norm(item1_vec)
        norm2 = np.linalg.norm(item2_vec)

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        similarity = float(np.dot(item1_vec, item2_vec) / (norm1 * norm2))
        return max(0.0, min(similarity, 1.0))

    def _find_neighbors(self, user_id: str, k: int) -> list[tuple[str, float]]:
        """Find k most similar users.

        Args:
            user_id: Target user ID.
            k: Number of neighbors to find.

        Returns:
            List of (neighbor_id, similarity) tuples sorted by similarity.
        """
        if user_id not in self.user_items:
            return []

        user_items = self.user_items[user_id]
        similarities: list[tuple[str, float]] = []

        for other_user_id, other_items in self.user_items.items():
            if other_user_id == user_id:
                continue

            # Check cache
            cache_key = (user_id, other_user_id)
            if cache_key in self.user_similarity_cache:
                similarity = self.user_similarity_cache[cache_key]
            else:
                similarity = self._compute_user_similarity(user_items, other_items)
                self.user_similarity_cache[cache_key] = similarity

            if similarity > 0.0:
                similarities.append((other_user_id, similarity))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def _predict_rating(self, user_id: str, item_id: str) -> float:
        """Predict user's rating for an item.

        Args:
            user_id: User ID.
            item_id: Item ID.

        Returns:
            Predicted rating (0.0 if cannot predict).
        """
        if self.method == "user_based":
            return self._predict_rating_user_based(user_id, item_id)
        else:
            return self._predict_rating_item_based(user_id, item_id)

    def _predict_rating_user_based(self, user_id: str, item_id: str) -> float:
        """Predict rating using user-based collaborative filtering.

        Args:
            user_id: User ID.
            item_id: Item ID.

        Returns:
            Predicted rating.
        """
        # Find similar users who rated this item
        neighbors = self._find_neighbors(user_id, self.n_neighbors)

        if not neighbors:
            return 0.0

        weighted_sum = 0.0
        similarity_sum = 0.0

        for neighbor_id, similarity in neighbors:
            if item_id in self.user_items[neighbor_id]:
                rating = self.user_items[neighbor_id][item_id]
                weighted_sum += similarity * rating
                similarity_sum += similarity

        if similarity_sum == 0.0:
            return 0.0

        predicted_rating = weighted_sum / similarity_sum
        return predicted_rating

    def _predict_rating_item_based(self, user_id: str, item_id: str) -> float:
        """Predict rating using item-based collaborative filtering.

        Args:
            user_id: User ID.
            item_id: Item ID.

        Returns:
            Predicted rating.
        """
        if user_id not in self.user_items:
            return 0.0

        user_items = self.user_items[user_id]
        item_users = self.item_users.get(item_id, {})

        if not item_users:
            return 0.0

        # Find similar items that the user has rated
        similarities: list[tuple[str, float]] = []

        for rated_item_id in user_items:
            if rated_item_id == item_id:
                continue

            # Check cache
            cache_key = (item_id, rated_item_id)
            if cache_key in self.item_similarity_cache:
                similarity = self.item_similarity_cache[cache_key]
            else:
                rated_item_users = self.item_users.get(rated_item_id, {})
                similarity = self._compute_item_similarity(item_users, rated_item_users)
                self.item_similarity_cache[cache_key] = similarity

            if similarity > 0.0:
                similarities.append((rated_item_id, similarity))

        if not similarities:
            return 0.0

        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = similarities[: self.n_neighbors]

        # Compute weighted average
        weighted_sum = 0.0
        similarity_sum = 0.0

        for similar_item_id, similarity in similarities:
            rating = user_items[similar_item_id]
            weighted_sum += similarity * rating
            similarity_sum += similarity

        if similarity_sum == 0.0:
            return 0.0

        predicted_rating = weighted_sum / similarity_sum
        return predicted_rating

    def _generate_explanation(self, user_id: str, item_id: str, score: float) -> str:  # noqa: ARG002
        """Generate explanation for recommendation.

        Args:
            user_id: User ID.
            item_id: Item ID.
            score: Predicted score.

        Returns:
            Human-readable explanation.
        """
        if self.method == "user_based":
            return f"Users similar to you rated this highly (predicted score: {score:.2f})"
        else:
            return f"Similar to items you've liked (predicted score: {score:.2f})"

    def _get_item_type(self, item_id: str) -> str:
        """Get the type of an item.

        This is a simple implementation that extracts type from item_id prefix.
        In production, this would query a database or metadata store.

        Args:
            item_id: Item ID.

        Returns:
            Item type string.
        """
        # Simple heuristic: extract prefix before underscore
        if "_" in item_id:
            return item_id.split("_")[0]
        return "unknown"
