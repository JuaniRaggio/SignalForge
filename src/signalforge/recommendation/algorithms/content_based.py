"""Content-based filtering recommendation algorithm.

This module implements content-based filtering that recommends items similar
to what the user has previously liked, based on item features.

The algorithm:
1. Builds a user preference profile from their portfolio/watchlist
2. Compares candidate items to the user's preferences
3. Ranks items by feature similarity

Supported features:
- Sector alignment
- Market capitalization matching
- Volatility preferences
- Signal type matching
- Recency boosting
"""

from __future__ import annotations

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


class ContentBasedRecommender(BaseRecommender):
    """Content-based filtering using item features.

    This recommender matches items to user preferences based on content features
    such as sector, market cap, volatility, and signal type.

    Attributes:
        algorithm_name: Name of the algorithm.
        feature_weights: Weights for different features in scoring.
        item_features: Database of item features.
    """

    algorithm_name = "content_based"

    def __init__(self, feature_weights: dict[str, float] | None = None):
        """Initialize the content-based recommender.

        Args:
            feature_weights: Optional custom weights for features.
        """
        # Default weights for different features
        self.feature_weights = feature_weights or {
            "sector": 0.3,
            "market_cap": 0.1,
            "volatility": 0.15,
            "signal_type": 0.2,
            "recency": 0.25,
        }

        # Validate weights sum to 1.0
        total_weight = sum(self.feature_weights.values())
        if not 0.99 <= total_weight <= 1.01:
            logger.warning(
                "feature_weights_not_normalized",
                total=total_weight,
                weights=self.feature_weights,
            )
            # Normalize weights
            self.feature_weights = {
                k: v / total_weight for k, v in self.feature_weights.items()
            }

        # Item feature database
        self.item_features: dict[str, dict[str, str | float | int]] = {}

        logger.info(
            "content_based_recommender_initialized",
            feature_weights=self.feature_weights,
        )

    async def recommend(
        self,
        request: RecommendationRequest,
        user_profile: ExplicitProfile,
        implicit_profile: ImplicitProfile | None = None,
    ) -> list[RecommendationItem]:
        """Generate recommendations based on content similarity.

        Args:
            request: Recommendation request.
            user_profile: Explicit user preferences.
            implicit_profile: Optional implicit user behavior.

        Returns:
            List of recommended items sorted by similarity score.
        """
        if not self.item_features:
            logger.warning("no_item_features_available")
            return []

        # Build user preference vector
        user_preferences = self._build_user_preference_vector(user_profile, implicit_profile)

        # Get candidate items
        candidate_items = list(self.item_features.keys())

        # Filter by item type if specified
        if request.item_types:
            candidate_items = [
                item_id
                for item_id in candidate_items
                if self.item_features[item_id].get("item_type") in request.item_types
            ]

        # Score each candidate
        scored_items: list[tuple[str, float]] = []
        for item_id in candidate_items:
            item_features = self.item_features[item_id]
            score = self._compute_item_score(item_features, user_preferences)
            scored_items.append((item_id, score))

        # Sort by score
        scored_items.sort(key=lambda x: x[1], reverse=True)

        # Convert to recommendation items
        recommendations = []
        for item_id, score in scored_items[: request.limit]:
            item_features = self.item_features[item_id]
            explanation = self._generate_explanation(item_features, user_preferences)

            recommendations.append(
                RecommendationItem(
                    item_id=item_id,
                    item_type=str(item_features.get("item_type", "unknown")),
                    score=score,
                    source="content_based",
                    explanation=explanation,
                    metadata={
                        "sector": str(item_features.get("sector", "")),
                        "signal_type": str(item_features.get("signal_type", "")),
                    },
                )
            )

        logger.info(
            "content_based_recommendations_generated",
            user_id=request.user_id,
            num_recommendations=len(recommendations),
        )

        return recommendations

    async def train(self, interaction_data: pl.DataFrame) -> None:
        """Update item feature database from interaction data.

        Args:
            interaction_data: DataFrame containing item features.
                Expected columns: item_id, sector, market_cap, volatility, signal_type, timestamp

        Raises:
            ValueError: If interaction_data has invalid schema.
        """
        required_columns = {"item_id"}
        actual_columns = set(interaction_data.columns)

        if not required_columns.issubset(actual_columns):
            missing = required_columns - actual_columns
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("training_content_based_filter", num_items=len(interaction_data))

        # Extract item features
        for row in interaction_data.iter_rows(named=True):
            item_id = str(row["item_id"])

            self.item_features[item_id] = {
                "item_type": str(row.get("item_type", "unknown")),
                "sector": str(row.get("sector", "")),
                "market_cap": float(row.get("market_cap", 0.0)),
                "volatility": float(row.get("volatility", 0.0)),
                "signal_type": str(row.get("signal_type", "")),
                "timestamp": int(row.get("timestamp", 0)),
            }

        logger.info(
            "content_based_filter_trained",
            num_items=len(self.item_features),
        )

    def _build_user_preference_vector(
        self,
        explicit: ExplicitProfile,
        implicit: ImplicitProfile | None,
    ) -> dict[str, str | float | list[str] | dict[str, float]]:
        """Build user preference vector from profiles.

        Args:
            explicit: Explicit user preferences.
            implicit: Optional implicit user behavior.

        Returns:
            Dictionary of user preferences for matching.
        """
        preferences: dict[str, str | float | list[str] | dict[str, float]] = {}

        # Sector preferences from explicit profile
        preferences["preferred_sectors"] = explicit.preferred_sectors

        # Risk tolerance maps to volatility preference
        risk_to_volatility = {
            "low": 0.1,
            "medium": 0.5,
            "high": 0.8,
        }
        preferences["preferred_volatility"] = risk_to_volatility[explicit.risk_tolerance]

        # Investment horizon
        preferences["investment_horizon"] = float(explicit.investment_horizon)

        # Incorporate implicit preferences if available
        if implicit:
            # Weight sectors by view count
            if implicit.viewed_sectors:
                total_views = sum(implicit.viewed_sectors.values())
                sector_weights = {
                    sector: count / total_views
                    for sector, count in implicit.viewed_sectors.items()
                }
                preferences["sector_weights"] = sector_weights

            # Use implicit volatility if available
            if implicit.preferred_volatility > 0.0:
                # Blend explicit and implicit volatility preferences
                explicit_vol = preferences["preferred_volatility"]
                if isinstance(explicit_vol, float):
                    preferences["preferred_volatility"] = (
                        0.6 * explicit_vol + 0.4 * implicit.preferred_volatility
                    )

            # Use implicit holding period
            if implicit.avg_holding_period > 0.0:
                preferences["avg_holding_period"] = implicit.avg_holding_period

        return preferences

    def _compute_item_score(
        self,
        item_features: dict[str, str | float | int],
        user_preferences: dict[str, str | float | list[str] | dict[str, float]],
    ) -> float:
        """Compute item relevance score.

        Args:
            item_features: Features of the candidate item.
            user_preferences: User preference vector.

        Returns:
            Relevance score between 0.0 and 1.0.
        """
        total_score = 0.0

        # Sector alignment
        item_sector = str(item_features.get("sector", ""))
        preferred_sectors = user_preferences.get("preferred_sectors", [])
        if isinstance(preferred_sectors, list):
            sector_score = self._get_sector_alignment(item_sector, preferred_sectors)
            total_score += self.feature_weights["sector"] * sector_score

        # Volatility matching
        item_volatility = float(item_features.get("volatility", 0.0))
        preferred_volatility = user_preferences.get("preferred_volatility", 0.5)
        if isinstance(preferred_volatility, float):
            volatility_score = self._get_volatility_match(item_volatility, preferred_volatility)
            total_score += self.feature_weights["volatility"] * volatility_score

        # Market cap matching (placeholder - would need user preferences)
        market_cap_score = 0.5  # Neutral score
        total_score += self.feature_weights["market_cap"] * market_cap_score

        # Signal type matching (placeholder - would need user preferences)
        signal_type_score = 0.5  # Neutral score
        total_score += self.feature_weights["signal_type"] * signal_type_score

        # Recency boost
        recency_score = self._get_recency_score(int(item_features.get("timestamp", 0)))
        total_score += self.feature_weights["recency"] * recency_score

        return max(0.0, min(total_score, 1.0))

    def _get_sector_alignment(self, item_sector: str, preferred_sectors: list[str]) -> float:
        """Score sector alignment.

        Args:
            item_sector: Sector of the candidate item.
            preferred_sectors: User's preferred sectors.

        Returns:
            Alignment score between 0.0 and 1.0.
        """
        if not item_sector or not preferred_sectors:
            return 0.5  # Neutral score

        # Exact match
        if item_sector in preferred_sectors:
            return 1.0

        # No match
        return 0.0

    def _get_volatility_match(self, item_volatility: float, preferred_volatility: float) -> float:
        """Score volatility matching.

        Args:
            item_volatility: Item's volatility level.
            preferred_volatility: User's preferred volatility level.

        Returns:
            Match score between 0.0 and 1.0.
        """
        if preferred_volatility == 0.0:
            return 0.5  # Neutral score

        # Calculate distance and convert to similarity
        distance = abs(item_volatility - preferred_volatility)
        similarity = max(0.0, 1.0 - distance)

        return similarity

    def _get_recency_score(self, timestamp: int) -> float:
        """Score item recency.

        Args:
            timestamp: Item timestamp (unix timestamp).

        Returns:
            Recency score between 0.0 and 1.0.
        """
        import time

        if timestamp == 0:
            return 0.5  # Neutral score for items without timestamp

        # Calculate age in days
        current_time = int(time.time())
        age_seconds = current_time - timestamp
        age_days = age_seconds / (24 * 3600)

        # Exponential decay: newer items score higher
        # Half-life of 7 days
        decay_rate = np.log(2) / 7.0
        recency_score = float(np.exp(-decay_rate * age_days))

        return max(0.0, min(recency_score, 1.0))

    def _generate_explanation(
        self,
        item_features: dict[str, str | float | int],
        user_preferences: dict[str, str | float | list[str] | dict[str, float]],
    ) -> str:
        """Generate explanation for recommendation.

        Args:
            item_features: Features of the recommended item.
            user_preferences: User preference vector.

        Returns:
            Human-readable explanation.
        """
        reasons = []

        # Check sector match
        item_sector = str(item_features.get("sector", ""))
        preferred_sectors = user_preferences.get("preferred_sectors", [])
        if isinstance(preferred_sectors, list) and item_sector in preferred_sectors:
            reasons.append(f"matches your preferred sector ({item_sector})")

        # Check volatility match
        item_volatility = float(item_features.get("volatility", 0.0))
        preferred_volatility = user_preferences.get("preferred_volatility", 0.5)
        if isinstance(preferred_volatility, float) and abs(item_volatility - preferred_volatility) < 0.2:
            reasons.append("matches your risk tolerance")

        # Check recency
        timestamp = int(item_features.get("timestamp", 0))
        if timestamp > 0:
            import time

            age_days = (int(time.time()) - timestamp) / (24 * 3600)
            if age_days < 1:
                reasons.append("recent signal")

        if reasons:
            return f"Recommended because it {', '.join(reasons)}"
        else:
            return "Matches your investment profile"
