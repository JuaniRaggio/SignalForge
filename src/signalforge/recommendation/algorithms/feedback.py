"""Feedback processing and anti-herding mechanisms for recommendations.

This module provides components for:
- Recording and processing user feedback on recommendations
- Tracking impressions, clicks, and outcomes
- Calculating rewards for bandit updates
- Analyzing algorithm performance
- Preventing herding behavior through diversification

The feedback system enables:
- Continuous improvement of recommendation quality
- Algorithm selection optimization via bandits
- Detection and mitigation of herding effects
- Performance monitoring and analytics

Examples:
    Recording feedback:

    >>> from signalforge.recommendation.algorithms.feedback import FeedbackProcessor
    >>>
    >>> processor = FeedbackProcessor(session)
    >>>
    >>> # Record impression
    >>> impression_id = await processor.record_impression(
    ...     user_id="user_123",
    ...     item_id="AAPL",
    ...     algorithm="content_based",
    ...     position=1,
    ...     context={"time": "morning"},
    ... )
    >>>
    >>> # Record click
    >>> await processor.record_click(impression_id, datetime.now())
    >>>
    >>> # Record outcome
    >>> await processor.record_outcome(
    ...     impression_id,
    ...     outcome_type="trade",
    ...     outcome_value=150.0,  # Profit
    ... )
    >>>
    >>> # Calculate reward for bandit
    >>> reward = await processor.calculate_reward(impression_id)

    Anti-herding:

    >>> from signalforge.recommendation.algorithms.feedback import AntiHerdingFilter
    >>>
    >>> filter = AntiHerdingFilter(herding_threshold=0.3)
    >>> filtered = filter.filter(recommendations, recent_popular)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from signalforge.core.logging import get_logger
from signalforge.recommendation.algorithms.schemas import RecommendationItem

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)


class FeedbackProcessor:
    """Process user feedback for recommendation improvement.

    This class handles recording and processing of user interactions with
    recommendations, including impressions, clicks, and final outcomes.
    It calculates rewards for bandit algorithm updates.

    Attributes:
        session: Database session for storing feedback.
    """

    def __init__(self, session: AsyncSession):
        """Initialize the feedback processor.

        Args:
            session: Async database session.
        """
        self.session = session
        self._impressions: dict[str, dict[str, str | int | float | datetime]] = {}
        self._clicks: dict[str, datetime] = {}
        self._outcomes: dict[str, tuple[str, float | None]] = {}

        logger.info("feedback_processor_initialized")

    async def record_impression(
        self,
        user_id: str,
        item_id: str,
        algorithm: str,
        position: int,
        context: dict[str, str | int | float | bool],  # noqa: ARG002
    ) -> str:
        """Record that item was shown to user.

        Args:
            user_id: User who saw the recommendation.
            item_id: Item that was recommended.
            algorithm: Algorithm that produced the recommendation.
            position: Position in the recommendation list (1-indexed).
            context: Context information for the recommendation.

        Returns:
            Impression ID for tracking.

        Raises:
            ValueError: If position is not positive.
        """
        if position < 1:
            raise ValueError(f"Position must be positive, got {position}")

        # Generate impression ID
        impression_id = f"{user_id}_{item_id}_{datetime.now().timestamp()}"

        # Store impression
        self._impressions[impression_id] = {
            "user_id": user_id,
            "item_id": item_id,
            "algorithm": algorithm,
            "position": position,
            "timestamp": datetime.now(),
        }

        logger.debug(
            "impression_recorded",
            impression_id=impression_id,
            user_id=user_id,
            item_id=item_id,
            algorithm=algorithm,
            position=position,
        )

        return impression_id

    async def record_click(
        self,
        impression_id: str,
        clicked_at: datetime,
    ) -> None:
        """Record click on recommended item.

        Args:
            impression_id: ID of the impression that was clicked.
            clicked_at: Timestamp of the click.

        Raises:
            ValueError: If impression_id is not found.
        """
        if impression_id not in self._impressions:
            raise ValueError(f"Impression {impression_id} not found")

        self._clicks[impression_id] = clicked_at

        impression = self._impressions[impression_id]
        logger.info(
            "click_recorded",
            impression_id=impression_id,
            user_id=impression["user_id"],
            item_id=impression["item_id"],
            algorithm=impression["algorithm"],
        )

    async def record_outcome(
        self,
        impression_id: str,
        outcome_type: str,
        outcome_value: float | None = None,
    ) -> None:
        """Record final outcome of recommendation.

        Args:
            impression_id: ID of the impression.
            outcome_type: Type of outcome: "trade", "watchlist_add", "ignore".
            outcome_value: Optional value (e.g., P&L for trades).

        Raises:
            ValueError: If impression_id is not found or outcome_type is invalid.
        """
        if impression_id not in self._impressions:
            raise ValueError(f"Impression {impression_id} not found")

        valid_outcomes = {"trade", "watchlist_add", "ignore"}
        if outcome_type not in valid_outcomes:
            raise ValueError(
                f"Invalid outcome_type '{outcome_type}', must be one of {valid_outcomes}"
            )

        self._outcomes[impression_id] = (outcome_type, outcome_value)

        impression = self._impressions[impression_id]
        logger.info(
            "outcome_recorded",
            impression_id=impression_id,
            user_id=impression["user_id"],
            item_id=impression["item_id"],
            outcome_type=outcome_type,
            outcome_value=outcome_value,
        )

    async def calculate_reward(
        self,
        impression_id: str,
    ) -> float:
        """Calculate reward for bandit update.

        Reward structure:
        - No click: 0.0
        - Click only: 0.3
        - Watchlist add: 0.5
        - Trade with profit: 1.0
        - Trade with loss: 0.2 (still engaged)
        - Ignore after click: 0.1

        Args:
            impression_id: ID of the impression.

        Returns:
            Reward value between 0.0 and 1.0.

        Raises:
            ValueError: If impression_id is not found.
        """
        if impression_id not in self._impressions:
            raise ValueError(f"Impression {impression_id} not found")

        # Check if clicked
        clicked = impression_id in self._clicks

        # Check outcome
        outcome = self._outcomes.get(impression_id)

        if outcome is None:
            # No outcome recorded yet
            reward = 0.3 if clicked else 0.0
        else:
            outcome_type, outcome_value = outcome

            if outcome_type == "ignore":
                reward = 0.1 if clicked else 0.0
            elif outcome_type == "watchlist_add":
                reward = 0.5
            elif outcome_type == "trade":
                reward = 1.0 if outcome_value is not None and outcome_value > 0.0 else 0.2
            else:
                reward = 0.0

        logger.debug(
            "reward_calculated",
            impression_id=impression_id,
            clicked=clicked,
            outcome=outcome,
            reward=reward,
        )

        return reward

    async def get_algorithm_performance(
        self,
        days: int = 30,
    ) -> dict[str, dict[str, float]]:
        """Get performance metrics by algorithm.

        Args:
            days: Number of days to look back.

        Returns:
            Dictionary mapping algorithm names to performance metrics:
            - impressions: Total impressions
            - clicks: Total clicks
            - ctr: Click-through rate
            - conversions: Total conversions (trades + watchlist adds)
            - conversion_rate: Conversion rate
            - avg_reward: Average reward
        """
        cutoff_time = datetime.now() - timedelta(days=days)

        # Filter impressions within time window
        recent_impressions = {
            imp_id: imp
            for imp_id, imp in self._impressions.items()
            if isinstance(imp.get("timestamp"), datetime)
            and isinstance(imp["timestamp"], datetime)
            and imp["timestamp"] >= cutoff_time
        }

        # Aggregate by algorithm
        performance: dict[str, dict[str, float]] = {}

        for imp_id, impression in recent_impressions.items():
            algorithm = str(impression["algorithm"])

            if algorithm not in performance:
                performance[algorithm] = {
                    "impressions": 0.0,
                    "clicks": 0.0,
                    "conversions": 0.0,
                    "total_reward": 0.0,
                }

            performance[algorithm]["impressions"] += 1.0

            # Check if clicked
            if imp_id in self._clicks:
                performance[algorithm]["clicks"] += 1.0

            # Check if converted
            if imp_id in self._outcomes:
                outcome_type, _ = self._outcomes[imp_id]
                if outcome_type in {"trade", "watchlist_add"}:
                    performance[algorithm]["conversions"] += 1.0

            # Add reward
            reward = await self.calculate_reward(imp_id)
            performance[algorithm]["total_reward"] += reward

        # Calculate rates
        for _algorithm, metrics in performance.items():
            impressions = metrics["impressions"]
            if impressions > 0:
                metrics["ctr"] = metrics["clicks"] / impressions
                metrics["conversion_rate"] = metrics["conversions"] / impressions
                metrics["avg_reward"] = metrics["total_reward"] / impressions
            else:
                metrics["ctr"] = 0.0
                metrics["conversion_rate"] = 0.0
                metrics["avg_reward"] = 0.0

        logger.info(
            "algorithm_performance_computed",
            days=days,
            num_algorithms=len(performance),
        )

        return performance

    def get_impression_count(self) -> int:
        """Get total number of impressions recorded.

        Returns:
            Total impression count.
        """
        return len(self._impressions)

    def get_click_count(self) -> int:
        """Get total number of clicks recorded.

        Returns:
            Total click count.
        """
        return len(self._clicks)

    def get_outcome_count(self) -> int:
        """Get total number of outcomes recorded.

        Returns:
            Total outcome count.
        """
        return len(self._outcomes)


class AntiHerdingFilter:
    """Prevent herding by diversifying recommendations.

    This filter detects when too many users are receiving the same
    recommendations and applies penalties to over-popular items,
    encouraging more diverse recommendations across the user base.

    Attributes:
        herding_threshold: Threshold above which items are penalized.
    """

    def __init__(self, herding_threshold: float = 0.3):
        """Initialize the anti-herding filter.

        Args:
            herding_threshold: Fraction of users receiving an item above
                which it's considered "too popular". Range: 0.0 to 1.0.

        Raises:
            ValueError: If threshold is not in [0.0, 1.0].
        """
        if not 0.0 <= herding_threshold <= 1.0:
            raise ValueError(
                f"herding_threshold must be in [0.0, 1.0], got {herding_threshold}"
            )

        self.herding_threshold = herding_threshold
        logger.info("anti_herding_filter_initialized", threshold=herding_threshold)

    def filter(
        self,
        recommendations: list[RecommendationItem],
        recent_popular: list[str],
    ) -> list[RecommendationItem]:
        """Reduce score of items that are too popular.

        Strategy:
        1. Calculate popularity for each item in recommendations
        2. If item appears in recent_popular above threshold, apply penalty
        3. Inject contrarian picks (items not in popular list)

        Args:
            recommendations: List of recommendation items.
            recent_popular: List of item IDs that are currently popular.

        Returns:
            Filtered and re-ranked recommendation items.
        """
        if not recommendations:
            return recommendations

        if not recent_popular:
            # No popular items to filter against
            logger.debug("no_popular_items_to_filter")
            return recommendations

        # Calculate popularity scores
        total_popular = len(recent_popular)
        popular_set = set(recent_popular)

        filtered_items: list[RecommendationItem] = []

        for item in recommendations:
            # Check if item is in popular list
            if item.item_id in popular_set:
                # Calculate how popular (frequency in recent_popular)
                popularity = recent_popular.count(item.item_id) / total_popular

                if popularity > self.herding_threshold:
                    # Apply penalty
                    penalty = self._calculate_popularity_penalty(
                        item.item_id, recent_popular
                    )
                    adjusted_score = item.score * (1.0 - penalty)

                    # Create modified item
                    filtered_item = RecommendationItem(
                        item_id=item.item_id,
                        item_type=item.item_type,
                        score=max(adjusted_score, 0.0),
                        source=item.source,
                        explanation=f"{item.explanation} (Anti-herding penalty applied)",
                        metadata={
                            **item.metadata,
                            "anti_herding_penalty": penalty,
                            "original_score": item.score,
                        },
                    )
                    filtered_items.append(filtered_item)

                    logger.debug(
                        "herding_penalty_applied",
                        item_id=item.item_id,
                        popularity=popularity,
                        penalty=penalty,
                        original_score=item.score,
                        adjusted_score=adjusted_score,
                    )
                else:
                    # Popular but not above threshold
                    filtered_items.append(item)
            else:
                # Not popular, keep as is
                filtered_items.append(item)

        # Re-sort by adjusted scores
        filtered_items.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "anti_herding_filter_applied",
            original_count=len(recommendations),
            filtered_count=len(filtered_items),
            num_popular_items=len(popular_set),
        )

        return filtered_items

    def _calculate_popularity_penalty(
        self,
        item_id: str,
        popular_ids: list[str],
    ) -> float:
        """Calculate penalty for popular items.

        Penalty increases linearly with popularity above the threshold.

        Args:
            item_id: Item identifier.
            popular_ids: List of popular item IDs.

        Returns:
            Penalty value between 0.0 and 1.0.
        """
        # Calculate frequency
        count = popular_ids.count(item_id)
        total = len(popular_ids)

        if total == 0:
            return 0.0

        popularity = count / total

        # Calculate penalty above threshold
        if popularity <= self.herding_threshold:
            return 0.0

        # Linear penalty from threshold to 1.0
        excess_popularity = popularity - self.herding_threshold
        max_excess = 1.0 - self.herding_threshold

        if max_excess == 0.0:
            return 0.0

        # Penalty ranges from 0.0 to 0.5 (max 50% score reduction)
        penalty = 0.5 * (excess_popularity / max_excess)

        return min(penalty, 0.5)

    def get_popularity_stats(
        self,
        recent_popular: list[str],
    ) -> dict[str, int | float]:
        """Get statistics about popularity distribution.

        Args:
            recent_popular: List of popular item IDs.

        Returns:
            Dictionary with popularity statistics.
        """
        if not recent_popular:
            return {
                "total_items": 0,
                "unique_items": 0,
                "max_frequency": 0,
                "max_popularity": 0.0,
            }

        # Count frequencies
        from collections import Counter

        counter = Counter(recent_popular)
        total = len(recent_popular)

        return {
            "total_items": total,
            "unique_items": len(counter),
            "max_frequency": counter.most_common(1)[0][1] if counter else 0,
            "max_popularity": (
                counter.most_common(1)[0][1] / total if counter and total > 0 else 0.0
            ),
        }
