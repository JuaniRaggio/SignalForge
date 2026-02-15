"""Feed generation for personalized signal recommendations.

This module provides the feed generation logic that creates personalized
feeds of trading signals for users. It handles:
- Feed generation from ranked signals
- Watchlist boosting
- Diversity enforcement
- Confidence filtering

Key Features:
- Personalized feed generation
- Configurable feed size and quality thresholds
- Watchlist signal boosting
- Sector diversity constraints
- Explanation generation

Examples:
    Generating a personalized feed:

    >>> from signalforge.recommendation.feed_generator import FeedGenerator, FeedConfig
    >>> from signalforge.recommendation.ranking import RankingEngine
    >>> config = FeedConfig(
    ...     max_signals_per_day=10,
    ...     min_confidence=0.5,
    ...     include_watchlist_boost=True
    ... )
    >>> ranking_engine = RankingEngine()
    >>> feed_gen = FeedGenerator(config, ranking_engine)
    >>> feed = feed_gen.generate_feed(user_profile, available_signals)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from signalforge.core.logging import get_logger
from signalforge.recommendation.item_model import SignalItem
from signalforge.recommendation.ranking import RankedSignal, RankingEngine
from signalforge.recommendation.user_model import UserProfile

logger = get_logger(__name__)


@dataclass
class FeedConfig:
    """Configuration for feed generation.

    Attributes:
        max_signals_per_day: Maximum number of signals to include in the feed.
        min_confidence: Minimum confidence score for inclusion (0.0 to 1.0).
        include_watchlist_boost: Whether to boost watchlist signals.
    """

    max_signals_per_day: int = 10
    min_confidence: float = 0.5
    include_watchlist_boost: bool = True

    def __post_init__(self) -> None:
        """Validate feed configuration."""
        if self.max_signals_per_day <= 0:
            raise ValueError(
                f"max_signals_per_day must be positive, got {self.max_signals_per_day}"
            )

        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be between 0.0 and 1.0, got {self.min_confidence}"
            )


@dataclass
class FeedItem:
    """A signal in the personalized feed with metadata.

    Attributes:
        signal: The underlying signal item.
        rank: Ranked signal with scores and explanation.
        position: Position in the feed (1-indexed).
        reason: Reason why this signal was included in the feed.
    """

    signal: SignalItem
    rank: RankedSignal
    position: int
    reason: str

    def __post_init__(self) -> None:
        """Validate feed item."""
        if self.position <= 0:
            raise ValueError(f"Position must be positive, got {self.position}")

        if not self.reason:
            raise ValueError("Reason cannot be empty")


class FeedGenerator:
    """Generator for personalized signal feeds.

    This class orchestrates the feed generation process, combining ranking,
    filtering, boosting, and diversity constraints to produce high-quality
    personalized feeds.

    Examples:
        >>> config = FeedConfig(max_signals_per_day=10, min_confidence=0.5)
        >>> ranking_engine = RankingEngine()
        >>> feed_gen = FeedGenerator(config, ranking_engine)
        >>> feed = feed_gen.generate_feed(user_profile, signals)
    """

    def __init__(self, config: FeedConfig, ranking_engine: RankingEngine) -> None:
        """Initialize the feed generator.

        Args:
            config: Feed generation configuration.
            ranking_engine: Ranking engine instance for scoring signals.
        """
        self._config = config
        self._ranking_engine = ranking_engine

        logger.info(
            "feed_generator_initialized",
            max_signals=config.max_signals_per_day,
            min_confidence=config.min_confidence,
            watchlist_boost=config.include_watchlist_boost,
        )

    @property
    def config(self) -> FeedConfig:
        """Get the feed configuration."""
        return self._config

    @property
    def ranking_engine(self) -> RankingEngine:
        """Get the ranking engine."""
        return self._ranking_engine

    def boost_watchlist(
        self,
        signals: list[RankedSignal],
        watchlist: list[str],
        boost: float = 1.5,
    ) -> list[RankedSignal]:
        """Boost ranking scores for watchlist signals.

        Args:
            signals: List of ranked signals.
            watchlist: List of symbols in user's watchlist.
            boost: Multiplicative boost factor (should be > 1.0).

        Returns:
            List of ranked signals with boosted scores (re-sorted).

        Raises:
            ValueError: If boost factor is <= 0.

        Examples:
            >>> feed_gen = FeedGenerator(config, ranking_engine)
            >>> boosted = feed_gen.boost_watchlist(
            ...     ranked_signals,
            ...     ["AAPL", "GOOGL"],
            ...     boost=1.5
            ... )
        """
        if boost <= 0.0:
            raise ValueError(f"Boost factor must be positive, got {boost}")

        if not watchlist:
            return signals

        logger.debug(
            "boosting_watchlist_signals",
            num_signals=len(signals),
            watchlist_size=len(watchlist),
            boost=boost,
        )

        boosted_signals: list[RankedSignal] = []
        boosted_count = 0

        # Note: We don't have direct access to signal symbols here
        # In a real implementation, we would need to pass the original
        # SignalItem objects or include symbol in RankedSignal
        # For now, we'll create new RankedSignal objects with boosted scores

        for ranked_signal in signals:
            # Check if signal_id contains a watchlist symbol
            # This is a simplification - in production, we'd have a better mapping
            is_watchlist = any(symbol in ranked_signal.signal_id for symbol in watchlist)

            if is_watchlist:
                # Apply boost to the score
                new_score = min(ranked_signal.score * boost, 1.0)  # Cap at 1.0

                # Create new RankedSignal with boosted score
                boosted_signal = RankedSignal(
                    signal_id=ranked_signal.signal_id,
                    score=new_score,
                    relevance=ranked_signal.relevance,
                    confidence=ranked_signal.confidence,
                    recency=ranked_signal.recency,
                    explanation=f"[Watchlist] {ranked_signal.explanation}",
                )
                boosted_signals.append(boosted_signal)
                boosted_count += 1
            else:
                boosted_signals.append(ranked_signal)

        # Re-sort by score
        boosted_signals.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "watchlist_signals_boosted",
            total_signals=len(signals),
            boosted_count=boosted_count,
        )

        return boosted_signals

    def ensure_diversity(
        self,
        feed: list[FeedItem],
        max_per_sector: int = 3,
    ) -> list[FeedItem]:
        """Ensure sector diversity in the feed.

        This method enforces a maximum number of signals per sector to
        prevent over-concentration in any single sector.

        Args:
            feed: List of feed items.
            max_per_sector: Maximum signals allowed per sector.

        Returns:
            List of feed items with diversity constraints applied.

        Raises:
            ValueError: If max_per_sector is non-positive.

        Examples:
            >>> feed_gen = FeedGenerator(config, ranking_engine)
            >>> diversified = feed_gen.ensure_diversity(feed, max_per_sector=3)
        """
        if max_per_sector <= 0:
            raise ValueError(f"max_per_sector must be positive, got {max_per_sector}")

        if not feed:
            return feed

        logger.debug(
            "ensuring_feed_diversity",
            num_items=len(feed),
            max_per_sector=max_per_sector,
        )

        sector_counts: dict[str, int] = Counter()
        diversified_feed: list[FeedItem] = []

        for feed_item in feed:
            sector = feed_item.signal.features.sector

            if sector_counts[sector] < max_per_sector:
                diversified_feed.append(feed_item)
                sector_counts[sector] += 1
            else:
                logger.debug(
                    "signal_excluded_for_diversity",
                    signal_id=feed_item.signal.signal_id,
                    sector=sector,
                    reason=f"sector_limit_reached_{max_per_sector}",
                )

        # Update positions after filtering
        for idx, item in enumerate(diversified_feed, start=1):
            # Create new FeedItem with updated position
            diversified_feed[idx - 1] = FeedItem(
                signal=item.signal,
                rank=item.rank,
                position=idx,
                reason=item.reason,
            )

        logger.info(
            "feed_diversity_ensured",
            original_count=len(feed),
            diversified_count=len(diversified_feed),
            sectors_represented=len(sector_counts),
        )

        return diversified_feed

    def _filter_by_confidence(
        self,
        ranked: list[RankedSignal],
        min_confidence: float,
    ) -> list[RankedSignal]:
        """Filter signals by minimum confidence threshold.

        Args:
            ranked: List of ranked signals.
            min_confidence: Minimum confidence score.

        Returns:
            Filtered list of ranked signals.
        """
        filtered = [s for s in ranked if s.confidence >= min_confidence]

        logger.debug(
            "signals_filtered_by_confidence",
            original_count=len(ranked),
            filtered_count=len(filtered),
            min_confidence=min_confidence,
        )

        return filtered

    def _generate_reason(
        self,
        signal: SignalItem,
        rank: RankedSignal,
        is_watchlist: bool,
    ) -> str:
        """Generate a reason string for why a signal was included.

        Args:
            signal: Signal item.
            rank: Ranked signal.
            is_watchlist: Whether signal is in user's watchlist.

        Returns:
            Reason string.
        """
        reasons = []

        # Watchlist status
        if is_watchlist:
            reasons.append("In your watchlist")

        # High relevance
        if rank.relevance >= 0.7:
            reasons.append("Highly relevant to your preferences")

        # High confidence
        if rank.confidence >= 0.7:
            reasons.append("High confidence signal")

        # Recent signal
        if rank.recency >= 0.8:
            reasons.append("Recently generated")

        # Strong expected return
        if signal.features.expected_return >= 0.05:
            reasons.append(f"Strong expected return ({signal.features.expected_return:.1%})")

        # Preferred sector
        if reasons:
            return "; ".join(reasons)
        else:
            return "Matches your investment profile"

    def generate_feed(
        self,
        user: UserProfile,
        available_signals: list[SignalItem],
        confidence_scores: dict[str, float] | None = None,
    ) -> list[FeedItem]:
        """Generate a personalized feed of signals for a user.

        This method orchestrates the complete feed generation pipeline:
        1. Rank signals using the ranking engine
        2. Filter by confidence threshold
        3. Boost watchlist signals (if enabled)
        4. Apply diversity constraints
        5. Limit to max feed size
        6. Generate reasons for inclusion

        Args:
            user: User profile.
            available_signals: List of available signal items.
            confidence_scores: Optional dictionary mapping signal_id to confidence.

        Returns:
            List of FeedItem objects representing the personalized feed.

        Examples:
            >>> feed_gen = FeedGenerator(config, ranking_engine)
            >>> confidence = {"sig_1": 0.9, "sig_2": 0.7}
            >>> feed = feed_gen.generate_feed(
            ...     user_profile,
            ...     available_signals,
            ...     confidence
            ... )
        """
        if not available_signals:
            logger.warning("empty_available_signals_list")
            return []

        logger.info(
            "generating_feed",
            user_id=user.user_id,
            num_available_signals=len(available_signals),
        )

        # Step 1: Filter by risk tolerance
        risk_filtered = self._ranking_engine.filter_by_risk_tolerance(
            available_signals, user.explicit.risk_tolerance
        )

        if not risk_filtered:
            logger.warning(
                "no_signals_after_risk_filtering",
                user_id=user.user_id,
                risk_tolerance=user.explicit.risk_tolerance,
            )
            return []

        # Step 2: Rank signals
        ranked_signals = self._ranking_engine.rank_signals(user, risk_filtered, confidence_scores)

        if not ranked_signals:
            logger.warning("no_signals_after_ranking", user_id=user.user_id)
            return []

        # Step 3: Filter by confidence threshold
        confidence_filtered = self._filter_by_confidence(
            ranked_signals, self._config.min_confidence
        )

        if not confidence_filtered:
            logger.warning(
                "no_signals_after_confidence_filtering",
                user_id=user.user_id,
                min_confidence=self._config.min_confidence,
            )
            return []

        # Step 4: Boost watchlist signals if enabled
        if self._config.include_watchlist_boost and user.explicit.watchlist:
            confidence_filtered = self.boost_watchlist(
                confidence_filtered, user.explicit.watchlist, boost=1.5
            )

        # Step 5: Limit to max feed size
        top_signals = confidence_filtered[: self._config.max_signals_per_day]

        # Step 6: Create FeedItem objects
        # We need to map ranked signals back to signal items
        signal_map = {s.signal_id: s for s in risk_filtered}
        feed_items: list[FeedItem] = []

        for idx, ranked in enumerate(top_signals, start=1):
            signal_item = signal_map.get(ranked.signal_id)
            if signal_item is None:
                logger.warning(
                    "signal_not_found_in_map",
                    signal_id=ranked.signal_id,
                )
                continue

            # Check if in watchlist
            is_watchlist = signal_item.features.symbol in user.explicit.watchlist

            # Generate reason
            reason = self._generate_reason(signal_item, ranked, is_watchlist)

            feed_item = FeedItem(
                signal=signal_item, rank=ranked, position=idx, reason=reason
            )

            feed_items.append(feed_item)

        # Step 7: Ensure diversity
        diversified_feed = self.ensure_diversity(feed_items, max_per_sector=3)

        logger.info(
            "feed_generated",
            user_id=user.user_id,
            feed_size=len(diversified_feed),
            avg_score=sum(item.rank.score for item in diversified_feed) / len(diversified_feed)
            if diversified_feed
            else 0.0,
        )

        return diversified_feed
