"""Ranking engine for scoring and ordering signals.

This module provides the core ranking logic for ordering signals based on
multiple factors including:
- Relevance to user preferences
- Signal confidence/quality
- Recency of the signal
- Diversity considerations

Key Features:
- Multi-factor ranking with configurable weights
- User-signal relevance scoring
- Time-based recency scoring
- Diversity penalty application
- Risk tolerance filtering

Examples:
    Ranking signals for a user:

    >>> from signalforge.recommendation.ranking import RankingEngine, RankingConfig
    >>> config = RankingConfig(
    ...     relevance_weight=0.5,
    ...     confidence_weight=0.3,
    ...     recency_weight=0.2,
    ...     diversity_penalty=0.1
    ... )
    >>> engine = RankingEngine(config)
    >>> ranked = engine.rank_signals(user_profile, signal_items)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from signalforge.core.logging import get_logger
from signalforge.recommendation.item_model import SignalItem
from signalforge.recommendation.user_model import UserProfile

logger = get_logger(__name__)


@dataclass
class RankingConfig:
    """Configuration for signal ranking.

    Attributes:
        relevance_weight: Weight for user-signal relevance score (0.0 to 1.0).
        confidence_weight: Weight for signal confidence/quality (0.0 to 1.0).
        recency_weight: Weight for signal freshness (0.0 to 1.0).
        diversity_penalty: Penalty factor for over-representation (0.0 to 1.0).
    """

    relevance_weight: float = 0.5
    confidence_weight: float = 0.3
    recency_weight: float = 0.2
    diversity_penalty: float = 0.1

    def __post_init__(self) -> None:
        """Validate ranking configuration."""
        if not 0.0 <= self.relevance_weight <= 1.0:
            raise ValueError(
                f"relevance_weight must be between 0.0 and 1.0, got {self.relevance_weight}"
            )

        if not 0.0 <= self.confidence_weight <= 1.0:
            raise ValueError(
                f"confidence_weight must be between 0.0 and 1.0, got {self.confidence_weight}"
            )

        if not 0.0 <= self.recency_weight <= 1.0:
            raise ValueError(
                f"recency_weight must be between 0.0 and 1.0, got {self.recency_weight}"
            )

        if not 0.0 <= self.diversity_penalty <= 1.0:
            raise ValueError(
                f"diversity_penalty must be between 0.0 and 1.0, got {self.diversity_penalty}"
            )

        # Weights should sum to approximately 1.0 (allow some tolerance)
        total_weight = self.relevance_weight + self.confidence_weight + self.recency_weight
        if not 0.95 <= total_weight <= 1.05:
            logger.warning(
                "ranking_weights_sum_warning",
                total=total_weight,
                expected=1.0,
            )


@dataclass
class RankedSignal:
    """A signal with its ranking score and components.

    Attributes:
        signal_id: Unique identifier for the signal.
        score: Overall ranking score (0.0 to 1.0).
        relevance: Relevance score component (0.0 to 1.0).
        confidence: Confidence score component (0.0 to 1.0).
        recency: Recency score component (0.0 to 1.0).
        explanation: Human-readable explanation of the ranking.
    """

    signal_id: str
    score: float
    relevance: float
    confidence: float
    recency: float
    explanation: str

    def __post_init__(self) -> None:
        """Validate ranked signal."""
        if not self.signal_id:
            raise ValueError("Signal ID cannot be empty")

        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")

        if not 0.0 <= self.relevance <= 1.0:
            raise ValueError(f"Relevance must be between 0.0 and 1.0, got {self.relevance}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        if not 0.0 <= self.recency <= 1.0:
            raise ValueError(f"Recency must be between 0.0 and 1.0, got {self.recency}")


class RankingEngine:
    """Engine for ranking and ordering signals for users.

    This class implements a multi-factor ranking algorithm that combines
    relevance, confidence, and recency to produce personalized signal rankings.

    Examples:
        >>> config = RankingConfig(
        ...     relevance_weight=0.5,
        ...     confidence_weight=0.3,
        ...     recency_weight=0.2
        ... )
        >>> engine = RankingEngine(config)
        >>> ranked = engine.rank_signals(user_profile, signals)
    """

    def __init__(self, config: RankingConfig | None = None) -> None:
        """Initialize the ranking engine.

        Args:
            config: Ranking configuration. If None, uses defaults.
        """
        self._config = config or RankingConfig()
        logger.info(
            "ranking_engine_initialized",
            relevance_weight=self._config.relevance_weight,
            confidence_weight=self._config.confidence_weight,
            recency_weight=self._config.recency_weight,
        )

    @property
    def config(self) -> RankingConfig:
        """Get the ranking configuration."""
        return self._config

    def calculate_relevance(self, user: UserProfile, signal: SignalItem) -> float:
        """Calculate relevance score between user and signal.

        This method computes how relevant a signal is to a user based on:
        - Embedding similarity
        - Sector preference match
        - Watchlist match
        - Risk tolerance alignment

        Args:
            user: User profile.
            signal: Signal item.

        Returns:
            Relevance score between 0.0 and 1.0.

        Examples:
            >>> engine = RankingEngine()
            >>> relevance = engine.calculate_relevance(user_profile, signal_item)
        """
        logger.debug(
            "calculating_relevance",
            user_id=user.user_id,
            signal_id=signal.signal_id,
        )

        # Embedding similarity (cosine similarity for normalized vectors)
        embedding_sim = float(np.dot(user.combined_embedding, signal.embedding))
        embedding_sim = max(0.0, min(1.0, (embedding_sim + 1.0) / 2.0))  # Map [-1,1] to [0,1]

        # Sector preference match
        sector_match = 0.0
        if signal.features.sector in user.explicit.preferred_sectors:
            sector_match = 1.0
        elif signal.features.sector in user.implicit.viewed_sectors:
            # Partial match based on view frequency
            total_sector_views = sum(user.implicit.viewed_sectors.values())
            sector_views = user.implicit.viewed_sectors[signal.features.sector]
            sector_match = min(sector_views / total_sector_views, 1.0) if total_sector_views > 0 else 0.0

        # Watchlist match
        watchlist_match = 1.0 if signal.features.symbol in user.explicit.watchlist else 0.0

        # Risk tolerance alignment
        risk_map = {"low": 0, "medium": 1, "high": 2}
        user_risk = risk_map[user.explicit.risk_tolerance]
        signal_risk = risk_map[signal.features.risk_level]
        risk_diff = abs(user_risk - signal_risk)
        risk_alignment = 1.0 - (risk_diff / 2.0)  # Max diff is 2, so normalize to [0, 1]

        # Combine factors (weighted average)
        relevance = (
            embedding_sim * 0.4 + sector_match * 0.3 + watchlist_match * 0.2 + risk_alignment * 0.1
        )

        logger.debug(
            "relevance_calculated",
            signal_id=signal.signal_id,
            relevance=relevance,
            embedding_sim=embedding_sim,
            sector_match=sector_match,
            watchlist_match=watchlist_match,
        )

        return float(relevance)

    def calculate_recency(self, signal: SignalItem, max_age_hours: int = 48) -> float:
        """Calculate recency score based on signal age.

        More recent signals receive higher scores. The score decays exponentially
        with age up to max_age_hours.

        Args:
            signal: Signal item.
            max_age_hours: Maximum age in hours for non-zero score.

        Returns:
            Recency score between 0.0 and 1.0.

        Examples:
            >>> engine = RankingEngine()
            >>> recency = engine.calculate_recency(signal_item, max_age_hours=48)
        """
        if max_age_hours <= 0:
            raise ValueError(f"max_age_hours must be positive, got {max_age_hours}")

        now = datetime.utcnow()
        age = now - signal.created_at
        age_hours = age.total_seconds() / 3600.0

        if age_hours < 0:
            logger.warning(
                "future_signal_timestamp",
                signal_id=signal.signal_id,
                created_at=signal.created_at,
            )
            return 1.0  # Future signals get max score

        if age_hours >= max_age_hours:
            return 0.0

        # Exponential decay: score = exp(-lambda * age)
        # At max_age_hours, score should be ~0.01
        decay_rate = -math.log(0.01) / max_age_hours
        recency_score = math.exp(-decay_rate * age_hours)

        logger.debug(
            "recency_calculated",
            signal_id=signal.signal_id,
            age_hours=age_hours,
            recency=recency_score,
        )

        return float(recency_score)

    def apply_diversity_penalty(
        self, ranked: list[RankedSignal], penalty: float = 0.1
    ) -> list[RankedSignal]:
        """Apply diversity penalty to over-represented sectors.

        This method reduces scores for signals from sectors that appear too
        frequently in the ranking to encourage diversity.

        Args:
            ranked: List of ranked signals (sorted by score).
            penalty: Penalty factor per additional occurrence (0.0 to 1.0).

        Returns:
            List of ranked signals with adjusted scores.

        Examples:
            >>> engine = RankingEngine()
            >>> diversified = engine.apply_diversity_penalty(ranked_signals, penalty=0.1)
        """
        if not 0.0 <= penalty <= 1.0:
            raise ValueError(f"Penalty must be between 0.0 and 1.0, got {penalty}")

        if not ranked:
            return ranked

        logger.debug("applying_diversity_penalty", num_signals=len(ranked), penalty=penalty)

        # This is a placeholder - in a real implementation, we would need
        # access to the signal features to determine sectors
        # For now, return as-is
        logger.warning("diversity_penalty_not_implemented_yet")
        return ranked

    def filter_by_risk_tolerance(
        self, signals: list[SignalItem], tolerance: str
    ) -> list[SignalItem]:
        """Filter signals by user's risk tolerance.

        This method filters out signals that don't match the user's risk profile:
        - low: Only low risk signals
        - medium: Low and medium risk signals
        - high: All risk levels

        Args:
            signals: List of signal items.
            tolerance: Risk tolerance level ("low", "medium", "high").

        Returns:
            Filtered list of signals.

        Raises:
            ValueError: If tolerance is invalid.

        Examples:
            >>> engine = RankingEngine()
            >>> filtered = engine.filter_by_risk_tolerance(signals, "medium")
        """
        if tolerance not in ("low", "medium", "high"):
            raise ValueError(f"Invalid risk tolerance: {tolerance}")

        logger.debug(
            "filtering_by_risk_tolerance",
            num_signals=len(signals),
            tolerance=tolerance,
        )

        if tolerance == "high":
            # Accept all risk levels
            return signals

        allowed_risks = {"low": ["low"], "medium": ["low", "medium"]}[tolerance]

        filtered = [s for s in signals if s.features.risk_level in allowed_risks]

        logger.info(
            "signals_filtered_by_risk",
            original_count=len(signals),
            filtered_count=len(filtered),
            tolerance=tolerance,
        )

        return filtered

    def rank_signals(
        self, user: UserProfile, signals: list[SignalItem], confidence_scores: dict[str, float] | None = None
    ) -> list[RankedSignal]:
        """Rank signals for a user based on multiple factors.

        This method combines relevance, confidence, and recency to produce
        a final ranking score for each signal.

        Args:
            user: User profile.
            signals: List of signal items to rank.
            confidence_scores: Optional dictionary mapping signal_id to confidence
                              scores. If not provided, defaults to 0.5 for all.

        Returns:
            List of RankedSignal objects sorted by score (descending).

        Examples:
            >>> engine = RankingEngine()
            >>> confidence = {"sig_1": 0.9, "sig_2": 0.7}
            >>> ranked = engine.rank_signals(user_profile, signals, confidence)
        """
        if not signals:
            logger.warning("empty_signals_list_for_ranking")
            return []

        logger.info(
            "ranking_signals",
            user_id=user.user_id,
            num_signals=len(signals),
        )

        # Default confidence if not provided
        if confidence_scores is None:
            confidence_scores = {}

        ranked_signals: list[RankedSignal] = []

        for signal in signals:
            # Calculate individual components
            relevance = self.calculate_relevance(user, signal)
            confidence = confidence_scores.get(signal.signal_id, 0.5)
            recency = self.calculate_recency(signal)

            # Validate confidence
            if not 0.0 <= confidence <= 1.0:
                logger.warning(
                    "invalid_confidence_score",
                    signal_id=signal.signal_id,
                    confidence=confidence,
                )
                confidence = 0.5

            # Calculate weighted overall score
            score = (
                relevance * self._config.relevance_weight
                + confidence * self._config.confidence_weight
                + recency * self._config.recency_weight
            )

            # Generate explanation
            explanation = self._generate_explanation(signal, relevance, confidence, recency)

            ranked_signal = RankedSignal(
                signal_id=signal.signal_id,
                score=score,
                relevance=relevance,
                confidence=confidence,
                recency=recency,
                explanation=explanation,
            )

            ranked_signals.append(ranked_signal)

        # Sort by score (descending)
        ranked_signals.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "signals_ranked",
            num_signals=len(ranked_signals),
            top_score=ranked_signals[0].score if ranked_signals else 0.0,
            avg_score=sum(s.score for s in ranked_signals) / len(ranked_signals)
            if ranked_signals
            else 0.0,
        )

        return ranked_signals

    def _generate_explanation(
        self,
        signal: SignalItem,
        relevance: float,
        confidence: float,
        recency: float,
    ) -> str:
        """Generate human-readable explanation for ranking.

        Args:
            signal: Signal item.
            relevance: Relevance score.
            confidence: Confidence score.
            recency: Recency score.

        Returns:
            Explanation string.
        """
        # Identify the dominant factor
        factors = {
            "relevance": (relevance, self._config.relevance_weight),
            "confidence": (confidence, self._config.confidence_weight),
            "recency": (recency, self._config.recency_weight),
        }

        weighted_factors = {name: value * weight for name, (value, weight) in factors.items()}
        dominant_factor = max(weighted_factors.items(), key=lambda x: x[1])[0]

        explanations = {
            "relevance": f"Highly relevant to your preferences (score: {relevance:.2f})",
            "confidence": f"High confidence signal (score: {confidence:.2f})",
            "recency": f"Recent signal (score: {recency:.2f})",
        }

        base_explanation = explanations[dominant_factor]

        # Add sector and symbol information
        full_explanation = (
            f"{base_explanation}. "
            f"{signal.features.symbol} ({signal.features.sector}), "
            f"expected return: {signal.features.expected_return:.1%}, "
            f"risk: {signal.features.risk_level}"
        )

        return full_explanation
