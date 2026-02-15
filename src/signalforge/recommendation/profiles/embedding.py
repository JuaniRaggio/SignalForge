"""User embedding generation for recommendations."""

from __future__ import annotations

import math
from datetime import datetime

import structlog

from signalforge.recommendation.profiles.schemas import (
    ExplicitProfile,
    ImplicitProfile,
    InvestmentHorizon,
    PortfolioPosition,
    RiskTolerance,
    UserEmbedding,
)

logger = structlog.get_logger(__name__)


class UserEmbeddingGenerator:
    """Generate user embeddings for recommendation."""

    def __init__(self, embedding_dim: int = 64) -> None:
        """Initialize user embedding generator.

        Args:
            embedding_dim: Dimension of embedding vector.
        """
        self._embedding_dim = embedding_dim
        logger.info("user_embedding_generator_initialized", dim=embedding_dim)

    def generate_embedding(
        self,
        explicit: ExplicitProfile,
        implicit: ImplicitProfile,
    ) -> UserEmbedding:
        """Generate user embedding from profile data.

        Process:
        1. Encode portfolio composition (sectors, size)
        2. Encode watchlist preferences
        3. Encode behavior patterns
        4. Encode risk tolerance and horizon
        5. Combine into single vector

        Args:
            explicit: Explicit user profile.
            implicit: Implicit user profile.

        Returns:
            User embedding vector.
        """
        # Allocate dimensions
        # 16 dims for portfolio, 8 for watchlist, 24 for behavior, 16 for preferences
        portfolio_vec = self._encode_portfolio(explicit.portfolio)[:16]
        watchlist_vec = self._encode_watchlist(explicit.watchlist)[:8]
        behavior_vec = self._encode_behavior(implicit)[:24]
        prefs_vec = self._encode_preferences(explicit)[:16]

        # Pad if necessary
        portfolio_vec = self._pad_vector(portfolio_vec, 16)
        watchlist_vec = self._pad_vector(watchlist_vec, 8)
        behavior_vec = self._pad_vector(behavior_vec, 24)
        prefs_vec = self._pad_vector(prefs_vec, 16)

        # Combine
        embedding = portfolio_vec + watchlist_vec + behavior_vec + prefs_vec

        # Normalize
        embedding = self._normalize_vector(embedding)

        logger.info("embedding_generated", user_id=explicit.user_id)

        return UserEmbedding(
            user_id=explicit.user_id,
            embedding=embedding,
            version=1,
            created_at=datetime.utcnow(),
        )

    def _encode_portfolio(self, portfolio: list[PortfolioPosition]) -> list[float]:
        """Encode portfolio as vector.

        Args:
            portfolio: List of portfolio positions.

        Returns:
            Portfolio encoding vector.
        """
        if not portfolio:
            return [0.0] * 16

        # Calculate total value
        total_value = sum(p.shares * p.cost_basis for p in portfolio)

        # Encode portfolio size (log scale)
        size_encoding = math.log(total_value + 1) / 20.0  # normalized to ~0-1

        # Encode number of positions
        position_count = len(portfolio) / 100.0  # normalized

        # Encode position concentration (Herfindahl index)
        if total_value > 0:
            weights = [(p.shares * p.cost_basis) / total_value for p in portfolio]
            concentration = sum(w * w for w in weights)
        else:
            concentration = 0.0

        # Encode average holding period (days)
        if portfolio:
            holding_periods = [
                (datetime.utcnow() - p.acquired_at).days for p in portfolio
            ]
            avg_holding = sum(holding_periods) / len(holding_periods)
            holding_encoding = math.log(avg_holding + 1) / 10.0
        else:
            holding_encoding = 0.0

        # Create vector with basic metrics
        vec = [
            size_encoding,
            position_count,
            concentration,
            holding_encoding,
        ]

        # Pad to 16 dimensions
        return self._pad_vector(vec, 16)

    def _encode_watchlist(self, watchlist: list[str]) -> list[float]:
        """Encode watchlist as vector.

        Args:
            watchlist: List of symbols in watchlist.

        Returns:
            Watchlist encoding vector.
        """
        if not watchlist:
            return [0.0] * 8

        # Encode watchlist size
        size = len(watchlist) / 50.0  # normalized

        # Create simple vector
        vec = [size]

        return self._pad_vector(vec, 8)

    def _encode_sectors(self, sectors: dict[str, float]) -> list[float]:
        """Encode sector preferences.

        Args:
            sectors: Dictionary mapping sector to preference score.

        Returns:
            Sector encoding vector.
        """
        if not sectors:
            return [0.0] * 8

        # Define common sectors
        common_sectors = [
            "Technology",
            "Financials",
            "Healthcare",
            "Consumer",
            "Industrial",
            "Energy",
            "Materials",
            "Utilities",
        ]

        # Encode as vector
        vec = [sectors.get(sector, 0.0) for sector in common_sectors]

        return vec

    def _encode_behavior(self, implicit: ImplicitProfile) -> list[float]:
        """Encode behavior patterns.

        Args:
            implicit: Implicit user profile.

        Returns:
            Behavior encoding vector.
        """
        engagement = implicit.engagement_pattern

        # Encode session metrics
        session_duration = min(engagement.avg_session_duration / 3600.0, 1.0)  # hours
        sessions_per_week = min(engagement.sessions_per_week / 20.0, 1.0)

        # Encode time preferences
        morning_pref = sum(1 for h in engagement.preferred_hours if 6 <= h < 12) / 5.0
        afternoon_pref = (
            sum(1 for h in engagement.preferred_hours if 12 <= h < 18) / 5.0
        )
        evening_pref = sum(1 for h in engagement.preferred_hours if 18 <= h < 24) / 5.0

        # Encode day preferences
        weekday_pref = sum(1 for d in engagement.preferred_days if d < 5) / 3.0
        weekend_pref = sum(1 for d in engagement.preferred_days if d >= 5) / 3.0

        # Encode content preferences
        content_prefs = list(engagement.content_type_preferences.values())[:3]
        while len(content_prefs) < 3:
            content_prefs.append(0.0)

        # Encode sector affinity
        sector_vec = self._encode_sectors(implicit.sector_affinity)

        # Combine
        vec = (
            [
                session_duration,
                sessions_per_week,
                morning_pref,
                afternoon_pref,
                evening_pref,
                weekday_pref,
                weekend_pref,
            ]
            + content_prefs
            + sector_vec
        )

        return vec[:24]

    def _encode_preferences(self, explicit: ExplicitProfile) -> list[float]:
        """Encode user preferences.

        Args:
            explicit: Explicit user profile.

        Returns:
            Preferences encoding vector.
        """
        # Encode risk tolerance
        risk_map = {
            RiskTolerance.CONSERVATIVE: [1.0, 0.0, 0.0],
            RiskTolerance.MODERATE: [0.0, 1.0, 0.0],
            RiskTolerance.AGGRESSIVE: [0.0, 0.0, 1.0],
        }
        risk_vec = risk_map[explicit.risk_tolerance]

        # Encode investment horizon
        horizon_map = {
            InvestmentHorizon.DAY_TRADE: [1.0, 0.0, 0.0, 0.0],
            InvestmentHorizon.SWING: [0.0, 1.0, 0.0, 0.0],
            InvestmentHorizon.POSITION: [0.0, 0.0, 1.0, 0.0],
            InvestmentHorizon.LONG_TERM: [0.0, 0.0, 0.0, 1.0],
        }
        horizon_vec = horizon_map[explicit.investment_horizon]

        # Encode notification preferences
        notif = explicit.notification_preferences
        notif_vec = [
            1.0 if notif.email_enabled else 0.0,
            1.0 if notif.push_enabled else 0.0,
            notif.max_alerts_per_day / 20.0,
        ]

        # Encode preferred sectors
        sector_count = len(explicit.preferred_sectors) / 10.0

        # Combine
        vec = risk_vec + horizon_vec + notif_vec + [sector_count]

        return self._pad_vector(vec, 16)

    def compute_similarity(
        self, emb1: UserEmbedding, emb2: UserEmbedding
    ) -> float:
        """Compute cosine similarity between embeddings.

        Args:
            emb1: First user embedding.
            emb2: Second user embedding.

        Returns:
            Cosine similarity score (0-1).
        """
        vec1 = emb1.embedding
        vec2 = emb2.embedding

        if len(vec1) != len(vec2):
            raise ValueError("Embeddings must have same dimension")

        # Compute dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=True))

        # Compute magnitudes
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        similarity = dot_product / (mag1 * mag2)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, similarity))

    def find_similar_users(
        self,
        target: UserEmbedding,
        candidates: list[UserEmbedding],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Find most similar users.

        Args:
            target: Target user embedding.
            candidates: List of candidate user embeddings.
            top_k: Number of top similar users to return.

        Returns:
            List of (user_id, similarity_score) tuples, sorted by similarity.
        """
        similarities: list[tuple[str, float]] = []

        for candidate in candidates:
            if candidate.user_id == target.user_id:
                continue

            similarity = self.compute_similarity(target, candidate)
            similarities.append((candidate.user_id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            "similar_users_found",
            user_id=target.user_id,
            found=len(similarities[:top_k]),
        )

        return similarities[:top_k]

    def _pad_vector(self, vec: list[float], target_dim: int) -> list[float]:
        """Pad vector to target dimension.

        Args:
            vec: Input vector.
            target_dim: Target dimension.

        Returns:
            Padded vector.
        """
        if len(vec) >= target_dim:
            return vec[:target_dim]

        return vec + [0.0] * (target_dim - len(vec))

    def _normalize_vector(self, vec: list[float]) -> list[float]:
        """Normalize vector to unit length.

        Args:
            vec: Input vector.

        Returns:
            Normalized vector.
        """
        magnitude = math.sqrt(sum(x * x for x in vec))

        if magnitude == 0:
            return vec

        return [x / magnitude for x in vec]
