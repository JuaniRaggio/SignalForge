"""Implicit user behavior tracking."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from signalforge.recommendation.profiles.schemas import (
    EngagementPattern,
    ImplicitProfile,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class ImplicitProfileTracker:
    """Track implicit user behavior."""

    def __init__(
        self, session: AsyncSession, redis_client: Any | None = None
    ) -> None:
        """Initialize implicit profile tracker.

        Args:
            session: SQLAlchemy async session for database operations.
            redis_client: Optional Redis client for caching.
        """
        self._session = session
        self._redis = redis_client
        self._profiles: dict[str, ImplicitProfile] = {}
        logger.info("implicit_profile_tracker_initialized")

    async def track_view(
        self, user_id: str, symbol: str, duration_seconds: float
    ) -> None:
        """Track symbol view event.

        Args:
            user_id: User identifier.
            symbol: Stock symbol viewed.
            duration_seconds: Duration of view in seconds.
        """
        if user_id not in self._profiles:
            self._profiles[user_id] = ImplicitProfile(user_id=user_id)

        profile = self._profiles[user_id]
        profile.viewed_symbols.append((symbol, datetime.utcnow(), duration_seconds))

        logger.info(
            "view_tracked",
            user_id=user_id,
            symbol=symbol,
            duration=duration_seconds,
        )

    async def track_click(self, user_id: str, signal_id: str) -> None:
        """Track signal click.

        Args:
            user_id: User identifier.
            signal_id: Signal identifier clicked.
        """
        if user_id not in self._profiles:
            self._profiles[user_id] = ImplicitProfile(user_id=user_id)

        profile = self._profiles[user_id]
        profile.clicked_signals.append((signal_id, datetime.utcnow()))

        logger.info("click_tracked", user_id=user_id, signal_id=signal_id)

    async def track_search(self, user_id: str, query: str) -> None:
        """Track search query.

        Args:
            user_id: User identifier.
            query: Search query string.
        """
        if user_id not in self._profiles:
            self._profiles[user_id] = ImplicitProfile(user_id=user_id)

        profile = self._profiles[user_id]
        profile.search_history.append((query, datetime.utcnow()))

        logger.info("search_tracked", user_id=user_id, query=query)

    async def get_implicit_profile(self, user_id: str) -> ImplicitProfile:
        """Build implicit profile from tracked events.

        Args:
            user_id: User identifier.

        Returns:
            User's implicit profile.
        """
        if user_id not in self._profiles:
            logger.info("creating_new_implicit_profile", user_id=user_id)
            self._profiles[user_id] = ImplicitProfile(user_id=user_id)

        profile = self._profiles[user_id]

        # Calculate derived metrics
        profile.sector_affinity = await self.calculate_sector_affinity(user_id)
        profile.engagement_pattern = await self.calculate_engagement_pattern(user_id)

        return profile

    async def calculate_sector_affinity(self, user_id: str) -> dict[str, float]:
        """Calculate sector affinity from behavior.

        Args:
            user_id: User identifier.

        Returns:
            Dictionary mapping sector to affinity score (0-1).
        """
        if user_id not in self._profiles:
            return {}

        profile = self._profiles[user_id]
        sector_scores: dict[str, float] = defaultdict(float)

        # Weight by view duration
        for symbol, _timestamp, duration in profile.viewed_symbols:
            # In production, map symbol to sector
            sector = self._map_symbol_to_sector(symbol)
            sector_scores[sector] += duration

        # Normalize scores
        if sector_scores:
            max_score = max(sector_scores.values())
            if max_score > 0:
                sector_scores = {
                    sector: score / max_score
                    for sector, score in sector_scores.items()
                }

        logger.info("sector_affinity_calculated", user_id=user_id)
        return dict(sector_scores)

    async def calculate_engagement_pattern(self, user_id: str) -> EngagementPattern:
        """Analyze engagement patterns.

        Args:
            user_id: User identifier.

        Returns:
            User's engagement pattern.
        """
        if user_id not in self._profiles:
            return EngagementPattern()

        profile = self._profiles[user_id]

        # Calculate session metrics
        all_timestamps = [ts for _, ts, _ in profile.viewed_symbols]
        all_timestamps.extend([ts for _, ts in profile.clicked_signals])
        all_timestamps.extend([ts for _, ts in profile.search_history])

        if not all_timestamps:
            return EngagementPattern()

        all_timestamps.sort()

        # Calculate sessions (gap > 30 min = new session)
        sessions: list[list[datetime]] = []
        current_session: list[datetime] = []

        for ts in all_timestamps:
            if not current_session:
                current_session.append(ts)
            elif (ts - current_session[-1]).total_seconds() > 1800:  # 30 min
                sessions.append(current_session)
                current_session = [ts]
            else:
                current_session.append(ts)

        if current_session:
            sessions.append(current_session)

        # Calculate metrics
        avg_duration = 0.0
        if sessions:
            durations = [
                (session[-1] - session[0]).total_seconds() for session in sessions
            ]
            avg_duration = sum(durations) / len(durations)

        # Calculate sessions per week
        if all_timestamps:
            time_span = (all_timestamps[-1] - all_timestamps[0]).days + 1
            weeks = max(time_span / 7.0, 1.0)
            sessions_per_week = len(sessions) / weeks
        else:
            sessions_per_week = 0.0

        # Calculate preferred hours
        hour_counts: dict[int, int] = defaultdict(int)
        for ts in all_timestamps:
            hour_counts[ts.hour] += 1

        preferred_hours = sorted(
            hour_counts.keys(), key=lambda h: hour_counts[h], reverse=True
        )[:5]

        # Calculate preferred days
        day_counts: dict[int, int] = defaultdict(int)
        for ts in all_timestamps:
            day_counts[ts.weekday()] += 1

        preferred_days = sorted(
            day_counts.keys(), key=lambda d: day_counts[d], reverse=True
        )[:3]

        # Content type preferences (simplified)
        content_prefs = {
            "symbols": len(profile.viewed_symbols) / max(len(all_timestamps), 1),
            "signals": len(profile.clicked_signals) / max(len(all_timestamps), 1),
            "search": len(profile.search_history) / max(len(all_timestamps), 1),
        }

        logger.info("engagement_pattern_calculated", user_id=user_id)

        return EngagementPattern(
            avg_session_duration=avg_duration,
            sessions_per_week=sessions_per_week,
            preferred_hours=preferred_hours,
            preferred_days=preferred_days,
            content_type_preferences=content_prefs,
        )

    def _map_symbol_to_sector(self, symbol: str) -> str:
        """Map stock symbol to sector.

        Args:
            symbol: Stock symbol.

        Returns:
            Sector name.

        Note:
            This is a simplified implementation. In production,
            this would query a database or external service.
        """
        # Simplified mapping
        tech_symbols = {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"}
        finance_symbols = {"JPM", "BAC", "WFC", "GS", "MS", "C"}
        health_symbols = {"JNJ", "UNH", "PFE", "ABBV", "TMO", "ABT"}

        if symbol in tech_symbols:
            return "Technology"
        elif symbol in finance_symbols:
            return "Financials"
        elif symbol in health_symbols:
            return "Healthcare"
        else:
            return "Other"
