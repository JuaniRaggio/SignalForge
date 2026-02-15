"""Explicit user profile management."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from signalforge.recommendation.profiles.schemas import (
    ExplicitProfile,
    PortfolioPosition,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class ExplicitProfileManager:
    """Manage explicit user profile data."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize explicit profile manager.

        Args:
            session: SQLAlchemy async session for database operations.
        """
        self._session = session
        self._profiles: dict[str, ExplicitProfile] = {}
        logger.info("explicit_profile_manager_initialized")

    async def create_profile(
        self, user_id: str, data: ExplicitProfile
    ) -> ExplicitProfile:
        """Create new user profile.

        Args:
            user_id: User identifier.
            data: Profile data.

        Returns:
            Created profile.
        """
        if user_id in self._profiles:
            logger.warning("profile_already_exists", user_id=user_id)
            raise ValueError(f"Profile already exists for user {user_id}")

        profile = ExplicitProfile(
            user_id=user_id,
            portfolio=data.portfolio,
            watchlist=data.watchlist,
            preferred_sectors=data.preferred_sectors,
            risk_tolerance=data.risk_tolerance,
            investment_horizon=data.investment_horizon,
            notification_preferences=data.notification_preferences,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        self._profiles[user_id] = profile
        logger.info("profile_created", user_id=user_id)
        return profile

    async def update_profile(
        self, user_id: str, updates: dict[str, object]
    ) -> ExplicitProfile:
        """Update profile fields.

        Args:
            user_id: User identifier.
            updates: Dictionary of fields to update.

        Returns:
            Updated profile.

        Raises:
            ValueError: If profile does not exist.
        """
        profile = await self.get_profile(user_id)
        if profile is None:
            logger.error("profile_not_found", user_id=user_id)
            raise ValueError(f"Profile not found for user {user_id}")

        # Update fields
        update_data = profile.model_dump()
        update_data.update(updates)
        update_data["updated_at"] = datetime.utcnow()

        updated_profile = ExplicitProfile(**update_data)
        self._profiles[user_id] = updated_profile

        logger.info("profile_updated", user_id=user_id, fields=list(updates.keys()))
        return updated_profile

    async def get_profile(self, user_id: str) -> ExplicitProfile | None:
        """Get user's explicit profile.

        Args:
            user_id: User identifier.

        Returns:
            User profile or None if not found.
        """
        return self._profiles.get(user_id)

    async def add_to_portfolio(
        self, user_id: str, position: PortfolioPosition
    ) -> None:
        """Add position to portfolio.

        Args:
            user_id: User identifier.
            position: Portfolio position to add.

        Raises:
            ValueError: If profile does not exist.
        """
        profile = await self.get_profile(user_id)
        if profile is None:
            logger.error("profile_not_found", user_id=user_id)
            raise ValueError(f"Profile not found for user {user_id}")

        # Check if symbol already exists
        existing_symbols = {p.symbol for p in profile.portfolio}
        if position.symbol in existing_symbols:
            # Update existing position
            profile.portfolio = [
                position if p.symbol == position.symbol else p
                for p in profile.portfolio
            ]
        else:
            profile.portfolio.append(position)

        profile.updated_at = datetime.utcnow()
        self._profiles[user_id] = profile

        logger.info(
            "portfolio_position_added",
            user_id=user_id,
            symbol=position.symbol,
        )

    async def remove_from_portfolio(self, user_id: str, symbol: str) -> None:
        """Remove position from portfolio.

        Args:
            user_id: User identifier.
            symbol: Stock symbol to remove.

        Raises:
            ValueError: If profile does not exist.
        """
        profile = await self.get_profile(user_id)
        if profile is None:
            logger.error("profile_not_found", user_id=user_id)
            raise ValueError(f"Profile not found for user {user_id}")

        profile.portfolio = [p for p in profile.portfolio if p.symbol != symbol]
        profile.updated_at = datetime.utcnow()
        self._profiles[user_id] = profile

        logger.info("portfolio_position_removed", user_id=user_id, symbol=symbol)

    async def update_watchlist(self, user_id: str, symbols: list[str]) -> None:
        """Update watchlist.

        Args:
            user_id: User identifier.
            symbols: List of symbols for watchlist.

        Raises:
            ValueError: If profile does not exist.
        """
        profile = await self.get_profile(user_id)
        if profile is None:
            logger.error("profile_not_found", user_id=user_id)
            raise ValueError(f"Profile not found for user {user_id}")

        profile.watchlist = symbols
        profile.updated_at = datetime.utcnow()
        self._profiles[user_id] = profile

        logger.info("watchlist_updated", user_id=user_id, count=len(symbols))

    async def get_portfolio_sectors(self, user_id: str) -> dict[str, float]:
        """Get sector allocation from portfolio.

        Args:
            user_id: User identifier.

        Returns:
            Dictionary mapping sector to allocation percentage.

        Raises:
            ValueError: If profile does not exist.
        """
        profile = await self.get_profile(user_id)
        if profile is None:
            logger.error("profile_not_found", user_id=user_id)
            raise ValueError(f"Profile not found for user {user_id}")

        if not profile.portfolio:
            return {}

        # Calculate total portfolio value
        total_value = sum(p.shares * p.cost_basis for p in profile.portfolio)

        if total_value == 0:
            return {}

        # For now, we'll use preferred_sectors as a proxy
        # In production, you'd map symbols to actual sectors
        sector_map: dict[str, float] = {}
        for position in profile.portfolio:
            position_value = position.shares * position.cost_basis
            # Map to first preferred sector (simplified)
            sector = (
                profile.preferred_sectors[0]
                if profile.preferred_sectors
                else "unknown"
            )
            sector_map[sector] = sector_map.get(sector, 0.0) + (
                position_value / total_value
            )

        logger.info("portfolio_sectors_calculated", user_id=user_id)
        return sector_map
