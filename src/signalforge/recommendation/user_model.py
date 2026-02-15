"""User modeling for personalized recommendations.

This module provides data structures and logic for building user profiles that
drive personalized signal recommendations. It supports:
- Explicit user preferences (stated preferences)
- Implicit user behavior tracking (observed activity)
- Combined profile generation
- Incremental profile updates

Key Features:
- Explicit preference modeling (risk tolerance, sectors, watchlist)
- Implicit behavior tracking (viewing history, preferences)
- Hybrid profile generation
- Profile embedding for similarity matching

Examples:
    Creating a user profile from preferences:

    >>> from signalforge.recommendation.user_model import UserModel
    >>> user_model = UserModel()
    >>> preferences = {
    ...     "risk_tolerance": "medium",
    ...     "investment_horizon": 30,
    ...     "preferred_sectors": ["Technology", "Healthcare"],
    ...     "watchlist": ["AAPL", "GOOGL", "MSFT"]
    ... }
    >>> explicit_profile = user_model.from_preferences(preferences)

    Updating implicit profile from activity:

    >>> activity = {"action": "view", "symbol": "AAPL", "sector": "Technology"}
    >>> implicit_profile = user_model.update_implicit(implicit_profile, activity)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from signalforge.core.logging import get_logger

logger = get_logger(__name__)

RiskTolerance = Literal["low", "medium", "high"]


@dataclass
class ExplicitProfile:
    """Explicit user preferences stated by the user.

    Attributes:
        risk_tolerance: User's risk tolerance level (low, medium, high).
        investment_horizon: Preferred holding period in days.
        preferred_sectors: List of preferred market sectors.
        watchlist: List of symbols the user is watching.
    """

    risk_tolerance: RiskTolerance
    investment_horizon: int
    preferred_sectors: list[str]
    watchlist: list[str]

    def __post_init__(self) -> None:
        """Validate explicit profile."""
        if self.risk_tolerance not in ("low", "medium", "high"):
            raise ValueError(f"Invalid risk tolerance: {self.risk_tolerance}")

        if self.investment_horizon <= 0:
            raise ValueError(f"Investment horizon must be positive, got {self.investment_horizon}")

        if not isinstance(self.preferred_sectors, list):
            raise TypeError("preferred_sectors must be a list")

        if not isinstance(self.watchlist, list):
            raise TypeError("watchlist must be a list")


@dataclass
class ImplicitProfile:
    """Implicit user preferences inferred from behavior.

    This profile is built by observing user actions such as viewing signals,
    clicking on sectors, or interacting with specific symbols.

    Attributes:
        viewed_sectors: Dictionary mapping sector names to view counts.
        viewed_symbols: Dictionary mapping symbols to view counts.
        avg_holding_period: Average holding period of viewed/interacted signals.
        preferred_volatility: Preferred volatility level inferred from interactions.
    """

    viewed_sectors: dict[str, int] = field(default_factory=dict)
    viewed_symbols: dict[str, int] = field(default_factory=dict)
    avg_holding_period: float = 0.0
    preferred_volatility: float = 0.0

    def __post_init__(self) -> None:
        """Validate implicit profile."""
        if self.avg_holding_period < 0.0:
            raise ValueError(
                f"Average holding period cannot be negative, got {self.avg_holding_period}"
            )

        if self.preferred_volatility < 0.0:
            raise ValueError(
                f"Preferred volatility cannot be negative, got {self.preferred_volatility}"
            )

        # Ensure view counts are non-negative
        for sector, count in self.viewed_sectors.items():
            if count < 0:
                raise ValueError(f"View count for sector {sector} cannot be negative: {count}")

        for symbol, count in self.viewed_symbols.items():
            if count < 0:
                raise ValueError(f"View count for symbol {symbol} cannot be negative: {count}")


@dataclass
class UserProfile:
    """Complete user profile combining explicit and implicit preferences.

    Attributes:
        user_id: Unique identifier for the user.
        explicit: Explicit preferences stated by user.
        implicit: Implicit preferences inferred from behavior.
        combined_embedding: Vector representation combining both profiles.
    """

    user_id: str
    explicit: ExplicitProfile
    implicit: ImplicitProfile
    combined_embedding: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Validate user profile."""
        if not self.user_id:
            raise ValueError("User ID cannot be empty")

        if len(self.combined_embedding) == 0:
            raise ValueError("Combined embedding cannot be empty")

        # Validate embedding contains valid values
        if np.any(np.isnan(self.combined_embedding)) or np.any(np.isinf(self.combined_embedding)):
            raise ValueError("Combined embedding contains invalid values (NaN or Inf)")


class UserModel:
    """Model for creating and managing user profiles.

    This class handles the creation of user profiles from both explicit
    preferences and implicit behavioral data, and combines them into a
    unified representation suitable for recommendation algorithms.

    Examples:
        >>> user_model = UserModel()
        >>> preferences = {
        ...     "risk_tolerance": "medium",
        ...     "investment_horizon": 30,
        ...     "preferred_sectors": ["Technology"],
        ...     "watchlist": ["AAPL"]
        ... }
        >>> explicit = user_model.from_preferences(preferences)
    """

    def __init__(self, embedding_dim: int = 32) -> None:
        """Initialize the user model.

        Args:
            embedding_dim: Dimensionality of user embedding vectors.
        """
        if embedding_dim <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {embedding_dim}")

        self._embedding_dim = embedding_dim
        logger.info("user_model_initialized", embedding_dim=embedding_dim)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimensionality."""
        return self._embedding_dim

    def from_preferences(self, preferences: dict[str, str | int | list[str]]) -> ExplicitProfile:
        """Create an explicit profile from user preferences.

        Args:
            preferences: Dictionary containing user preferences with keys:
                - risk_tolerance: "low", "medium", or "high"
                - investment_horizon: Preferred holding period in days
                - preferred_sectors: List of sector names
                - watchlist: List of symbol tickers

        Returns:
            ExplicitProfile object.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If values are invalid.

        Examples:
            >>> user_model = UserModel()
            >>> prefs = {
            ...     "risk_tolerance": "medium",
            ...     "investment_horizon": 30,
            ...     "preferred_sectors": ["Technology"],
            ...     "watchlist": ["AAPL", "MSFT"]
            ... }
            >>> explicit = user_model.from_preferences(prefs)
        """
        required_fields = [
            "risk_tolerance",
            "investment_horizon",
            "preferred_sectors",
            "watchlist",
        ]

        missing_fields = [field for field in required_fields if field not in preferences]
        if missing_fields:
            raise KeyError(f"Missing required fields: {missing_fields}")

        try:
            # Extract and validate fields
            investment_horizon_value = preferences["investment_horizon"]
            if not isinstance(investment_horizon_value, int):
                investment_horizon_value = int(investment_horizon_value)  # type: ignore[arg-type]

            preferred_sectors_value = preferences["preferred_sectors"]
            if not isinstance(preferred_sectors_value, list):
                raise TypeError("preferred_sectors must be a list")

            watchlist_value = preferences["watchlist"]
            if not isinstance(watchlist_value, list):
                raise TypeError("watchlist must be a list")

            profile = ExplicitProfile(
                risk_tolerance=str(preferences["risk_tolerance"]),  # type: ignore[arg-type]
                investment_horizon=investment_horizon_value,
                preferred_sectors=preferred_sectors_value,
                watchlist=watchlist_value,
            )

            logger.info(
                "explicit_profile_created",
                risk_tolerance=profile.risk_tolerance,
                num_sectors=len(profile.preferred_sectors),
                num_watchlist=len(profile.watchlist),
            )

            return profile

        except (ValueError, TypeError) as e:
            logger.error("explicit_profile_creation_failed", error=str(e), preferences=preferences)
            raise ValueError(f"Failed to create explicit profile: {e}") from e

    def from_activity(self, activities: list[dict[str, str | int | float]]) -> ImplicitProfile:
        """Create an implicit profile from user activity history.

        Args:
            activities: List of activity dictionaries, each containing:
                - action: Type of action (e.g., "view", "click")
                - symbol: Stock symbol (optional)
                - sector: Market sector (optional)
                - holding_period: Holding period in days (optional)
                - volatility: Signal volatility (optional)

        Returns:
            ImplicitProfile object with aggregated behavioral data.

        Raises:
            ValueError: If activities list is empty or invalid.

        Examples:
            >>> user_model = UserModel()
            >>> activities = [
            ...     {"action": "view", "symbol": "AAPL", "sector": "Technology"},
            ...     {"action": "view", "symbol": "GOOGL", "sector": "Technology"},
            ...     {"action": "click", "symbol": "MSFT", "sector": "Technology"}
            ... ]
            >>> implicit = user_model.from_activity(activities)
        """
        if not activities:
            logger.warning("empty_activity_list")
            return ImplicitProfile()

        viewed_sectors: dict[str, int] = defaultdict(int)
        viewed_symbols: dict[str, int] = defaultdict(int)
        holding_periods: list[float] = []
        volatilities: list[float] = []

        for activity in activities:
            # Count sector views
            if "sector" in activity and activity["sector"]:
                sector = str(activity["sector"])
                viewed_sectors[sector] += 1

            # Count symbol views
            if "symbol" in activity and activity["symbol"]:
                symbol = str(activity["symbol"])
                viewed_symbols[symbol] += 1

            # Collect holding periods
            if "holding_period" in activity and activity["holding_period"]:
                try:
                    holding_periods.append(float(activity["holding_period"]))
                except (ValueError, TypeError):
                    # Skip invalid values
                    logger.debug("invalid_holding_period_in_activity", activity=activity)

            # Collect volatility preferences
            if "volatility" in activity and activity["volatility"]:
                try:
                    volatilities.append(float(activity["volatility"]))
                except (ValueError, TypeError):
                    # Skip invalid values
                    logger.debug("invalid_volatility_in_activity", activity=activity)

        # Calculate averages
        avg_holding_period = float(np.mean(holding_periods)) if holding_periods else 0.0
        preferred_volatility = float(np.mean(volatilities)) if volatilities else 0.0

        profile = ImplicitProfile(
            viewed_sectors=dict(viewed_sectors),
            viewed_symbols=dict(viewed_symbols),
            avg_holding_period=avg_holding_period,
            preferred_volatility=preferred_volatility,
        )

        logger.info(
            "implicit_profile_created",
            num_activities=len(activities),
            unique_sectors=len(viewed_sectors),
            unique_symbols=len(viewed_symbols),
            avg_holding_period=avg_holding_period,
        )

        return profile

    def _create_explicit_embedding(self, explicit: ExplicitProfile) -> NDArray[np.float64]:
        """Create embedding vector from explicit preferences.

        Args:
            explicit: Explicit user profile.

        Returns:
            Embedding vector as numpy array.
        """
        # Risk tolerance encoding
        risk_encoding = {"low": [1.0, 0.0, 0.0], "medium": [0.0, 1.0, 0.0], "high": [0.0, 0.0, 1.0]}
        risk_vec = risk_encoding[explicit.risk_tolerance]

        # Investment horizon (normalized to ~3 months)
        horizon_norm = min(explicit.investment_horizon / 90.0, 1.0)

        # Sector preferences (simple count, normalized)
        num_sectors = len(explicit.preferred_sectors)
        sectors_norm = min(num_sectors / 10.0, 1.0)  # Normalize to max 10 sectors

        # Watchlist size (normalized)
        watchlist_size = len(explicit.watchlist)
        watchlist_norm = min(watchlist_size / 20.0, 1.0)  # Normalize to max 20 symbols

        # Combine features
        feature_vector = np.array(
            [horizon_norm, sectors_norm, watchlist_norm, *risk_vec], dtype=np.float64
        )

        return feature_vector

    def _create_implicit_embedding(self, implicit: ImplicitProfile) -> NDArray[np.float64]:
        """Create embedding vector from implicit behavior.

        Args:
            implicit: Implicit user profile.

        Returns:
            Embedding vector as numpy array.
        """
        # Total sector views (normalized)
        total_sector_views = sum(implicit.viewed_sectors.values())
        sector_views_norm = min(total_sector_views / 100.0, 1.0)

        # Total symbol views (normalized)
        total_symbol_views = sum(implicit.viewed_symbols.values())
        symbol_views_norm = min(total_symbol_views / 100.0, 1.0)

        # Holding period preference (normalized to ~3 months)
        holding_norm = min(implicit.avg_holding_period / 90.0, 1.0)

        # Volatility preference (capped at 1.0)
        volatility_norm = min(implicit.preferred_volatility, 1.0)

        # Sector diversity (unique sectors / total views)
        sector_diversity = (
            len(implicit.viewed_sectors) / total_sector_views if total_sector_views > 0 else 0.0
        )

        # Combine features
        feature_vector = np.array(
            [
                sector_views_norm,
                symbol_views_norm,
                holding_norm,
                volatility_norm,
                sector_diversity,
            ],
            dtype=np.float64,
        )

        return feature_vector

    def combine_profiles(
        self, explicit: ExplicitProfile, implicit: ImplicitProfile, weight: float = 0.7
    ) -> NDArray[np.float64]:
        """Combine explicit and implicit profiles into a single embedding.

        Args:
            explicit: Explicit user profile.
            implicit: Implicit user profile.
            weight: Weight for explicit profile (0.0 to 1.0).
                   Implicit profile receives (1 - weight).

        Returns:
            Combined embedding vector, L2 normalized.

        Raises:
            ValueError: If weight is not in [0.0, 1.0].

        Examples:
            >>> user_model = UserModel()
            >>> combined = user_model.combine_profiles(
            ...     explicit_profile,
            ...     implicit_profile,
            ...     weight=0.7
            ... )
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")

        logger.debug("combining_profiles", weight=weight)

        # Create individual embeddings
        explicit_emb = self._create_explicit_embedding(explicit)
        implicit_emb = self._create_implicit_embedding(implicit)

        # Combine with weighted average
        combined = np.concatenate([explicit_emb * weight, implicit_emb * (1 - weight)])

        # Pad or truncate to target dimension
        if len(combined) < self._embedding_dim:
            padding = np.zeros(self._embedding_dim - len(combined), dtype=np.float64)
            combined = np.concatenate([combined, padding])
        elif len(combined) > self._embedding_dim:
            combined = combined[: self._embedding_dim]

        # L2 normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        logger.debug(
            "profiles_combined",
            dimension=len(combined),
            norm=float(np.linalg.norm(combined)),
        )

        return combined

    def update_implicit(
        self, profile: ImplicitProfile, activity: dict[str, str | int | float]
    ) -> ImplicitProfile:
        """Update an implicit profile with new activity data.

        This method incrementally updates the implicit profile with new user
        activity, maintaining running averages and counts.

        Args:
            profile: Existing implicit profile to update.
            activity: New activity data dictionary with optional keys:
                - symbol: Stock symbol
                - sector: Market sector
                - holding_period: Holding period in days
                - volatility: Signal volatility

        Returns:
            Updated ImplicitProfile.

        Examples:
            >>> user_model = UserModel()
            >>> profile = ImplicitProfile()
            >>> activity = {"symbol": "AAPL", "sector": "Technology"}
            >>> updated = user_model.update_implicit(profile, activity)
        """
        # Create a copy to avoid mutating the original
        viewed_sectors = profile.viewed_sectors.copy()
        viewed_symbols = profile.viewed_symbols.copy()

        # Update sector views
        if "sector" in activity and activity["sector"]:
            sector = str(activity["sector"])
            viewed_sectors[sector] = viewed_sectors.get(sector, 0) + 1

        # Update symbol views
        if "symbol" in activity and activity["symbol"]:
            symbol = str(activity["symbol"])
            viewed_symbols[symbol] = viewed_symbols.get(symbol, 0) + 1

        # Update holding period (running average)
        avg_holding_period = profile.avg_holding_period
        if "holding_period" in activity and activity["holding_period"]:
            try:
                new_period = float(activity["holding_period"])
                total_views = sum(viewed_symbols.values())
                if total_views > 1:
                    # Update running average
                    avg_holding_period = (
                        avg_holding_period * (total_views - 1) + new_period
                    ) / total_views
                else:
                    avg_holding_period = new_period
            except (ValueError, TypeError):
                pass

        # Update volatility preference (running average)
        preferred_volatility = profile.preferred_volatility
        if "volatility" in activity and activity["volatility"]:
            try:
                new_volatility = float(activity["volatility"])
                total_views = sum(viewed_symbols.values())
                if total_views > 1:
                    # Update running average
                    preferred_volatility = (
                        preferred_volatility * (total_views - 1) + new_volatility
                    ) / total_views
                else:
                    preferred_volatility = new_volatility
            except (ValueError, TypeError):
                pass

        updated_profile = ImplicitProfile(
            viewed_sectors=viewed_sectors,
            viewed_symbols=viewed_symbols,
            avg_holding_period=avg_holding_period,
            preferred_volatility=preferred_volatility,
        )

        logger.debug(
            "implicit_profile_updated",
            total_sector_views=sum(viewed_sectors.values()),
            total_symbol_views=sum(viewed_symbols.values()),
        )

        return updated_profile

    def create_user_profile(
        self,
        user_id: str,
        explicit: ExplicitProfile,
        implicit: ImplicitProfile,
        weight: float = 0.7,
    ) -> UserProfile:
        """Create a complete user profile from explicit and implicit components.

        This is a convenience method that combines profile components and
        generates the final embedding.

        Args:
            user_id: Unique identifier for the user.
            explicit: Explicit user preferences.
            implicit: Implicit user behavior.
            weight: Weight for explicit profile in combination.

        Returns:
            Complete UserProfile with combined embedding.

        Examples:
            >>> user_model = UserModel()
            >>> profile = user_model.create_user_profile(
            ...     "user_123",
            ...     explicit_profile,
            ...     implicit_profile,
            ...     weight=0.7
            ... )
        """
        combined_embedding = self.combine_profiles(explicit, implicit, weight)

        user_profile = UserProfile(
            user_id=user_id,
            explicit=explicit,
            implicit=implicit,
            combined_embedding=combined_embedding,
        )

        logger.info(
            "user_profile_created",
            user_id=user_id,
            risk_tolerance=explicit.risk_tolerance,
            num_viewed_sectors=len(implicit.viewed_sectors),
        )

        return user_profile
