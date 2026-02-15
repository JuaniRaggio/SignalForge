"""Profile resolver for determining user output configuration."""

from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog

from signalforge.models.user import ExperienceLevel, User

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class OutputConfig:
    """Configuration for output adaptation based on user experience level."""

    level: ExperienceLevel
    include_glossary: bool
    include_raw_data: bool
    max_complexity: int


@dataclass(frozen=True)
class ResolvedProfile:
    """Resolved user profile with output configuration."""

    user_id: UUID
    output_config: OutputConfig
    preferences: dict[str, Any]


class ProfileResolver:
    """Resolves user profile to determine output adaptation strategy."""

    def __init__(self) -> None:
        """Initialize the profile resolver."""
        self._logger = logger.bind(component="profile_resolver")

    def resolve(self, user: User) -> ResolvedProfile:
        """
        Resolve user profile to output configuration.

        Args:
            user: User model instance

        Returns:
            ResolvedProfile with output configuration and preferences
        """
        self._logger.info(
            "resolving_user_profile",
            user_id=str(user.id),
            experience_level=user.experience_level.value,
        )

        base_config = self.get_default_config(user.experience_level)

        overrides: dict[str, Any] = {}
        if user.notification_preferences:
            prefs = user.notification_preferences
            if "include_raw_data" in prefs:
                overrides["include_raw_data"] = prefs["include_raw_data"]
            if "include_glossary" in prefs:
                overrides["include_glossary"] = prefs["include_glossary"]
            if "max_complexity" in prefs:
                overrides["max_complexity"] = prefs["max_complexity"]

        final_config = (
            self.override_config(base_config, overrides) if overrides else base_config
        )

        preferences = {
            "risk_tolerance": user.risk_tolerance.value,
            "investment_horizon": user.investment_horizon.value,
            "preferred_sectors": user.preferred_sectors or [],
            "watchlist": user.watchlist or [],
        }

        return ResolvedProfile(
            user_id=user.id,
            output_config=final_config,
            preferences=preferences,
        )

    def get_default_config(self, level: ExperienceLevel) -> OutputConfig:
        """
        Get default output configuration for experience level.

        Args:
            level: User experience level

        Returns:
            Default OutputConfig for the level

        Configuration by level:
            - CASUAL: Simple guidance, glossary enabled, no raw data, low complexity
            - INFORMED: Contextual interpretation, optional glossary, minimal raw, medium complexity
            - ACTIVE: Technical analysis, no glossary, some raw data, high complexity
            - QUANT: Raw data focus, no glossary, full raw data, maximum complexity
        """
        configs = {
            ExperienceLevel.CASUAL: OutputConfig(
                level=ExperienceLevel.CASUAL,
                include_glossary=True,
                include_raw_data=False,
                max_complexity=3,
            ),
            ExperienceLevel.INFORMED: OutputConfig(
                level=ExperienceLevel.INFORMED,
                include_glossary=True,
                include_raw_data=False,
                max_complexity=6,
            ),
            ExperienceLevel.ACTIVE: OutputConfig(
                level=ExperienceLevel.ACTIVE,
                include_glossary=False,
                include_raw_data=True,
                max_complexity=8,
            ),
            ExperienceLevel.QUANT: OutputConfig(
                level=ExperienceLevel.QUANT,
                include_glossary=False,
                include_raw_data=True,
                max_complexity=10,
            ),
        }

        config = configs[level]
        self._logger.debug(
            "default_config_retrieved",
            level=level.value,
            config=config,
        )
        return config

    def override_config(
        self, base: OutputConfig, overrides: dict[str, Any]
    ) -> OutputConfig:
        """
        Override base configuration with user-specific preferences.

        Args:
            base: Base OutputConfig to override
            overrides: Dictionary of override values

        Returns:
            New OutputConfig with overrides applied
        """
        self._logger.debug(
            "applying_config_overrides",
            base_level=base.level.value,
            overrides=overrides,
        )

        return OutputConfig(
            level=base.level,
            include_glossary=overrides.get("include_glossary", base.include_glossary),
            include_raw_data=overrides.get("include_raw_data", base.include_raw_data),
            max_complexity=overrides.get("max_complexity", base.max_complexity),
        )
