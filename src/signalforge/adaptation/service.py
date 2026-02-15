"""Adaptation service for orchestrating content adaptation."""

from typing import Any

import structlog

from signalforge.adaptation.formatters.base import BaseFormatter
from signalforge.adaptation.profile_resolver import ProfileResolver
from signalforge.adaptation.template_engine import TemplateEngine
from signalforge.models.user import ExperienceLevel, User

logger = structlog.get_logger(__name__)


class AdaptationService:
    """
    Main service orchestrating content adaptation.

    Coordinates profile resolution, formatting, and template rendering
    to deliver user-specific content.
    """

    def __init__(
        self,
        profile_resolver: ProfileResolver,
        template_engine: TemplateEngine,
        formatters: dict[ExperienceLevel, BaseFormatter],
    ) -> None:
        """
        Initialize adaptation service.

        Args:
            profile_resolver: Profile resolver instance
            template_engine: Template engine instance
            formatters: Dictionary mapping experience level to formatter
        """
        self._profile_resolver = profile_resolver
        self._template_engine = template_engine
        self._formatters = formatters
        self._logger = logger.bind(component="adaptation_service")
        self._logger.info(
            "adaptation_service_initialized",
            formatter_count=len(formatters),
        )

    def adapt(self, data: dict[str, Any], user: User) -> dict[str, Any]:
        """
        Adapt generic data to user experience level.

        Args:
            data: Raw data to adapt
            user: User model instance

        Returns:
            Adapted data formatted for user level
        """
        self._logger.info(
            "adapting_content",
            user_id=str(user.id),
            experience_level=user.experience_level.value,
        )

        profile = self._profile_resolver.resolve(user)

        formatter = self._formatters.get(profile.output_config.level)
        if not formatter:
            self._logger.warning(
                "formatter_not_found",
                level=profile.output_config.level.value,
            )
            return data

        formatted = formatter.format(data)

        formatted["_user_preferences"] = profile.preferences
        formatted["_output_config"] = {
            "level": profile.output_config.level.value,
            "include_glossary": profile.output_config.include_glossary,
            "include_raw_data": profile.output_config.include_raw_data,
            "max_complexity": profile.output_config.max_complexity,
        }

        self._logger.debug(
            "content_adapted",
            user_id=str(user.id),
            formatter_used=formatter.__class__.__name__,
        )

        return formatted

    def adapt_signal(self, signal: dict[str, Any], user: User) -> dict[str, Any]:
        """
        Adapt trading signal to user experience level.

        Args:
            signal: Raw signal data
            user: User model instance

        Returns:
            Adapted signal formatted for user level
        """
        self._logger.info(
            "adapting_signal",
            user_id=str(user.id),
            signal_type=signal.get("type"),
        )

        signal_with_type = signal.copy()
        signal_with_type["_content_type"] = "signal"

        adapted = self.adapt(signal_with_type, user)

        profile = self._profile_resolver.resolve(user)
        if "description" in adapted and profile.output_config.include_glossary:
            adapted["description"] = self._template_engine.render(
                "signal", {"description": adapted["description"]}, profile.output_config
            )

        return adapted

    def adapt_news(self, news: dict[str, Any], user: User) -> dict[str, Any]:
        """
        Adapt news article to user experience level.

        Args:
            news: Raw news data
            user: User model instance

        Returns:
            Adapted news formatted for user level
        """
        self._logger.info(
            "adapting_news",
            user_id=str(user.id),
            news_title=news.get("title", "unknown"),
        )

        news_with_type = news.copy()
        news_with_type["_content_type"] = "news"

        adapted = self.adapt(news_with_type, user)

        profile = self._profile_resolver.resolve(user)

        if "summary" in adapted and profile.output_config.max_complexity < 10:
            adapted["summary"] = self._template_engine.simplify_text(
                adapted["summary"], profile.output_config.max_complexity
            )

        if "summary" in adapted and profile.output_config.include_glossary:
            adapted["summary"] = self._template_engine._glossary.inject_tooltips(
                adapted["summary"]
            )

        return adapted

    def get_formatter_for_user(self, user: User) -> BaseFormatter:
        """
        Get formatter instance for user.

        Args:
            user: User model instance

        Returns:
            Formatter instance for user's experience level
        """
        profile = self._profile_resolver.resolve(user)
        formatter = self._formatters.get(profile.output_config.level)

        if not formatter:
            self._logger.error(
                "no_formatter_available",
                level=profile.output_config.level.value,
            )
            raise ValueError(
                f"No formatter available for level: {profile.output_config.level.value}"
            )

        return formatter
