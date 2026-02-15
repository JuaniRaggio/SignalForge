"""Base formatter abstract class."""

from abc import ABC, abstractmethod
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class BaseFormatter(ABC):
    """
    Abstract base class for content formatters.

    Each formatter adapts data for a specific user experience level.
    """

    def __init__(self) -> None:
        """Initialize base formatter."""
        self._logger = logger.bind(
            component="formatter", formatter_type=self.__class__.__name__
        )
        self._logger.debug("formatter_initialized")

    @abstractmethod
    def format(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Format data for specific user experience level.

        Args:
            data: Raw input data

        Returns:
            Formatted data adapted to user level
        """
        pass

    def _extract_core_fields(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Extract core fields common across all formatters.

        Args:
            data: Input data

        Returns:
            Dictionary with core fields
        """
        core: dict[str, Any] = {}

        if "symbol" in data:
            core["symbol"] = data["symbol"]
        if "timestamp" in data:
            core["timestamp"] = data["timestamp"]
        if "type" in data:
            core["type"] = data["type"]

        return core

    def _safe_get(
        self, data: dict[str, Any], key: str, default: Any = None
    ) -> Any:
        """
        Safely get value from dictionary.

        Args:
            data: Dictionary to query
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            Value or default
        """
        return data.get(key, default)

    def _format_number(
        self, value: float | int, precision: int = 2, percentage: bool = False
    ) -> str:
        """
        Format number for display.

        Args:
            value: Number to format
            precision: Decimal places
            percentage: Whether to format as percentage

        Returns:
            Formatted string
        """
        if percentage:
            return f"{value * 100:.{precision}f}%"
        return f"{value:.{precision}f}"

    def _round_value(self, value: float, precision: int = 2) -> float:
        """
        Round value to precision.

        Args:
            value: Value to round
            precision: Decimal places

        Returns:
            Rounded value
        """
        return round(value, precision)
