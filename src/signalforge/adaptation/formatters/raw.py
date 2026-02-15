"""Raw formatter for quantitative/professional users."""

from typing import Any

from signalforge.adaptation.formatters.base import BaseFormatter


class RawFormatter(BaseFormatter):
    """
    Raw formatter for QUANT experience level.

    Returns complete JSON data without modifications.
    Includes all technical fields and preserves precision.
    """

    def format(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Format data for quantitative users.

        Returns complete, unmodified data with all fields.

        Args:
            data: Raw input data

        Returns:
            Complete data dictionary with all fields preserved
        """
        self._logger.debug("formatting_raw_data", keys=list(data.keys()))

        formatted = data.copy()

        formatted["_meta"] = {
            "formatter": "raw",
            "experience_level": "quant",
            "precision": "full",
            "fields_included": "all",
        }

        return formatted
