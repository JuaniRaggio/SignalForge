"""Analysis formatter for active traders."""

from typing import Any

from signalforge.adaptation.formatters.base import BaseFormatter


class AnalysisFormatter(BaseFormatter):
    """
    Analysis formatter for ACTIVE experience level.

    Provides technical data with brief explanations.
    Highlights key metrics and actionable insights.
    No excessive simplification.
    """

    def format(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Format data for active traders.

        Includes technical metrics with concise context.

        Args:
            data: Raw input data

        Returns:
            Formatted data with technical analysis focus
        """
        self._logger.debug("formatting_analysis_data", keys=list(data.keys()))

        formatted = self._extract_core_fields(data)

        if "signal" in data:
            formatted["signal"] = self._format_signal(data["signal"])

        if "metrics" in data:
            formatted["metrics"] = self._format_metrics(data["metrics"])

        if "indicators" in data:
            formatted["indicators"] = self._format_indicators(data["indicators"])

        if "recommendations" in data:
            formatted["recommendations"] = self._format_recommendations(
                data["recommendations"]
            )

        formatted["_meta"] = {
            "formatter": "analysis",
            "experience_level": "active",
            "focus": "technical_metrics",
        }

        return formatted

    def _format_signal(self, signal: dict[str, Any]) -> dict[str, Any]:
        """Format signal data for active traders."""
        formatted: dict[str, Any] = {
            "type": self._safe_get(signal, "type"),
            "strength": self._safe_get(signal, "strength"),
            "confidence": self._format_number(
                self._safe_get(signal, "confidence", 0.0), 1, percentage=True
            ),
        }

        if "direction" in signal:
            formatted["direction"] = signal["direction"]

        if "price_target" in signal:
            formatted["price_target"] = self._round_value(signal["price_target"], 2)

        if "stop_loss" in signal:
            formatted["stop_loss"] = self._round_value(signal["stop_loss"], 2)

        if "risk_reward" in signal:
            formatted["risk_reward_ratio"] = self._round_value(signal["risk_reward"], 2)

        return formatted

    def _format_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Format metrics highlighting key values."""
        formatted: dict[str, Any] = {}

        key_metrics = [
            "rsi",
            "macd",
            "volume",
            "volatility",
            "sharpe_ratio",
            "max_drawdown",
            "alpha",
            "beta",
        ]

        for metric in key_metrics:
            if metric in metrics:
                value = metrics[metric]
                if isinstance(value, (int, float)):
                    formatted[metric] = self._round_value(value, 2)
                else:
                    formatted[metric] = value

        if "price" in metrics:
            formatted["current_price"] = self._round_value(metrics["price"], 2)

        if "change_percent" in metrics:
            formatted["change"] = self._format_number(
                metrics["change_percent"], 2, percentage=True
            )

        return formatted

    def _format_indicators(self, indicators: dict[str, Any]) -> dict[str, Any]:
        """Format technical indicators."""
        formatted: dict[str, Any] = {}

        if "moving_averages" in indicators:
            ma = indicators["moving_averages"]
            formatted["moving_averages"] = {
                k: self._round_value(v, 2) if isinstance(v, (int, float)) else v
                for k, v in ma.items()
            }

        if "oscillators" in indicators:
            formatted["oscillators"] = indicators["oscillators"]

        if "trend" in indicators:
            formatted["trend"] = {
                "direction": indicators["trend"].get("direction"),
                "strength": indicators["trend"].get("strength"),
            }

        return formatted

    def _format_recommendations(
        self, recommendations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Format actionable recommendations."""
        formatted = []

        for rec in recommendations[:5]:
            formatted_rec: dict[str, Any] = {
                "action": self._safe_get(rec, "action"),
                "rationale": self._safe_get(rec, "rationale"),
            }

            if "priority" in rec:
                formatted_rec["priority"] = rec["priority"]

            if "timeframe" in rec:
                formatted_rec["timeframe"] = rec["timeframe"]

            formatted.append(formatted_rec)

        return formatted
