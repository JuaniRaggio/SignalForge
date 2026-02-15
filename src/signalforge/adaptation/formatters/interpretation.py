"""Interpretation formatter for informed investors."""

from typing import Any

from signalforge.adaptation.formatters.base import BaseFormatter


class InterpretationFormatter(BaseFormatter):
    """
    Interpretation formatter for INFORMED experience level.

    Adds market context and explains why things matter.
    Less raw data, more insight and comparative analysis.
    """

    def format(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Format data for informed investors.

        Focuses on context, comparisons, and significance.

        Args:
            data: Raw input data

        Returns:
            Formatted data with contextual interpretation
        """
        self._logger.debug("formatting_interpretation_data", keys=list(data.keys()))

        formatted = self._extract_core_fields(data)

        if "signal" in data:
            formatted["signal"] = self._format_signal_with_context(data["signal"])

        if "metrics" in data:
            formatted["key_insights"] = self._extract_key_insights(data["metrics"])

        if "market_context" in data:
            formatted["market_context"] = data["market_context"]
        else:
            formatted["market_context"] = self._generate_market_context(data)

        if "recommendations" in data:
            formatted["recommendations"] = self._format_recommendations_with_why(
                data["recommendations"]
            )

        if "comparisons" in data:
            formatted["comparisons"] = data["comparisons"]

        formatted["_meta"] = {
            "formatter": "interpretation",
            "experience_level": "informed",
            "focus": "context_and_significance",
        }

        return formatted

    def _format_signal_with_context(self, signal: dict[str, Any]) -> dict[str, Any]:
        """Format signal with contextual explanation."""
        formatted: dict[str, Any] = {
            "type": self._safe_get(signal, "type"),
            "direction": self._safe_get(signal, "direction"),
            "strength": self._safe_get(signal, "strength"),
        }

        confidence = self._safe_get(signal, "confidence", 0.0)
        formatted["confidence"] = self._format_number(confidence, 0, percentage=True)

        if confidence > 0.7:
            formatted["interpretation"] = "High confidence signal"
        elif confidence > 0.5:
            formatted["interpretation"] = "Moderate confidence signal"
        else:
            formatted["interpretation"] = "Low confidence signal, use caution"

        if "rationale" in signal:
            formatted["why_it_matters"] = signal["rationale"]

        return formatted

    def _extract_key_insights(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract and explain key insights from metrics."""
        insights: list[dict[str, Any]] = []

        if "rsi" in metrics:
            rsi = metrics["rsi"]
            if rsi > 70:
                insights.append(
                    {
                        "metric": "RSI",
                        "value": self._round_value(rsi, 1),
                        "interpretation": "Overbought condition, potential pullback ahead",
                    }
                )
            elif rsi < 30:
                insights.append(
                    {
                        "metric": "RSI",
                        "value": self._round_value(rsi, 1),
                        "interpretation": "Oversold condition, potential bounce opportunity",
                    }
                )

        if "volatility" in metrics:
            vol = metrics["volatility"]
            insights.append(
                {
                    "metric": "Volatility",
                    "value": self._format_number(vol, 1, percentage=True),
                    "interpretation": self._interpret_volatility(vol),
                }
            )

        if "sharpe_ratio" in metrics:
            sharpe = metrics["sharpe_ratio"]
            insights.append(
                {
                    "metric": "Risk-Adjusted Return",
                    "value": self._round_value(sharpe, 2),
                    "interpretation": self._interpret_sharpe(sharpe),
                }
            )

        if "change_percent" in metrics:
            change = metrics["change_percent"]
            insights.append(
                {
                    "metric": "Price Change",
                    "value": self._format_number(change, 2, percentage=True),
                    "interpretation": self._interpret_price_change(change),
                }
            )

        return insights

    def _generate_market_context(self, data: dict[str, Any]) -> dict[str, Any]:
        """Generate market context from available data."""
        context: dict[str, Any] = {}

        if "sector" in data:
            context["sector"] = data["sector"]

        if "industry" in data:
            context["industry"] = data["industry"]

        if "market_cap" in data:
            market_cap = data["market_cap"]
            if market_cap > 10_000_000_000:
                context["size"] = "Large Cap"
            elif market_cap > 2_000_000_000:
                context["size"] = "Mid Cap"
            else:
                context["size"] = "Small Cap"

        return context

    def _format_recommendations_with_why(
        self, recommendations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Format recommendations explaining rationale."""
        formatted = []

        for rec in recommendations[:3]:
            formatted_rec: dict[str, Any] = {
                "action": self._safe_get(rec, "action"),
                "why": self._safe_get(rec, "rationale", "Based on current analysis"),
            }

            if "expected_outcome" in rec:
                formatted_rec["expected_outcome"] = rec["expected_outcome"]

            if "timeframe" in rec:
                formatted_rec["timeframe"] = rec["timeframe"]

            if "risk_level" in rec:
                formatted_rec["risk"] = rec["risk_level"]

            formatted.append(formatted_rec)

        return formatted

    def _interpret_volatility(self, volatility: float) -> str:
        """Interpret volatility level."""
        if volatility > 0.4:
            return "High volatility - expect larger price swings"
        if volatility > 0.2:
            return "Moderate volatility - normal price fluctuation"
        return "Low volatility - stable price movement"

    def _interpret_sharpe(self, sharpe: float) -> str:
        """Interpret Sharpe ratio."""
        if sharpe > 2.0:
            return "Excellent risk-adjusted returns"
        if sharpe > 1.0:
            return "Good risk-adjusted returns"
        if sharpe > 0:
            return "Positive but modest risk-adjusted returns"
        return "Poor risk-adjusted returns"

    def _interpret_price_change(self, change: float) -> str:
        """Interpret price change."""
        abs_change = abs(change)
        direction = "gain" if change > 0 else "loss"

        if abs_change > 0.05:
            return f"Significant {direction}"
        if abs_change > 0.02:
            return f"Moderate {direction}"
        return f"Minor {direction}"
