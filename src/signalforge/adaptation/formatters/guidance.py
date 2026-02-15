"""Guidance formatter for casual observers."""

from typing import Any

from signalforge.adaptation.formatters.base import BaseFormatter


class GuidanceFormatter(BaseFormatter):
    """
    Guidance formatter for CASUAL experience level.

    Simple language without jargon.
    Clear explanations of basic concepts.
    Actionable recommendations with plain English.
    """

    def format(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Format data for casual observers.

        Simplifies language and provides clear guidance.

        Args:
            data: Raw input data

        Returns:
            Formatted data with simplified guidance
        """
        self._logger.debug("formatting_guidance_data", keys=list(data.keys()))

        formatted = self._extract_core_fields(data)

        if "signal" in data:
            formatted["simple_summary"] = self._create_simple_summary(data["signal"])

        if "metrics" in data:
            formatted["what_this_means"] = self._explain_metrics_simply(data["metrics"])

        if "recommendations" in data:
            formatted["what_to_do"] = self._create_simple_actions(
                data["recommendations"]
            )

        formatted["key_points"] = self._extract_key_points(data)

        formatted["_meta"] = {
            "formatter": "guidance",
            "experience_level": "casual",
            "focus": "simple_explanations",
        }

        return formatted

    def _create_simple_summary(self, signal: dict[str, Any]) -> dict[str, Any]:
        """Create simple summary of signal."""
        summary: dict[str, Any] = {}

        direction = self._safe_get(signal, "direction", "neutral")
        strength = self._safe_get(signal, "strength", "moderate")

        if direction == "bullish" or direction == "buy":
            summary["outlook"] = "Positive"
            summary["simple_explanation"] = (
                f"This is a {strength} positive signal. "
                "The stock may be a good buying opportunity."
            )
        elif direction == "bearish" or direction == "sell":
            summary["outlook"] = "Negative"
            summary["simple_explanation"] = (
                f"This is a {strength} negative signal. "
                "You may want to be cautious or consider selling."
            )
        else:
            summary["outlook"] = "Neutral"
            summary["simple_explanation"] = (
                "The signal is neutral. No strong indication to buy or sell right now."
            )

        confidence = self._safe_get(signal, "confidence", 0.5)
        if confidence > 0.7:
            summary["reliability"] = "High - This signal is quite reliable"
        elif confidence > 0.5:
            summary["reliability"] = "Medium - Use with other information"
        else:
            summary["reliability"] = "Low - Be cautious with this signal"

        return summary

    def _explain_metrics_simply(
        self, metrics: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Explain metrics in simple terms."""
        explanations: list[dict[str, Any]] = []

        if "price" in metrics and "change_percent" in metrics:
            price = metrics["price"]
            change = metrics["change_percent"]
            direction = "up" if change > 0 else "down"
            explanations.append(
                {
                    "what": "Current Price",
                    "value": f"${self._round_value(price, 2)}",
                    "meaning": f"The stock is {direction} {abs(self._round_value(change * 100, 1))}% today",
                }
            )

        if "rsi" in metrics:
            rsi = metrics["rsi"]
            if rsi > 70:
                explanations.append(
                    {
                        "what": "Market Momentum",
                        "value": "Overbought",
                        "meaning": "The stock may have risen too fast and could drop soon",
                    }
                )
            elif rsi < 30:
                explanations.append(
                    {
                        "what": "Market Momentum",
                        "value": "Oversold",
                        "meaning": "The stock may have dropped too much and could recover",
                    }
                )
            else:
                explanations.append(
                    {
                        "what": "Market Momentum",
                        "value": "Normal",
                        "meaning": "The stock is trading at reasonable levels",
                    }
                )

        if "volatility" in metrics:
            vol = metrics["volatility"]
            if vol > 0.4:
                level = "High"
                meaning = "This stock moves a lot. Higher risk but potentially higher reward"
            elif vol > 0.2:
                level = "Medium"
                meaning = "This stock has normal price swings"
            else:
                level = "Low"
                meaning = "This stock is stable with small price changes"

            explanations.append({"what": "Price Stability", "value": level, "meaning": meaning})

        if "volume" in metrics:
            volume = metrics["volume"]
            explanations.append(
                {
                    "what": "Trading Activity",
                    "value": self._format_large_number(volume),
                    "meaning": "Number of shares traded today",
                }
            )

        return explanations

    def _create_simple_actions(
        self, recommendations: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Create simple actionable recommendations."""
        actions = []

        for rec in recommendations[:3]:
            action_type = self._safe_get(rec, "action", "monitor")
            rationale = self._safe_get(rec, "rationale", "")

            simple_action: dict[str, Any] = {
                "suggestion": self._simplify_action(action_type),
                "reason": self._simplify_rationale(rationale),
            }

            if "risk_level" in rec:
                risk = rec["risk_level"]
                simple_action["risk"] = self._explain_risk(risk)

            actions.append(simple_action)

        if not actions:
            actions.append(
                {
                    "suggestion": "Keep watching this stock",
                    "reason": "Wait for clearer signals before taking action",
                    "risk": "Low risk - just monitoring",
                }
            )

        return actions

    def _extract_key_points(self, data: dict[str, Any]) -> list[str]:
        """Extract key points in simple language."""
        points = []

        if "signal" in data:
            signal = data["signal"]
            direction = self._safe_get(signal, "direction", "neutral")
            if direction == "bullish" or direction == "buy":
                points.append("The trend looks positive")
            elif direction == "bearish" or direction == "sell":
                points.append("The trend looks negative")

        if "metrics" in data:
            metrics = data["metrics"]
            if "change_percent" in metrics:
                change = metrics["change_percent"]
                if abs(change) > 0.05:
                    points.append(
                        f"Price moved significantly ({self._format_number(change, 1, percentage=True)})"
                    )

        if "sector" in data:
            points.append(f"Part of {data['sector']} sector")

        if not points:
            points.append("Information is still being analyzed")

        return points

    def _simplify_action(self, action: str) -> str:
        """Simplify action recommendation."""
        action_lower = action.lower()

        simplifications = {
            "strong_buy": "Consider buying - strong signal",
            "buy": "Consider buying",
            "hold": "Keep if you own it, wait if you don't",
            "sell": "Consider selling",
            "strong_sell": "Consider selling - strong signal",
            "monitor": "Watch and wait",
            "accumulate": "Buy small amounts over time",
            "reduce": "Sell some of your shares",
        }

        return simplifications.get(action_lower, "Watch this stock")

    def _simplify_rationale(self, rationale: str) -> str:
        """Simplify rationale to plain English."""
        if not rationale:
            return "Based on current market conditions"

        simplified = rationale.replace("leverage", "use")
        simplified = simplified.replace("volatility", "price swings")
        simplified = simplified.replace("momentum", "trend")
        simplified = simplified.replace("overbought", "risen too fast")
        simplified = simplified.replace("oversold", "dropped too much")

        return simplified

    def _explain_risk(self, risk: str) -> str:
        """Explain risk level simply."""
        risk_lower = risk.lower()

        explanations = {
            "low": "Low risk - relatively safe",
            "medium": "Medium risk - some chance of loss",
            "high": "High risk - significant chance of loss",
            "very_high": "Very high risk - only for experienced investors",
        }

        return explanations.get(risk_lower, "Risk level not specified")

    def _format_large_number(self, num: float | int) -> str:
        """Format large numbers in readable form."""
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.1f}B"
        if num >= 1_000_000:
            return f"{num / 1_000_000:.1f}M"
        if num >= 1_000:
            return f"{num / 1_000:.1f}K"
        return str(num)
