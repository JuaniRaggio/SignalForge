"""Translate technical SHAP values into trader-friendly explanations.

This module provides translation services that convert technical SHAP values
and feature names into natural language explanations that traders can understand.
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


class ExplanationTranslator:
    """Translate technical SHAP values into trader-friendly explanations.

    This class converts technical feature names and SHAP values into
    natural language that traders can understand and act upon.
    """

    # Feature category mappings for better explanations
    FEATURE_CATEGORIES = {
        "momentum": ["rsi", "momentum", "roc", "williams"],
        "trend": ["macd", "ema", "sma", "adx", "aroon"],
        "volatility": ["volatility", "atr", "bollinger", "std"],
        "volume": ["volume", "obv", "vwap", "mfi"],
        "price": ["close", "open", "high", "low", "price"],
    }

    # Direction mapping
    DIRECTION_PHRASES = {
        "positive": {
            "momentum": "supporting upward movement",
            "trend": "confirming bullish trend",
            "volatility": "increasing uncertainty",
            "volume": "showing strong interest",
            "price": "pushing price higher",
        },
        "negative": {
            "momentum": "indicating downward pressure",
            "trend": "confirming bearish trend",
            "volatility": "reducing market uncertainty",
            "volume": "showing weak participation",
            "price": "pushing price lower",
        },
    }

    def __init__(self) -> None:
        """Initialize explanation translator."""
        logger.info("explanation_translator_initialized")

    def _get_feature_category(self, feature_name: str) -> str:
        """Get the category of a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Category name (momentum, trend, volatility, volume, price, or other)
        """
        feature_lower = feature_name.lower()

        for category, keywords in self.FEATURE_CATEGORIES.items():
            if any(keyword in feature_lower for keyword in keywords):
                return category

        return "other"

    def translate_shap_values(
        self,
        shap_values: list[tuple[str, float]],
        _feature_names: list[str],
        prediction_type: str = "price_movement",
    ) -> str:
        """Convert SHAP values to readable text.

        Args:
            shap_values: List of (feature_name, shap_value) tuples
            feature_names: List of all feature names
            prediction_type: Type of prediction (price_movement, volatility, etc.)

        Returns:
            Human-readable explanation text
        """
        if not shap_values:
            return "No significant features contributed to this prediction."

        explanations = []

        for feature_name, shap_value in shap_values:
            direction = "positive" if shap_value > 0 else "negative"
            explanation = self.format_feature_impact(
                feature_name,
                shap_value,
                direction,
            )
            if explanation:
                explanations.append(explanation)

        result = " ".join(explanations)

        logger.debug(
            "translated_shap_values",
            num_features=len(shap_values),
            prediction_type=prediction_type,
        )

        return result

    def generate_confidence_explanation(
        self,
        confidence_score: float,
        top_features: list[tuple[str, float, str]],
    ) -> str:
        """Explain confidence level based on features.

        Args:
            confidence_score: Confidence score (0-1)
            top_features: List of (feature_name, impact, direction) tuples

        Returns:
            Confidence explanation text
        """
        # Categorize confidence
        if confidence_score >= 0.8:
            confidence_level = "high"
            confidence_phrase = "We have high confidence in this prediction"
        elif confidence_score >= 0.6:
            confidence_level = "moderate"
            confidence_phrase = "We have moderate confidence in this prediction"
        else:
            confidence_level = "low"
            confidence_phrase = "This prediction has lower confidence"

        # Check feature alignment
        if not top_features:
            return f"{confidence_phrase}."

        directions = [direction for _, _, direction in top_features]
        positive_count = sum(1 for d in directions if d == "positive")
        negative_count = len(directions) - positive_count

        if positive_count == len(directions) or negative_count == len(directions):
            alignment = "all indicators are aligned in the same direction"
        elif abs(positive_count - negative_count) <= 1:
            alignment = "indicators show mixed signals"
        else:
            alignment = "indicators show some disagreement"

        result = f"{confidence_phrase} because {alignment}."

        logger.debug(
            "generated_confidence_explanation",
            confidence_level=confidence_level,
            num_features=len(top_features),
        )

        return result

    def format_feature_impact(
        self,
        feature_name: str,
        impact_value: float,
        direction: str,
    ) -> str:
        """Format single feature impact into readable text.

        Args:
            feature_name: Name of the feature
            impact_value: SHAP value (contribution)
            direction: Direction of impact (positive or negative)

        Returns:
            Formatted feature impact description
        """
        category = self._get_feature_category(feature_name)
        impact_magnitude = abs(impact_value)

        # Get base phrase for category and direction
        phrase_dict = self.DIRECTION_PHRASES.get(direction, {})
        base_phrase = phrase_dict.get(category, "affecting the prediction")

        # Determine strength
        if impact_magnitude > 0.5:
            strength = "strongly"
        elif impact_magnitude > 0.2:
            strength = "moderately"
        else:
            strength = "slightly"

        # Create specific explanations based on feature type
        if "rsi" in feature_name.lower():
            if direction == "positive":
                return f"RSI indicator is {strength} {base_phrase}, suggesting overbought conditions favor this move."
            return f"RSI indicator is {strength} {base_phrase}, suggesting oversold conditions limit this move."

        elif "macd" in feature_name.lower():
            if direction == "positive":
                return f"MACD {strength} {base_phrase}, with bullish crossover signals."
            return f"MACD {strength} {base_phrase}, with bearish crossover signals."

        elif "volume" in feature_name.lower():
            if direction == "positive":
                return f"Trading volume is {strength} {base_phrase}, indicating strong conviction."
            return f"Trading volume is {strength} {base_phrase}, indicating weak participation."

        elif "volatility" in feature_name.lower():
            if direction == "positive":
                return f"Market volatility is {strength} increasing uncertainty in this prediction."
            return f"Low volatility is {strength} supporting stability in this prediction."

        elif any(
            trend in feature_name.lower() for trend in ["ema", "sma", "moving_average"]
        ):
            if direction == "positive":
                return f"Moving average trends are {strength} {base_phrase}."
            return f"Moving average trends are {strength} {base_phrase}."

        elif "momentum" in feature_name.lower():
            if direction == "positive":
                return f"Price momentum is {strength} supporting upward movement."
            return f"Price momentum is {strength} indicating downward pressure."

        # Generic fallback
        readable_name = feature_name.replace("_", " ").title()
        return f"{readable_name} is {strength} {base_phrase}."

    def create_summary(
        self,
        explanation: str,
        max_features: int = 5,
    ) -> str:
        """Create concise summary from detailed explanation.

        Args:
            explanation: Full explanation text
            max_features: Maximum number of features to include in summary

        Returns:
            Concise summary
        """
        # Split by sentences (roughly)
        sentences = [s.strip() for s in explanation.split(".") if s.strip()]

        # Take first max_features sentences
        summary_sentences = sentences[:max_features]

        # Add conclusion
        if len(sentences) > max_features:
            summary = ". ".join(summary_sentences) + f". ({len(sentences) - max_features} additional factors considered.)"
        else:
            summary = ". ".join(summary_sentences) + "."

        logger.debug(
            "created_summary",
            original_length=len(explanation),
            summary_length=len(summary),
        )

        return summary

    def translate_feature_importance(
        self,
        feature_importance: dict[str, float],
        top_k: int = 10,
    ) -> dict[str, str]:
        """Translate feature importance scores to readable descriptions.

        Args:
            feature_importance: Dictionary of feature names to importance scores
            top_k: Number of top features to translate

        Returns:
            Dictionary of feature names to readable descriptions
        """
        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        translations = {}

        for _, (feature_name, importance) in enumerate(sorted_features[:top_k], 1):
            category = self._get_feature_category(feature_name)
            magnitude = abs(importance)

            if magnitude > 0.5:
                impact_level = "critical"
            elif magnitude > 0.3:
                impact_level = "significant"
            elif magnitude > 0.1:
                impact_level = "moderate"
            else:
                impact_level = "minor"

            readable_name = feature_name.replace("_", " ").title()
            translations[feature_name] = (
                f"{readable_name} ({category}) - {impact_level} impact on predictions"
            )

        logger.debug(
            "translated_feature_importance",
            total_features=len(feature_importance),
            translated_count=len(translations),
        )

        return translations
