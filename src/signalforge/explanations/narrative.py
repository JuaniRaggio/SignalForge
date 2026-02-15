"""Natural language narrative generation for predictions."""

from typing import ClassVar

import structlog

from .schemas import FeatureContribution, PredictionExplanation

logger = structlog.get_logger(__name__)


class NarrativeGenerator:
    """Generate natural language narratives from explanations."""

    TEMPLATES: ClassVar[dict[str, str]] = {
        "bullish_strong": (
            "The model is strongly bullish on {symbol} primarily due to {top_reason}. "
            "{supporting_factors}"
        ),
        "bullish_moderate": (
            "{symbol} shows moderate bullish signals, with {top_reason} being the main driver. "
            "{supporting_factors}"
        ),
        "neutral": "The model sees mixed signals for {symbol}. {balance_explanation}",
        "bearish_moderate": (
            "{symbol} faces headwinds from {top_reason}. {mitigating_factors}"
        ),
        "bearish_strong": (
            "Strong bearish outlook for {symbol} driven by {top_reason}. "
            "{additional_concerns}"
        ),
    }

    FEATURE_DESCRIPTIONS: ClassVar[dict[str, dict[str, str]]] = {
        "rsi_14": {
            "high": "overbought conditions",
            "low": "oversold conditions",
            "neutral": "neutral RSI",
        },
        "rsi": {
            "high": "overbought momentum",
            "low": "oversold momentum",
            "neutral": "balanced momentum",
        },
        "macd_histogram": {
            "positive": "positive momentum",
            "negative": "negative momentum",
        },
        "macd": {
            "positive": "bullish MACD crossover",
            "negative": "bearish MACD crossover",
        },
        "sma_50_200_cross": {
            "golden": "golden cross pattern",
            "death": "death cross pattern",
        },
        "sma_cross": {
            "positive": "bullish moving average alignment",
            "negative": "bearish moving average alignment",
        },
        "volume_surge": {
            "high": "unusual volume activity",
            "low": "thin trading volume",
        },
        "volume": {
            "high": "strong volume confirmation",
            "low": "weak volume support",
            "normal": "normal trading volume",
        },
        "sentiment_score": {
            "positive": "positive market sentiment",
            "negative": "negative market sentiment",
        },
        "sentiment": {
            "positive": "favorable sentiment readings",
            "negative": "negative sentiment backdrop",
        },
        "volatility": {
            "high": "elevated volatility levels",
            "low": "low volatility environment",
        },
        "trend_strength": {
            "high": "strong trending behavior",
            "low": "weak trend formation",
        },
    }

    def __init__(self) -> None:
        """Initialize narrative generator."""
        logger.info("narrative_generator_initialized")

    def generate_narrative(
        self,
        explanation: PredictionExplanation,
        include_technical_details: bool = False,
    ) -> str:
        """
        Generate narrative from explanation.

        Args:
            explanation: The prediction explanation
            include_technical_details: Include technical SHAP values

        Returns:
            Natural language narrative
        """
        # Determine overall sentiment
        sentiment = self._determine_sentiment(explanation)

        # Select appropriate template
        template = self._select_template(explanation.prediction, sentiment)

        # Generate feature descriptions
        top_reason = self._describe_feature(explanation.top_features[0])

        supporting_factors = self._describe_supporting_factors(
            explanation.top_features[1:],
            sentiment,
        )

        # Fill in template
        narrative = template.format(
            symbol=explanation.symbol,
            top_reason=top_reason,
            supporting_factors=supporting_factors,
            balance_explanation=self._explain_balance(explanation),
            mitigating_factors=supporting_factors,
            additional_concerns=supporting_factors,
        )

        # Add technical details if requested
        if include_technical_details:
            narrative += f" (Base: {explanation.base_value:.4f}, Prediction: {explanation.prediction:.4f})"

        logger.info(
            "generated_narrative",
            symbol=explanation.symbol,
            sentiment=sentiment,
            length=len(narrative),
        )

        return narrative

    def _determine_sentiment(self, explanation: PredictionExplanation) -> str:
        """
        Determine overall sentiment from explanation.

        Args:
            explanation: The prediction explanation

        Returns:
            Sentiment label: bullish_strong, bullish_moderate, neutral, bearish_moderate, bearish_strong
        """
        prediction = explanation.prediction
        base = explanation.base_value
        deviation = prediction - base

        # Calculate relative strength
        relative_deviation = deviation / abs(base) if abs(base) > 0 else deviation

        # Check feature alignment
        if explanation.top_features:
            positive_count = sum(1 for f in explanation.top_features if f.direction == "positive")
            negative_count = len(explanation.top_features) - positive_count

            # Strong alignment indicates strong signal
            alignment_ratio = max(positive_count, negative_count) / len(explanation.top_features)

            if relative_deviation > 0.1:
                if alignment_ratio > 0.7:
                    return "bullish_strong"
                return "bullish_moderate"
            elif relative_deviation < -0.1:
                if alignment_ratio > 0.7:
                    return "bearish_strong"
                return "bearish_moderate"

        return "neutral"

    def _select_template(self, _prediction: float, sentiment: str) -> str:
        """
        Select appropriate narrative template.

        Args:
            _prediction: Model prediction value (unused, reserved for future use)
            sentiment: Determined sentiment label

        Returns:
            Template string
        """
        return self.TEMPLATES.get(sentiment, self.TEMPLATES["neutral"])

    def _describe_feature(self, contribution: FeatureContribution) -> str:
        """
        Generate description for a feature.

        Args:
            contribution: Feature contribution object

        Returns:
            Human-readable feature description
        """
        feature_name = contribution.feature_name.lower()
        feature_value = contribution.feature_value
        direction = contribution.direction

        # Try to find matching feature description
        for key, descriptions in self.FEATURE_DESCRIPTIONS.items():
            if key in feature_name:
                # Determine value category
                if "rsi" in key:
                    if feature_value > 70:
                        category = "high"
                    elif feature_value < 30:
                        category = "low"
                    else:
                        category = "neutral"
                elif "macd" in key or "sma" in key:
                    category = "positive" if feature_value > 0 else "negative"
                elif "volume" in key:
                    if feature_value > 1.5:
                        category = "high"
                    elif feature_value < 0.5:
                        category = "low"
                    else:
                        category = "normal"
                elif "sentiment" in key:
                    category = "positive" if feature_value > 0 else "negative"
                elif "volatility" in key or "trend" in key:
                    category = "high" if feature_value > 1.0 else "low"
                else:
                    category = "positive" if direction == "positive" else "negative"

                return descriptions.get(category, contribution.human_readable)

        # Fallback to human_readable from contribution
        return contribution.human_readable

    def _describe_supporting_factors(
        self,
        factors: list[FeatureContribution],
        _sentiment: str,
    ) -> str:
        """
        Describe supporting factors.

        Args:
            factors: List of supporting feature contributions
            _sentiment: Overall sentiment (unused, reserved for future use)

        Returns:
            Combined description of supporting factors
        """
        if not factors:
            return "No significant supporting factors identified."

        descriptions = [self._describe_feature(f) for f in factors[:3]]

        if len(descriptions) == 1:
            return descriptions[0]
        elif len(descriptions) == 2:
            return self._combine_factors(descriptions, conjunction="and")
        else:
            # Three or more - use oxford comma
            return self._combine_factors(descriptions, conjunction="and")

    def _combine_factors(
        self,
        factors: list[str],
        conjunction: str = "and",
    ) -> str:
        """
        Combine multiple factors into readable text.

        Args:
            factors: List of factor descriptions
            conjunction: Conjunction word (and/or)

        Returns:
            Combined text
        """
        if len(factors) == 0:
            return ""
        elif len(factors) == 1:
            return factors[0]
        elif len(factors) == 2:
            return f"{factors[0]} {conjunction} {factors[1]}"
        else:
            # Oxford comma for 3+
            return f"{', '.join(factors[:-1])}, {conjunction} {factors[-1]}"

    def _explain_balance(self, explanation: PredictionExplanation) -> str:
        """
        Explain balanced/neutral signals.

        Args:
            explanation: The prediction explanation

        Returns:
            Explanation of balanced signals
        """
        positive_features = [f for f in explanation.top_features if f.direction == "positive"]
        negative_features = [f for f in explanation.top_features if f.direction == "negative"]

        if not positive_features and not negative_features:
            return "All features show minimal impact on the prediction."

        positive_desc = self._combine_factors(
            [self._describe_feature(f) for f in positive_features[:2]]
        )
        negative_desc = self._combine_factors(
            [self._describe_feature(f) for f in negative_features[:2]]
        )

        if positive_features and negative_features:
            return f"Bullish signals from {positive_desc} are offset by bearish signals from {negative_desc}."
        elif positive_features:
            return f"Mild bullish bias from {positive_desc}."
        else:
            return f"Mild bearish bias from {negative_desc}."

    def generate_summary(
        self,
        explanations: list[PredictionExplanation],
    ) -> str:
        """
        Generate summary for multiple predictions.

        Args:
            explanations: List of prediction explanations

        Returns:
            Summary narrative
        """
        if not explanations:
            return "No predictions to summarize."

        # Count sentiments
        sentiments = [self._determine_sentiment(exp) for exp in explanations]
        bullish_count = sum(1 for s in sentiments if "bullish" in s)
        bearish_count = sum(1 for s in sentiments if "bearish" in s)
        neutral_count = len(sentiments) - bullish_count - bearish_count

        # Average prediction
        avg_prediction = sum(exp.prediction for exp in explanations) / len(explanations)

        summary = (
            f"Analysis of {len(explanations)} predictions: "
            f"{bullish_count} bullish, {bearish_count} bearish, {neutral_count} neutral. "
            f"Average prediction: {avg_prediction:.4f}. "
        )

        # Most common top feature
        top_features = [exp.top_features[0].feature_name for exp in explanations if exp.top_features]
        if top_features:
            most_common = max(set(top_features), key=top_features.count)
            summary += f"Most influential feature: {most_common}."

        logger.info(
            "generated_summary",
            num_predictions=len(explanations),
            bullish=bullish_count,
            bearish=bearish_count,
        )

        return summary
