"""Explanation generation for personalized recommendations.

This module provides human-readable explanations for why specific items
were recommended to users. It uses template-based explanation generation
with context filling.

Key Features:
- Template-based explanation generation
- Context-aware template selection
- Support for multiple recommendation sources
- Batch explanation generation

Examples:
    Generating an explanation:

    >>> from signalforge.recommendation.feed.explainer import RecommendationExplainer
    >>> explainer = RecommendationExplainer()
    >>> explanation = explainer.generate_explanation(
    ...     item,
    ...     user_profile,
    ...     "content_based"
    ... )
"""

from __future__ import annotations

from typing import Any

from signalforge.core.logging import get_logger
from signalforge.recommendation.user_model import ExplicitProfile

logger = get_logger(__name__)


class RecommendationExplainer:
    """Generate human-readable explanations for recommendations.

    This class provides context-aware explanation generation using
    templates and user/item metadata.

    Examples:
        >>> explainer = RecommendationExplainer()
        >>> explanation = explainer.generate_explanation(item, profile, "collaborative")
    """

    EXPLANATION_TEMPLATES: dict[str, str] = {
        "portfolio_related": "This is relevant because you hold {related_symbol} in {sector}",
        "watchlist_match": "You've been watching {symbol} - here's a new signal",
        "sector_interest": "Based on your interest in {sector} stocks",
        "similar_users": "Traders with similar profiles found this valuable",
        "event_driven": "Upcoming {event_type} may impact this stock",
        "risk_appropriate": "Matches your {risk_level} risk tolerance",
        "trending": "This is gaining attention in the market",
        "high_confidence": "High confidence signal with {confidence:.0%} accuracy",
        "sector_momentum": "{sector} sector showing strong momentum",
        "volatility_match": "Volatility level matches your preference",
        "holding_period_match": "Holding period aligns with your {horizon}-term strategy",
        "content_based": "Similar to signals you've viewed before",
        "collaborative": "Recommended based on similar trader activity",
        "hybrid": "Recommended based on your preferences and similar traders",
    }

    def __init__(self) -> None:
        """Initialize the recommendation explainer."""
        logger.info("recommendation_explainer_initialized")

    def generate_explanation(
        self,
        item: Any,
        user_profile: ExplicitProfile,
        recommendation_source: str,
    ) -> str:
        """Generate human-readable explanation for a recommendation.

        This method analyzes the item, user profile, and recommendation
        source to generate a contextual explanation for why the item
        was recommended.

        Args:
            item: Recommended item (FeedItem or similar).
            user_profile: User's explicit profile.
            recommendation_source: Source of recommendation
                (content_based, collaborative, hybrid, etc).

        Returns:
            Human-readable explanation string.

        Examples:
            >>> explainer = RecommendationExplainer()
            >>> explanation = explainer.generate_explanation(
            ...     feed_item,
            ...     user_profile,
            ...     "content_based"
            ... )
        """
        logger.debug(
            "generating_explanation",
            item_id=getattr(item, "item_id", "unknown"),
            source=recommendation_source,
        )

        # Select appropriate template
        template_key = self._select_template(item, user_profile, recommendation_source)

        # Build context for template filling
        context = self._build_context(item, user_profile, recommendation_source)

        # Fill template with context
        explanation = self._fill_template(self.EXPLANATION_TEMPLATES[template_key], context)

        logger.debug(
            "explanation_generated",
            item_id=getattr(item, "item_id", "unknown"),
            template=template_key,
        )

        return explanation

    def _select_template(
        self,
        item: Any,
        user_profile: ExplicitProfile,
        source: str,
    ) -> str:
        """Select the most appropriate explanation template.

        This method examines the item and user profile to determine
        which explanation template best fits the recommendation context.

        Args:
            item: Recommended item.
            user_profile: User's explicit profile.
            source: Recommendation source.

        Returns:
            Template key from EXPLANATION_TEMPLATES.
        """
        # Check for watchlist match (highest priority)
        if hasattr(item, "symbol") and item.symbol in user_profile.watchlist:
            return "watchlist_match"

        # Check for sector interest
        if hasattr(item, "metadata") and "sector" in item.metadata:
            sector = item.metadata["sector"]
            if sector in user_profile.preferred_sectors:
                return "sector_interest"

        # Check for high confidence
        if hasattr(item, "metadata") and "confidence" in item.metadata:
            confidence = item.metadata.get("confidence", 0.0)
            if confidence >= 0.8:
                return "high_confidence"

        # Check for event-driven
        if hasattr(item, "metadata") and "event_type" in item.metadata:
            return "event_driven"

        # Use source-based template as fallback
        source_templates = {
            "content_based": "content_based",
            "collaborative": "collaborative",
            "hybrid": "hybrid",
        }

        return source_templates.get(source, "trending")

    def _build_context(
        self,
        item: Any,
        user_profile: ExplicitProfile,
        source: str,
    ) -> dict[str, Any]:
        """Build context dictionary for template filling.

        Args:
            item: Recommended item.
            user_profile: User's explicit profile.
            source: Recommendation source.

        Returns:
            Context dictionary with keys for template variables.
        """
        context: dict[str, Any] = {}

        # Add symbol if available
        if hasattr(item, "symbol") and item.symbol:
            context["symbol"] = item.symbol

        # Add sector if available
        if hasattr(item, "metadata") and "sector" in item.metadata:
            context["sector"] = item.metadata["sector"]

        # Add confidence if available
        if hasattr(item, "metadata") and "confidence" in item.metadata:
            context["confidence"] = item.metadata["confidence"]

        # Add event type if available
        if hasattr(item, "metadata") and "event_type" in item.metadata:
            context["event_type"] = item.metadata["event_type"]

        # Add risk level from user profile
        context["risk_level"] = user_profile.risk_tolerance

        # Add investment horizon
        horizon_map = {
            (0, 7): "short",
            (7, 90): "medium",
            (90, float("inf")): "long",
        }
        horizon = "medium"
        for (min_days, max_days), label in horizon_map.items():
            if min_days <= user_profile.investment_horizon < max_days:
                horizon = label
                break
        context["horizon"] = horizon

        # Add related symbol (for portfolio-related recommendations)
        if user_profile.watchlist:
            context["related_symbol"] = user_profile.watchlist[0]

        return context

    def _fill_template(self, template: str, context: dict[str, Any]) -> str:
        """Fill template string with context values.

        This method safely fills template placeholders with values from
        the context dictionary, handling missing keys gracefully.

        Args:
            template: Template string with {placeholder} markers.
            context: Dictionary of values to fill into template.

        Returns:
            Filled template string.

        Examples:
            >>> explainer = RecommendationExplainer()
            >>> template = "Recommended for {sector} investors"
            >>> context = {"sector": "Technology"}
            >>> explainer._fill_template(template, context)
            'Recommended for Technology investors'
        """
        try:
            return template.format(**context)
        except KeyError as e:
            logger.warning(
                "template_fill_error",
                template=template,
                missing_key=str(e),
            )
            # Return template with partial filling
            filled = template
            for key, value in context.items():
                filled = filled.replace(f"{{{key}}}", str(value))
            return filled

    def generate_batch_explanations(
        self,
        items: list[Any],
        user_profile: ExplicitProfile,
    ) -> dict[str, str]:
        """Generate explanations for multiple items in batch.

        This is more efficient than calling generate_explanation repeatedly
        as it can optimize context building and template selection.

        Args:
            items: List of recommended items.
            user_profile: User's explicit profile.

        Returns:
            Dictionary mapping item_id to explanation string.

        Examples:
            >>> explainer = RecommendationExplainer()
            >>> explanations = explainer.generate_batch_explanations(
            ...     [item1, item2, item3],
            ...     user_profile
            ... )
        """
        logger.debug("generating_batch_explanations", num_items=len(items))

        explanations: dict[str, str] = {}

        for item in items:
            item_id = getattr(item, "item_id", f"item_{id(item)}")
            try:
                # For batch processing, we assume content-based source
                # In real implementation, each item might have its own source
                explanation = self.generate_explanation(item, user_profile, "content_based")
                explanations[item_id] = explanation
            except Exception as e:
                logger.error(
                    "explanation_generation_failed",
                    item_id=item_id,
                    error=str(e),
                )
                explanations[item_id] = "Recommended based on your preferences"

        logger.info(
            "batch_explanations_generated",
            num_items=len(items),
            num_explanations=len(explanations),
        )

        return explanations
