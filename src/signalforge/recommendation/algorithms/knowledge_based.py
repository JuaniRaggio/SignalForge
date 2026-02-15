"""Knowledge-based recommendation algorithm.

This module implements rule-based recommendations using domain knowledge
about finance, trading, and portfolio management.

The algorithm applies business rules for:
- Portfolio diversification
- Risk-appropriate recommendations
- Event-driven recommendations (earnings, FOMC, etc.)
- Sector rotation strategies
- Market condition awareness

This recommender doesn't require training on historical data, but uses
expert knowledge encoded as rules.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import polars as pl

from signalforge.core.logging import get_logger
from signalforge.recommendation.algorithms.base import BaseRecommender
from signalforge.recommendation.algorithms.schemas import (
    RecommendationItem,
    RecommendationRequest,
)
from signalforge.recommendation.user_model import ExplicitProfile, ImplicitProfile, RiskTolerance

logger = get_logger(__name__)


@dataclass
class RecommendationRule:
    """A recommendation rule with condition and action.

    Attributes:
        rule_id: Unique identifier for the rule.
        condition: Python expression or callable to evaluate.
        action: Type of action (boost, filter, inject).
        weight: Weight or multiplier for the action.
        description: Human-readable description of the rule.
    """

    rule_id: str
    condition: str | Callable[..., bool]
    action: str  # "boost", "filter", "inject"
    weight: float
    description: str

    def __post_init__(self) -> None:
        """Validate rule."""
        if self.action not in ("boost", "filter", "inject"):
            raise ValueError(f"Invalid action: {self.action}")

        if self.weight < 0.0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")


class KnowledgeBasedRecommender(BaseRecommender):
    """Rule-based recommendations using domain knowledge.

    This recommender applies business rules and domain expertise to generate
    recommendations. It doesn't learn from data but uses pre-defined rules.

    Attributes:
        algorithm_name: Name of the algorithm.
        rules: List of recommendation rules to apply.
        item_database: Database of items available for recommendation.
    """

    algorithm_name = "knowledge_based"

    def __init__(self) -> None:
        """Initialize the knowledge-based recommender."""
        self.rules = self._load_rules()
        self.item_database: dict[str, dict[str, str | float | int | list[str]]] = {}

        logger.info("knowledge_based_recommender_initialized", num_rules=len(self.rules))

    async def recommend(
        self,
        request: RecommendationRequest,
        user_profile: ExplicitProfile,
        implicit_profile: ImplicitProfile | None = None,  # noqa: ARG002
    ) -> list[RecommendationItem]:
        """Apply business rules to generate recommendations.

        Args:
            request: Recommendation request.
            user_profile: Explicit user preferences.
            implicit_profile: Optional implicit user behavior.

        Returns:
            List of recommended items after applying rules.
        """
        if not self.item_database:
            logger.warning("no_item_database_available")
            return []

        # Get candidate items
        candidates = self._get_candidate_items(request)

        # Apply rules in sequence
        candidates = self._apply_diversification_rules(user_profile, candidates)
        candidates = self._apply_risk_rules(user_profile.risk_tolerance, candidates)

        # Apply event rules if context is provided
        if request.context:
            candidates = self._apply_event_rules(request.context, candidates)

        # Apply sector rotation rules
        candidates = self._apply_sector_rotation_rules(user_profile, candidates)

        # Sort by score and limit
        candidates.sort(key=lambda x: x.score, reverse=True)
        recommendations = candidates[: request.limit]

        logger.info(
            "knowledge_based_recommendations_generated",
            user_id=request.user_id,
            num_recommendations=len(recommendations),
        )

        return recommendations

    async def train(self, interaction_data: pl.DataFrame) -> None:
        """Update item database (knowledge-based doesn't need training).

        Args:
            interaction_data: DataFrame containing item information.
                Expected columns: item_id, item_type, sector, risk_level, upcoming_events
        """
        required_columns = {"item_id"}
        actual_columns = set(interaction_data.columns)

        if not required_columns.issubset(actual_columns):
            missing = required_columns - actual_columns
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("updating_item_database", num_items=len(interaction_data))

        # Update item database
        for row in interaction_data.iter_rows(named=True):
            item_id = str(row["item_id"])

            self.item_database[item_id] = {
                "item_type": str(row.get("item_type", "unknown")),
                "sector": str(row.get("sector", "")),
                "risk_level": str(row.get("risk_level", "medium")),
                "volatility": float(row.get("volatility", 0.5)),
                "upcoming_events": list(row.get("upcoming_events", [])) if row.get("upcoming_events") else [],
                "market_cap": float(row.get("market_cap", 0.0)),
            }

        logger.info("item_database_updated", num_items=len(self.item_database))

    def _load_rules(self) -> list[RecommendationRule]:
        """Load recommendation rules.

        Returns:
            List of recommendation rules.
        """
        rules = [
            RecommendationRule(
                rule_id="diversify_sectors",
                condition="sector_overweight",
                action="filter",
                weight=0.5,
                description="Filter items from overweight sectors to promote diversification",
            ),
            RecommendationRule(
                rule_id="match_risk_tolerance",
                condition="risk_mismatch",
                action="filter",
                weight=1.0,
                description="Filter items that don't match user's risk tolerance",
            ),
            RecommendationRule(
                rule_id="boost_upcoming_events",
                condition="has_upcoming_event",
                action="boost",
                weight=1.3,
                description="Boost items with upcoming catalysts (earnings, FOMC, etc.)",
            ),
            RecommendationRule(
                rule_id="sector_rotation",
                condition="sector_rotation_opportunity",
                action="boost",
                weight=1.2,
                description="Boost sectors showing rotation opportunities",
            ),
            RecommendationRule(
                rule_id="avoid_high_concentration",
                condition="portfolio_concentration_high",
                action="filter",
                weight=0.3,
                description="Avoid further concentration in already high-weight positions",
            ),
        ]

        return rules

    def _get_candidate_items(
        self, request: RecommendationRequest
    ) -> list[RecommendationItem]:
        """Get initial candidate items.

        Args:
            request: Recommendation request.

        Returns:
            List of candidate recommendation items.
        """
        candidates = []

        for item_id, item_data in self.item_database.items():
            # Filter by item type if specified
            if request.item_types and item_data["item_type"] not in request.item_types:
                continue

            # Create recommendation item with neutral score
            candidates.append(
                RecommendationItem(
                    item_id=item_id,
                    item_type=str(item_data["item_type"]),
                    score=0.5,  # Neutral starting score
                    source="knowledge_based",
                    explanation="Based on investment rules",
                    metadata={
                        "sector": str(item_data["sector"]),
                        "risk_level": str(item_data["risk_level"]),
                    },
                )
            )

        return candidates

    def _apply_diversification_rules(
        self,
        user_profile: ExplicitProfile,
        candidates: list[RecommendationItem],
    ) -> list[RecommendationItem]:
        """Apply portfolio diversification rules.

        If user is overweight in certain sectors, filter or downweight
        additional recommendations from those sectors.

        Args:
            user_profile: User's explicit profile.
            candidates: Candidate recommendation items.

        Returns:
            Filtered/adjusted list of candidates.
        """
        # Get user's sector exposure from watchlist
        watchlist_sectors: dict[str, int] = {}
        for symbol in user_profile.watchlist:
            # In production, would query database for sector
            # For now, use a simple heuristic
            if symbol in self.item_database:
                sector = str(self.item_database[symbol].get("sector", ""))
                watchlist_sectors[sector] = watchlist_sectors.get(sector, 0) + 1

        if not watchlist_sectors:
            return candidates

        # Calculate total watchlist size
        total_watchlist = len(user_profile.watchlist)

        # Find overweight sectors (more than 30% of watchlist)
        overweight_threshold = 0.3
        overweight_sectors = {
            sector for sector, count in watchlist_sectors.items()
            if count / total_watchlist > overweight_threshold
        }

        if not overweight_sectors:
            return candidates

        # Downweight items from overweight sectors
        adjusted_candidates = []
        for candidate in candidates:
            sector = str(candidate.metadata.get("sector", ""))
            if sector in overweight_sectors:
                # Reduce score for diversification
                candidate.score *= 0.7
                candidate.explanation = f"Diversification: {candidate.explanation}"

            adjusted_candidates.append(candidate)

        logger.debug(
            "diversification_rules_applied",
            overweight_sectors=list(overweight_sectors),
            num_candidates=len(adjusted_candidates),
        )

        return adjusted_candidates

    def _apply_risk_rules(
        self,
        risk_tolerance: RiskTolerance,
        candidates: list[RecommendationItem],
    ) -> list[RecommendationItem]:
        """Filter by risk tolerance.

        Args:
            risk_tolerance: User's risk tolerance level.
            candidates: Candidate recommendation items.

        Returns:
            Filtered list matching risk tolerance.
        """
        # Map risk tolerance to acceptable risk levels
        acceptable_risks = {
            "low": {"low"},
            "medium": {"low", "medium"},
            "high": {"low", "medium", "high"},
        }

        accepted = acceptable_risks[risk_tolerance]

        filtered_candidates = []
        for candidate in candidates:
            risk_level = str(candidate.metadata.get("risk_level", "medium"))

            if risk_level in accepted:
                filtered_candidates.append(candidate)
            else:
                logger.debug(
                    "filtered_by_risk",
                    item_id=candidate.item_id,
                    item_risk=risk_level,
                    user_risk=risk_tolerance,
                )

        logger.debug(
            "risk_rules_applied",
            risk_tolerance=risk_tolerance,
            original_count=len(candidates),
            filtered_count=len(filtered_candidates),
        )

        return filtered_candidates

    def _apply_event_rules(
        self,
        context: dict[str, str | int | float | bool],
        candidates: list[RecommendationItem],
    ) -> list[RecommendationItem]:
        """Boost items with upcoming events.

        Args:
            context: Context information including market conditions.
            candidates: Candidate recommendation items.

        Returns:
            Adjusted list with event-based boosts.
        """
        # Check if we're looking for event-driven opportunities
        event_focus = context.get("event_driven", False)

        if not event_focus:
            return candidates

        adjusted_candidates = []
        for candidate in candidates:
            item_data = self.item_database.get(candidate.item_id, {})
            upcoming_events_data = item_data.get("upcoming_events", [])

            if upcoming_events_data and isinstance(upcoming_events_data, list):
                # Boost items with upcoming events
                candidate.score *= 1.3
                events_str = ", ".join(str(e) for e in upcoming_events_data)
                candidate.explanation = f"Upcoming events: {events_str}"

            adjusted_candidates.append(candidate)

        logger.debug(
            "event_rules_applied",
            num_candidates=len(adjusted_candidates),
        )

        return adjusted_candidates

    def _apply_sector_rotation_rules(
        self,
        user_profile: ExplicitProfile,
        candidates: list[RecommendationItem],
    ) -> list[RecommendationItem]:
        """Apply sector rotation rules.

        Boost sectors that are underrepresented in user's portfolio
        to encourage rotation and diversification.

        Args:
            user_profile: User's explicit profile.
            candidates: Candidate recommendation items.

        Returns:
            Adjusted list with sector rotation boosts.
        """
        # Get user's current sector exposure
        user_sectors = set()
        for symbol in user_profile.watchlist:
            if symbol in self.item_database:
                sector = str(self.item_database[symbol].get("sector", ""))
                if sector:
                    user_sectors.add(sector)

        # Get all available sectors from candidates
        candidate_sectors = {
            str(c.metadata.get("sector", ""))
            for c in candidates
            if c.metadata.get("sector")
        }

        # Find underrepresented sectors
        underrepresented = candidate_sectors - user_sectors

        if not underrepresented:
            return candidates

        # Boost items from underrepresented sectors
        adjusted_candidates = []
        for candidate in candidates:
            sector = str(candidate.metadata.get("sector", ""))
            if sector in underrepresented:
                candidate.score *= 1.2
                candidate.explanation = f"Sector rotation opportunity: {candidate.explanation}"

            adjusted_candidates.append(candidate)

        logger.debug(
            "sector_rotation_rules_applied",
            underrepresented_sectors=list(underrepresented),
            num_candidates=len(adjusted_candidates),
        )

        return adjusted_candidates
