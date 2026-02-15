"""Detect contradictions and divergences between analyst opinions.

This module provides capabilities to analyze multiple analyst opinions and detect
contradictions, divergences, and consensus strength. It is designed for financial
analysis where analysts may have conflicting views on the same security.

Key Features:
- Rating divergence detection
- Price target divergence detection
- Sentiment divergence detection
- Temporal contradiction detection (same source changing opinion)
- Consensus strength calculation
- Divergence summarization

Examples:
    Analyze analyst opinions for contradictions:

    >>> from signalforge.nlp.contradiction_detector import ContradictionDetector, AnalystOpinion
    >>> from datetime import datetime
    >>>
    >>> opinions = [
    ...     AnalystOpinion(source="Source1", analyst="John", firm="Firm1",
    ...                    date=datetime.now(), rating="buy", target_price=150.0,
    ...                    sentiment_score=0.8, key_points=["Strong growth"],
    ...                    text="Positive outlook"),
    ...     AnalystOpinion(source="Source2", analyst="Jane", firm="Firm2",
    ...                    date=datetime.now(), rating="sell", target_price=100.0,
    ...                    sentiment_score=-0.6, key_points=["Weak fundamentals"],
    ...                    text="Negative outlook")
    ... ]
    >>> detector = ContradictionDetector()
    >>> analysis = detector.analyze_opinions(opinions)
    >>> print(f"Consensus strength: {analysis.consensus_strength:.2f}")
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from signalforge.core.logging import get_logger

logger = get_logger(__name__)


class ContradictionType(str, Enum):
    """Types of contradictions that can be detected."""

    RATING_DIVERGENCE = "rating_divergence"
    TARGET_DIVERGENCE = "target_divergence"
    SENTIMENT_DIVERGENCE = "sentiment_divergence"
    TEMPORAL_CONTRADICTION = "temporal_contradiction"
    FACT_CONTRADICTION = "fact_contradiction"


class DivergenceSeverity(str, Enum):
    """Severity levels for divergences."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class AnalystOpinion:
    """Represents a single analyst's opinion.

    Attributes:
        source: Source of the opinion (e.g., news outlet, research platform).
        analyst: Name of the analyst (if available).
        firm: Firm or institution (if available).
        date: Date of the opinion.
        rating: Analyst rating (e.g., buy, sell, hold).
        target_price: Price target if provided.
        sentiment_score: Normalized sentiment score from -1 (bearish) to 1 (bullish).
        key_points: List of key points from the opinion.
        text: Full text of the opinion.
    """

    source: str
    analyst: str | None
    firm: str | None
    date: datetime
    rating: str | None
    target_price: float | None
    sentiment_score: float
    key_points: list[str]
    text: str

    def __post_init__(self) -> None:
        """Validate analyst opinion fields."""
        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError(f"Sentiment score must be between -1.0 and 1.0, got {self.sentiment_score}")
        if self.target_price is not None and self.target_price <= 0:
            raise ValueError(f"Target price must be positive, got {self.target_price}")


@dataclass
class Contradiction:
    """Represents a detected contradiction between two opinions.

    Attributes:
        type: Type of contradiction detected.
        severity: Severity level of the divergence.
        opinion_a: First opinion in the contradiction.
        opinion_b: Second opinion in the contradiction.
        description: Human-readable description of the contradiction.
        divergence_score: Quantitative measure of divergence (0-1).
        implications: List of potential implications.
    """

    type: ContradictionType
    severity: DivergenceSeverity
    opinion_a: AnalystOpinion
    opinion_b: AnalystOpinion
    description: str
    divergence_score: float
    implications: list[str]

    def __post_init__(self) -> None:
        """Validate contradiction fields."""
        if not 0.0 <= self.divergence_score <= 1.0:
            raise ValueError(f"Divergence score must be between 0.0 and 1.0, got {self.divergence_score}")


@dataclass
class DivergenceAnalysis:
    """Results of analyzing multiple opinions for divergences.

    Attributes:
        symbol: Stock symbol being analyzed.
        contradictions: List of detected contradictions.
        consensus_strength: Strength of consensus (0 = no consensus, 1 = unanimous).
        bullish_count: Number of bullish opinions.
        bearish_count: Number of bearish opinions.
        neutral_count: Number of neutral opinions.
        key_disagreement_topics: Main topics of disagreement.
        recommendation: Overall recommendation based on analysis.
    """

    symbol: str
    contradictions: list[Contradiction]
    consensus_strength: float
    bullish_count: int
    bearish_count: int
    neutral_count: int
    key_disagreement_topics: list[str]
    recommendation: str

    def __post_init__(self) -> None:
        """Validate divergence analysis fields."""
        if not 0.0 <= self.consensus_strength <= 1.0:
            raise ValueError(f"Consensus strength must be between 0.0 and 1.0, got {self.consensus_strength}")
        if self.bullish_count < 0 or self.bearish_count < 0 or self.neutral_count < 0:
            raise ValueError("Opinion counts must be non-negative")


class ContradictionDetector:
    """Detect contradictions between analyst opinions.

    This class analyzes multiple analyst opinions to detect various types of
    contradictions and divergences, calculate consensus strength, and provide
    actionable summaries.

    Examples:
        >>> detector = ContradictionDetector()
        >>> analysis = detector.analyze_opinions(opinions)
        >>> print(f"Found {len(analysis.contradictions)} contradictions")
    """

    # Rating mapping for numerical comparison
    RATING_MAP: dict[str, float] = {
        "strong_buy": 2.0,
        "buy": 1.0,
        "outperform": 1.0,
        "hold": 0.0,
        "neutral": 0.0,
        "underperform": -1.0,
        "sell": -1.0,
        "strong_sell": -2.0,
    }

    def __init__(self) -> None:
        """Initialize the contradiction detector."""
        logger.info("contradiction_detector_initialized")

    def analyze_opinions(
        self,
        opinions: list[AnalystOpinion],
        symbol: str = "UNKNOWN",
    ) -> DivergenceAnalysis:
        """Analyze a set of opinions for contradictions.

        Args:
            opinions: List of analyst opinions to analyze.
            symbol: Stock symbol being analyzed.

        Returns:
            DivergenceAnalysis containing all detected contradictions and metrics.

        Raises:
            ValueError: If opinions list is empty.

        Examples:
            >>> detector = ContradictionDetector()
            >>> analysis = detector.analyze_opinions(opinions, "AAPL")
            >>> print(analysis.consensus_strength)
        """
        if not opinions:
            raise ValueError("Opinions list cannot be empty")

        logger.info("analyzing_opinions", num_opinions=len(opinions), symbol=symbol)

        # Detect all types of contradictions
        contradictions: list[Contradiction] = []

        rating_contradictions = self.detect_rating_divergence(opinions)
        contradictions.extend(rating_contradictions)

        target_contradictions = self.detect_target_divergence(opinions)
        contradictions.extend(target_contradictions)

        sentiment_contradictions = self.detect_sentiment_divergence(opinions)
        contradictions.extend(sentiment_contradictions)

        temporal_contradictions = self.detect_temporal_contradiction(opinions)
        contradictions.extend(temporal_contradictions)

        # Calculate consensus strength
        consensus_strength = self.calculate_consensus_strength(opinions)

        # Count sentiment distribution
        bullish_count = sum(1 for op in opinions if op.sentiment_score > 0.2)
        bearish_count = sum(1 for op in opinions if op.sentiment_score < -0.2)
        neutral_count = len(opinions) - bullish_count - bearish_count

        # Extract key disagreement topics
        key_topics = self.extract_key_disagreement_topics(contradictions)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            bullish_count, bearish_count, neutral_count, consensus_strength
        )

        analysis = DivergenceAnalysis(
            symbol=symbol,
            contradictions=contradictions,
            consensus_strength=consensus_strength,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            key_disagreement_topics=key_topics,
            recommendation=recommendation,
        )

        logger.info(
            "analysis_complete",
            num_contradictions=len(contradictions),
            consensus_strength=consensus_strength,
            bullish=bullish_count,
            bearish=bearish_count,
            neutral=neutral_count,
        )

        return analysis

    def detect_rating_divergence(
        self,
        opinions: list[AnalystOpinion],
        threshold: float = 0.3,
    ) -> list[Contradiction]:
        """Detect divergence in analyst ratings.

        Args:
            opinions: List of analyst opinions.
            threshold: Minimum divergence score to flag (0-1).

        Returns:
            List of detected rating contradictions.

        Examples:
            >>> detector = ContradictionDetector()
            >>> contradictions = detector.detect_rating_divergence(opinions)
        """
        contradictions: list[Contradiction] = []

        # Get opinions with ratings
        rated_opinions = [op for op in opinions if op.rating is not None]

        if len(rated_opinions) < 2:
            return contradictions

        # Compare each pair of opinions
        for i in range(len(rated_opinions)):
            for j in range(i + 1, len(rated_opinions)):
                op_a = rated_opinions[i]
                op_b = rated_opinions[j]

                # Convert ratings to numerical values
                rating_a = self._normalize_rating(op_a.rating)
                rating_b = self._normalize_rating(op_b.rating)

                if rating_a is None or rating_b is None:
                    continue

                # Calculate divergence score (0-1 scale)
                divergence_score = abs(rating_a - rating_b) / 4.0

                if divergence_score >= threshold:
                    severity = self._calculate_severity(divergence_score)
                    description = f"{op_a.source} rates {op_a.rating} while {op_b.source} rates {op_b.rating}"

                    implications = self._generate_rating_implications(
                        op_a.rating or "", op_b.rating or "", divergence_score
                    )

                    contradiction = Contradiction(
                        type=ContradictionType.RATING_DIVERGENCE,
                        severity=severity,
                        opinion_a=op_a,
                        opinion_b=op_b,
                        description=description,
                        divergence_score=divergence_score,
                        implications=implications,
                    )
                    contradictions.append(contradiction)

        logger.debug("rating_divergence_detected", count=len(contradictions))
        return contradictions

    def detect_target_divergence(
        self,
        opinions: list[AnalystOpinion],
        threshold_percent: float = 30.0,
    ) -> list[Contradiction]:
        """Detect divergence in price targets.

        Args:
            opinions: List of analyst opinions.
            threshold_percent: Minimum percentage difference to flag.

        Returns:
            List of detected target price contradictions.

        Examples:
            >>> detector = ContradictionDetector()
            >>> contradictions = detector.detect_target_divergence(opinions)
        """
        contradictions: list[Contradiction] = []

        # Get opinions with target prices
        target_opinions = [op for op in opinions if op.target_price is not None]

        if len(target_opinions) < 2:
            return contradictions

        # Compare each pair
        for i in range(len(target_opinions)):
            for j in range(i + 1, len(target_opinions)):
                op_a = target_opinions[i]
                op_b = target_opinions[j]

                target_a = op_a.target_price
                target_b = op_b.target_price

                if target_a is None or target_b is None:
                    continue

                # Calculate percentage difference
                max_target = max(target_a, target_b)
                min_target = min(target_a, target_b)
                percent_diff = ((max_target - min_target) / min_target) * 100

                if percent_diff >= threshold_percent:
                    # Normalize to 0-1 scale
                    divergence_score = min(percent_diff / 100.0, 1.0)
                    severity = self._calculate_severity(divergence_score)

                    description = (
                        f"{op_a.source} targets ${target_a:.2f} while "
                        f"{op_b.source} targets ${target_b:.2f} ({percent_diff:.1f}% difference)"
                    )

                    implications = [
                        "Wide price target range indicates high uncertainty",
                        "Potential upside/downside varies significantly between analysts",
                    ]

                    contradiction = Contradiction(
                        type=ContradictionType.TARGET_DIVERGENCE,
                        severity=severity,
                        opinion_a=op_a,
                        opinion_b=op_b,
                        description=description,
                        divergence_score=divergence_score,
                        implications=implications,
                    )
                    contradictions.append(contradiction)

        logger.debug("target_divergence_detected", count=len(contradictions))
        return contradictions

    def detect_sentiment_divergence(
        self,
        opinions: list[AnalystOpinion],
        threshold: float = 0.5,
    ) -> list[Contradiction]:
        """Detect conflicting sentiment between opinions.

        Args:
            opinions: List of analyst opinions.
            threshold: Minimum sentiment difference to flag (0-2).

        Returns:
            List of detected sentiment contradictions.

        Examples:
            >>> detector = ContradictionDetector()
            >>> contradictions = detector.detect_sentiment_divergence(opinions)
        """
        contradictions: list[Contradiction] = []

        if len(opinions) < 2:
            return contradictions

        # Compare each pair
        for i in range(len(opinions)):
            for j in range(i + 1, len(opinions)):
                op_a = opinions[i]
                op_b = opinions[j]

                # Calculate sentiment divergence (range: 0-2)
                sentiment_diff = abs(op_a.sentiment_score - op_b.sentiment_score)

                if sentiment_diff >= threshold:
                    # Normalize to 0-1 scale
                    divergence_score = sentiment_diff / 2.0
                    severity = self._calculate_severity(divergence_score)

                    sentiment_a_label = self._sentiment_label(op_a.sentiment_score)
                    sentiment_b_label = self._sentiment_label(op_b.sentiment_score)

                    description = (
                        f"{op_a.source} has {sentiment_a_label} sentiment "
                        f"({op_a.sentiment_score:.2f}) while {op_b.source} has "
                        f"{sentiment_b_label} sentiment ({op_b.sentiment_score:.2f})"
                    )

                    implications = [
                        "Conflicting market sentiment from different sources",
                        "Consider reviewing underlying assumptions and data",
                    ]

                    contradiction = Contradiction(
                        type=ContradictionType.SENTIMENT_DIVERGENCE,
                        severity=severity,
                        opinion_a=op_a,
                        opinion_b=op_b,
                        description=description,
                        divergence_score=divergence_score,
                        implications=implications,
                    )
                    contradictions.append(contradiction)

        logger.debug("sentiment_divergence_detected", count=len(contradictions))
        return contradictions

    def detect_temporal_contradiction(
        self,
        opinions: list[AnalystOpinion],
        same_source: bool = True,
    ) -> list[Contradiction]:
        """Detect when same source changes opinion significantly.

        Args:
            opinions: List of analyst opinions.
            same_source: Only check opinions from the same source.

        Returns:
            List of detected temporal contradictions.

        Examples:
            >>> detector = ContradictionDetector()
            >>> contradictions = detector.detect_temporal_contradiction(opinions)
        """
        contradictions: list[Contradiction] = []

        if len(opinions) < 2:
            return contradictions

        # Group by source/analyst
        grouped: dict[str, list[AnalystOpinion]] = defaultdict(list)

        for opinion in opinions:
            key = opinion.source
            if opinion.analyst and same_source:
                key = f"{opinion.source}_{opinion.analyst}"
            grouped[key].append(opinion)

        # Check each group for temporal contradictions
        for key, group_opinions in grouped.items():
            if len(group_opinions) < 2:
                continue

            # Sort by date
            sorted_opinions = sorted(group_opinions, key=lambda x: x.date)

            # Compare consecutive opinions
            for i in range(len(sorted_opinions) - 1):
                op_old = sorted_opinions[i]
                op_new = sorted_opinions[i + 1]

                # Check for rating change
                if op_old.rating and op_new.rating:
                    rating_old = self._normalize_rating(op_old.rating)
                    rating_new = self._normalize_rating(op_new.rating)

                    if rating_old is not None and rating_new is not None:
                        rating_change = abs(rating_new - rating_old)
                        if rating_change >= 2.0:  # Changed by 2+ levels
                            days_diff = (op_new.date - op_old.date).days

                            divergence_score = min(rating_change / 4.0, 1.0)
                            severity = self._calculate_severity(divergence_score)

                            description = (
                                f"{key} changed rating from {op_old.rating} to "
                                f"{op_new.rating} over {days_diff} days"
                            )

                            implications = [
                                "Significant change in analyst's view",
                                "May indicate new information or changed market conditions",
                            ]

                            contradiction = Contradiction(
                                type=ContradictionType.TEMPORAL_CONTRADICTION,
                                severity=severity,
                                opinion_a=op_old,
                                opinion_b=op_new,
                                description=description,
                                divergence_score=divergence_score,
                                implications=implications,
                            )
                            contradictions.append(contradiction)

        logger.debug("temporal_contradictions_detected", count=len(contradictions))
        return contradictions

    def calculate_consensus_strength(
        self,
        opinions: list[AnalystOpinion],
    ) -> float:
        """Calculate how strong the consensus is.

        Args:
            opinions: List of analyst opinions.

        Returns:
            Consensus strength (0 = no consensus, 1 = unanimous).

        Examples:
            >>> detector = ContradictionDetector()
            >>> strength = detector.calculate_consensus_strength(opinions)
            >>> print(f"Consensus: {strength:.2%}")
        """
        if not opinions:
            return 0.0

        if len(opinions) == 1:
            return 1.0

        # Calculate variance in sentiment scores
        sentiment_scores = [op.sentiment_score for op in opinions]
        mean_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        variance = sum((s - mean_sentiment) ** 2 for s in sentiment_scores) / len(sentiment_scores)

        # Normalize variance to 0-1 scale (max variance = 1 when all scores are at extremes)
        max_variance = 1.0
        normalized_variance = min(variance / max_variance, 1.0)

        # Consensus strength is inverse of variance
        consensus = 1.0 - normalized_variance

        logger.debug(
            "consensus_calculated",
            consensus_strength=consensus,
            mean_sentiment=mean_sentiment,
            variance=variance,
        )

        return consensus

    def extract_key_disagreement_topics(
        self,
        contradictions: list[Contradiction],
    ) -> list[str]:
        """Extract main topics of disagreement.

        Args:
            contradictions: List of detected contradictions.

        Returns:
            List of key disagreement topics.

        Examples:
            >>> detector = ContradictionDetector()
            >>> topics = detector.extract_key_disagreement_topics(contradictions)
        """
        if not contradictions:
            return []

        # Count key points from contradicting opinions
        topic_counts: dict[str, int] = defaultdict(int)

        for contradiction in contradictions:
            for point in contradiction.opinion_a.key_points:
                topic_counts[point.lower()] += 1
            for point in contradiction.opinion_b.key_points:
                topic_counts[point.lower()] += 1

        # Sort by frequency and return top topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        top_topics = [topic for topic, _ in sorted_topics[:5]]

        logger.debug("disagreement_topics_extracted", count=len(top_topics))
        return top_topics

    def generate_divergence_summary(
        self,
        analysis: DivergenceAnalysis,
    ) -> str:
        """Generate human-readable summary of divergences.

        Args:
            analysis: Divergence analysis results.

        Returns:
            Human-readable summary string.

        Examples:
            >>> detector = ContradictionDetector()
            >>> summary = detector.generate_divergence_summary(analysis)
            >>> print(summary)
        """
        lines: list[str] = []

        lines.append(f"Divergence Analysis for {analysis.symbol}")
        lines.append("=" * 50)
        lines.append(f"Total Opinions Analyzed: {analysis.bullish_count + analysis.bearish_count + analysis.neutral_count}")
        lines.append(f"Bullish: {analysis.bullish_count}, Bearish: {analysis.bearish_count}, Neutral: {analysis.neutral_count}")
        lines.append(f"Consensus Strength: {analysis.consensus_strength:.2%}")
        lines.append(f"Contradictions Found: {len(analysis.contradictions)}")
        lines.append("")

        if analysis.contradictions:
            lines.append("Key Contradictions:")
            for i, contradiction in enumerate(analysis.contradictions[:5], 1):
                lines.append(f"{i}. [{contradiction.severity.upper()}] {contradiction.description}")

        if analysis.key_disagreement_topics:
            lines.append("")
            lines.append("Key Disagreement Topics:")
            for topic in analysis.key_disagreement_topics:
                lines.append(f"  - {topic}")

        lines.append("")
        lines.append(f"Recommendation: {analysis.recommendation}")

        summary = "\n".join(lines)
        logger.debug("divergence_summary_generated", length=len(summary))
        return summary

    def compare_to_consensus(
        self,
        opinion: AnalystOpinion,
        consensus: DivergenceAnalysis,
    ) -> dict[str, Any]:
        """Compare a single opinion to the consensus.

        Args:
            opinion: Single analyst opinion to compare.
            consensus: Consensus analysis to compare against.

        Returns:
            Dictionary with comparison metrics (alignment, consensus_direction,
            opinion_direction, sentiment_diff, is_contrarian, consensus_strength).

        Examples:
            >>> detector = ContradictionDetector()
            >>> comparison = detector.compare_to_consensus(opinion, consensus)
            >>> print(comparison['alignment'])
        """
        total_opinions = consensus.bullish_count + consensus.bearish_count + consensus.neutral_count

        if total_opinions == 0:
            return {
                "alignment": "unknown",
                "sentiment_diff": 0.0,
                "is_contrarian": False,
            }

        # Determine consensus direction
        if consensus.bullish_count > consensus.bearish_count and consensus.bullish_count > consensus.neutral_count:
            consensus_direction = "bullish"
            consensus_sentiment = 0.5
        elif consensus.bearish_count > consensus.bullish_count and consensus.bearish_count > consensus.neutral_count:
            consensus_direction = "bearish"
            consensus_sentiment = -0.5
        else:
            consensus_direction = "neutral"
            consensus_sentiment = 0.0

        # Determine opinion direction
        opinion_direction = self._sentiment_label(opinion.sentiment_score)

        # Calculate alignment
        alignment = "aligned" if opinion_direction == consensus_direction else "divergent"
        is_contrarian = alignment == "divergent" and consensus.consensus_strength > 0.7

        sentiment_diff = abs(opinion.sentiment_score - consensus_sentiment)

        result = {
            "alignment": alignment,
            "consensus_direction": consensus_direction,
            "opinion_direction": opinion_direction,
            "sentiment_diff": sentiment_diff,
            "is_contrarian": is_contrarian,
            "consensus_strength": consensus.consensus_strength,
        }

        logger.debug("opinion_compared_to_consensus", alignment=alignment, is_contrarian=is_contrarian)
        return result

    def _normalize_rating(self, rating: str | None) -> float | None:
        """Normalize rating to numerical value.

        Args:
            rating: Rating string.

        Returns:
            Numerical rating value or None if invalid.
        """
        if rating is None:
            return None

        normalized = rating.lower().replace(" ", "_")
        return self.RATING_MAP.get(normalized)

    def _calculate_severity(self, divergence_score: float) -> DivergenceSeverity:
        """Calculate severity based on divergence score.

        Args:
            divergence_score: Divergence score (0-1).

        Returns:
            DivergenceSeverity enum value.
        """
        if divergence_score >= 0.75:
            return DivergenceSeverity.EXTREME
        elif divergence_score >= 0.5:
            return DivergenceSeverity.HIGH
        elif divergence_score >= 0.3:
            return DivergenceSeverity.MEDIUM
        else:
            return DivergenceSeverity.LOW

    def _sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label.

        Args:
            score: Sentiment score (-1 to 1).

        Returns:
            Sentiment label string.
        """
        if score > 0.2:
            return "bullish"
        elif score < -0.2:
            return "bearish"
        else:
            return "neutral"

    def _generate_rating_implications(
        self,
        _rating_a: str,
        _rating_b: str,
        divergence_score: float,
    ) -> list[str]:
        """Generate implications for rating divergence.

        Args:
            _rating_a: First rating (currently unused, reserved for future use).
            _rating_b: Second rating (currently unused, reserved for future use).
            divergence_score: Divergence score.

        Returns:
            List of implication strings.
        """
        implications: list[str] = []

        if divergence_score >= 0.75:
            implications.append("Extreme divergence suggests fundamentally different analysis frameworks")
            implications.append("High uncertainty - recommend independent research")
        elif divergence_score >= 0.5:
            implications.append("Significant disagreement on stock outlook")
            implications.append("May indicate valuation methodology differences")
        else:
            implications.append("Moderate disagreement within normal analyst variance")

        return implications

    def _generate_recommendation(
        self,
        bullish: int,
        bearish: int,
        neutral: int,
        consensus_strength: float,
    ) -> str:
        """Generate overall recommendation.

        Args:
            bullish: Number of bullish opinions.
            bearish: Number of bearish opinions.
            neutral: Number of neutral opinions.
            consensus_strength: Consensus strength (0-1).

        Returns:
            Recommendation string.
        """
        total = bullish + bearish + neutral

        if total == 0:
            return "Insufficient data for recommendation"

        # Determine majority
        if bullish > bearish and bullish > neutral:
            direction = "bullish"
        elif bearish > bullish and bearish > neutral:
            direction = "bearish"
        else:
            direction = "neutral"

        # Consider consensus strength
        if consensus_strength > 0.7:
            confidence = "Strong"
        elif consensus_strength > 0.5:
            confidence = "Moderate"
        else:
            confidence = "Weak"

        recommendation = f"{confidence} {direction} consensus"

        if consensus_strength < 0.5:
            recommendation += " - High disagreement, exercise caution"

        return recommendation
