"""Score urgency/timeliness of financial news.

This module provides functionality to assess the urgency and time-sensitivity
of financial news content using pattern matching and temporal analysis.

Key Features:
- Urgency level classification (critical, high, medium, low, archive)
- Temporal signal detection
- Time decay calculation
- Immediate action detection
- Relevance window estimation
- Batch scoring support

Examples:
    Score urgency of a news article:

    >>> from signalforge.nlp.urgency_scorer import UrgencyScorer
    >>>
    >>> scorer = UrgencyScorer()
    >>> text = "BREAKING: Company announces major acquisition"
    >>> result = scorer.score_urgency(text, title="Breaking News")
    >>> print(f"Urgency: {result.level}, Score: {result.score:.2f}")
    Urgency: critical, Score: 0.95

    Check if immediate action required:

    >>> text = "Trading halted due to pending news"
    >>> requires_action = scorer.requires_immediate_action(text)
    >>> print(f"Immediate action: {requires_action}")
    Immediate action: True
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

from signalforge.core.logging import get_logger

logger = get_logger(__name__)


class UrgencyLevel(str, Enum):
    """Urgency level classification."""

    CRITICAL = "critical"  # Immediate action needed
    HIGH = "high"  # Same day relevance
    MEDIUM = "medium"  # This week relevance
    LOW = "low"  # General information
    ARCHIVE = "archive"  # Historical, no urgency


@dataclass
class UrgencyResult:
    """Result of urgency scoring.

    Attributes:
        level: Urgency level classification.
        score: Urgency score from 0.0 (no urgency) to 1.0 (maximum urgency).
        temporal_signals: List of detected urgency phrases.
        action_required: Whether immediate action is needed.
        time_sensitivity_hours: Estimated hours until content becomes stale.
        confidence: Confidence in the urgency assessment.
    """

    level: UrgencyLevel
    score: float
    temporal_signals: list[str]
    action_required: bool
    time_sensitivity_hours: int | None
    confidence: float

    def __post_init__(self) -> None:
        """Validate urgency result fields."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be between 0.0 and 1.0, got {self.score}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

        if self.time_sensitivity_hours is not None and self.time_sensitivity_hours < 0:
            raise ValueError(
                f"time_sensitivity_hours must be non-negative, got {self.time_sensitivity_hours}"
            )


class UrgencyScorer:
    """Score the urgency/timeliness of news content.

    This scorer uses pattern matching to identify urgency signals and
    classifies content based on time sensitivity and action requirements.

    Examples:
        >>> scorer = UrgencyScorer()
        >>> text = "Company reports earnings beat this morning"
        >>> result = scorer.score_urgency(text)
        >>> result.level
        <UrgencyLevel.HIGH: 'high'>
    """

    CRITICAL_SIGNALS = [
        r"\bbreaking\b",
        r"just announced",
        r"\bhalted\b",
        r"\bsuspended\b",
        r"\bbankruptcy\b",
        r"\bfraud\b",
        r"sec investigation",
        r"fda reject(?:ed|s|ion)?",
        r"ceo resign(?:ed|s|ation)?",
        r"\bimmediate\b",
        r"\bemergency\b",
        r"\bcrisis\b",
        r"trading (?:halt|suspended)",
        r"circuit breaker",
        r"\bcrash(?:ed|ing)?\b",
        r"market (?:crash|collapse)",
    ]

    HIGH_SIGNALS = [
        r"\btoday\b",
        r"this morning",
        r"this afternoon",
        r"\breports?\b",
        r"\bannounces?\b",
        r"\bwarns?\b",
        r"guidance cut",
        r"\bdowngrade[sd]?\b",
        r"\bupgrade[sd]?\b",
        r"\bbeat[s]?\b",
        r"\bmiss(?:ed|es)?\b",
        r"earnings (?:call|report)",
        r"press release",
        r"\bjust\b",
        r"right now",
        r"breaking",
        r"developing",
    ]

    MEDIUM_SIGNALS = [
        r"this week",
        r"\bupcoming\b",
        r"\bexpects?\b",
        r"plans to",
        r"\bconsidering\b",
        r"\bexploring\b",
        r"in talks",
        r"negotiations?",
        r"\bsoon\b",
        r"next week",
        r"next month",
        r"\bscheduled\b",
    ]

    LOW_SIGNALS = [
        r"\bcould\b",
        r"\bmight\b",
        r"\bmay\b",
        r"\banalysis\b",
        r"\bopinion\b",
        r"\boutlook\b",
        r"long.?term",
        r"\bhistor(?:y|ical)\b",
        r"\bbackground\b",
        r"\bcontext\b",
        r"\btrend\b",
    ]

    ARCHIVE_SIGNALS = [
        r"last year",
        r"years? ago",
        r"decades? ago",
        r"\bhistoric(?:al)?\b",
        r"in \d{4}",  # References to specific past years
        r"previously",
        r"in the past",
    ]

    def __init__(self) -> None:
        """Initialize the urgency scorer."""
        self.logger = get_logger(__name__)

        # Compile regex patterns
        self._critical_regexes = [re.compile(p, re.IGNORECASE) for p in self.CRITICAL_SIGNALS]
        self._high_regexes = [re.compile(p, re.IGNORECASE) for p in self.HIGH_SIGNALS]
        self._medium_regexes = [re.compile(p, re.IGNORECASE) for p in self.MEDIUM_SIGNALS]
        self._low_regexes = [re.compile(p, re.IGNORECASE) for p in self.LOW_SIGNALS]
        self._archive_regexes = [re.compile(p, re.IGNORECASE) for p in self.ARCHIVE_SIGNALS]

        self.logger.info("urgency_scorer_initialized")

    def score_urgency(
        self,
        text: str,
        title: str | None = None,
        published_at: datetime | None = None,
    ) -> UrgencyResult:
        """Score the urgency of a piece of content.

        Args:
            text: Content text to analyze.
            title: Optional title (weighted more heavily).
            published_at: Publication timestamp for time decay calculation.

        Returns:
            UrgencyResult with urgency classification and metadata.

        Examples:
            >>> scorer = UrgencyScorer()
            >>> text = "Breaking: Major merger announced"
            >>> result = scorer.score_urgency(text)
            >>> result.level
            <UrgencyLevel.CRITICAL: 'critical'>
        """
        if not text or not text.strip():
            raise ValueError("text cannot be empty")

        # Combine title and text, weighting title more heavily
        full_text = text
        if title:
            # Repeat title to give it more weight in pattern matching
            full_text = f"{title} {title} {text}"

        # Detect temporal signals
        temporal_signals = self.detect_temporal_signals(full_text)

        # Calculate base urgency score
        score = 0.0
        detected_level = UrgencyLevel.LOW

        # Check critical signals (score: 0.9-1.0)
        critical_matches = sum(
            1 for regex in self._critical_regexes if regex.search(full_text)
        )
        if critical_matches > 0:
            score = max(score, 0.9 + min(critical_matches * 0.05, 0.1))
            detected_level = UrgencyLevel.CRITICAL

        # Check high urgency signals (score: 0.6-0.8)
        elif any(regex.search(full_text) for regex in self._high_regexes):
            high_matches = sum(1 for regex in self._high_regexes if regex.search(full_text))
            score = max(score, 0.6 + min(high_matches * 0.05, 0.2))
            detected_level = UrgencyLevel.HIGH

        # Check medium urgency signals (score: 0.3-0.5)
        elif any(regex.search(full_text) for regex in self._medium_regexes):
            medium_matches = sum(
                1 for regex in self._medium_regexes if regex.search(full_text)
            )
            score = max(score, 0.3 + min(medium_matches * 0.05, 0.2))
            detected_level = UrgencyLevel.MEDIUM

        # Check archive signals (score: 0.0-0.1)
        elif any(regex.search(full_text) for regex in self._archive_regexes):
            score = 0.05
            detected_level = UrgencyLevel.ARCHIVE

        # Check low urgency signals (score: 0.1-0.3)
        elif any(regex.search(full_text) for regex in self._low_regexes):
            score = 0.2
            detected_level = UrgencyLevel.LOW

        else:
            # No strong signals, default to low
            score = 0.25
            detected_level = UrgencyLevel.LOW

        # Apply time decay if publication time is known
        final_level = detected_level
        if published_at is not None:
            final_level = self.calculate_time_decay(published_at, detected_level)
            # Adjust score based on time decay
            if final_level.value != detected_level.value:
                level_scores = {
                    UrgencyLevel.CRITICAL: 1.0,
                    UrgencyLevel.HIGH: 0.7,
                    UrgencyLevel.MEDIUM: 0.4,
                    UrgencyLevel.LOW: 0.2,
                    UrgencyLevel.ARCHIVE: 0.05,
                }
                score = level_scores[final_level]

        # Check if immediate action required
        action_required = self.requires_immediate_action(full_text)

        # Estimate time sensitivity
        time_sensitivity_hours = self.estimate_relevance_window(final_level)

        # Calculate confidence based on number of signals
        confidence = min(0.7 + len(temporal_signals) * 0.05, 1.0)

        result = UrgencyResult(
            level=final_level,
            score=score,
            temporal_signals=temporal_signals,
            action_required=action_required,
            time_sensitivity_hours=time_sensitivity_hours,
            confidence=confidence,
        )

        self.logger.debug(
            "urgency_scored",
            level=final_level.value,
            score=score,
            num_signals=len(temporal_signals),
            action_required=action_required,
        )

        return result

    def detect_temporal_signals(self, text: str) -> list[str]:
        """Extract temporal/urgency signals from text.

        Args:
            text: Text to analyze.

        Returns:
            List of detected urgency phrases.

        Examples:
            >>> scorer = UrgencyScorer()
            >>> text = "Breaking news: Company announces today"
            >>> signals = scorer.detect_temporal_signals(text)
            >>> "breaking" in [s.lower() for s in signals]
            True
        """
        if not text:
            return []

        signals = []

        # Collect all pattern matches
        all_regexes = [
            (self._critical_regexes, "critical"),
            (self._high_regexes, "high"),
            (self._medium_regexes, "medium"),
            (self._low_regexes, "low"),
            (self._archive_regexes, "archive"),
        ]

        for regexes, _category in all_regexes:
            for regex in regexes:
                match = regex.search(text)
                if match:
                    signals.append(match.group(0))

        # Remove duplicates while preserving order
        seen = set()
        unique_signals = []
        for signal in signals:
            if signal.lower() not in seen:
                seen.add(signal.lower())
                unique_signals.append(signal)

        self.logger.debug("temporal_signals_detected", num_signals=len(unique_signals))

        return unique_signals

    def calculate_time_decay(
        self, published_at: datetime, base_urgency: UrgencyLevel
    ) -> UrgencyLevel:
        """Adjust urgency based on time since publication.

        Args:
            published_at: Publication timestamp.
            base_urgency: Initial urgency level.

        Returns:
            Adjusted urgency level after time decay.

        Examples:
            >>> scorer = UrgencyScorer()
            >>> from datetime import datetime, timedelta
            >>> old_time = datetime.now() - timedelta(days=7)
            >>> level = scorer.calculate_time_decay(old_time, UrgencyLevel.HIGH)
            >>> level in [UrgencyLevel.MEDIUM, UrgencyLevel.LOW]
            True
        """
        now = datetime.now(UTC)

        # Handle future timestamps
        if published_at > now:
            self.logger.warning("future_timestamp", published_at=published_at.isoformat())
            return base_urgency

        time_diff = now - published_at

        # Archive content doesn't decay
        if base_urgency == UrgencyLevel.ARCHIVE:
            return UrgencyLevel.ARCHIVE

        # Critical urgency decay
        if base_urgency == UrgencyLevel.CRITICAL:
            if time_diff < timedelta(hours=1):
                return UrgencyLevel.CRITICAL
            elif time_diff < timedelta(hours=6):
                return UrgencyLevel.HIGH
            elif time_diff < timedelta(days=1):
                return UrgencyLevel.MEDIUM
            else:
                return UrgencyLevel.LOW

        # High urgency decay
        if base_urgency == UrgencyLevel.HIGH:
            if time_diff < timedelta(hours=6):
                return UrgencyLevel.HIGH
            elif time_diff < timedelta(days=1):
                return UrgencyLevel.MEDIUM
            elif time_diff < timedelta(days=3):
                return UrgencyLevel.LOW
            else:
                return UrgencyLevel.ARCHIVE

        # Medium urgency decay
        if base_urgency == UrgencyLevel.MEDIUM:
            if time_diff < timedelta(days=2):
                return UrgencyLevel.MEDIUM
            elif time_diff < timedelta(days=7):
                return UrgencyLevel.LOW
            else:
                return UrgencyLevel.ARCHIVE

        # Low urgency decay
        if base_urgency == UrgencyLevel.LOW:
            if time_diff < timedelta(days=7):
                return UrgencyLevel.LOW
            else:
                return UrgencyLevel.ARCHIVE

        return base_urgency

    def requires_immediate_action(self, text: str) -> bool:
        """Determine if content requires immediate action.

        Args:
            text: Text to analyze.

        Returns:
            True if immediate action required, False otherwise.

        Examples:
            >>> scorer = UrgencyScorer()
            >>> text = "Trading halted, immediate action required"
            >>> scorer.requires_immediate_action(text)
            True
        """
        if not text:
            return False

        # Check for critical signals
        critical_count = sum(1 for regex in self._critical_regexes if regex.search(text))

        # Requires immediate action if multiple critical signals or specific phrases
        action_patterns = [
            r"immediate action",
            r"act now",
            r"urgent",
            r"breaking",
            r"halted",
            r"suspended",
        ]

        action_regexes = [re.compile(p, re.IGNORECASE) for p in action_patterns]
        action_count = sum(1 for regex in action_regexes if regex.search(text))

        requires_action = critical_count >= 2 or action_count >= 1

        self.logger.debug(
            "action_requirement_checked",
            requires_action=requires_action,
            critical_count=critical_count,
            action_count=action_count,
        )

        return requires_action

    def estimate_relevance_window(self, urgency_level: UrgencyLevel) -> int:
        """Estimate hours until content becomes stale.

        Args:
            urgency_level: Urgency level classification.

        Returns:
            Estimated hours of relevance.

        Examples:
            >>> scorer = UrgencyScorer()
            >>> hours = scorer.estimate_relevance_window(UrgencyLevel.CRITICAL)
            >>> hours
            1
        """
        window_map = {
            UrgencyLevel.CRITICAL: 1,  # 1 hour
            UrgencyLevel.HIGH: 6,  # 6 hours
            UrgencyLevel.MEDIUM: 48,  # 2 days
            UrgencyLevel.LOW: 168,  # 1 week
            UrgencyLevel.ARCHIVE: 8760,  # 1 year (effectively indefinite)
        }

        hours = window_map.get(urgency_level, 24)

        self.logger.debug(
            "relevance_window_estimated", urgency_level=urgency_level.value, hours=hours
        )

        return hours

    def batch_score(self, texts: list[str]) -> list[UrgencyResult]:
        """Score multiple texts efficiently.

        Args:
            texts: List of texts to score.

        Returns:
            List of UrgencyResult objects, one per input text.

        Examples:
            >>> scorer = UrgencyScorer()
            >>> texts = ["Breaking news", "Analysis of trends"]
            >>> results = scorer.batch_score(texts)
            >>> len(results)
            2
        """
        if not texts:
            return []

        results = []
        for i, text in enumerate(texts):
            try:
                result = self.score_urgency(text)
                results.append(result)
            except ValueError as e:
                self.logger.warning(
                    "batch_score_item_failed", index=i, error=str(e), text_preview=text[:100]
                )
                # Add a low urgency result for failed items
                results.append(
                    UrgencyResult(
                        level=UrgencyLevel.LOW,
                        score=0.0,
                        temporal_signals=[],
                        action_required=False,
                        time_sensitivity_hours=168,
                        confidence=0.0,
                    )
                )

        self.logger.info("batch_scoring_complete", total_texts=len(texts), successful=len(results))

        return results
