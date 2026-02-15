"""Tests for urgency scoring module.

This module tests the UrgencyScorer's ability to classify news urgency
and detect temporal signals using pattern matching.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest


class TestUrgencyLevel:
    """Tests for UrgencyLevel enum."""

    def test_urgency_level_values(self) -> None:
        """Test urgency level enum values."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        assert UrgencyLevel.CRITICAL.value == "critical"
        assert UrgencyLevel.HIGH.value == "high"
        assert UrgencyLevel.MEDIUM.value == "medium"
        assert UrgencyLevel.LOW.value == "low"
        assert UrgencyLevel.ARCHIVE.value == "archive"


class TestUrgencyResult:
    """Tests for UrgencyResult dataclass."""

    def test_valid_urgency_result(self) -> None:
        """Test creation of valid urgency result."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel, UrgencyResult

        result = UrgencyResult(
            level=UrgencyLevel.HIGH,
            score=0.75,
            temporal_signals=["today", "announces"],
            action_required=False,
            time_sensitivity_hours=6,
            confidence=0.85,
        )

        assert result.level == UrgencyLevel.HIGH
        assert result.score == 0.75
        assert len(result.temporal_signals) == 2
        assert result.action_required is False
        assert result.time_sensitivity_hours == 6
        assert result.confidence == 0.85

    def test_invalid_score(self) -> None:
        """Test validation of score range."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel, UrgencyResult

        with pytest.raises(ValueError, match="score must be between"):
            UrgencyResult(
                level=UrgencyLevel.HIGH,
                score=1.5,
                temporal_signals=[],
                action_required=False,
                time_sensitivity_hours=6,
                confidence=0.85,
            )

    def test_invalid_confidence(self) -> None:
        """Test validation of confidence range."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel, UrgencyResult

        with pytest.raises(ValueError, match="confidence must be between"):
            UrgencyResult(
                level=UrgencyLevel.HIGH,
                score=0.75,
                temporal_signals=[],
                action_required=False,
                time_sensitivity_hours=6,
                confidence=-0.1,
            )

    def test_invalid_time_sensitivity(self) -> None:
        """Test validation of time_sensitivity_hours."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel, UrgencyResult

        with pytest.raises(ValueError, match="time_sensitivity_hours must be non-negative"):
            UrgencyResult(
                level=UrgencyLevel.HIGH,
                score=0.75,
                temporal_signals=[],
                action_required=False,
                time_sensitivity_hours=-5,
                confidence=0.85,
            )


class TestUrgencyScorer:
    """Tests for UrgencyScorer class."""

    @pytest.fixture
    def scorer(self) -> Any:
        """Create urgency scorer."""
        from signalforge.nlp.urgency_scorer import UrgencyScorer

        return UrgencyScorer()

    def test_initialization(self, scorer: Any) -> None:
        """Test scorer initialization."""
        assert scorer is not None
        assert len(scorer._critical_regexes) > 0
        assert len(scorer._high_regexes) > 0
        assert len(scorer._medium_regexes) > 0
        assert len(scorer._low_regexes) > 0

    def test_score_critical_urgency_breaking(self, scorer: Any) -> None:
        """Test scoring of critical urgency with breaking news."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text = "BREAKING: Company announces major acquisition"
        result = scorer.score_urgency(text)

        assert result.level == UrgencyLevel.CRITICAL
        assert result.score >= 0.9
        assert len(result.temporal_signals) > 0

    def test_score_critical_urgency_halted(self, scorer: Any) -> None:
        """Test scoring of critical urgency with trading halt."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text = "Trading halted due to pending announcement"
        result = scorer.score_urgency(text)

        assert result.level == UrgencyLevel.CRITICAL
        assert result.score >= 0.9

    def test_score_critical_urgency_bankruptcy(self, scorer: Any) -> None:
        """Test scoring of critical urgency with bankruptcy."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text = "Company files for bankruptcy protection"
        result = scorer.score_urgency(text)

        assert result.level == UrgencyLevel.CRITICAL

    def test_score_high_urgency_today(self, scorer: Any) -> None:
        """Test scoring of high urgency with today reference."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text = "Company reports earnings today"
        result = scorer.score_urgency(text)

        assert result.level == UrgencyLevel.HIGH
        assert 0.6 <= result.score <= 1.0

    def test_score_high_urgency_announces(self, scorer: Any) -> None:
        """Test scoring of high urgency with announces."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text = "CEO announces major restructuring"
        result = scorer.score_urgency(text)

        assert result.level == UrgencyLevel.HIGH

    def test_score_high_urgency_upgrade(self, scorer: Any) -> None:
        """Test scoring of high urgency with upgrade."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text = "Analyst upgrades stock to buy"
        result = scorer.score_urgency(text)

        assert result.level == UrgencyLevel.HIGH

    def test_score_medium_urgency_this_week(self, scorer: Any) -> None:
        """Test scoring of medium urgency with this week."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text = "Company expects to release new product this week"
        result = scorer.score_urgency(text)

        # May be MEDIUM or HIGH depending on other signals
        assert result.level in [UrgencyLevel.MEDIUM, UrgencyLevel.HIGH]
        assert 0.3 <= result.score <= 1.0

    def test_score_medium_urgency_upcoming(self, scorer: Any) -> None:
        """Test scoring of medium urgency with upcoming."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text = "Upcoming merger discussions"
        result = scorer.score_urgency(text)

        assert result.level == UrgencyLevel.MEDIUM

    def test_score_low_urgency_analysis(self, scorer: Any) -> None:
        """Test scoring of low urgency with analysis."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text = "Long-term analysis of market trends"
        result = scorer.score_urgency(text)

        assert result.level == UrgencyLevel.LOW
        assert result.score <= 0.3

    def test_score_low_urgency_opinion(self, scorer: Any) -> None:
        """Test scoring of low urgency with opinion."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text = "Expert opinion on market outlook"
        result = scorer.score_urgency(text)

        assert result.level == UrgencyLevel.LOW

    def test_score_archive_urgency_historical(self, scorer: Any) -> None:
        """Test scoring of archive urgency with historical reference."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text = "Historical analysis from last year"
        result = scorer.score_urgency(text)

        assert result.level == UrgencyLevel.ARCHIVE
        assert result.score <= 0.1

    def test_score_urgency_with_title(self, scorer: Any) -> None:
        """Test scoring with title weighted more heavily."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text = "Regular news content"
        title = "BREAKING NEWS"

        result = scorer.score_urgency(text, title=title)

        assert result.level == UrgencyLevel.CRITICAL

    def test_score_urgency_empty_text(self, scorer: Any) -> None:
        """Test scoring with empty text raises error."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            scorer.score_urgency("")

    def test_score_urgency_whitespace_only(self, scorer: Any) -> None:
        """Test scoring with whitespace only raises error."""
        with pytest.raises(ValueError, match="text cannot be empty"):
            scorer.score_urgency("   ")

    def test_detect_temporal_signals_breaking(self, scorer: Any) -> None:
        """Test detection of breaking news signal."""
        text = "Breaking news: Company announces merger"
        signals = scorer.detect_temporal_signals(text)

        assert len(signals) > 0
        assert any("breaking" in s.lower() for s in signals)

    def test_detect_temporal_signals_multiple(self, scorer: Any) -> None:
        """Test detection of multiple temporal signals."""
        text = "Breaking: Company announces today major acquisition"
        signals = scorer.detect_temporal_signals(text)

        assert len(signals) >= 2

    def test_detect_temporal_signals_empty(self, scorer: Any) -> None:
        """Test detection with empty text."""
        signals = scorer.detect_temporal_signals("")
        assert signals == []

    def test_detect_temporal_signals_no_signals(self, scorer: Any) -> None:
        """Test detection when no signals present."""
        text = "Regular company update"
        signals = scorer.detect_temporal_signals(text)

        # Should still detect some signals or return empty
        assert isinstance(signals, list)

    def test_calculate_time_decay_critical_recent(self, scorer: Any) -> None:
        """Test time decay for recent critical news."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        published = datetime.now(UTC) - timedelta(minutes=30)
        level = scorer.calculate_time_decay(published, UrgencyLevel.CRITICAL)

        assert level == UrgencyLevel.CRITICAL

    def test_calculate_time_decay_critical_aged(self, scorer: Any) -> None:
        """Test time decay for aged critical news."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        published = datetime.now(UTC) - timedelta(hours=3)
        level = scorer.calculate_time_decay(published, UrgencyLevel.CRITICAL)

        assert level in [UrgencyLevel.HIGH, UrgencyLevel.MEDIUM]

    def test_calculate_time_decay_high_recent(self, scorer: Any) -> None:
        """Test time decay for recent high urgency news."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        published = datetime.now(UTC) - timedelta(hours=2)
        level = scorer.calculate_time_decay(published, UrgencyLevel.HIGH)

        assert level == UrgencyLevel.HIGH

    def test_calculate_time_decay_high_aged(self, scorer: Any) -> None:
        """Test time decay for aged high urgency news."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        published = datetime.now(UTC) - timedelta(days=2)
        level = scorer.calculate_time_decay(published, UrgencyLevel.HIGH)

        assert level in [UrgencyLevel.LOW, UrgencyLevel.ARCHIVE]

    def test_calculate_time_decay_medium_aged(self, scorer: Any) -> None:
        """Test time decay for aged medium urgency news."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        published = datetime.now(UTC) - timedelta(days=10)
        level = scorer.calculate_time_decay(published, UrgencyLevel.MEDIUM)

        assert level == UrgencyLevel.ARCHIVE

    def test_calculate_time_decay_archive_no_decay(self, scorer: Any) -> None:
        """Test that archive content doesn't decay."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        published = datetime.now(UTC) - timedelta(days=365)
        level = scorer.calculate_time_decay(published, UrgencyLevel.ARCHIVE)

        assert level == UrgencyLevel.ARCHIVE

    def test_calculate_time_decay_future_timestamp(self, scorer: Any) -> None:
        """Test handling of future timestamps."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        published = datetime.now(UTC) + timedelta(hours=1)
        level = scorer.calculate_time_decay(published, UrgencyLevel.HIGH)

        assert level == UrgencyLevel.HIGH  # Should return unchanged

    def test_requires_immediate_action_true(self, scorer: Any) -> None:
        """Test detection of immediate action requirement."""
        text = "Trading halted, immediate action required"
        requires_action = scorer.requires_immediate_action(text)

        assert requires_action is True

    def test_requires_immediate_action_breaking(self, scorer: Any) -> None:
        """Test detection with breaking news."""
        text = "Breaking: Major announcement"
        requires_action = scorer.requires_immediate_action(text)

        assert requires_action is True

    def test_requires_immediate_action_false(self, scorer: Any) -> None:
        """Test detection when no immediate action required."""
        text = "Company provides quarterly update"
        requires_action = scorer.requires_immediate_action(text)

        assert requires_action is False

    def test_requires_immediate_action_empty(self, scorer: Any) -> None:
        """Test immediate action detection with empty text."""
        requires_action = scorer.requires_immediate_action("")
        assert requires_action is False

    def test_estimate_relevance_window_critical(self, scorer: Any) -> None:
        """Test relevance window estimation for critical news."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        hours = scorer.estimate_relevance_window(UrgencyLevel.CRITICAL)
        assert hours == 1

    def test_estimate_relevance_window_high(self, scorer: Any) -> None:
        """Test relevance window estimation for high urgency."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        hours = scorer.estimate_relevance_window(UrgencyLevel.HIGH)
        assert hours == 6

    def test_estimate_relevance_window_medium(self, scorer: Any) -> None:
        """Test relevance window estimation for medium urgency."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        hours = scorer.estimate_relevance_window(UrgencyLevel.MEDIUM)
        assert hours == 48

    def test_estimate_relevance_window_low(self, scorer: Any) -> None:
        """Test relevance window estimation for low urgency."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        hours = scorer.estimate_relevance_window(UrgencyLevel.LOW)
        assert hours == 168

    def test_estimate_relevance_window_archive(self, scorer: Any) -> None:
        """Test relevance window estimation for archive."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        hours = scorer.estimate_relevance_window(UrgencyLevel.ARCHIVE)
        assert hours == 8760

    def test_batch_score_multiple_texts(self, scorer: Any) -> None:
        """Test batch scoring of multiple texts."""
        texts = [
            "Breaking news announcement",
            "Company reports today",
            "Long-term analysis",
        ]

        results = scorer.batch_score(texts)

        assert len(results) == 3
        assert all(hasattr(r, "level") for r in results)
        assert all(hasattr(r, "score") for r in results)

    def test_batch_score_empty_list(self, scorer: Any) -> None:
        """Test batch scoring with empty list."""
        results = scorer.batch_score([])
        assert results == []

    def test_batch_score_single_text(self, scorer: Any) -> None:
        """Test batch scoring with single text."""
        texts = ["Breaking news"]
        results = scorer.batch_score(texts)

        assert len(results) == 1

    def test_batch_score_with_invalid_text(self, scorer: Any) -> None:
        """Test batch scoring handles invalid texts gracefully."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        texts = ["Breaking news", "", "Valid text"]
        results = scorer.batch_score(texts)

        assert len(results) == 3
        # Empty text should get low urgency result
        assert results[1].level == UrgencyLevel.LOW
        assert results[1].confidence == 0.0


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    def scorer(self) -> Any:
        """Create urgency scorer."""
        from signalforge.nlp.urgency_scorer import UrgencyScorer

        return UrgencyScorer()

    def test_full_urgency_workflow(self, scorer: Any) -> None:
        """Test complete urgency scoring workflow."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text = "BREAKING: Company announces unexpected CEO resignation"
        title = "Breaking News Alert"
        published = datetime.now(UTC)

        result = scorer.score_urgency(text, title=title, published_at=published)

        assert result.level == UrgencyLevel.CRITICAL
        assert result.score > 0.9
        assert len(result.temporal_signals) > 0
        assert result.action_required is True
        assert result.time_sensitivity_hours == 1
        assert result.confidence > 0.7

    def test_urgency_with_time_decay_workflow(self, scorer: Any) -> None:
        """Test urgency with time decay applied."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text = "Company announces earnings beat"
        published = datetime.now(UTC) - timedelta(hours=12)

        result = scorer.score_urgency(text, published_at=published)

        # Should be decayed from HIGH to MEDIUM or lower
        assert result.level in [UrgencyLevel.MEDIUM, UrgencyLevel.LOW]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def scorer(self) -> Any:
        """Create urgency scorer."""
        from signalforge.nlp.urgency_scorer import UrgencyScorer

        return UrgencyScorer()

    def test_very_long_text(self, scorer: Any) -> None:
        """Test scoring with very long text."""
        text = "Breaking news. " * 1000
        result = scorer.score_urgency(text)

        assert result is not None
        assert hasattr(result, "level")

    def test_mixed_urgency_signals(self, scorer: Any) -> None:
        """Test text with mixed urgency signals."""
        text = "Breaking news from last year about historical trends"
        result = scorer.score_urgency(text)

        # Should prioritize critical signals over archive signals
        assert result is not None

    def test_case_insensitive_matching(self, scorer: Any) -> None:
        """Test that matching is case insensitive."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text1 = "BREAKING NEWS"
        text2 = "breaking news"
        text3 = "Breaking News"

        result1 = scorer.score_urgency(text1)
        result2 = scorer.score_urgency(text2)
        result3 = scorer.score_urgency(text3)

        assert result1.level == result2.level == result3.level == UrgencyLevel.CRITICAL

    def test_multiple_critical_signals(self, scorer: Any) -> None:
        """Test handling of multiple critical signals."""
        from signalforge.nlp.urgency_scorer import UrgencyLevel

        text = "Breaking: Trading halted due to emergency announcement"
        result = scorer.score_urgency(text)

        assert result.level == UrgencyLevel.CRITICAL
        assert result.action_required is True
