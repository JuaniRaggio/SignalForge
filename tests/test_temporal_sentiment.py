"""Tests for temporal sentiment analysis module.

This module tests the TemporalSentimentAnalyzer with mocked sentiment analyzer
to avoid requiring actual model downloads during testing.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    pass


@pytest.fixture
def mock_sentiment_analyzer() -> MagicMock:
    """Create a mock sentiment analyzer."""
    mock = MagicMock()

    # Mock analyze_batch to return predictable results
    def mock_analyze_batch(texts: list[str]) -> list[Any]:
        results = []
        for text in texts:
            # Simple heuristic for testing
            if "beat" in text.lower() or "strong" in text.lower() or "positive" in text.lower():
                result = MagicMock()
                result.label = "positive"
                result.confidence = 0.9
                result.scores = {"positive": 0.9, "negative": 0.05, "neutral": 0.05}
            elif "miss" in text.lower() or "weak" in text.lower() or "negative" in text.lower():
                result = MagicMock()
                result.label = "negative"
                result.confidence = 0.85
                result.scores = {"positive": 0.05, "negative": 0.85, "neutral": 0.1}
            else:
                result = MagicMock()
                result.label = "neutral"
                result.confidence = 0.7
                result.scores = {"positive": 0.3, "negative": 0.2, "neutral": 0.5}
            results.append(result)
        return results

    mock.analyze_batch = mock_analyze_batch
    return mock


@pytest.fixture
def temporal_analyzer(mock_sentiment_analyzer: MagicMock) -> Any:
    """Create TemporalSentimentAnalyzer with mocked dependencies."""
    from signalforge.nlp.temporal_sentiment import TemporalSentimentAnalyzer

    return TemporalSentimentAnalyzer(sentiment_analyzer=mock_sentiment_analyzer)


class TestSentimentTrend:
    """Tests for SentimentTrend enum."""

    def test_sentiment_trend_values(self) -> None:
        """Test sentiment trend enum values."""
        from signalforge.nlp.temporal_sentiment import SentimentTrend

        assert SentimentTrend.IMPROVING.value == "improving"
        assert SentimentTrend.STABLE.value == "stable"
        assert SentimentTrend.DETERIORATING.value == "deteriorating"


class TestTemporalSentimentResult:
    """Tests for TemporalSentimentResult dataclass."""

    def test_valid_temporal_sentiment_result(self) -> None:
        """Test creation of valid temporal sentiment result."""
        from signalforge.nlp.temporal_sentiment import (
            SentimentTrend,
            TemporalSentimentResult,
        )

        result = TemporalSentimentResult(
            current_score=0.8,
            baseline_score=0.5,
            delta=0.3,
            z_score=2.5,
            trend=SentimentTrend.IMPROVING,
            confidence=0.9,
            sample_size=10,
            time_period_days=30,
        )

        assert result.current_score == 0.8
        assert result.baseline_score == 0.5
        assert result.delta == 0.3
        assert result.z_score == 2.5
        assert result.trend == SentimentTrend.IMPROVING
        assert result.confidence == 0.9
        assert result.sample_size == 10
        assert result.time_period_days == 30

    def test_invalid_current_score(self) -> None:
        """Test validation of current_score range."""
        from signalforge.nlp.temporal_sentiment import (
            SentimentTrend,
            TemporalSentimentResult,
        )

        with pytest.raises(ValueError, match="current_score must be between"):
            TemporalSentimentResult(
                current_score=1.5,
                baseline_score=0.5,
                delta=0.3,
                z_score=2.5,
                trend=SentimentTrend.IMPROVING,
                confidence=0.9,
                sample_size=10,
                time_period_days=30,
            )

    def test_invalid_baseline_score(self) -> None:
        """Test validation of baseline_score range."""
        from signalforge.nlp.temporal_sentiment import (
            SentimentTrend,
            TemporalSentimentResult,
        )

        with pytest.raises(ValueError, match="baseline_score must be between"):
            TemporalSentimentResult(
                current_score=0.8,
                baseline_score=-1.5,
                delta=0.3,
                z_score=2.5,
                trend=SentimentTrend.IMPROVING,
                confidence=0.9,
                sample_size=10,
                time_period_days=30,
            )

    def test_invalid_confidence(self) -> None:
        """Test validation of confidence range."""
        from signalforge.nlp.temporal_sentiment import (
            SentimentTrend,
            TemporalSentimentResult,
        )

        with pytest.raises(ValueError, match="confidence must be between"):
            TemporalSentimentResult(
                current_score=0.8,
                baseline_score=0.5,
                delta=0.3,
                z_score=2.5,
                trend=SentimentTrend.IMPROVING,
                confidence=1.5,
                sample_size=10,
                time_period_days=30,
            )

    def test_invalid_sample_size(self) -> None:
        """Test validation of sample_size."""
        from signalforge.nlp.temporal_sentiment import (
            SentimentTrend,
            TemporalSentimentResult,
        )

        with pytest.raises(ValueError, match="sample_size must be non-negative"):
            TemporalSentimentResult(
                current_score=0.8,
                baseline_score=0.5,
                delta=0.3,
                z_score=2.5,
                trend=SentimentTrend.IMPROVING,
                confidence=0.9,
                sample_size=-1,
                time_period_days=30,
            )


class TestSectorSentiment:
    """Tests for SectorSentiment dataclass."""

    def test_valid_sector_sentiment(self) -> None:
        """Test creation of valid sector sentiment."""
        from signalforge.nlp.temporal_sentiment import SectorSentiment, SentimentTrend

        result = SectorSentiment(
            sector="Technology",
            sentiment_score=0.6,
            article_count=50,
            top_topics=["AI", "Cloud Computing"],
            trend=SentimentTrend.IMPROVING,
        )

        assert result.sector == "Technology"
        assert result.sentiment_score == 0.6
        assert result.article_count == 50
        assert result.top_topics == ["AI", "Cloud Computing"]
        assert result.trend == SentimentTrend.IMPROVING

    def test_invalid_sector_empty(self) -> None:
        """Test validation of empty sector."""
        from signalforge.nlp.temporal_sentiment import SectorSentiment, SentimentTrend

        with pytest.raises(ValueError, match="sector cannot be empty"):
            SectorSentiment(
                sector="",
                sentiment_score=0.6,
                article_count=50,
                top_topics=[],
                trend=SentimentTrend.IMPROVING,
            )

    def test_invalid_sentiment_score(self) -> None:
        """Test validation of sentiment_score range."""
        from signalforge.nlp.temporal_sentiment import SectorSentiment, SentimentTrend

        with pytest.raises(ValueError, match="sentiment_score must be between"):
            SectorSentiment(
                sector="Technology",
                sentiment_score=2.0,
                article_count=50,
                top_topics=[],
                trend=SentimentTrend.IMPROVING,
            )


class TestTemporalSentimentAnalyzer:
    """Tests for TemporalSentimentAnalyzer class."""

    def test_initialization(self, temporal_analyzer: Any) -> None:
        """Test analyzer initialization."""
        assert temporal_analyzer is not None
        assert temporal_analyzer._sentiment_analyzer is not None

    def test_normalize_sentiment_score_positive(self, temporal_analyzer: Any) -> None:
        """Test normalization of positive sentiment."""
        score = temporal_analyzer._normalize_sentiment_score("positive", 0.9)
        assert score == 0.9

    def test_normalize_sentiment_score_negative(self, temporal_analyzer: Any) -> None:
        """Test normalization of negative sentiment."""
        score = temporal_analyzer._normalize_sentiment_score("negative", 0.8)
        assert score == -0.8

    def test_normalize_sentiment_score_neutral(self, temporal_analyzer: Any) -> None:
        """Test normalization of neutral sentiment."""
        score = temporal_analyzer._normalize_sentiment_score("neutral", 0.7)
        assert score == 0.0

    def test_calculate_average_sentiment(self, temporal_analyzer: Any) -> None:
        """Test average sentiment calculation."""
        texts = ["Strong earnings beat", "Positive guidance", "Revenue growth"]
        avg_score, avg_confidence = temporal_analyzer._calculate_average_sentiment(texts)

        assert avg_score > 0  # Should be positive
        assert 0 <= avg_confidence <= 1

    def test_calculate_average_sentiment_empty(self, temporal_analyzer: Any) -> None:
        """Test average sentiment with empty texts."""
        avg_score, avg_confidence = temporal_analyzer._calculate_average_sentiment([])

        assert avg_score == 0.0
        assert avg_confidence == 0.0

    def test_calculate_z_score(self, temporal_analyzer: Any) -> None:
        """Test z-score calculation."""
        baseline_scores = [0.1, 0.2, 0.15, 0.18, 0.22]
        current_score = 0.8

        z_score = temporal_analyzer._calculate_z_score(current_score, baseline_scores)

        assert z_score > 0  # Should be positive since current is higher

    def test_calculate_z_score_insufficient_data(self, temporal_analyzer: Any) -> None:
        """Test z-score with insufficient baseline data."""
        z_score = temporal_analyzer._calculate_z_score(0.5, [0.1])
        assert z_score == 0.0

    def test_calculate_z_score_no_variance(self, temporal_analyzer: Any) -> None:
        """Test z-score with no variance in baseline."""
        baseline_scores = [0.5, 0.5, 0.5]
        z_score = temporal_analyzer._calculate_z_score(0.5, baseline_scores)
        assert z_score == 0.0

    def test_determine_trend_improving(self, temporal_analyzer: Any) -> None:
        """Test trend determination for improving sentiment."""
        from signalforge.nlp.temporal_sentiment import SentimentTrend

        trend = temporal_analyzer._determine_trend(delta=0.3, z_score=2.5, threshold_std=1.0)
        assert trend == SentimentTrend.IMPROVING

    def test_determine_trend_deteriorating(self, temporal_analyzer: Any) -> None:
        """Test trend determination for deteriorating sentiment."""
        from signalforge.nlp.temporal_sentiment import SentimentTrend

        trend = temporal_analyzer._determine_trend(delta=-0.3, z_score=-2.5, threshold_std=1.0)
        assert trend == SentimentTrend.DETERIORATING

    def test_determine_trend_stable(self, temporal_analyzer: Any) -> None:
        """Test trend determination for stable sentiment."""
        from signalforge.nlp.temporal_sentiment import SentimentTrend

        trend = temporal_analyzer._determine_trend(delta=0.1, z_score=0.5, threshold_std=1.0)
        assert trend == SentimentTrend.STABLE

    def test_analyze_sentiment_delta_improving(self, temporal_analyzer: Any) -> None:
        """Test sentiment delta analysis showing improvement."""
        from signalforge.nlp.temporal_sentiment import SentimentTrend

        current_texts = ["Strong earnings beat", "Positive guidance"]
        baseline_texts = [
            "Neutral performance",
            "In line with expectations",
            "Normal results",
            "Average quarter",
            "Standard update",
        ]

        result = temporal_analyzer.analyze_sentiment_delta("AAPL", current_texts, baseline_texts)

        assert result.current_score > result.baseline_score
        assert result.delta > 0
        # Trend may be IMPROVING or STABLE depending on z-score threshold
        assert result.trend in [SentimentTrend.IMPROVING, SentimentTrend.STABLE]
        assert result.sample_size == 2
        assert result.confidence > 0

    def test_analyze_sentiment_delta_no_baseline(self, temporal_analyzer: Any) -> None:
        """Test sentiment delta with no baseline."""
        current_texts = ["Strong earnings beat"]

        result = temporal_analyzer.analyze_sentiment_delta("AAPL", current_texts)

        assert result.baseline_score == 0.0
        assert result.delta == result.current_score
        assert result.sample_size == 1

    def test_analyze_sentiment_delta_empty_texts(self, temporal_analyzer: Any) -> None:
        """Test sentiment delta with empty texts raises error."""
        with pytest.raises(ValueError, match="texts cannot be empty"):
            temporal_analyzer.analyze_sentiment_delta("AAPL", [])

    def test_calculate_sector_baseline_placeholder(self, temporal_analyzer: Any) -> None:
        """Test sector baseline calculation (placeholder implementation)."""
        from signalforge.nlp.temporal_sentiment import SentimentTrend

        result = temporal_analyzer.calculate_sector_baseline("Technology", lookback_days=30)

        assert result.sector == "Technology"
        assert result.sentiment_score == 0.0
        assert result.article_count == 0
        assert result.trend == SentimentTrend.STABLE

    def test_detect_sentiment_shift_no_shift(self, temporal_analyzer: Any) -> None:
        """Test sentiment shift detection with no significant shift."""
        texts = ["Neutral news"]

        is_shift = temporal_analyzer.detect_sentiment_shift("AAPL", texts, threshold_std=2.0)

        assert isinstance(is_shift, bool)

    def test_detect_sentiment_shift_empty_texts(self, temporal_analyzer: Any) -> None:
        """Test sentiment shift detection with empty texts."""
        is_shift = temporal_analyzer.detect_sentiment_shift("AAPL", [], threshold_std=2.0)
        assert is_shift is False

    def test_compare_to_sector(self, temporal_analyzer: Any) -> None:
        """Test sector comparison."""
        texts = ["Strong performance"]

        result = temporal_analyzer.compare_to_sector("AAPL", "Technology", texts)

        assert "relative_sentiment" in result
        assert "percentile" in result
        assert "is_outlier" in result
        assert isinstance(result["relative_sentiment"], float)
        assert isinstance(result["percentile"], float)
        assert isinstance(result["is_outlier"], bool)

    def test_compare_to_sector_empty_texts(self, temporal_analyzer: Any) -> None:
        """Test sector comparison with empty texts raises error."""
        with pytest.raises(ValueError, match="texts cannot be empty"):
            temporal_analyzer.compare_to_sector("AAPL", "Technology", [])

    def test_get_sentiment_momentum(self, temporal_analyzer: Any) -> None:
        """Test sentiment momentum calculation."""
        now = datetime.now()
        texts_by_date = {
            now - timedelta(days=2): ["Weak performance"],
            now - timedelta(days=1): ["Neutral news"],
            now: ["Strong results"],
        }

        momentum = temporal_analyzer.get_sentiment_momentum("AAPL", texts_by_date)

        assert isinstance(momentum, float)
        assert momentum > 0  # Should show positive momentum

    def test_get_sentiment_momentum_insufficient_data(self, temporal_analyzer: Any) -> None:
        """Test momentum with insufficient data."""
        texts_by_date = {datetime.now(): ["Some news"]}

        momentum = temporal_analyzer.get_sentiment_momentum("AAPL", texts_by_date)
        assert momentum == 0.0

    def test_get_sentiment_momentum_empty(self, temporal_analyzer: Any) -> None:
        """Test momentum with empty data."""
        momentum = temporal_analyzer.get_sentiment_momentum("AAPL", {})
        assert momentum == 0.0

    def test_analyze_sentiment_by_source(self, temporal_analyzer: Any) -> None:
        """Test sentiment breakdown by source."""
        texts_with_sources = [
            ("Strong earnings", "Reuters"),
            ("Weak guidance", "Reuters"),
            ("Positive outlook", "Bloomberg"),
        ]

        result = temporal_analyzer.analyze_sentiment_by_source(texts_with_sources)

        assert "Reuters" in result
        assert "Bloomberg" in result
        assert isinstance(result["Reuters"], float)
        assert isinstance(result["Bloomberg"], float)

    def test_analyze_sentiment_by_source_empty(self, temporal_analyzer: Any) -> None:
        """Test source analysis with empty data."""
        result = temporal_analyzer.analyze_sentiment_by_source([])
        assert result == {}

    def test_analyze_sentiment_by_source_single_source(self, temporal_analyzer: Any) -> None:
        """Test source analysis with single source."""
        texts_with_sources = [
            ("News 1", "Reuters"),
            ("News 2", "Reuters"),
        ]

        result = temporal_analyzer.analyze_sentiment_by_source(texts_with_sources)

        assert len(result) == 1
        assert "Reuters" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_text_analysis(self, temporal_analyzer: Any) -> None:
        """Test analysis with single text."""
        result = temporal_analyzer.analyze_sentiment_delta("AAPL", ["Single text"])

        assert result.sample_size == 1
        assert result.confidence > 0

    def test_very_long_text_list(self, temporal_analyzer: Any) -> None:
        """Test analysis with many texts."""
        texts = ["Test text"] * 100

        result = temporal_analyzer.analyze_sentiment_delta("AAPL", texts)

        assert result.sample_size == 100

    def test_mixed_sentiment_texts(self, temporal_analyzer: Any) -> None:
        """Test analysis with mixed sentiment."""
        texts = [
            "Strong positive results",
            "Weak negative performance",
            "Neutral outlook",
        ]

        result = temporal_analyzer.analyze_sentiment_delta("AAPL", texts)

        assert -1.0 <= result.current_score <= 1.0
        assert result.sample_size == 3
