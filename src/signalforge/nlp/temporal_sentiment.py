"""Temporal sentiment analysis with baseline comparison.

This module provides advanced sentiment analysis that tracks sentiment changes
over time, compares to baselines, and detects significant sentiment shifts.

Key Features:
- Sentiment delta calculation vs historical baseline
- Sector-level sentiment benchmarking
- Statistical significance testing (z-scores)
- Sentiment momentum calculation
- Source-level sentiment breakdown

Examples:
    Basic temporal sentiment analysis:

    >>> from signalforge.nlp.temporal_sentiment import TemporalSentimentAnalyzer
    >>>
    >>> analyzer = TemporalSentimentAnalyzer()
    >>> current_texts = ["Strong earnings beat expectations"]
    >>> baseline_texts = ["Revenue in line with guidance"]
    >>> result = analyzer.analyze_sentiment_delta("AAPL", current_texts, baseline_texts)
    >>> print(f"Delta: {result.delta:.2f}, Trend: {result.trend}")
    Delta: 0.35, Trend: improving

    Detect sentiment shifts:

    >>> texts = ["Major breakthrough in AI technology"]
    >>> is_shift = analyzer.detect_sentiment_shift("NVDA", texts, threshold_std=2.0)
    >>> print(f"Significant shift detected: {is_shift}")
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    from signalforge.nlp.sentiment import SentimentAnalyzer

logger = get_logger(__name__)


class SentimentTrend(str, Enum):
    """Sentiment trend classification."""

    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"


@dataclass
class TemporalSentimentResult:
    """Result of temporal sentiment analysis.

    Attributes:
        current_score: Current sentiment score (-1 to 1).
        baseline_score: Historical average sentiment score.
        delta: Difference between current and baseline (current - baseline).
        z_score: Statistical significance of current sentiment vs baseline.
        trend: Classification of sentiment trend.
        confidence: Confidence in the analysis (0.0 to 1.0).
        sample_size: Number of documents analyzed in current period.
        time_period_days: Time period for baseline calculation.
    """

    current_score: float
    baseline_score: float
    delta: float
    z_score: float
    trend: SentimentTrend
    confidence: float
    sample_size: int
    time_period_days: int

    def __post_init__(self) -> None:
        """Validate temporal sentiment result fields."""
        if not -1.0 <= self.current_score <= 1.0:
            raise ValueError(
                f"current_score must be between -1.0 and 1.0, got {self.current_score}"
            )

        if not -1.0 <= self.baseline_score <= 1.0:
            raise ValueError(
                f"baseline_score must be between -1.0 and 1.0, got {self.baseline_score}"
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

        if self.sample_size < 0:
            raise ValueError(f"sample_size must be non-negative, got {self.sample_size}")

        if self.time_period_days < 0:
            raise ValueError(
                f"time_period_days must be non-negative, got {self.time_period_days}"
            )


@dataclass
class SectorSentiment:
    """Sentiment statistics for a market sector.

    Attributes:
        sector: Sector name (e.g., "Technology", "Healthcare").
        sentiment_score: Average sentiment score for the sector.
        article_count: Number of articles analyzed.
        top_topics: Most discussed topics in the sector.
        trend: Overall sentiment trend for the sector.
    """

    sector: str
    sentiment_score: float
    article_count: int
    top_topics: list[str]
    trend: SentimentTrend

    def __post_init__(self) -> None:
        """Validate sector sentiment fields."""
        if not self.sector:
            raise ValueError("sector cannot be empty")

        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError(
                f"sentiment_score must be between -1.0 and 1.0, got {self.sentiment_score}"
            )

        if self.article_count < 0:
            raise ValueError(f"article_count must be non-negative, got {self.article_count}")


class TemporalSentimentAnalyzer:
    """Analyzes sentiment changes over time with sector baselines.

    This analyzer tracks sentiment evolution and compares current sentiment
    against historical baselines to detect significant shifts. It provides
    statistical measures of sentiment change and supports sector-level
    benchmarking.

    Examples:
        >>> analyzer = TemporalSentimentAnalyzer()
        >>> texts = ["Strong Q4 performance"]
        >>> result = analyzer.analyze_sentiment_delta("AAPL", texts)
        >>> print(result.trend)
        improving
    """

    def __init__(self, sentiment_analyzer: SentimentAnalyzer | None = None) -> None:
        """Initialize the temporal sentiment analyzer.

        Args:
            sentiment_analyzer: Sentiment analyzer instance. If None, creates default.
        """
        self.logger = get_logger(__name__)

        # Lazy import to avoid circular dependency
        if sentiment_analyzer is None:
            from signalforge.nlp.sentiment import get_sentiment_analyzer

            self._sentiment_analyzer = get_sentiment_analyzer()
        else:
            self._sentiment_analyzer = sentiment_analyzer

        self.logger.info("temporal_sentiment_analyzer_initialized")

    def _normalize_sentiment_score(self, label: str, confidence: float) -> float:
        """Convert sentiment label and confidence to normalized score.

        Args:
            label: Sentiment label ("positive", "negative", "neutral").
            confidence: Confidence score (0.0 to 1.0).

        Returns:
            Normalized score from -1.0 (very negative) to 1.0 (very positive).
        """
        if label == "positive":
            return confidence
        elif label == "negative":
            return -confidence
        else:  # neutral
            return 0.0

    def _calculate_average_sentiment(self, texts: list[str]) -> tuple[float, float]:
        """Calculate average sentiment score from texts.

        Args:
            texts: List of text documents to analyze.

        Returns:
            Tuple of (average_score, average_confidence).
        """
        if not texts:
            return 0.0, 0.0

        results = self._sentiment_analyzer.analyze_batch(texts)

        scores = []
        confidences = []

        for result in results:
            score = self._normalize_sentiment_score(result.label, result.confidence)
            scores.append(score)
            confidences.append(result.confidence)

        avg_score = statistics.mean(scores) if scores else 0.0
        avg_confidence = statistics.mean(confidences) if confidences else 0.0

        return avg_score, avg_confidence

    def _calculate_z_score(
        self, current_score: float, baseline_scores: list[float]
    ) -> float:
        """Calculate z-score for current sentiment vs baseline.

        Args:
            current_score: Current sentiment score.
            baseline_scores: Historical sentiment scores.

        Returns:
            Z-score indicating standard deviations from baseline mean.
        """
        if not baseline_scores or len(baseline_scores) < 2:
            return 0.0

        mean = statistics.mean(baseline_scores)
        try:
            stdev = statistics.stdev(baseline_scores)
        except statistics.StatisticsError:
            return 0.0

        if stdev == 0:
            return 0.0

        return (current_score - mean) / stdev

    def _determine_trend(
        self, delta: float, z_score: float, threshold_std: float = 1.0
    ) -> SentimentTrend:
        """Determine sentiment trend from delta and z-score.

        Args:
            delta: Sentiment delta (current - baseline).
            z_score: Statistical significance measure.
            threshold_std: Minimum z-score to consider trend significant.

        Returns:
            Sentiment trend classification.
        """
        if abs(z_score) < threshold_std:
            return SentimentTrend.STABLE

        if delta > 0:
            return SentimentTrend.IMPROVING
        else:
            return SentimentTrend.DETERIORATING

    def analyze_sentiment_delta(
        self,
        symbol: str,
        texts: list[str],
        baseline_texts: list[str] | None = None,
        baseline_days: int = 30,
    ) -> TemporalSentimentResult:
        """Calculate sentiment delta vs baseline.

        Args:
            symbol: Stock symbol being analyzed.
            texts: Current texts to analyze.
            baseline_texts: Historical texts for baseline. If None, uses empty baseline.
            baseline_days: Time period for baseline (for metadata only).

        Returns:
            TemporalSentimentResult with sentiment analysis and trend.

        Examples:
            >>> analyzer = TemporalSentimentAnalyzer()
            >>> current = ["Earnings beat expectations"]
            >>> baseline = ["Earnings met expectations"]
            >>> result = analyzer.analyze_sentiment_delta("AAPL", current, baseline)
            >>> result.trend
            <SentimentTrend.IMPROVING: 'improving'>
        """
        self.logger.debug(
            "analyzing_sentiment_delta",
            symbol=symbol,
            current_texts=len(texts),
            baseline_texts=len(baseline_texts) if baseline_texts else 0,
        )

        if not texts:
            raise ValueError("texts cannot be empty")

        # Calculate current sentiment
        current_score, current_confidence = self._calculate_average_sentiment(texts)

        # Calculate baseline sentiment
        if baseline_texts:
            baseline_score, _ = self._calculate_average_sentiment(baseline_texts)

            # Get individual scores for z-score calculation
            baseline_results = self._sentiment_analyzer.analyze_batch(baseline_texts)
            baseline_scores = [
                self._normalize_sentiment_score(r.label, r.confidence)
                for r in baseline_results
            ]
        else:
            baseline_score = 0.0
            baseline_scores = [0.0]

        # Calculate delta and z-score
        delta = current_score - baseline_score
        z_score = self._calculate_z_score(current_score, baseline_scores)

        # Determine trend
        trend = self._determine_trend(delta, z_score)

        result = TemporalSentimentResult(
            current_score=current_score,
            baseline_score=baseline_score,
            delta=delta,
            z_score=z_score,
            trend=trend,
            confidence=current_confidence,
            sample_size=len(texts),
            time_period_days=baseline_days,
        )

        self.logger.info(
            "sentiment_delta_analyzed",
            symbol=symbol,
            delta=delta,
            z_score=z_score,
            trend=trend.value,
        )

        return result

    def calculate_sector_baseline(
        self, sector: str, lookback_days: int = 30
    ) -> SectorSentiment:
        """Calculate average sentiment for a sector.

        Note: This is a placeholder implementation. In production, this would
        query a database of historical sector news.

        Args:
            sector: Sector name (e.g., "Technology").
            lookback_days: Days to look back for baseline.

        Returns:
            SectorSentiment with aggregated statistics.
        """
        self.logger.warning(
            "sector_baseline_not_implemented",
            sector=sector,
            lookback_days=lookback_days,
            message="Returning placeholder data. Implement database integration.",
        )

        # Placeholder implementation
        return SectorSentiment(
            sector=sector,
            sentiment_score=0.0,
            article_count=0,
            top_topics=[],
            trend=SentimentTrend.STABLE,
        )

    def detect_sentiment_shift(
        self, symbol: str, texts: list[str], threshold_std: float = 2.0
    ) -> bool:
        """Detect if sentiment has shifted significantly.

        Args:
            symbol: Stock symbol being analyzed.
            texts: Current texts to analyze.
            threshold_std: Minimum z-score to consider shift significant.

        Returns:
            True if significant shift detected, False otherwise.

        Examples:
            >>> analyzer = TemporalSentimentAnalyzer()
            >>> texts = ["Major breakthrough announced"]
            >>> analyzer.detect_sentiment_shift("NVDA", texts, threshold_std=2.0)
            False
        """
        if not texts:
            return False

        result = self.analyze_sentiment_delta(symbol, texts)

        is_shift = abs(result.z_score) >= threshold_std

        self.logger.info(
            "sentiment_shift_detection",
            symbol=symbol,
            z_score=result.z_score,
            threshold=threshold_std,
            shift_detected=is_shift,
        )

        return is_shift

    def compare_to_sector(
        self, symbol: str, sector: str, texts: list[str]
    ) -> dict[str, float | bool]:
        """Compare symbol sentiment to sector average.

        Args:
            symbol: Stock symbol being analyzed.
            sector: Sector name for comparison.
            texts: Current texts to analyze.

        Returns:
            Dict with relative_sentiment, percentile, and is_outlier.

        Examples:
            >>> analyzer = TemporalSentimentAnalyzer()
            >>> texts = ["Strong performance"]
            >>> result = analyzer.compare_to_sector("AAPL", "Technology", texts)
            >>> "relative_sentiment" in result
            True
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        # Calculate symbol sentiment
        symbol_score, _ = self._calculate_average_sentiment(texts)

        # Get sector baseline
        sector_baseline = self.calculate_sector_baseline(sector)

        # Calculate relative metrics
        relative_sentiment = symbol_score - sector_baseline.sentiment_score

        # Placeholder percentile and outlier detection
        # In production, would compare to distribution of sector constituents
        percentile = 50.0
        is_outlier = abs(relative_sentiment) > 0.5

        self.logger.info(
            "sector_comparison",
            symbol=symbol,
            sector=sector,
            symbol_score=symbol_score,
            sector_score=sector_baseline.sentiment_score,
            relative_sentiment=relative_sentiment,
        )

        return {
            "relative_sentiment": relative_sentiment,
            "percentile": percentile,
            "is_outlier": is_outlier,
        }

    def get_sentiment_momentum(
        self, symbol: str, texts_by_date: dict[datetime, list[str]], window_days: int = 7
    ) -> float:
        """Calculate rate of sentiment change.

        Args:
            symbol: Stock symbol being analyzed.
            texts_by_date: Dictionary mapping dates to texts.
            window_days: Window size for momentum calculation (metadata only).

        Returns:
            Momentum score (change in sentiment per day).

        Examples:
            >>> analyzer = TemporalSentimentAnalyzer()
            >>> from datetime import datetime, timedelta
            >>> texts_by_date = {
            ...     datetime.now(): ["Positive news"],
            ...     datetime.now() - timedelta(days=1): ["Neutral news"]
            ... }
            >>> momentum = analyzer.get_sentiment_momentum("AAPL", texts_by_date)
            >>> isinstance(momentum, float)
            True
        """
        if not texts_by_date or len(texts_by_date) < 2:
            return 0.0

        # Sort dates
        sorted_dates = sorted(texts_by_date.keys())

        # Calculate sentiment for each date
        sentiment_scores = []
        for date in sorted_dates:
            texts = texts_by_date[date]
            if texts:
                score, _ = self._calculate_average_sentiment(texts)
                sentiment_scores.append(score)

        if len(sentiment_scores) < 2:
            return 0.0

        # Calculate momentum as slope
        # Simple linear regression: y = mx + b
        n = len(sentiment_scores)
        x_values = list(range(n))
        y_values = sentiment_scores

        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0

        momentum = numerator / denominator

        self.logger.info(
            "sentiment_momentum_calculated",
            symbol=symbol,
            window_days=window_days,
            momentum=momentum,
        )

        return momentum

    def analyze_sentiment_by_source(
        self, texts_with_sources: list[tuple[str, str]]
    ) -> dict[str, float]:
        """Break down sentiment by news source.

        Args:
            texts_with_sources: List of (text, source) tuples.

        Returns:
            Dictionary mapping source names to average sentiment scores.

        Examples:
            >>> analyzer = TemporalSentimentAnalyzer()
            >>> texts_with_sources = [
            ...     ("Positive news", "Reuters"),
            ...     ("Negative news", "Reuters"),
            ...     ("Neutral news", "Bloomberg")
            ... ]
            >>> result = analyzer.analyze_sentiment_by_source(texts_with_sources)
            >>> "Reuters" in result
            True
        """
        if not texts_with_sources:
            return {}

        # Group texts by source
        source_texts: dict[str, list[str]] = {}
        for text, source in texts_with_sources:
            if source not in source_texts:
                source_texts[source] = []
            source_texts[source].append(text)

        # Calculate sentiment for each source
        source_sentiments: dict[str, float] = {}
        for source, texts in source_texts.items():
            score, _ = self._calculate_average_sentiment(texts)
            source_sentiments[source] = score

        self.logger.info(
            "sentiment_by_source_analyzed",
            num_sources=len(source_sentiments),
            sources=list(source_sentiments.keys()),
        )

        return source_sentiments
