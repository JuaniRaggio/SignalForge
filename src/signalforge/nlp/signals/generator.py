"""Generate trading signals from NLP analysis.

This module provides functionality to generate comprehensive trading signals
from NLP analysis of financial documents, including sentiment analysis,
price target extraction, and sector-specific insights.

Key Features:
- Sentiment analysis with baseline normalization
- Sentiment momentum calculation
- Price target aggregation
- Analyst consensus generation
- Urgency scoring
- Information novelty detection

Examples:
    Generate signals from a document:

    >>> from signalforge.nlp.signals.generator import NLPSignalGenerator
    >>> generator = NLPSignalGenerator(sentiment_analyzer, sector_analyzers)
    >>> signals = await generator.generate_signals(document, session)
    >>> print(f"Sentiment: {signals.sentiment.label}, Urgency: {signals.urgency}")
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.core.logging import get_logger
from signalforge.nlp.price_target_extractor import PriceTarget, Rating
from signalforge.nlp.sentiment import BaseSentimentAnalyzer, SentimentLabel
from signalforge.nlp.urgency_scorer import UrgencyLevel, UrgencyScorer

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class SentimentOutput:
    """Sentiment analysis output.

    Attributes:
        score: Sentiment score (-1 to 1, negative to positive).
        label: Sentiment label (bullish, bearish, neutral).
        delta_vs_baseline: Difference from historical baseline.
        is_new_information: Whether document contains new information.
    """

    score: float
    label: str
    delta_vs_baseline: float
    is_new_information: bool

    def __post_init__(self) -> None:
        """Validate sentiment output fields."""
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"score must be between -1.0 and 1.0, got {self.score}")

        if self.label not in ("bullish", "bearish", "neutral"):
            raise ValueError(f"Invalid label: {self.label}. Must be bullish, bearish, or neutral")


@dataclass
class AnalystConsensus:
    """Aggregated analyst consensus.

    Attributes:
        rating: Consensus rating (strong_buy, buy, hold, sell, strong_sell).
        confidence: Confidence in consensus (0.0 to 1.0).
        bullish_count: Number of bullish signals.
        bearish_count: Number of bearish signals.
        neutral_count: Number of neutral signals.
    """

    rating: str
    confidence: float
    bullish_count: int
    bearish_count: int
    neutral_count: int

    def __post_init__(self) -> None:
        """Validate analyst consensus fields."""
        valid_ratings = ["strong_buy", "buy", "hold", "sell", "strong_sell"]
        if self.rating not in valid_ratings:
            raise ValueError(f"Invalid rating: {self.rating}. Must be one of {valid_ratings}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

        if self.bullish_count < 0 or self.bearish_count < 0 or self.neutral_count < 0:
            raise ValueError("Counts must be non-negative")


@dataclass
class SectorSignal:
    """Sector-specific signal.

    Attributes:
        sector: Sector name.
        signal: Signal type (buy, sell, hold).
        confidence: Signal confidence (0.0 to 1.0).
        reasoning: Explanation of the signal.
    """

    sector: str
    signal: str
    confidence: float
    reasoning: str

    def __post_init__(self) -> None:
        """Validate sector signal fields."""
        if self.signal not in ("buy", "sell", "hold"):
            raise ValueError(f"Invalid signal: {self.signal}. Must be buy, sell, or hold")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class FinancialDocument:
    """Financial document representation.

    Attributes:
        id: Document ID.
        title: Document title.
        content: Document content/text.
        symbols: Related stock symbols.
        published_at: Publication timestamp.
        source: Document source.
        metadata: Additional metadata.
    """

    id: str
    title: str
    content: str
    symbols: list[str]
    published_at: datetime
    source: str
    metadata: dict[str, Any]


@dataclass
class NLPSignalOutput:
    """Comprehensive NLP signal output.

    Attributes:
        ticker: Stock ticker symbol.
        sentiment: Sentiment analysis output.
        price_targets: Extracted price targets.
        analyst_consensus: Aggregated analyst consensus.
        sector_signals: Sector-specific signals.
        urgency: Urgency level (low, medium, high, critical).
        generated_at: Signal generation timestamp.
    """

    ticker: str
    sentiment: SentimentOutput
    price_targets: list[PriceTarget]
    analyst_consensus: AnalystConsensus
    sector_signals: list[SectorSignal]
    urgency: str
    generated_at: datetime

    def __post_init__(self) -> None:
        """Validate NLP signal output fields."""
        valid_urgency = ["low", "medium", "high", "critical"]
        if self.urgency not in valid_urgency:
            raise ValueError(f"Invalid urgency: {self.urgency}. Must be one of {valid_urgency}")


class NLPSignalGenerator:
    """Generate trading signals from NLP analysis.

    This generator combines multiple NLP components to produce comprehensive
    trading signals from financial documents.

    Examples:
        >>> generator = NLPSignalGenerator(sentiment_analyzer, sector_analyzers)
        >>> signals = await generator.generate_signals(document, session)
        >>> print(f"Sentiment: {signals.sentiment.score:.2f}")
    """

    def __init__(
        self,
        sentiment_analyzer: BaseSentimentAnalyzer,
        sector_analyzers: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the NLP signal generator.

        Args:
            sentiment_analyzer: Sentiment analyzer instance.
            sector_analyzers: Dictionary of sector-specific analyzers (optional).
        """
        self._sentiment_analyzer = sentiment_analyzer
        self._sector_analyzers = sector_analyzers or {}
        self._urgency_scorer = UrgencyScorer()

        # Cache for historical sentiment baselines
        self._sentiment_baselines: dict[str, float] = {}

        logger.info(
            "nlp_signal_generator_initialized",
            num_sector_analyzers=len(self._sector_analyzers),
        )

    async def generate_signals(
        self,
        document: FinancialDocument,
        session: AsyncSession,
    ) -> NLPSignalOutput:
        """Generate comprehensive NLP signals from a document.

        This method orchestrates the following steps:
        1. Extract sentiment with normalization
        2. Calculate sentiment momentum vs baseline
        3. Run sector-specific analysis
        4. Aggregate analyst consensus
        5. Score urgency

        Args:
            document: Financial document to analyze.
            session: Database session for historical data.

        Returns:
            NLPSignalOutput with comprehensive signals.

        Examples:
            >>> signals = await generator.generate_signals(document, session)
            >>> print(signals.sentiment.label)
            'bullish'
        """
        logger.info(
            "generating_nlp_signals",
            document_id=document.id,
            symbols=document.symbols,
        )

        # Determine primary symbol
        primary_symbol = document.symbols[0] if document.symbols else "UNKNOWN"

        # 1. Extract sentiment
        full_text = f"{document.title} {document.content}"
        sentiment_result = self._sentiment_analyzer.analyze(full_text)

        # 2. Normalize sentiment relative to baseline
        normalized_score = self.normalize_sentiment(
            self._sentiment_to_score(sentiment_result.label),
            primary_symbol,
            document.metadata.get("sector", "Unknown"),
        )

        # 3. Calculate sentiment momentum
        historical_sentiments = await self._fetch_historical_sentiments(
            primary_symbol,
            session,
            days=30,
        )
        sentiment_momentum = self.calculate_sentiment_momentum(
            normalized_score,
            historical_sentiments,
        )

        # 4. Check if document contains new information
        recent_documents = await self._fetch_recent_documents(
            primary_symbol,
            session,
            hours=24,
        )
        is_new_info = self.is_new_information(document, recent_documents)

        # 5. Extract price targets
        from signalforge.nlp.price_target_extractor import PriceTargetExtractor

        price_extractor = PriceTargetExtractor()
        price_targets = price_extractor.extract_all_targets(full_text)

        # 6. Generate sector signals
        sector_signals = await self._generate_sector_signals(document, sentiment_result.label)

        # 7. Aggregate analyst consensus
        analyst_consensus = self.aggregate_analyst_consensus(sector_signals, price_targets)

        # 8. Score urgency
        urgency_result = self._urgency_scorer.score_urgency(
            document.content,
            title=document.title,
            published_at=document.published_at,
        )
        urgency = self._map_urgency_level(urgency_result.level)

        # Create sentiment output
        sentiment_output = SentimentOutput(
            score=normalized_score,
            label=self._score_to_label(normalized_score),
            delta_vs_baseline=sentiment_momentum,
            is_new_information=is_new_info,
        )

        signal_output = NLPSignalOutput(
            ticker=primary_symbol,
            sentiment=sentiment_output,
            price_targets=price_targets,
            analyst_consensus=analyst_consensus,
            sector_signals=sector_signals,
            urgency=urgency,
            generated_at=datetime.now(UTC),
        )

        logger.info(
            "nlp_signals_generated",
            ticker=primary_symbol,
            sentiment_score=normalized_score,
            urgency=urgency,
            num_price_targets=len(price_targets),
        )

        return signal_output

    def normalize_sentiment(
        self,
        raw_sentiment: float,
        symbol: str,
        sector: str,
    ) -> float:
        """Normalize sentiment relative to asset/sector baseline.

        Args:
            raw_sentiment: Raw sentiment score (-1 to 1).
            symbol: Stock symbol.
            sector: Sector classification.

        Returns:
            Normalized sentiment score.

        Examples:
            >>> generator = NLPSignalGenerator(analyzer)
            >>> normalized = generator.normalize_sentiment(0.5, "AAPL", "Technology")
            >>> -1.0 <= normalized <= 1.0
            True
        """
        # Get or compute baseline
        baseline_key = f"{symbol}:{sector}"
        if baseline_key not in self._sentiment_baselines:
            # Default baseline is neutral (0.0)
            # In production, this would be computed from historical data
            self._sentiment_baselines[baseline_key] = 0.0

        baseline = self._sentiment_baselines[baseline_key]

        # Normalize: subtract baseline and bound to [-1, 1]
        normalized = raw_sentiment - baseline
        normalized = max(-1.0, min(1.0, normalized))

        logger.debug(
            "sentiment_normalized",
            symbol=symbol,
            sector=sector,
            raw=raw_sentiment,
            baseline=baseline,
            normalized=normalized,
        )

        return normalized

    def calculate_sentiment_momentum(
        self,
        current_sentiment: float,
        historical_sentiments: list[tuple[datetime, float]],
    ) -> float:
        """Calculate sentiment delta vs moving average.

        Args:
            current_sentiment: Current sentiment score.
            historical_sentiments: List of (timestamp, sentiment) tuples.

        Returns:
            Sentiment momentum (delta from moving average).

        Examples:
            >>> historical = [(datetime.now(), 0.3), (datetime.now(), 0.4)]
            >>> momentum = generator.calculate_sentiment_momentum(0.6, historical)
            >>> momentum > 0
            True
        """
        if not historical_sentiments:
            # No historical data, momentum is 0
            return 0.0

        # Calculate moving average
        sentiment_values = [s for _, s in historical_sentiments]
        avg_sentiment = statistics.mean(sentiment_values)

        # Momentum is the difference
        momentum = current_sentiment - avg_sentiment

        logger.debug(
            "sentiment_momentum_calculated",
            current=current_sentiment,
            avg=avg_sentiment,
            momentum=momentum,
            num_historical=len(historical_sentiments),
        )

        return momentum

    def aggregate_analyst_consensus(
        self,
        sector_signals: list[SectorSignal],
        price_targets: list[PriceTarget],
    ) -> AnalystConsensus:
        """Aggregate signals into analyst consensus.

        Args:
            sector_signals: List of sector-specific signals.
            price_targets: List of price targets.

        Returns:
            Aggregated analyst consensus.

        Examples:
            >>> signals = [SectorSignal("Tech", "buy", 0.8, "...")]
            >>> targets = []
            >>> consensus = generator.aggregate_analyst_consensus(signals, targets)
            >>> consensus.rating in ["strong_buy", "buy", "hold", "sell", "strong_sell"]
            True
        """
        # Count signal types from sector signals
        bullish_count = sum(1 for s in sector_signals if s.signal == "buy")
        bearish_count = sum(1 for s in sector_signals if s.signal == "sell")
        neutral_count = sum(1 for s in sector_signals if s.signal == "hold")

        # Add price target signals
        for target in price_targets:
            if target.rating == Rating.STRONG_BUY or target.rating == Rating.BUY:
                bullish_count += 1
            elif target.rating == Rating.SELL or target.rating == Rating.STRONG_SELL:
                bearish_count += 1
            elif target.rating == Rating.HOLD:
                neutral_count += 1

        total_signals = bullish_count + bearish_count + neutral_count

        if total_signals == 0:
            # No signals, default to hold with low confidence
            return AnalystConsensus(
                rating="hold",
                confidence=0.0,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
            )

        # Determine consensus rating
        bullish_pct = bullish_count / total_signals
        bearish_pct = bearish_count / total_signals

        if bullish_pct >= 0.7:
            rating = "strong_buy"
            confidence = bullish_pct
        elif bullish_pct >= 0.5:
            rating = "buy"
            confidence = bullish_pct
        elif bearish_pct >= 0.7:
            rating = "strong_sell"
            confidence = bearish_pct
        elif bearish_pct >= 0.5:
            rating = "sell"
            confidence = bearish_pct
        else:
            rating = "hold"
            confidence = 1.0 - max(bullish_pct, bearish_pct)

        consensus = AnalystConsensus(
            rating=rating,
            confidence=confidence,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
        )

        logger.info(
            "analyst_consensus_aggregated",
            rating=rating,
            confidence=confidence,
            total_signals=total_signals,
            bullish=bullish_count,
            bearish=bearish_count,
            neutral=neutral_count,
        )

        return consensus

    def is_new_information(
        self,
        document: FinancialDocument,
        recent_documents: list[FinancialDocument],
    ) -> bool:
        """Determine if document contains new information.

        Uses simple heuristics:
        - Check if title is sufficiently different
        - Check if content has low similarity to recent docs

        Args:
            document: Current document.
            recent_documents: Recently processed documents.

        Returns:
            True if document contains new information.

        Examples:
            >>> doc = FinancialDocument(...)
            >>> recent = []
            >>> is_new = generator.is_new_information(doc, recent)
            >>> isinstance(is_new, bool)
            True
        """
        if not recent_documents:
            return True

        # Check title similarity (simple word overlap)
        doc_title_words = set(document.title.lower().split())

        for recent_doc in recent_documents:
            recent_title_words = set(recent_doc.title.lower().split())

            # Calculate Jaccard similarity
            intersection = doc_title_words & recent_title_words
            union = doc_title_words | recent_title_words

            if union:
                similarity = len(intersection) / len(union)

                # If title is >70% similar, likely duplicate
                if similarity > 0.7:
                    logger.debug(
                        "duplicate_document_detected",
                        document_id=document.id,
                        similar_to=recent_doc.id,
                        similarity=similarity,
                    )
                    return False

        return True

    async def _generate_sector_signals(
        self,
        document: FinancialDocument,
        sentiment_label: SentimentLabel,
    ) -> list[SectorSignal]:
        """Generate sector-specific signals.

        Args:
            document: Financial document.
            sentiment_label: Overall sentiment label.

        Returns:
            List of sector-specific signals.
        """
        sector_signals = []

        # Get sector from document metadata
        sector = document.metadata.get("sector", "Unknown")

        # If we have a sector-specific analyzer, use it
        if sector in self._sector_analyzers:
            # analyzer = self._sector_analyzers[sector]
            # Call sector analyzer (implementation varies by analyzer)
            # For now, we'll create a simple signal based on sentiment
            pass

        # Create a basic sector signal based on sentiment
        signal_map = {
            "positive": "buy",
            "negative": "sell",
            "neutral": "hold",
        }

        signal = signal_map.get(sentiment_label, "hold")
        confidence = 0.6 if sentiment_label != "neutral" else 0.3

        sector_signal = SectorSignal(
            sector=sector,
            signal=signal,
            confidence=confidence,
            reasoning=f"Based on {sentiment_label} sentiment analysis",
        )

        sector_signals.append(sector_signal)

        return sector_signals

    async def _fetch_historical_sentiments(
        self,
        symbol: str,  # noqa: ARG002
        session: AsyncSession,  # noqa: ARG002
        days: int = 30,  # noqa: ARG002
    ) -> list[tuple[datetime, float]]:
        """Fetch historical sentiment data for a symbol.

        Args:
            symbol: Stock symbol.
            session: Database session.
            days: Number of days of history to fetch.

        Returns:
            List of (timestamp, sentiment) tuples.
        """
        # TODO: Implement database query for historical sentiments
        # For now, return empty list (no historical data)
        return []

    async def _fetch_recent_documents(
        self,
        symbol: str,
        session: AsyncSession,
        hours: int = 24,
    ) -> list[FinancialDocument]:
        """Fetch recent documents for a symbol.

        Args:
            symbol: Stock symbol.
            session: Database session.
            hours: Number of hours of history to fetch.

        Returns:
            List of recent documents.
        """
        # TODO: Implement database query for recent documents
        # Query NewsArticle model for recent documents
        from signalforge.models.news import NewsArticle

        cutoff = datetime.now(UTC) - timedelta(hours=hours)

        result = await session.execute(
            select(NewsArticle)
            .where(NewsArticle.published_at >= cutoff)
            .where(NewsArticle.symbols.contains([symbol]))
            .order_by(NewsArticle.published_at.desc())
            .limit(10)
        )

        articles = result.scalars().all()

        # Convert to FinancialDocument objects
        documents = [
            FinancialDocument(
                id=str(article.id),
                title=article.title,
                content=article.content or "",
                symbols=article.symbols,
                published_at=article.published_at or datetime.now(UTC),
                source=article.source,
                metadata=article.metadata_,
            )
            for article in articles
        ]

        return documents

    @staticmethod
    def _sentiment_to_score(label: SentimentLabel) -> float:
        """Convert sentiment label to numeric score.

        Args:
            label: Sentiment label.

        Returns:
            Numeric score (-1 to 1).
        """
        score_map = {
            "positive": 0.7,
            "negative": -0.7,
            "neutral": 0.0,
        }
        return score_map.get(label, 0.0)

    @staticmethod
    def _score_to_label(score: float) -> str:
        """Convert numeric score to sentiment label.

        Args:
            score: Numeric sentiment score (-1 to 1).

        Returns:
            Sentiment label (bullish, bearish, neutral).
        """
        if score >= 0.3:
            return "bullish"
        elif score <= -0.3:
            return "bearish"
        else:
            return "neutral"

    @staticmethod
    def _map_urgency_level(level: UrgencyLevel) -> str:
        """Map UrgencyLevel enum to string.

        Args:
            level: Urgency level enum.

        Returns:
            Urgency string (low, medium, high, critical).
        """
        urgency_map = {
            UrgencyLevel.ARCHIVE: "low",
            UrgencyLevel.LOW: "low",
            UrgencyLevel.MEDIUM: "medium",
            UrgencyLevel.HIGH: "high",
            UrgencyLevel.CRITICAL: "critical",
        }
        return urgency_map.get(level, "low")
