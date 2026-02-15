"""Tests for NLP signal generator."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from signalforge.nlp.price_target_extractor import PriceTarget, Rating, TargetAction
from signalforge.nlp.sentiment import SentimentResult
from signalforge.nlp.signals.generator import (
    AnalystConsensus,
    FinancialDocument,
    NLPSignalGenerator,
    NLPSignalOutput,
    SectorSignal,
    SentimentOutput,
)


class MockSentimentAnalyzer:
    """Mock sentiment analyzer for testing."""

    def analyze(self, text: str) -> SentimentResult:
        """Mock sentiment analysis."""
        if "strong" in text.lower() or "beat" in text.lower():
            return SentimentResult(
                text=text,
                label="positive",
                confidence=0.9,
                scores={"positive": 0.9, "negative": 0.05, "neutral": 0.05},
            )
        elif "weak" in text.lower() or "miss" in text.lower():
            return SentimentResult(
                text=text,
                label="negative",
                confidence=0.85,
                scores={"positive": 0.05, "negative": 0.85, "neutral": 0.1},
            )
        else:
            return SentimentResult(
                text=text,
                label="neutral",
                confidence=0.7,
                scores={"positive": 0.3, "negative": 0.3, "neutral": 0.4},
            )


@pytest.fixture
def mock_sentiment_analyzer() -> MockSentimentAnalyzer:
    """Create mock sentiment analyzer."""
    return MockSentimentAnalyzer()


@pytest.fixture
def generator(mock_sentiment_analyzer: MockSentimentAnalyzer) -> NLPSignalGenerator:
    """Create NLP signal generator."""
    return NLPSignalGenerator(mock_sentiment_analyzer)


@pytest.fixture
def sample_document() -> FinancialDocument:
    """Create sample document for testing."""
    return FinancialDocument(
        id="test-doc-1",
        title="Apple Reports Strong Q4 Earnings",
        content="Apple Inc. reported strong Q4 earnings that beat analyst expectations.",
        symbols=["AAPL"],
        published_at=datetime.now(UTC),
        source="test",
        metadata={"sector": "Information Technology"},
    )


@pytest.fixture
async def mock_db() -> AsyncSession:
    """Create mock database session."""
    mock_session = AsyncMock(spec=AsyncSession)

    # Create a mock result that supports scalars().all()
    mock_scalars_result = MagicMock()
    mock_scalars_result.all.return_value = []

    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars_result
    mock_result.scalar_one_or_none.return_value = None

    mock_session.execute = AsyncMock(return_value=mock_result)
    return mock_session


# Test SentimentOutput validation


def test_sentiment_output_valid():
    """Test valid sentiment output creation."""
    sentiment = SentimentOutput(
        score=0.7,
        label="bullish",
        delta_vs_baseline=0.2,
        is_new_information=True,
    )
    assert sentiment.score == 0.7
    assert sentiment.label == "bullish"


def test_sentiment_output_invalid_score():
    """Test sentiment output with invalid score."""
    with pytest.raises(ValueError, match="score must be between -1.0 and 1.0"):
        SentimentOutput(
            score=1.5,
            label="bullish",
            delta_vs_baseline=0.0,
            is_new_information=True,
        )


def test_sentiment_output_invalid_label():
    """Test sentiment output with invalid label."""
    with pytest.raises(ValueError, match="Invalid label"):
        SentimentOutput(
            score=0.5,
            label="invalid",
            delta_vs_baseline=0.0,
            is_new_information=True,
        )


# Test AnalystConsensus validation


def test_analyst_consensus_valid():
    """Test valid analyst consensus creation."""
    consensus = AnalystConsensus(
        rating="buy",
        confidence=0.8,
        bullish_count=5,
        bearish_count=1,
        neutral_count=2,
    )
    assert consensus.rating == "buy"
    assert consensus.confidence == 0.8


def test_analyst_consensus_invalid_rating():
    """Test analyst consensus with invalid rating."""
    with pytest.raises(ValueError, match="Invalid rating"):
        AnalystConsensus(
            rating="invalid",
            confidence=0.8,
            bullish_count=5,
            bearish_count=1,
            neutral_count=2,
        )


def test_analyst_consensus_negative_counts():
    """Test analyst consensus with negative counts."""
    with pytest.raises(ValueError, match="Counts must be non-negative"):
        AnalystConsensus(
            rating="buy",
            confidence=0.8,
            bullish_count=-1,
            bearish_count=1,
            neutral_count=2,
        )


# Test SectorSignal validation


def test_sector_signal_valid():
    """Test valid sector signal creation."""
    signal = SectorSignal(
        sector="Technology",
        signal="buy",
        confidence=0.75,
        reasoning="Strong fundamentals",
    )
    assert signal.signal == "buy"
    assert signal.confidence == 0.75


def test_sector_signal_invalid_signal():
    """Test sector signal with invalid signal type."""
    with pytest.raises(ValueError, match="Invalid signal"):
        SectorSignal(
            sector="Technology",
            signal="invalid",
            confidence=0.75,
            reasoning="Test",
        )


# Test normalize_sentiment


def test_normalize_sentiment_above_baseline(generator: NLPSignalGenerator):
    """Test sentiment normalization above baseline."""
    generator._sentiment_baselines["AAPL:Technology"] = 0.2
    normalized = generator.normalize_sentiment(0.7, "AAPL", "Technology")
    assert normalized == pytest.approx(0.5)  # 0.7 - 0.2


def test_normalize_sentiment_below_baseline(generator: NLPSignalGenerator):
    """Test sentiment normalization below baseline."""
    generator._sentiment_baselines["AAPL:Technology"] = 0.5
    normalized = generator.normalize_sentiment(-0.3, "AAPL", "Technology")
    assert normalized == -0.8  # -0.3 - 0.5


def test_normalize_sentiment_bounded(generator: NLPSignalGenerator):
    """Test sentiment normalization is bounded to [-1, 1]."""
    generator._sentiment_baselines["AAPL:Technology"] = -0.5
    normalized = generator.normalize_sentiment(0.8, "AAPL", "Technology")
    assert normalized == 1.0  # capped at 1.0


def test_normalize_sentiment_default_baseline(generator: NLPSignalGenerator):
    """Test sentiment normalization with default baseline."""
    normalized = generator.normalize_sentiment(0.6, "NEW", "Tech")
    assert normalized == 0.6  # baseline is 0.0 by default


# Test calculate_sentiment_momentum


def test_calculate_sentiment_momentum_positive(generator: NLPSignalGenerator):
    """Test positive sentiment momentum."""
    historical = [
        (datetime.now(UTC) - timedelta(days=i), 0.3 + i * 0.05) for i in range(5)
    ]
    momentum = generator.calculate_sentiment_momentum(0.7, historical)
    assert momentum > 0  # current > average


def test_calculate_sentiment_momentum_negative(generator: NLPSignalGenerator):
    """Test negative sentiment momentum."""
    historical = [
        (datetime.now(UTC) - timedelta(days=i), 0.6) for i in range(5)
    ]
    momentum = generator.calculate_sentiment_momentum(0.3, historical)
    assert momentum < 0  # current < average


def test_calculate_sentiment_momentum_no_history(generator: NLPSignalGenerator):
    """Test sentiment momentum with no history."""
    momentum = generator.calculate_sentiment_momentum(0.5, [])
    assert momentum == 0.0


# Test aggregate_analyst_consensus


def test_aggregate_consensus_bullish(generator: NLPSignalGenerator):
    """Test consensus aggregation for bullish signals."""
    signals = [
        SectorSignal("Tech", "buy", 0.8, ""),
        SectorSignal("Tech", "buy", 0.7, ""),
        SectorSignal("Tech", "hold", 0.6, ""),
    ]
    consensus = generator.aggregate_analyst_consensus(signals, [])
    assert consensus.rating in ["buy", "strong_buy"]
    assert consensus.bullish_count == 2


def test_aggregate_consensus_bearish(generator: NLPSignalGenerator):
    """Test consensus aggregation for bearish signals."""
    signals = [
        SectorSignal("Tech", "sell", 0.8, ""),
        SectorSignal("Tech", "sell", 0.7, ""),
        SectorSignal("Tech", "hold", 0.6, ""),
    ]
    consensus = generator.aggregate_analyst_consensus(signals, [])
    assert consensus.rating in ["sell", "strong_sell"]
    assert consensus.bearish_count == 2


def test_aggregate_consensus_with_price_targets(generator: NLPSignalGenerator):
    """Test consensus aggregation with price targets."""
    signals = [SectorSignal("Tech", "buy", 0.8, "")]
    price_targets = [
        PriceTarget(
            symbol="AAPL",
            target_price=200.0,
            current_price=150.0,
            upside_percent=33.3,
            action=TargetAction.UPGRADE,
            rating=Rating.BUY,
            analyst=None,
            firm=None,
            date=datetime.now(UTC),
            confidence=0.8,
            source_text="Test",
        )
    ]
    consensus = generator.aggregate_analyst_consensus(signals, price_targets)
    assert consensus.bullish_count == 2  # 1 signal + 1 target


def test_aggregate_consensus_empty(generator: NLPSignalGenerator):
    """Test consensus aggregation with no signals."""
    consensus = generator.aggregate_analyst_consensus([], [])
    assert consensus.rating == "hold"
    assert consensus.confidence == 0.0


# Test is_new_information


def test_is_new_information_no_recent(generator: NLPSignalGenerator):
    """Test new information detection with no recent documents."""
    doc = FinancialDocument(
        id="1",
        title="New Article",
        content="Content",
        symbols=["AAPL"],
        published_at=datetime.now(UTC),
        source="test",
        metadata={},
    )
    assert generator.is_new_information(doc, []) is True


def test_is_new_information_duplicate(generator: NLPSignalGenerator):
    """Test new information detection with duplicate title."""
    doc1 = FinancialDocument(
        id="1",
        title="Apple Reports Strong Earnings Beat",
        content="Content",
        symbols=["AAPL"],
        published_at=datetime.now(UTC),
        source="test",
        metadata={},
    )
    doc2 = FinancialDocument(
        id="2",
        title="Apple Reports Strong Earnings Beat Today",
        content="Different content",
        symbols=["AAPL"],
        published_at=datetime.now(UTC) - timedelta(hours=1),
        source="test",
        metadata={},
    )
    assert generator.is_new_information(doc1, [doc2]) is False


def test_is_new_information_different(generator: NLPSignalGenerator):
    """Test new information detection with different title."""
    doc1 = FinancialDocument(
        id="1",
        title="Apple Announces New Product",
        content="Content",
        symbols=["AAPL"],
        published_at=datetime.now(UTC),
        source="test",
        metadata={},
    )
    doc2 = FinancialDocument(
        id="2",
        title="Microsoft Reports Earnings",
        content="Different content",
        symbols=["MSFT"],
        published_at=datetime.now(UTC) - timedelta(hours=1),
        source="test",
        metadata={},
    )
    assert generator.is_new_information(doc1, [doc2]) is True


# Test generate_signals integration


@pytest.mark.asyncio
async def test_generate_signals_success(
    generator: NLPSignalGenerator,
    sample_document: FinancialDocument,
    mock_db: AsyncSession,
):
    """Test successful signal generation."""
    signals = await generator.generate_signals(sample_document, mock_db)

    assert isinstance(signals, NLPSignalOutput)
    assert signals.ticker == "AAPL"
    assert signals.sentiment.label in ["bullish", "bearish", "neutral"]
    assert -1.0 <= signals.sentiment.score <= 1.0
    assert signals.urgency in ["low", "medium", "high", "critical"]
    assert isinstance(signals.analyst_consensus, AnalystConsensus)


@pytest.mark.asyncio
async def test_generate_signals_positive_sentiment(
    generator: NLPSignalGenerator,
    mock_db: AsyncSession,
):
    """Test signal generation with positive sentiment."""
    doc = FinancialDocument(
        id="test",
        title="Strong Earnings Beat",
        content="Company reported strong earnings that beat expectations.",
        symbols=["AAPL"],
        published_at=datetime.now(UTC),
        source="test",
        metadata={"sector": "Technology"},
    )

    signals = await generator.generate_signals(doc, mock_db)
    assert signals.sentiment.label == "bullish"
    assert signals.sentiment.score > 0


@pytest.mark.asyncio
async def test_generate_signals_negative_sentiment(
    generator: NLPSignalGenerator,
    mock_db: AsyncSession,
):
    """Test signal generation with negative sentiment."""
    doc = FinancialDocument(
        id="test",
        title="Weak Earnings Miss",
        content="Company reported weak earnings that missed expectations.",
        symbols=["AAPL"],
        published_at=datetime.now(UTC),
        source="test",
        metadata={"sector": "Technology"},
    )

    signals = await generator.generate_signals(doc, mock_db)
    assert signals.sentiment.label == "bearish"
    assert signals.sentiment.score < 0


# Test helper methods


def test_sentiment_to_score(generator: NLPSignalGenerator):
    """Test sentiment label to score conversion."""
    assert generator._sentiment_to_score("positive") == 0.7
    assert generator._sentiment_to_score("negative") == -0.7
    assert generator._sentiment_to_score("neutral") == 0.0


def test_score_to_label(generator: NLPSignalGenerator):
    """Test score to sentiment label conversion."""
    assert generator._score_to_label(0.5) == "bullish"
    assert generator._score_to_label(-0.5) == "bearish"
    assert generator._score_to_label(0.1) == "neutral"
