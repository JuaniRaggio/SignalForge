"""Tests for contradiction detector module.

This module tests the contradiction detection capabilities for analyst opinions,
including rating divergence, target divergence, sentiment divergence, and
temporal contradictions.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from signalforge.nlp.contradiction_detector import (
    AnalystOpinion,
    Contradiction,
    ContradictionDetector,
    ContradictionType,
    DivergenceAnalysis,
    DivergenceSeverity,
)


class TestAnalystOpinion:
    """Tests for AnalystOpinion dataclass."""

    def test_valid_analyst_opinion(self) -> None:
        """Test creation of valid analyst opinion."""
        opinion = AnalystOpinion(
            source="Bloomberg",
            analyst="John Doe",
            firm="Goldman Sachs",
            date=datetime.now(),
            rating="buy",
            target_price=150.0,
            sentiment_score=0.8,
            key_points=["Strong growth", "Market leader"],
            text="Positive outlook for the company",
        )

        assert opinion.source == "Bloomberg"
        assert opinion.analyst == "John Doe"
        assert opinion.sentiment_score == 0.8

    def test_invalid_sentiment_score_high(self) -> None:
        """Test that sentiment score above 1.0 raises error."""
        with pytest.raises(ValueError, match="Sentiment score must be between -1.0 and 1.0"):
            AnalystOpinion(
                source="Source",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating=None,
                target_price=None,
                sentiment_score=1.5,
                key_points=[],
                text="Test",
            )

    def test_invalid_sentiment_score_low(self) -> None:
        """Test that sentiment score below -1.0 raises error."""
        with pytest.raises(ValueError, match="Sentiment score must be between -1.0 and 1.0"):
            AnalystOpinion(
                source="Source",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating=None,
                target_price=None,
                sentiment_score=-1.5,
                key_points=[],
                text="Test",
            )

    def test_invalid_target_price(self) -> None:
        """Test that negative target price raises error."""
        with pytest.raises(ValueError, match="Target price must be positive"):
            AnalystOpinion(
                source="Source",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating=None,
                target_price=-100.0,
                sentiment_score=0.0,
                key_points=[],
                text="Test",
            )

    def test_optional_fields(self) -> None:
        """Test that optional fields can be None."""
        opinion = AnalystOpinion(
            source="Source",
            analyst=None,
            firm=None,
            date=datetime.now(),
            rating=None,
            target_price=None,
            sentiment_score=0.0,
            key_points=[],
            text="Test",
        )

        assert opinion.analyst is None
        assert opinion.firm is None
        assert opinion.rating is None
        assert opinion.target_price is None


class TestContradiction:
    """Tests for Contradiction dataclass."""

    def test_valid_contradiction(self) -> None:
        """Test creation of valid contradiction."""
        op_a = AnalystOpinion(
            source="Source1",
            analyst=None,
            firm=None,
            date=datetime.now(),
            rating="buy",
            target_price=None,
            sentiment_score=0.8,
            key_points=[],
            text="Positive",
        )
        op_b = AnalystOpinion(
            source="Source2",
            analyst=None,
            firm=None,
            date=datetime.now(),
            rating="sell",
            target_price=None,
            sentiment_score=-0.6,
            key_points=[],
            text="Negative",
        )

        contradiction = Contradiction(
            type=ContradictionType.RATING_DIVERGENCE,
            severity=DivergenceSeverity.HIGH,
            opinion_a=op_a,
            opinion_b=op_b,
            description="Test contradiction",
            divergence_score=0.75,
            implications=["High uncertainty"],
        )

        assert contradiction.type == ContradictionType.RATING_DIVERGENCE
        assert contradiction.severity == DivergenceSeverity.HIGH
        assert contradiction.divergence_score == 0.75

    def test_invalid_divergence_score(self) -> None:
        """Test that invalid divergence score raises error."""
        op_a = AnalystOpinion(
            source="Source1",
            analyst=None,
            firm=None,
            date=datetime.now(),
            rating=None,
            target_price=None,
            sentiment_score=0.0,
            key_points=[],
            text="Test",
        )

        with pytest.raises(ValueError, match="Divergence score must be between 0.0 and 1.0"):
            Contradiction(
                type=ContradictionType.RATING_DIVERGENCE,
                severity=DivergenceSeverity.LOW,
                opinion_a=op_a,
                opinion_b=op_a,
                description="Test",
                divergence_score=1.5,
                implications=[],
            )


class TestDivergenceAnalysis:
    """Tests for DivergenceAnalysis dataclass."""

    def test_valid_divergence_analysis(self) -> None:
        """Test creation of valid divergence analysis."""
        analysis = DivergenceAnalysis(
            symbol="AAPL",
            contradictions=[],
            consensus_strength=0.8,
            bullish_count=5,
            bearish_count=2,
            neutral_count=1,
            key_disagreement_topics=["valuation", "growth"],
            recommendation="Strong bullish consensus",
        )

        assert analysis.symbol == "AAPL"
        assert analysis.consensus_strength == 0.8
        assert analysis.bullish_count == 5

    def test_invalid_consensus_strength(self) -> None:
        """Test that invalid consensus strength raises error."""
        with pytest.raises(ValueError, match="Consensus strength must be between 0.0 and 1.0"):
            DivergenceAnalysis(
                symbol="AAPL",
                contradictions=[],
                consensus_strength=1.5,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                key_disagreement_topics=[],
                recommendation="",
            )

    def test_negative_opinion_counts(self) -> None:
        """Test that negative opinion counts raise error."""
        with pytest.raises(ValueError, match="Opinion counts must be non-negative"):
            DivergenceAnalysis(
                symbol="AAPL",
                contradictions=[],
                consensus_strength=0.5,
                bullish_count=-1,
                bearish_count=0,
                neutral_count=0,
                key_disagreement_topics=[],
                recommendation="",
            )


class TestContradictionDetector:
    """Tests for ContradictionDetector class."""

    @pytest.fixture
    def detector(self) -> ContradictionDetector:
        """Create a contradiction detector instance."""
        return ContradictionDetector()

    @pytest.fixture
    def sample_opinions(self) -> list[AnalystOpinion]:
        """Create sample analyst opinions for testing."""
        return [
            AnalystOpinion(
                source="Source1",
                analyst="Analyst1",
                firm="Firm1",
                date=datetime.now(),
                rating="buy",
                target_price=150.0,
                sentiment_score=0.8,
                key_points=["Strong growth", "Market leader"],
                text="Positive outlook",
            ),
            AnalystOpinion(
                source="Source2",
                analyst="Analyst2",
                firm="Firm2",
                date=datetime.now(),
                rating="sell",
                target_price=100.0,
                sentiment_score=-0.6,
                key_points=["Weak fundamentals", "Overvalued"],
                text="Negative outlook",
            ),
            AnalystOpinion(
                source="Source3",
                analyst="Analyst3",
                firm="Firm3",
                date=datetime.now(),
                rating="hold",
                target_price=125.0,
                sentiment_score=0.1,
                key_points=["Wait and see"],
                text="Neutral stance",
            ),
        ]

    def test_analyze_opinions_empty_list(self, detector: ContradictionDetector) -> None:
        """Test that analyzing empty list raises error."""
        with pytest.raises(ValueError, match="Opinions list cannot be empty"):
            detector.analyze_opinions([])

    def test_analyze_opinions_basic(
        self, detector: ContradictionDetector, sample_opinions: list[AnalystOpinion]
    ) -> None:
        """Test basic opinion analysis."""
        analysis = detector.analyze_opinions(sample_opinions, "AAPL")

        assert analysis.symbol == "AAPL"
        assert len(analysis.contradictions) > 0
        assert 0.0 <= analysis.consensus_strength <= 1.0
        assert analysis.bullish_count == 1
        assert analysis.bearish_count == 1
        assert analysis.neutral_count == 1

    def test_detect_rating_divergence_basic(
        self, detector: ContradictionDetector, sample_opinions: list[AnalystOpinion]
    ) -> None:
        """Test basic rating divergence detection."""
        contradictions = detector.detect_rating_divergence(sample_opinions)

        assert len(contradictions) > 0
        assert all(c.type == ContradictionType.RATING_DIVERGENCE for c in contradictions)

    def test_detect_rating_divergence_threshold(self, detector: ContradictionDetector) -> None:
        """Test rating divergence with custom threshold."""
        opinions = [
            AnalystOpinion(
                source="S1",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating="buy",
                target_price=None,
                sentiment_score=0.5,
                key_points=[],
                text="Test",
            ),
            AnalystOpinion(
                source="S2",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating="hold",
                target_price=None,
                sentiment_score=0.0,
                key_points=[],
                text="Test",
            ),
        ]

        # Low threshold should detect divergence
        contradictions = detector.detect_rating_divergence(opinions, threshold=0.1)
        assert len(contradictions) > 0

        # High threshold should not detect divergence
        contradictions = detector.detect_rating_divergence(opinions, threshold=0.9)
        assert len(contradictions) == 0

    def test_detect_rating_divergence_no_ratings(self, detector: ContradictionDetector) -> None:
        """Test rating divergence with no ratings."""
        opinions = [
            AnalystOpinion(
                source="S1",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating=None,
                target_price=None,
                sentiment_score=0.5,
                key_points=[],
                text="Test",
            )
        ]

        contradictions = detector.detect_rating_divergence(opinions)
        assert len(contradictions) == 0

    def test_detect_target_divergence(
        self, detector: ContradictionDetector, sample_opinions: list[AnalystOpinion]
    ) -> None:
        """Test target price divergence detection."""
        contradictions = detector.detect_target_divergence(sample_opinions)

        assert len(contradictions) > 0
        assert all(c.type == ContradictionType.TARGET_DIVERGENCE for c in contradictions)

    def test_detect_target_divergence_threshold(self, detector: ContradictionDetector) -> None:
        """Test target divergence with custom threshold."""
        opinions = [
            AnalystOpinion(
                source="S1",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating=None,
                target_price=100.0,
                sentiment_score=0.0,
                key_points=[],
                text="Test",
            ),
            AnalystOpinion(
                source="S2",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating=None,
                target_price=150.0,
                sentiment_score=0.0,
                key_points=[],
                text="Test",
            ),
        ]

        # Should detect 50% divergence
        contradictions = detector.detect_target_divergence(opinions, threshold_percent=40.0)
        assert len(contradictions) > 0

        # Should not detect with high threshold
        contradictions = detector.detect_target_divergence(opinions, threshold_percent=60.0)
        assert len(contradictions) == 0

    def test_detect_sentiment_divergence(
        self, detector: ContradictionDetector, sample_opinions: list[AnalystOpinion]
    ) -> None:
        """Test sentiment divergence detection."""
        contradictions = detector.detect_sentiment_divergence(sample_opinions)

        assert len(contradictions) > 0
        assert all(c.type == ContradictionType.SENTIMENT_DIVERGENCE for c in contradictions)

    def test_detect_sentiment_divergence_threshold(self, detector: ContradictionDetector) -> None:
        """Test sentiment divergence with custom threshold."""
        opinions = [
            AnalystOpinion(
                source="S1",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating=None,
                target_price=None,
                sentiment_score=0.8,
                key_points=[],
                text="Test",
            ),
            AnalystOpinion(
                source="S2",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating=None,
                target_price=None,
                sentiment_score=-0.8,
                key_points=[],
                text="Test",
            ),
        ]

        # Should detect large divergence
        contradictions = detector.detect_sentiment_divergence(opinions, threshold=0.5)
        assert len(contradictions) > 0

        # Should not detect with very high threshold
        contradictions = detector.detect_sentiment_divergence(opinions, threshold=1.8)
        assert len(contradictions) == 0

    def test_temporal_contradiction_same_source(self, detector: ContradictionDetector) -> None:
        """Test temporal contradiction detection from same source."""
        base_date = datetime.now()

        opinions = [
            AnalystOpinion(
                source="Source1",
                analyst="Analyst1",
                firm=None,
                date=base_date,
                rating="strong_buy",
                target_price=None,
                sentiment_score=0.9,
                key_points=[],
                text="Test",
            ),
            AnalystOpinion(
                source="Source1",
                analyst="Analyst1",
                firm=None,
                date=base_date + timedelta(days=30),
                rating="strong_sell",
                target_price=None,
                sentiment_score=-0.9,
                key_points=[],
                text="Test",
            ),
        ]

        contradictions = detector.detect_temporal_contradiction(opinions, same_source=True)
        assert len(contradictions) > 0
        assert all(c.type == ContradictionType.TEMPORAL_CONTRADICTION for c in contradictions)

    def test_temporal_contradiction_different_sources(
        self, detector: ContradictionDetector
    ) -> None:
        """Test temporal contradiction not detected from different sources."""
        base_date = datetime.now()

        opinions = [
            AnalystOpinion(
                source="Source1",
                analyst="Analyst1",
                firm=None,
                date=base_date,
                rating="strong_buy",
                target_price=None,
                sentiment_score=0.9,
                key_points=[],
                text="Test",
            ),
            AnalystOpinion(
                source="Source2",
                analyst="Analyst2",
                firm=None,
                date=base_date + timedelta(days=30),
                rating="strong_sell",
                target_price=None,
                sentiment_score=-0.9,
                key_points=[],
                text="Test",
            ),
        ]

        # Should not detect temporal contradiction from different sources
        contradictions = detector.detect_temporal_contradiction(opinions, same_source=True)
        assert len(contradictions) == 0

    def test_calculate_consensus_strength_unanimous(
        self, detector: ContradictionDetector
    ) -> None:
        """Test consensus strength calculation for unanimous opinions."""
        opinions = [
            AnalystOpinion(
                source=f"Source{i}",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating="buy",
                target_price=None,
                sentiment_score=0.8,
                key_points=[],
                text="Test",
            )
            for i in range(5)
        ]

        consensus = detector.calculate_consensus_strength(opinions)
        assert consensus > 0.9  # Should be very high for identical opinions

    def test_calculate_consensus_strength_divergent(
        self, detector: ContradictionDetector
    ) -> None:
        """Test consensus strength calculation for divergent opinions."""
        opinions = [
            AnalystOpinion(
                source="S1",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating=None,
                target_price=None,
                sentiment_score=1.0,
                key_points=[],
                text="Test",
            ),
            AnalystOpinion(
                source="S2",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating=None,
                target_price=None,
                sentiment_score=-1.0,
                key_points=[],
                text="Test",
            ),
        ]

        consensus = detector.calculate_consensus_strength(opinions)
        assert consensus < 0.5  # Should be low for opposite opinions

    def test_calculate_consensus_strength_empty(self, detector: ContradictionDetector) -> None:
        """Test consensus strength with empty list."""
        consensus = detector.calculate_consensus_strength([])
        assert consensus == 0.0

    def test_calculate_consensus_strength_single(self, detector: ContradictionDetector) -> None:
        """Test consensus strength with single opinion."""
        opinions = [
            AnalystOpinion(
                source="S1",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating=None,
                target_price=None,
                sentiment_score=0.5,
                key_points=[],
                text="Test",
            )
        ]

        consensus = detector.calculate_consensus_strength(opinions)
        assert consensus == 1.0  # Single opinion is unanimous

    def test_extract_key_disagreement_topics(
        self, detector: ContradictionDetector, sample_opinions: list[AnalystOpinion]
    ) -> None:
        """Test extraction of key disagreement topics."""
        # First get contradictions
        contradictions = detector.detect_rating_divergence(sample_opinions)

        topics = detector.extract_key_disagreement_topics(contradictions)

        assert isinstance(topics, list)
        assert len(topics) <= 5  # Should return top 5

    def test_extract_key_disagreement_topics_empty(
        self, detector: ContradictionDetector
    ) -> None:
        """Test topic extraction with empty contradictions."""
        topics = detector.extract_key_disagreement_topics([])
        assert topics == []

    def test_generate_divergence_summary(
        self, detector: ContradictionDetector, sample_opinions: list[AnalystOpinion]
    ) -> None:
        """Test divergence summary generation."""
        analysis = detector.analyze_opinions(sample_opinions, "AAPL")
        summary = detector.generate_divergence_summary(analysis)

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "AAPL" in summary
        assert "Consensus Strength" in summary

    def test_compare_to_consensus_aligned(self, detector: ContradictionDetector) -> None:
        """Test comparing opinion to consensus when aligned."""
        opinions = [
            AnalystOpinion(
                source=f"Source{i}",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating="buy",
                target_price=None,
                sentiment_score=0.7,
                key_points=[],
                text="Test",
            )
            for i in range(5)
        ]

        consensus = detector.analyze_opinions(opinions)

        new_opinion = AnalystOpinion(
            source="NewSource",
            analyst=None,
            firm=None,
            date=datetime.now(),
            rating="buy",
            target_price=None,
            sentiment_score=0.8,
            key_points=[],
            text="Test",
        )

        comparison = detector.compare_to_consensus(new_opinion, consensus)

        assert comparison["alignment"] == "aligned"
        assert comparison["is_contrarian"] is False

    def test_compare_to_consensus_divergent(self, detector: ContradictionDetector) -> None:
        """Test comparing opinion to consensus when divergent."""
        opinions = [
            AnalystOpinion(
                source=f"Source{i}",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating="buy",
                target_price=None,
                sentiment_score=0.7,
                key_points=[],
                text="Test",
            )
            for i in range(5)
        ]

        consensus = detector.analyze_opinions(opinions)

        new_opinion = AnalystOpinion(
            source="NewSource",
            analyst=None,
            firm=None,
            date=datetime.now(),
            rating="sell",
            target_price=None,
            sentiment_score=-0.8,
            key_points=[],
            text="Test",
        )

        comparison = detector.compare_to_consensus(new_opinion, consensus)

        assert comparison["alignment"] == "divergent"
        # Contrarian if strong consensus and divergent
        assert isinstance(comparison["is_contrarian"], bool)

    def test_empty_opinions(self, detector: ContradictionDetector) -> None:
        """Test handling of empty opinions list."""
        with pytest.raises(ValueError):
            detector.analyze_opinions([])

    def test_single_opinion(self, detector: ContradictionDetector) -> None:
        """Test analysis with single opinion."""
        opinion = AnalystOpinion(
            source="Source1",
            analyst=None,
            firm=None,
            date=datetime.now(),
            rating="buy",
            target_price=150.0,
            sentiment_score=0.8,
            key_points=["Growth"],
            text="Positive",
        )

        analysis = detector.analyze_opinions([opinion], "AAPL")

        assert analysis.consensus_strength == 1.0
        assert len(analysis.contradictions) == 0

    def test_extreme_divergence(self, detector: ContradictionDetector) -> None:
        """Test detection of extreme divergence."""
        opinions = [
            AnalystOpinion(
                source="S1",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating="strong_buy",
                target_price=200.0,
                sentiment_score=1.0,
                key_points=[],
                text="Test",
            ),
            AnalystOpinion(
                source="S2",
                analyst=None,
                firm=None,
                date=datetime.now(),
                rating="strong_sell",
                target_price=50.0,
                sentiment_score=-1.0,
                key_points=[],
                text="Test",
            ),
        ]

        analysis = detector.analyze_opinions(opinions)

        # Should detect extreme divergence
        extreme_contradictions = [
            c for c in analysis.contradictions if c.severity == DivergenceSeverity.EXTREME
        ]
        assert len(extreme_contradictions) > 0

    def test_mixed_divergence_types(
        self, detector: ContradictionDetector, sample_opinions: list[AnalystOpinion]
    ) -> None:
        """Test detection of multiple divergence types."""
        analysis = detector.analyze_opinions(sample_opinions)

        # Should detect different types
        types_detected = {c.type for c in analysis.contradictions}
        assert len(types_detected) > 1  # Multiple types detected
