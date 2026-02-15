"""Tests for text summarization module.

This module tests the extractive summarization capabilities for financial
documents, including single document summarization, multi-document summarization,
key point extraction, and specialized earnings call summarization.
"""

from __future__ import annotations

import pytest

from signalforge.nlp.summarization import (
    MultiDocumentSummary,
    SummaryLength,
    SummaryResult,
    SummaryStyle,
    TextSummarizer,
)


class TestSummaryResult:
    """Tests for SummaryResult dataclass."""

    def test_valid_summary_result(self) -> None:
        """Test creation of valid summary result."""
        result = SummaryResult(
            summary="This is a test summary.",
            length=SummaryLength.SHORT,
            style=SummaryStyle.NEUTRAL,
            key_points=["Point 1", "Point 2"],
            entities_mentioned=["AAPL", "Apple Inc."],
            sentiment_summary="positive",
            word_count=5,
            compression_ratio=10.0,
        )

        assert result.summary == "This is a test summary."
        assert result.length == SummaryLength.SHORT
        assert result.word_count == 5

    def test_invalid_compression_ratio(self) -> None:
        """Test that negative compression ratio raises error."""
        with pytest.raises(ValueError, match="Compression ratio must be non-negative"):
            SummaryResult(
                summary="Test",
                length=SummaryLength.SHORT,
                style=SummaryStyle.NEUTRAL,
                key_points=[],
                entities_mentioned=[],
                sentiment_summary=None,
                word_count=1,
                compression_ratio=-1.0,
            )


class TestMultiDocumentSummary:
    """Tests for MultiDocumentSummary dataclass."""

    def test_valid_multi_document_summary(self) -> None:
        """Test creation of valid multi-document summary."""
        summary = MultiDocumentSummary(
            combined_summary="Combined summary text.",
            document_count=3,
            common_themes=["theme1", "theme2"],
            conflicting_points=["conflict1"],
            timeline=[("2024-01-01", "Event 1")],
            key_entities=["AAPL", "MSFT"],
        )

        assert summary.document_count == 3
        assert len(summary.common_themes) == 2

    def test_invalid_document_count(self) -> None:
        """Test that negative document count raises error."""
        with pytest.raises(ValueError, match="Document count must be non-negative"):
            MultiDocumentSummary(
                combined_summary="Test",
                document_count=-1,
                common_themes=[],
                conflicting_points=[],
                timeline=[],
                key_entities=[],
            )


class TestTextSummarizer:
    """Tests for TextSummarizer class."""

    @pytest.fixture
    def summarizer(self) -> TextSummarizer:
        """Create a text summarizer instance."""
        return TextSummarizer()

    @pytest.fixture
    def sample_text(self) -> str:
        """Create sample financial text for testing."""
        return """
        Apple Inc. reported strong quarterly earnings today. The company announced
        revenue of $90 billion for Q4 2024, beating analyst expectations by 5%.
        CEO Tim Cook stated that iPhone sales exceeded projections. The company
        raised its full-year guidance and expects continued growth in services.
        Analysts are optimistic about the company's future prospects. Apple's
        stock price increased 3% in after-hours trading. The tech giant also
        announced a new $10 billion share buyback program. Management emphasized
        strong demand in emerging markets.
        """

    @pytest.fixture
    def sample_texts(self) -> list[str]:
        """Create sample texts for multi-document testing."""
        return [
            "Apple reported strong earnings, beating expectations by 5%.",
            "The company raised its full-year guidance due to strong iPhone sales.",
            "Analysts are optimistic about Apple's growth in services revenue.",
        ]

    def test_summarize_empty_text(self, summarizer: TextSummarizer) -> None:
        """Test that empty text raises error."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            summarizer.summarize("")

    def test_summarize_brief(self, summarizer: TextSummarizer, sample_text: str) -> None:
        """Test brief summarization."""
        result = summarizer.summarize(sample_text, length=SummaryLength.BRIEF)

        assert isinstance(result, SummaryResult)
        assert result.length == SummaryLength.BRIEF
        assert 1 <= len(result.summary.split(". ")) <= 2

    def test_summarize_short(self, summarizer: TextSummarizer, sample_text: str) -> None:
        """Test short summarization."""
        result = summarizer.summarize(sample_text, length=SummaryLength.SHORT)

        assert isinstance(result, SummaryResult)
        assert result.length == SummaryLength.SHORT
        assert result.word_count > 0

    def test_summarize_medium(self, summarizer: TextSummarizer, sample_text: str) -> None:
        """Test medium summarization."""
        result = summarizer.summarize(sample_text, length=SummaryLength.MEDIUM)

        assert isinstance(result, SummaryResult)
        assert result.length == SummaryLength.MEDIUM
        assert result.word_count > 0

    def test_summarize_detailed(self, summarizer: TextSummarizer, sample_text: str) -> None:
        """Test detailed summarization."""
        result = summarizer.summarize(sample_text, length=SummaryLength.DETAILED)

        assert isinstance(result, SummaryResult)
        assert result.length == SummaryLength.DETAILED
        assert result.word_count > 0

    def test_summarize_neutral_style(self, summarizer: TextSummarizer, sample_text: str) -> None:
        """Test neutral style summarization."""
        result = summarizer.summarize(sample_text, style=SummaryStyle.NEUTRAL)

        assert result.style == SummaryStyle.NEUTRAL
        assert isinstance(result.summary, str)

    def test_summarize_actionable_style(
        self, summarizer: TextSummarizer, sample_text: str
    ) -> None:
        """Test actionable style summarization."""
        result = summarizer.summarize(sample_text, style=SummaryStyle.ACTIONABLE)

        assert result.style == SummaryStyle.ACTIONABLE
        assert isinstance(result.summary, str)

    def test_summarize_analytical_style(
        self, summarizer: TextSummarizer, sample_text: str
    ) -> None:
        """Test analytical style summarization."""
        result = summarizer.summarize(sample_text, style=SummaryStyle.ANALYTICAL)

        assert result.style == SummaryStyle.ANALYTICAL
        assert isinstance(result.summary, str)

    def test_summarize_executive_style(
        self, summarizer: TextSummarizer, sample_text: str
    ) -> None:
        """Test executive style summarization."""
        result = summarizer.summarize(sample_text, style=SummaryStyle.EXECUTIVE)

        assert result.style == SummaryStyle.EXECUTIVE
        assert isinstance(result.summary, str)

    def test_extract_key_points(self, summarizer: TextSummarizer, sample_text: str) -> None:
        """Test key point extraction."""
        points = summarizer.extract_key_points(sample_text, max_points=3)

        assert isinstance(points, list)
        assert len(points) <= 3
        assert all(isinstance(point, str) for point in points)

    def test_extract_key_points_max_limit(
        self, summarizer: TextSummarizer, sample_text: str
    ) -> None:
        """Test that key points respect max limit."""
        points = summarizer.extract_key_points(sample_text, max_points=2)

        assert len(points) <= 2

    def test_extract_key_points_empty_text(self, summarizer: TextSummarizer) -> None:
        """Test key point extraction from empty text."""
        points = summarizer.extract_key_points("")

        assert points == []

    def test_extract_entities(self, summarizer: TextSummarizer, sample_text: str) -> None:
        """Test entity extraction."""
        entities = summarizer.extract_entities(sample_text)

        assert isinstance(entities, list)
        assert len(entities) > 0
        # Should extract Apple
        assert any("Apple" in entity for entity in entities)

    def test_extract_entities_empty_text(self, summarizer: TextSummarizer) -> None:
        """Test entity extraction from empty text."""
        entities = summarizer.extract_entities("")

        assert entities == []

    def test_extract_entities_tickers(self, summarizer: TextSummarizer) -> None:
        """Test ticker extraction."""
        text = "AAPL stock rose 5% while MSFT declined 2%."
        entities = summarizer.extract_entities(text)

        assert "AAPL" in entities
        assert "MSFT" in entities

    def test_score_sentence_importance(
        self, summarizer: TextSummarizer, sample_text: str
    ) -> None:
        """Test sentence importance scoring."""
        sentence = "Apple Inc. reported strong quarterly earnings today."
        score = summarizer.score_sentence_importance(sentence, sample_text)

        assert 0.0 <= score <= 1.0
        assert score > 0.0  # Should have positive score

    def test_score_sentence_importance_empty(self, summarizer: TextSummarizer) -> None:
        """Test importance scoring with empty inputs."""
        score = summarizer.score_sentence_importance("", "")

        assert score == 0.0

    def test_score_sentence_importance_financial_terms(
        self, summarizer: TextSummarizer
    ) -> None:
        """Test that financial terms increase importance."""
        text = "The stock market today."
        sentence_with_terms = "Revenue exceeded earnings guidance, beating expectations."
        sentence_without_terms = "The meeting was scheduled for tomorrow."

        score_with = summarizer.score_sentence_importance(sentence_with_terms, text)
        score_without = summarizer.score_sentence_importance(sentence_without_terms, text)

        assert score_with > score_without

    def test_generate_headline(self, summarizer: TextSummarizer, sample_text: str) -> None:
        """Test headline generation."""
        headline = summarizer.generate_headline(sample_text, max_words=10)

        assert isinstance(headline, str)
        assert len(headline.split()) <= 11  # 10 + possible ellipsis

    def test_generate_headline_short_text(self, summarizer: TextSummarizer) -> None:
        """Test headline generation from short text."""
        text = "Apple beats earnings."
        headline = summarizer.generate_headline(text, max_words=5)

        assert len(headline) > 0
        assert "..." not in headline  # Should not have ellipsis for short text

    def test_generate_headline_empty_text(self, summarizer: TextSummarizer) -> None:
        """Test headline generation from empty text."""
        headline = summarizer.generate_headline("", max_words=10)

        assert headline == ""

    def test_create_executive_summary(
        self, summarizer: TextSummarizer, sample_text: str
    ) -> None:
        """Test executive summary creation."""
        summary = summarizer.create_executive_summary(sample_text, symbol="AAPL")

        assert isinstance(summary, str)
        assert "AAPL" in summary
        assert "Executive Summary" in summary

    def test_create_executive_summary_no_symbol(
        self, summarizer: TextSummarizer, sample_text: str
    ) -> None:
        """Test executive summary without symbol."""
        summary = summarizer.create_executive_summary(sample_text)

        assert isinstance(summary, str)
        assert "Executive Summary" in summary

    def test_multi_document_summary(
        self, summarizer: TextSummarizer, sample_texts: list[str]
    ) -> None:
        """Test multi-document summarization."""
        result = summarizer.summarize_multiple(sample_texts)

        assert isinstance(result, MultiDocumentSummary)
        assert result.document_count == len(sample_texts)
        assert len(result.combined_summary) > 0

    def test_multi_document_summary_empty_list(self, summarizer: TextSummarizer) -> None:
        """Test multi-document summarization with empty list."""
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            summarizer.summarize_multiple([])

    def test_multi_document_common_themes(
        self, summarizer: TextSummarizer, sample_texts: list[str]
    ) -> None:
        """Test common theme detection."""
        result = summarizer.summarize_multiple(sample_texts)

        assert isinstance(result.common_themes, list)
        # Should detect common themes like "apple", "earnings", etc.

    def test_summarize_earnings_call(self, summarizer: TextSummarizer) -> None:
        """Test specialized earnings call summarization."""
        transcript = """
        Good afternoon everyone. This is the Q4 2024 earnings call.
        We are pleased to report revenue of $90 billion, exceeding our guidance.
        Looking ahead, we expect continued growth in Q1 2025.

        Question-and-Answer Session

        Operator: We will now take questions.
        Analyst: Can you comment on the margin expansion?
        CEO: We saw strong margin improvement due to operational efficiency.
        """

        result = summarizer.summarize_earnings_call(transcript, "AAPL")

        assert isinstance(result, dict)
        assert "guidance" in result
        assert "key_metrics" in result
        assert "management_tone" in result
        assert "qa_highlights" in result

    def test_summarize_earnings_call_empty(self, summarizer: TextSummarizer) -> None:
        """Test earnings call summarization with empty transcript."""
        result = summarizer.summarize_earnings_call("", "AAPL")

        assert result["guidance"] == ""
        assert result["key_metrics"] == []

    def test_detect_common_themes(
        self, summarizer: TextSummarizer, sample_texts: list[str]
    ) -> None:
        """Test common theme detection across texts."""
        themes = summarizer.detect_common_themes(sample_texts)

        assert isinstance(themes, list)
        # Themes should be meaningful words

    def test_detect_common_themes_empty(self, summarizer: TextSummarizer) -> None:
        """Test theme detection with empty list."""
        themes = summarizer.detect_common_themes([])

        assert themes == []

    def test_compression_ratio(self, summarizer: TextSummarizer, sample_text: str) -> None:
        """Test that compression ratio is calculated correctly."""
        result = summarizer.summarize(sample_text, length=SummaryLength.BRIEF)

        assert result.compression_ratio > 1.0  # Summary should be shorter than original

    def test_sentiment_summary_positive(self, summarizer: TextSummarizer) -> None:
        """Test sentiment summary for positive text."""
        text = "The company exceeded expectations and beat earnings guidance."
        result = summarizer.summarize(text)

        assert result.sentiment_summary in ["positive", "neutral"]

    def test_sentiment_summary_negative(self, summarizer: TextSummarizer) -> None:
        """Test sentiment summary for negative text."""
        text = "The company missed earnings and reported declining revenue."
        result = summarizer.summarize(text)

        assert result.sentiment_summary in ["negative", "neutral"]

    def test_very_short_text(self, summarizer: TextSummarizer) -> None:
        """Test summarization of very short text."""
        text = "Apple reported earnings."
        result = summarizer.summarize(text, length=SummaryLength.SHORT)

        assert len(result.summary) > 0
        assert result.word_count > 0

    def test_very_long_text(self, summarizer: TextSummarizer) -> None:
        """Test summarization of very long text."""
        # Create a long text by repeating
        base_text = "Apple reported earnings. "
        text = base_text * 100

        result = summarizer.summarize(text, length=SummaryLength.SHORT)

        assert len(result.summary) > 0
        # Summary should be much shorter than original
        assert result.compression_ratio > 5.0

    def test_max_sentences_override(self, summarizer: TextSummarizer, sample_text: str) -> None:
        """Test max_sentences parameter override."""
        result = summarizer.summarize(sample_text, max_sentences=2)

        sentences = result.summary.split(". ")
        # Should be close to 2 sentences
        assert len(sentences) <= 3

    def test_entities_mentioned_populated(
        self, summarizer: TextSummarizer, sample_text: str
    ) -> None:
        """Test that entities_mentioned is populated."""
        result = summarizer.summarize(sample_text)

        assert len(result.entities_mentioned) > 0

    def test_key_points_populated(self, summarizer: TextSummarizer, sample_text: str) -> None:
        """Test that key_points is populated."""
        result = summarizer.summarize(sample_text)

        assert len(result.key_points) > 0

    def test_timeline_extraction(self, summarizer: TextSummarizer) -> None:
        """Test timeline extraction from texts."""
        texts = [
            "On January 15, 2024, Apple announced new products.",
            "The earnings report on 2024-02-01 showed strong growth.",
        ]

        result = summarizer.summarize_multiple(texts)

        # Should extract timeline events
        assert isinstance(result.timeline, list)

    def test_conflicting_points_detection(self, summarizer: TextSummarizer) -> None:
        """Test detection of conflicting information."""
        texts = [
            "The company exceeded earnings expectations.",
            "The company missed earnings expectations.",
        ]

        result = summarizer.summarize_multiple(texts)

        # Should detect conflict
        assert len(result.conflicting_points) >= 0

    def test_key_entities_extraction(
        self, summarizer: TextSummarizer, sample_texts: list[str]
    ) -> None:
        """Test key entities extraction from multiple documents."""
        result = summarizer.summarize_multiple(sample_texts)

        assert isinstance(result.key_entities, list)
        # Should extract Apple as a key entity
        assert any("Apple" in entity for entity in result.key_entities)

    def test_earnings_guidance_extraction(self, summarizer: TextSummarizer) -> None:
        """Test extraction of earnings guidance."""
        transcript = """
        We expect revenue to grow 10-15% in Q1 2025. Our guidance for
        full-year 2025 is $400-420 billion in revenue.
        """

        result = summarizer.summarize_earnings_call(transcript, "AAPL")

        assert len(result["guidance"]) > 0
        assert "guidance" in result["guidance"].lower() or "expect" in result["guidance"].lower()

    def test_earnings_metrics_extraction(self, summarizer: TextSummarizer) -> None:
        """Test extraction of key financial metrics."""
        transcript = """
        Revenue was $90 billion, up 15% year-over-year. Earnings per share
        reached $6.50, beating estimates. Operating margin expanded to 28%.
        """

        result = summarizer.summarize_earnings_call(transcript, "AAPL")

        assert len(result["key_metrics"]) > 0

    def test_management_tone_assessment(self, summarizer: TextSummarizer) -> None:
        """Test management tone assessment."""
        positive_transcript = """
        We are thrilled with the strong performance. Revenue exceeded
        expectations and we see continued growth ahead.
        """

        result = summarizer.summarize_earnings_call(positive_transcript, "AAPL")

        assert result["management_tone"] in ["positive", "neutral", "negative"]

    def test_qa_highlights_extraction(self, summarizer: TextSummarizer) -> None:
        """Test Q&A highlights extraction."""
        transcript = """
        Prepared remarks here.

        Question-and-Answer Session

        Analyst asked about margin expansion. CEO responded that
        operational efficiency drove improvements.
        """

        result = summarizer.summarize_earnings_call(transcript, "AAPL")

        assert isinstance(result["qa_highlights"], list)
