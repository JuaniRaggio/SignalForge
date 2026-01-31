"""Tests for topic extraction module."""

from unittest.mock import MagicMock, patch

import pytest

from signalforge.nlp.topics import (
    DEFAULT_DIVERSITY,
    DEFAULT_NGRAM_RANGE,
    DEFAULT_TOP_N,
    BaseTopicExtractor,
    EmbeddingTopicExtractor,
    TopicExtractionConfig,
    TopicExtractionResult,
    TopicKeyword,
    extract_keyphrases,
    extract_topics,
    get_topic_extractor,
)


class TestTopicKeyword:
    """Tests for TopicKeyword dataclass."""

    def test_valid_keyword(self) -> None:
        """Test creating a valid keyword."""
        keyword = TopicKeyword(keyword="earnings growth", score=0.85)
        assert keyword.keyword == "earnings growth"
        assert keyword.score == 0.85

    def test_empty_keyword_raises_error(self) -> None:
        """Test that empty keyword raises ValueError."""
        with pytest.raises(ValueError, match="Keyword cannot be empty"):
            TopicKeyword(keyword="", score=0.5)

    def test_whitespace_only_keyword_raises_error(self) -> None:
        """Test that whitespace-only keyword raises ValueError."""
        with pytest.raises(ValueError, match="Keyword cannot be empty"):
            TopicKeyword(keyword="   ", score=0.5)

    def test_invalid_score_below_zero_raises_error(self) -> None:
        """Test that score below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            TopicKeyword(keyword="test", score=-0.1)

    def test_invalid_score_above_one_raises_error(self) -> None:
        """Test that score above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Score must be between 0.0 and 1.0"):
            TopicKeyword(keyword="test", score=1.5)

    def test_edge_case_scores(self) -> None:
        """Test edge case scores (0.0 and 1.0)."""
        keyword_min = TopicKeyword(keyword="test", score=0.0)
        assert keyword_min.score == 0.0

        keyword_max = TopicKeyword(keyword="test", score=1.0)
        assert keyword_max.score == 1.0


class TestTopicExtractionResult:
    """Tests for TopicExtractionResult dataclass."""

    def test_valid_result(self) -> None:
        """Test creating a valid result."""
        keywords = [
            TopicKeyword(keyword="earnings", score=0.9),
            TopicKeyword(keyword="revenue", score=0.8),
        ]
        result = TopicExtractionResult(
            text="Test text",
            keywords=keywords,
            top_n=5,
        )
        assert result.text == "Test text"
        assert len(result.keywords) == 2
        assert result.top_n == 5

    def test_empty_keywords_list(self) -> None:
        """Test result with empty keywords list."""
        result = TopicExtractionResult(
            text="Test text",
            keywords=[],
            top_n=5,
        )
        assert len(result.keywords) == 0

    def test_invalid_top_n_raises_error(self) -> None:
        """Test that top_n <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="top_n must be positive"):
            TopicExtractionResult(
                text="Test text",
                keywords=[],
                top_n=0,
            )


class TestTopicExtractionConfig:
    """Tests for TopicExtractionConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = TopicExtractionConfig()
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.top_n == DEFAULT_TOP_N
        assert config.keyphrase_ngram_range == DEFAULT_NGRAM_RANGE
        assert config.diversity == DEFAULT_DIVERSITY
        assert config.use_mmr is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = TopicExtractionConfig(
            model_name="custom-model",
            top_n=10,
            keyphrase_ngram_range=(1, 3),
            diversity=0.7,
            use_mmr=False,
        )
        assert config.model_name == "custom-model"
        assert config.top_n == 10
        assert config.keyphrase_ngram_range == (1, 3)
        assert config.diversity == 0.7
        assert config.use_mmr is False

    def test_invalid_top_n_raises_error(self) -> None:
        """Test that invalid top_n raises ValueError."""
        with pytest.raises(ValueError, match="top_n must be positive"):
            TopicExtractionConfig(top_n=0)

    def test_invalid_ngram_range_min_raises_error(self) -> None:
        """Test that invalid ngram min raises ValueError."""
        with pytest.raises(ValueError, match="min_n must be at least 1"):
            TopicExtractionConfig(keyphrase_ngram_range=(0, 2))

    def test_invalid_ngram_range_max_raises_error(self) -> None:
        """Test that invalid ngram max raises ValueError."""
        with pytest.raises(ValueError, match="max_n must be >= min_n"):
            TopicExtractionConfig(keyphrase_ngram_range=(3, 2))

    def test_invalid_diversity_below_zero_raises_error(self) -> None:
        """Test that diversity below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="diversity must be between 0.0 and 1.0"):
            TopicExtractionConfig(diversity=-0.1)

    def test_invalid_diversity_above_one_raises_error(self) -> None:
        """Test that diversity above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="diversity must be between 0.0 and 1.0"):
            TopicExtractionConfig(diversity=1.5)

    def test_edge_case_diversity(self) -> None:
        """Test edge case diversity values."""
        config_min = TopicExtractionConfig(diversity=0.0)
        assert config_min.diversity == 0.0

        config_max = TopicExtractionConfig(diversity=1.0)
        assert config_max.diversity == 1.0


class TestBaseTopicExtractor:
    """Tests for BaseTopicExtractor abstract class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseTopicExtractor()  # type: ignore[abstract]


class MockEmbedder:
    """Mock embedder for testing."""

    def __init__(self) -> None:
        """Initialize mock embedder."""
        self.dimension = 384
        self.model_name = "mock-model"

    def encode(self, text: str) -> MagicMock:
        """Mock encode method."""
        result = MagicMock()
        result.text = text
        # Generate deterministic embedding based on text hash
        result.embedding = [float(hash(text) % 100) / 100.0] * self.dimension
        result.model_name = self.model_name
        result.dimension = self.dimension
        return result

    def encode_batch(self, texts: list[str]) -> list[MagicMock]:
        """Mock encode_batch method."""
        return [self.encode(text) for text in texts]


class TestEmbeddingTopicExtractor:
    """Tests for EmbeddingTopicExtractor class."""

    @pytest.fixture
    def mock_embedder(self) -> MockEmbedder:
        """Fixture for mock embedder."""
        return MockEmbedder()

    @pytest.fixture
    def extractor(self, mock_embedder: MockEmbedder) -> EmbeddingTopicExtractor:
        """Fixture for extractor with mocked embedder."""
        config = TopicExtractionConfig(top_n=5, use_mmr=False)
        extractor = EmbeddingTopicExtractor(config)
        extractor._embedder = mock_embedder  # type: ignore[assignment]
        return extractor

    def test_initialization(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test extractor initialization."""
        assert extractor._config.top_n == 5
        assert extractor._config.use_mmr is False
        assert extractor._embedder is not None

    def test_extract_candidates_simple_text(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test candidate extraction from simple text."""
        text = "Apple reported strong earnings growth."
        candidates = extractor._extract_candidates(text)

        assert len(candidates) > 0
        assert any("Apple" in c for c in candidates)
        assert any("earnings" in c.lower() for c in candidates)

    def test_extract_candidates_filters_stopwords(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test that candidates filter out stopword-only n-grams."""
        text = "The company is a leader."
        candidates = extractor._extract_candidates(text)

        # Should not contain pure stopwords
        assert "the" not in [c.lower() for c in candidates]
        assert "is" not in [c.lower() for c in candidates]
        assert "a" not in [c.lower() for c in candidates]

        # Should contain meaningful words
        assert any("company" in c.lower() for c in candidates)
        assert any("leader" in c.lower() for c in candidates)

    def test_tokenize_preserves_financial_symbols(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test that tokenization preserves financial symbols."""
        text = "Stock price $150 increased 15%"
        tokens = extractor._tokenize(text)

        assert "$150" in tokens or "$" in tokens or "150" in tokens
        assert "15%" in tokens or "15" in tokens

    def test_is_valid_candidate(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test candidate validation."""
        # Valid candidates
        assert extractor._is_valid_candidate("earnings growth")
        assert extractor._is_valid_candidate("Apple")
        assert extractor._is_valid_candidate("15%")
        assert extractor._is_valid_candidate("$150")

        # Invalid candidates
        assert not extractor._is_valid_candidate("")
        assert not extractor._is_valid_candidate("   ")
        assert not extractor._is_valid_candidate("...")
        assert not extractor._is_valid_candidate("123")  # Pure number without %
        assert not extractor._is_valid_candidate("a")  # Too short

    def test_extract_simple_text(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test extraction from simple text."""
        text = "Apple reported strong earnings growth with revenue increasing significantly."
        result = extractor.extract(text)

        assert result.text == text
        assert len(result.keywords) > 0
        assert result.top_n == 5
        assert all(isinstance(kw, TopicKeyword) for kw in result.keywords)
        assert all(0.0 <= kw.score <= 1.0 for kw in result.keywords)

    def test_extract_empty_text_raises_error(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            extractor.extract("")

    def test_extract_whitespace_only_raises_error(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            extractor.extract("   ")

    def test_extract_respects_top_n(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test that extraction respects top_n configuration."""
        text = "Apple reported strong earnings growth with revenue increasing significantly."
        result = extractor.extract(text)

        assert len(result.keywords) <= extractor._config.top_n

    def test_extract_with_mmr(self, mock_embedder: MockEmbedder) -> None:
        """Test extraction with MMR enabled."""
        config = TopicExtractionConfig(top_n=3, use_mmr=True, diversity=0.7)
        extractor = EmbeddingTopicExtractor(config)
        extractor._embedder = mock_embedder  # type: ignore[assignment]

        text = "Apple reported strong earnings growth with revenue increasing significantly."
        result = extractor.extract(text)

        assert len(result.keywords) > 0
        assert len(result.keywords) <= 3

    def test_maximal_marginal_relevance(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test MMR algorithm."""
        doc_embedding = [0.5] * 384
        candidate_embeddings = [
            ("keyword1", [0.6] * 384),
            ("keyword2", [0.5] * 384),
            ("keyword3", [0.4] * 384),
            ("keyword4", [0.3] * 384),
        ]

        selected = extractor._maximal_marginal_relevance(
            doc_embedding=doc_embedding,
            candidate_embeddings=candidate_embeddings,
            top_n=3,
            diversity=0.5,
        )

        assert len(selected) == 3
        assert all(isinstance(item, tuple) for item in selected)
        assert all(len(item) == 2 for item in selected)

    def test_extract_batch(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test batch extraction."""
        texts = [
            "Apple reported strong earnings.",
            "Tesla stock price surged.",
            "Market volatility increased.",
        ]
        results = extractor.extract_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, TopicExtractionResult) for r in results)
        assert all(r.text == text for r, text in zip(results, texts, strict=True))

    def test_extract_batch_empty_list_raises_error(
        self, extractor: EmbeddingTopicExtractor
    ) -> None:
        """Test that empty texts list raises ValueError."""
        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            extractor.extract_batch([])

    def test_extract_batch_with_empty_text(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test batch extraction with empty text."""
        texts = ["Apple reported earnings.", "", "Tesla stock surged."]
        results = extractor.extract_batch(texts)

        assert len(results) == 3
        assert len(results[1].keywords) == 0  # Empty text should have no keywords

    def test_extract_no_candidates(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test extraction when no valid candidates are found."""
        text = "a the is"  # Only stopwords
        result = extractor.extract(text)

        assert result.text == text
        assert len(result.keywords) == 0


class TestFactoryFunctions:
    """Tests for factory and convenience functions."""

    @patch("signalforge.nlp.topics.EmbeddingTopicExtractor")
    def test_get_topic_extractor_default(self, mock_extractor_class: MagicMock) -> None:
        """Test getting default extractor instance."""
        # Reset singleton
        import signalforge.nlp.topics as topics_module

        topics_module._default_extractor = None

        get_topic_extractor()
        mock_extractor_class.assert_called_once()

    @patch("signalforge.nlp.topics.EmbeddingTopicExtractor")
    def test_get_topic_extractor_custom_config(self, mock_extractor_class: MagicMock) -> None:
        """Test getting extractor with custom config."""
        config = TopicExtractionConfig(top_n=10)
        get_topic_extractor(config)
        mock_extractor_class.assert_called_once_with(config)

    def test_extract_topics_convenience(self) -> None:
        """Test convenience function for topic extraction."""
        with patch("signalforge.nlp.topics.get_topic_extractor") as mock_get:
            mock_extractor = MagicMock()
            mock_result = TopicExtractionResult(
                text="test",
                keywords=[TopicKeyword(keyword="test", score=0.9)],
                top_n=5,
            )
            mock_extractor.extract.return_value = mock_result
            mock_get.return_value = mock_extractor

            result = extract_topics("Test text")

            mock_get.assert_called_once()
            mock_extractor.extract.assert_called_once_with("Test text")
            assert result == mock_result

    def test_extract_keyphrases_convenience(self) -> None:
        """Test convenience function for keyphrase extraction."""
        with patch("signalforge.nlp.topics.extract_topics") as mock_extract:
            mock_result = TopicExtractionResult(
                text="test",
                keywords=[
                    TopicKeyword(keyword="earnings", score=0.9),
                    TopicKeyword(keyword="revenue", score=0.8),
                ],
                top_n=5,
            )
            mock_extract.return_value = mock_result

            keyphrases = extract_keyphrases("Test text", top_n=3)

            assert keyphrases == ["earnings", "revenue"]

    def test_extract_keyphrases_invalid_top_n(self) -> None:
        """Test that invalid top_n raises ValueError."""
        with pytest.raises(ValueError, match="top_n must be positive"):
            extract_keyphrases("Test text", top_n=0)


class TestIntegration:
    """Integration tests with actual embedding models (if available)."""

    @pytest.mark.skip(reason="Requires actual embedding model - slow test")
    def test_full_extraction_pipeline(self) -> None:
        """Test full extraction pipeline with real embeddings."""
        text = """
        Apple Inc. reported strong quarterly earnings today, beating analyst expectations.
        Revenue increased by 15% year-over-year, driven by robust iPhone sales.
        The company's stock price surged 5% in after-hours trading.
        """

        config = TopicExtractionConfig(top_n=5, use_mmr=True, diversity=0.6)
        extractor = EmbeddingTopicExtractor(config)

        result = extractor.extract(text)

        assert len(result.keywords) > 0
        assert len(result.keywords) <= 5

        # Check that keywords are relevant
        keywords_text = " ".join(kw.keyword.lower() for kw in result.keywords)
        assert any(
            term in keywords_text
            for term in ["earnings", "revenue", "apple", "stock", "iphone", "sales"]
        )

    @pytest.mark.skip(reason="Requires actual embedding model - slow test")
    def test_batch_extraction_pipeline(self) -> None:
        """Test batch extraction pipeline with real embeddings."""
        texts = [
            "Apple reported record earnings in Q4.",
            "Tesla stock surged after positive delivery numbers.",
            "Federal Reserve raised interest rates by 0.25%.",
        ]

        extractor = EmbeddingTopicExtractor()
        results = extractor.extract_batch(texts)

        assert len(results) == 3
        assert all(len(r.keywords) > 0 for r in results)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def extractor(self) -> EmbeddingTopicExtractor:
        """Fixture for extractor with mock embedder."""
        config = TopicExtractionConfig(top_n=3)
        extractor = EmbeddingTopicExtractor(config)
        extractor._embedder = MockEmbedder()  # type: ignore[assignment]
        return extractor

    def test_very_short_text(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test extraction from very short text."""
        text = "Apple"
        result = extractor.extract(text)

        assert result.text == text
        # May or may not have keywords depending on candidate extraction

    def test_very_long_text(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test extraction from very long text."""
        text = "Apple reported earnings. " * 1000
        result = extractor.extract(text)

        assert result.text == text
        assert len(result.keywords) <= extractor._config.top_n

    def test_text_with_special_characters(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test extraction from text with special characters."""
        text = "Apple's Q4 earnings: $50B revenue (up 15%) @company"
        result = extractor.extract(text)

        assert result.text == text

    def test_text_with_numbers_and_percentages(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test extraction from text with numbers and percentages."""
        text = "Revenue increased 15% to $50 billion in Q4 2023"
        result = extractor.extract(text)

        assert result.text == text
        # Should extract meaningful phrases, not just numbers

    def test_unicode_text(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test extraction from text with unicode characters."""
        text = "Apple reported strong earnings in 2023. Revenue increased significantly."
        result = extractor.extract(text)

        assert result.text == text

    def test_mixed_case_text(self, extractor: EmbeddingTopicExtractor) -> None:
        """Test extraction from mixed case text."""
        text = "APPLE reported STRONG EARNINGS growth"
        result = extractor.extract(text)

        assert result.text == text
        assert len(result.keywords) > 0
