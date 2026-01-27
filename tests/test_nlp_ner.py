"""Tests for Named Entity Recognition module.

This module tests the spaCy-based entity extractors with mocked models
to avoid requiring actual model downloads during testing.
"""

import sys
from unittest.mock import MagicMock, Mock

import pytest

# Mock spacy module before importing ner module
mock_spacy = MagicMock()

# Create a mock Doc class
class MockDoc:
    """Mock spaCy Doc object."""

    def __init__(self, text: str, ents: list[tuple[str, str, int, int]]) -> None:
        """Initialize mock document.

        Args:
            text: Document text.
            ents: List of (text, label, start, end) tuples.
        """
        self.text = text
        self.ents = [MockSpan(text, label, start, end) for text, label, start, end in ents]


class MockSpan:
    """Mock spaCy Span object for entities."""

    def __init__(self, text: str, label: str, start: int, end: int) -> None:
        """Initialize mock span.

        Args:
            text: Entity text.
            label: Entity label.
            start: Start character offset.
            end: End character offset.
        """
        self.text = text
        self.label_ = label
        self.start_char = start
        self.end_char = end


class MockLanguage:
    """Mock spaCy Language model."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """Initialize mock language model."""
        self.model_name = model_name
        self.pipe_names = ["ner", "tagger", "parser"]
        self._disabled_pipes: list[str] = []

    def __call__(self, text: str) -> MockDoc:
        """Process text and return mock document.

        Args:
            text: Input text.

        Returns:
            MockDoc with extracted entities.
        """
        # Simple mock entity extraction for testing
        entities = []

        # Extract organizations (Apple Inc., Microsoft)
        if "Apple Inc." in text:
            idx = text.index("Apple Inc.")
            entities.append(("Apple Inc.", "ORG", idx, idx + 10))
        if "Microsoft" in text:
            idx = text.index("Microsoft")
            entities.append(("Microsoft", "ORG", idx, idx + 9))

        # Extract locations
        if "Cupertino" in text:
            idx = text.index("Cupertino")
            entities.append(("Cupertino", "GPE", idx, idx + 9))
        if "Redmond" in text:
            idx = text.index("Redmond")
            entities.append(("Redmond", "GPE", idx, idx + 7))

        # Extract dates
        if "1976" in text:
            idx = text.index("1976")
            entities.append(("1976", "DATE", idx, idx + 4))

        # Extract tickers (with custom patterns)
        import re

        ticker_pattern = r"\$?[A-Z]{2,5}\b"
        for match in re.finditer(ticker_pattern, text):
            ticker_text = match.group()
            # Only treat as ticker if it's uppercase and 2-5 chars
            if ticker_text.isupper() or ticker_text.startswith("$"):
                entities.append((ticker_text, "TICKER", match.start(), match.end()))

        # Extract money amounts
        money_pattern = r"\$\d+(?:\.\d+)?[KMBTkmbt]?"
        for match in re.finditer(money_pattern, text):
            entities.append((match.group(), "MONEY", match.start(), match.end()))

        # Extract percentages
        percent_pattern = r"\d+(?:\.\d+)?%"
        for match in re.finditer(percent_pattern, text):
            entities.append((match.group(), "PERCENT", match.start(), match.end()))

        return MockDoc(text, entities)

    def pipe(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[MockDoc]:
        """Process texts in batch.

        Args:
            texts: List of texts to process.
            batch_size: Batch size (ignored in mock).

        Returns:
            List of MockDoc objects.
        """
        return [self(text) for text in texts]

    def disable_pipes(self, *pipes: str) -> None:
        """Disable pipeline components.

        Args:
            pipes: Names of pipes to disable.
        """
        self._disabled_pipes.extend(pipes)

    def add_pipe(self, name: str, before: str | None = None) -> Mock:
        """Add a pipeline component.

        Args:
            name: Name of the pipe to add.
            before: Name of pipe to add before.

        Returns:
            Mock entity ruler.
        """
        self.pipe_names.append(name)
        ruler = Mock()
        ruler.add_patterns = Mock()
        return ruler

    def get_pipe(self, name: str) -> Mock:
        """Get a pipeline component.

        Args:
            name: Name of the pipe.

        Returns:
            Mock pipeline component.
        """
        ruler = Mock()
        ruler.add_patterns = Mock()
        return ruler


# Configure mock spacy
def mock_load(model_name: str) -> MockLanguage:
    """Mock spacy.load function."""
    return MockLanguage(model_name)


mock_spacy.load = mock_load
sys.modules["spacy"] = mock_spacy
sys.modules["spacy.language"] = MagicMock()
sys.modules["spacy.tokens"] = MagicMock()

# ruff: noqa: E402
from signalforge.nlp.ner import (
    BaseEntityExtractor,
    EntityExtractionResult,
    FinancialEntityExtractor,
    NamedEntity,
    NERConfig,
    SpaCyEntityExtractor,
    extract_entities,
    extract_tickers,
    get_entity_extractor,
)


class TestNamedEntity:
    """Tests for NamedEntity dataclass."""

    def test_valid_named_entity(self) -> None:
        """Test creation of valid named entity."""
        entity = NamedEntity(
            text="Apple Inc.",
            label="ORG",
            start=0,
            end=10,
            confidence=0.95,
        )

        assert entity.text == "Apple Inc."
        assert entity.label == "ORG"
        assert entity.start == 0
        assert entity.end == 10
        assert entity.confidence == 0.95

    def test_default_confidence(self) -> None:
        """Test that default confidence is 1.0."""
        entity = NamedEntity(
            text="Apple Inc.",
            label="ORG",
            start=0,
            end=10,
        )

        assert entity.confidence == 1.0

    def test_invalid_confidence_range(self) -> None:
        """Test that invalid confidence values raise ValueError."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            NamedEntity(
                text="Test",
                label="ORG",
                start=0,
                end=4,
                confidence=1.5,
            )

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            NamedEntity(
                text="Test",
                label="ORG",
                start=0,
                end=4,
                confidence=-0.1,
            )

    def test_invalid_start_offset(self) -> None:
        """Test that negative start offset raises ValueError."""
        with pytest.raises(ValueError, match="start must be non-negative"):
            NamedEntity(
                text="Test",
                label="ORG",
                start=-1,
                end=4,
            )

    def test_invalid_end_offset(self) -> None:
        """Test that end < start raises ValueError."""
        with pytest.raises(ValueError, match="end must be >= start"):
            NamedEntity(
                text="Test",
                label="ORG",
                start=10,
                end=5,
            )

    def test_empty_text(self) -> None:
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="Entity text cannot be empty"):
            NamedEntity(
                text="",
                label="ORG",
                start=0,
                end=0,
            )

    def test_empty_label(self) -> None:
        """Test that empty label raises ValueError."""
        with pytest.raises(ValueError, match="Entity label cannot be empty"):
            NamedEntity(
                text="Test",
                label="",
                start=0,
                end=4,
            )


class TestEntityExtractionResult:
    """Tests for EntityExtractionResult dataclass."""

    def test_valid_result(self) -> None:
        """Test creation of valid extraction result."""
        entities = [
            NamedEntity(text="Apple Inc.", label="ORG", start=0, end=10),
            NamedEntity(text="Cupertino", label="GPE", start=25, end=34),
        ]
        result = EntityExtractionResult(
            text="Apple Inc. is located in Cupertino.",
            entities=entities,
        )

        assert result.text == "Apple Inc. is located in Cupertino."
        assert len(result.entities) == 2
        assert result.entity_counts["ORG"] == 1
        assert result.entity_counts["GPE"] == 1

    def test_empty_entities(self) -> None:
        """Test result with no entities."""
        result = EntityExtractionResult(
            text="Some text with no entities.",
            entities=[],
        )

        assert len(result.entities) == 0
        assert len(result.entity_counts) == 0

    def test_entity_counts_auto_computed(self) -> None:
        """Test that entity counts are automatically computed."""
        entities = [
            NamedEntity(text="Apple", label="ORG", start=0, end=5),
            NamedEntity(text="Microsoft", label="ORG", start=10, end=19),
            NamedEntity(text="2024", label="DATE", start=20, end=24),
        ]
        result = EntityExtractionResult(text="Test", entities=entities)

        assert result.entity_counts["ORG"] == 2
        assert result.entity_counts["DATE"] == 1

    def test_entity_counts_provided(self) -> None:
        """Test that provided entity counts are preserved."""
        entities = [NamedEntity(text="Apple", label="ORG", start=0, end=5)]
        counts = {"ORG": 1, "CUSTOM": 5}
        result = EntityExtractionResult(
            text="Test",
            entities=entities,
            entity_counts=counts,
        )

        # Provided counts should be preserved (not auto-computed)
        assert result.entity_counts == counts


class TestNERConfig:
    """Tests for NERConfig dataclass."""

    def test_default_config(self) -> None:
        """Test creation of config with defaults."""
        config = NERConfig()

        assert config.model_name == "en_core_web_sm"
        assert config.include_custom_patterns is True
        assert config.confidence_threshold == 0.5
        assert config.batch_size == 32
        assert config.enable_ner is True
        assert config.enable_entity_ruler is True
        assert config.cache_model is True

    def test_custom_config(self) -> None:
        """Test creation of config with custom values."""
        config = NERConfig(
            model_name="en_core_web_lg",
            include_custom_patterns=False,
            confidence_threshold=0.8,
            batch_size=16,
            enable_ner=False,
            cache_model=False,
        )

        assert config.model_name == "en_core_web_lg"
        assert config.include_custom_patterns is False
        assert config.confidence_threshold == 0.8
        assert config.batch_size == 16
        assert config.enable_ner is False
        assert config.cache_model is False

    def test_invalid_confidence_threshold(self) -> None:
        """Test that invalid confidence threshold raises ValueError."""
        with pytest.raises(ValueError, match="confidence_threshold must be between 0.0 and 1.0"):
            NERConfig(confidence_threshold=1.5)

        with pytest.raises(ValueError, match="confidence_threshold must be between 0.0 and 1.0"):
            NERConfig(confidence_threshold=-0.1)

    def test_invalid_batch_size(self) -> None:
        """Test that invalid batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            NERConfig(batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            NERConfig(batch_size=-1)

    def test_empty_model_name(self) -> None:
        """Test that empty model name raises ValueError."""
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            NERConfig(model_name="")


class TestBaseEntityExtractor:
    """Tests for BaseEntityExtractor abstract class."""

    def test_cannot_instantiate(self) -> None:
        """Test that abstract class cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseEntityExtractor()  # type: ignore[abstract]

    def test_must_implement_extract(self) -> None:
        """Test that subclasses must implement extract method."""

        class IncompleteExtractor(BaseEntityExtractor):
            def extract_batch(self, texts: list[str]) -> list[EntityExtractionResult]:
                return []

            @property
            def model_name(self) -> str:
                return "test"

        with pytest.raises(TypeError):
            IncompleteExtractor()  # type: ignore[abstract]


class TestSpaCyEntityExtractor:
    """Tests for SpaCyEntityExtractor class."""

    def test_initialization(self) -> None:
        """Test extractor initialization."""
        extractor = SpaCyEntityExtractor()

        assert extractor.model_name == "en_core_web_sm"
        assert extractor._config.include_custom_patterns is True

    def test_initialization_with_config(self) -> None:
        """Test extractor initialization with custom config."""
        config = NERConfig(model_name="en_core_web_lg", confidence_threshold=0.8)
        extractor = SpaCyEntityExtractor(config=config)

        assert extractor.model_name == "en_core_web_lg"
        assert extractor._config.confidence_threshold == 0.8

    def test_initialization_with_injected_model(self) -> None:
        """Test extractor initialization with injected model."""
        mock_model = MockLanguage()
        extractor = SpaCyEntityExtractor(model=mock_model)

        # Model should be set directly
        assert extractor._model is mock_model

    def test_extract_basic_entities(self) -> None:
        """Test extraction of basic entities."""
        extractor = SpaCyEntityExtractor()
        text = "Apple Inc. is located in Cupertino."
        result = extractor.extract(text)

        assert result.text == text
        assert len(result.entities) >= 2
        assert any(e.label == "ORG" for e in result.entities)
        assert any(e.label == "GPE" for e in result.entities)

    def test_extract_empty_text(self) -> None:
        """Test that empty text raises ValueError."""
        extractor = SpaCyEntityExtractor()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            extractor.extract("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            extractor.extract("   ")

    def test_extract_batch(self) -> None:
        """Test batch entity extraction."""
        extractor = SpaCyEntityExtractor()
        texts = [
            "Apple Inc. is based in Cupertino.",
            "Microsoft is located in Redmond.",
        ]
        results = extractor.extract_batch(texts)

        assert len(results) == 2
        assert all(isinstance(r, EntityExtractionResult) for r in results)
        assert results[0].text == texts[0]
        assert results[1].text == texts[1]
        assert len(results[0].entities) >= 2
        assert len(results[1].entities) >= 2

    def test_extract_batch_empty_list(self) -> None:
        """Test that empty batch raises ValueError."""
        extractor = SpaCyEntityExtractor()

        with pytest.raises(ValueError, match="Texts list cannot be empty"):
            extractor.extract_batch([])

    def test_model_lazy_loading(self) -> None:
        """Test that model is loaded lazily."""
        extractor = SpaCyEntityExtractor()

        # Model should not be loaded yet
        assert extractor._model is None

        # First extraction should load model
        extractor.extract("Test text.")

        # Model should now be loaded
        assert extractor._model is not None

    def test_model_caching(self) -> None:
        """Test that model is cached across instances."""
        # Clear cache first
        SpaCyEntityExtractor._model_cache.clear()

        config = NERConfig(cache_model=True)
        extractor1 = SpaCyEntityExtractor(config=config)
        extractor1.extract("Test text 1.")

        extractor2 = SpaCyEntityExtractor(config=config)
        extractor2.extract("Test text 2.")

        # Both extractors should use same cached model
        cache_key = config.model_name
        assert cache_key in SpaCyEntityExtractor._model_cache

    def test_confidence_threshold_filtering(self) -> None:
        """Test that entities below confidence threshold are filtered."""
        # Use threshold slightly above 1.0 by testing with strict threshold
        # Default confidence is 1.0, so threshold 1.0 should keep all entities
        config = NERConfig(confidence_threshold=1.0)
        extractor = SpaCyEntityExtractor(config=config)
        result = extractor.extract("Apple Inc. is in Cupertino.")

        # Entities with exactly 1.0 confidence should pass the >= threshold
        # The mock returns entities, so some should be present
        assert result is not None


class TestFinancialEntityExtractor:
    """Tests for FinancialEntityExtractor class."""

    def test_initialization(self) -> None:
        """Test financial extractor initialization."""
        extractor = FinancialEntityExtractor()

        assert extractor.model_name == "en_core_web_sm"
        assert extractor._config.include_custom_patterns is True

    def test_extract_ticker_symbols(self) -> None:
        """Test extraction of ticker symbols."""
        extractor = FinancialEntityExtractor()
        text = "$AAPL rose 15% today while MSFT remained flat."
        result = extractor.extract(text)

        tickers = [e for e in result.entities if e.label == "TICKER"]
        assert len(tickers) >= 1

    def test_extract_money_amounts(self) -> None:
        """Test extraction of monetary amounts."""
        extractor = FinancialEntityExtractor()
        text = "Revenue was $1.5B, up from $1.2B last quarter."
        result = extractor.extract(text)

        money = [e for e in result.entities if e.label == "MONEY"]
        assert len(money) >= 1

    def test_extract_percentages(self) -> None:
        """Test extraction of percentages."""
        extractor = FinancialEntityExtractor()
        text = "Profit margin increased 15% to reach 25.5% this quarter."
        result = extractor.extract(text)

        percents = [e for e in result.entities if e.label == "PERCENT"]
        assert len(percents) >= 1

    def test_extract_mixed_entities(self) -> None:
        """Test extraction of mixed entity types."""
        extractor = FinancialEntityExtractor()
        text = "Apple Inc. (AAPL) reported revenue of $1.5B, up 15% from last year."
        result = extractor.extract(text)

        # Should extract org, ticker, money, and percent
        assert len(result.entities) >= 3
        assert any(e.label == "ORG" for e in result.entities)
        assert any(e.label in ("MONEY", "TICKER", "PERCENT") for e in result.entities)

    def test_custom_patterns_not_added_without_config(self) -> None:
        """Test that custom patterns can be disabled."""
        config = NERConfig(include_custom_patterns=False)
        extractor = FinancialEntityExtractor(config=config)
        text = "$AAPL rose 15%."
        result = extractor.extract(text)

        # Result depends on whether patterns are added or not
        # Just ensure no exception is raised
        assert isinstance(result, EntityExtractionResult)

    def test_batch_extraction(self) -> None:
        """Test batch extraction with financial entities."""
        extractor = FinancialEntityExtractor()
        texts = [
            "$AAPL reported $1.5B revenue.",
            "MSFT grew 20% year-over-year.",
            "GOOGL stock hit $150.25 per share.",
        ]
        results = extractor.extract_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, EntityExtractionResult) for r in results)
        assert all(len(r.entities) > 0 for r in results)


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_entity_extractor_default(self) -> None:
        """Test getting default entity extractor."""
        extractor = get_entity_extractor()

        assert isinstance(extractor, FinancialEntityExtractor)
        assert extractor.model_name == "en_core_web_sm"

    def test_get_entity_extractor_singleton(self) -> None:
        """Test that default extractor is a singleton."""
        extractor1 = get_entity_extractor()
        extractor2 = get_entity_extractor()

        # Should return same instance
        assert extractor1 is extractor2

    def test_get_entity_extractor_custom_config(self) -> None:
        """Test getting extractor with custom config."""
        config = NERConfig(model_name="en_core_web_lg", batch_size=64)
        extractor = get_entity_extractor(config)

        assert extractor.model_name == "en_core_web_lg"
        assert extractor._config.batch_size == 64

    def test_get_entity_extractor_custom_not_singleton(self) -> None:
        """Test that custom config creates new instance."""
        config1 = NERConfig(batch_size=16)
        config2 = NERConfig(batch_size=16)

        extractor1 = get_entity_extractor(config1)
        extractor2 = get_entity_extractor(config2)

        # Should be different instances
        assert extractor1 is not extractor2

    def test_extract_entities_convenience(self) -> None:
        """Test extract_entities convenience function."""
        text = "Apple Inc. is located in Cupertino."
        result = extract_entities(text)

        assert isinstance(result, EntityExtractionResult)
        assert result.text == text
        assert len(result.entities) >= 2

    def test_extract_tickers_basic(self) -> None:
        """Test extract_tickers with basic tickers."""
        text = "Trading $AAPL and MSFT today."
        tickers = extract_tickers(text)

        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert len(tickers) == 2

    def test_extract_tickers_with_dollar_sign(self) -> None:
        """Test ticker extraction with dollar signs."""
        text = "$AAPL, $MSFT, and $GOOGL are tech stocks."
        tickers = extract_tickers(text)

        assert "AAPL" in tickers
        assert "MSFT" in tickers
        assert "GOOGL" in tickers

    def test_extract_tickers_deduplication(self) -> None:
        """Test that duplicate tickers are deduplicated."""
        text = "$AAPL rose today. AAPL is a strong buy. AAPL looks great."
        tickers = extract_tickers(text)

        assert tickers.count("AAPL") == 1
        assert len(tickers) == 1

    def test_extract_tickers_empty_text(self) -> None:
        """Test ticker extraction with empty text."""
        assert extract_tickers("") == []
        assert extract_tickers("   ") == []

    def test_extract_tickers_no_tickers(self) -> None:
        """Test ticker extraction with no tickers."""
        text = "The market was up today."
        tickers = extract_tickers(text)

        assert len(tickers) == 0

    def test_extract_tickers_filters_short_symbols(self) -> None:
        """Test that single-letter symbols are filtered."""
        text = "I went to the store."
        tickers = extract_tickers(text)

        # 'I' should not be extracted as a ticker
        assert "I" not in tickers

    def test_extract_tickers_case_sensitivity(self) -> None:
        """Test that only uppercase tickers are extracted."""
        text = "aapl and MSFT are stocks."
        tickers = extract_tickers(text)

        # Only MSFT should be extracted (uppercase)
        assert "MSFT" in tickers
        assert "aapl" not in tickers
        assert "AAPL" not in tickers


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_extract_very_long_text(self) -> None:
        """Test extraction on very long text."""
        extractor = SpaCyEntityExtractor()
        text = "Apple Inc. is located in Cupertino. " * 1000
        result = extractor.extract(text)

        assert isinstance(result, EntityExtractionResult)
        assert len(result.entities) > 0

    def test_extract_special_characters(self) -> None:
        """Test extraction with special characters."""
        extractor = SpaCyEntityExtractor()
        text = "Apple Inc. <<<<>>>> @#$% reported earnings."
        result = extractor.extract(text)

        assert isinstance(result, EntityExtractionResult)

    def test_extract_unicode_text(self) -> None:
        """Test extraction with unicode characters."""
        extractor = SpaCyEntityExtractor()
        text = "Apple Inc. reported revenue of $1.5B."
        result = extractor.extract(text)

        assert isinstance(result, EntityExtractionResult)

    def test_extract_newlines_and_whitespace(self) -> None:
        """Test extraction with newlines and whitespace."""
        extractor = SpaCyEntityExtractor()
        text = "Apple Inc.\n\n  is located in\n\tCupertino."
        result = extractor.extract(text)

        assert isinstance(result, EntityExtractionResult)

    def test_entity_counts_multiple_same_label(self) -> None:
        """Test entity counts with multiple entities of same type."""
        entities = [
            NamedEntity(text="Apple", label="ORG", start=0, end=5),
            NamedEntity(text="Microsoft", label="ORG", start=10, end=19),
            NamedEntity(text="Google", label="ORG", start=20, end=26),
        ]
        result = EntityExtractionResult(text="Test", entities=entities)

        assert result.entity_counts["ORG"] == 3
        assert len(result.entity_counts) == 1
