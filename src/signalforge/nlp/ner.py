"""Named Entity Recognition for financial text using spaCy.

This module provides NER capabilities specifically tuned for financial text,
including extraction of standard entities (people, organizations, dates) and
financial-specific entities (tickers, money values, percentages).

Key Features:
- spaCy-based entity extraction with lazy model loading
- Custom financial entity patterns (tickers, money, percentages)
- Batch processing support
- Mock-friendly design for testing
- Confidence scores for entities

Examples:
    Basic entity extraction:

    >>> from signalforge.nlp.ner import extract_entities
    >>>
    >>> text = "Apple Inc. (AAPL) reported revenue of $1.5B, up 15% from last year."
    >>> result = extract_entities(text)
    >>> print(f"Found {len(result.entities)} entities")
    Found 5 entities

    Extract only ticker symbols:

    >>> from signalforge.nlp.ner import extract_tickers
    >>> text = "Trading $AAPL and MSFT today."
    >>> tickers = extract_tickers(text)
    >>> print(tickers)
    ['AAPL', 'MSFT']

    Batch processing with custom configuration:

    >>> from signalforge.nlp.ner import get_entity_extractor, NERConfig
    >>> config = NERConfig(include_custom_patterns=True, confidence_threshold=0.8)
    >>> extractor = get_entity_extractor(config)
    >>> texts = ["Market update...", "Earnings report..."]
    >>> results = extractor.extract_batch(texts)
"""

from __future__ import annotations

import re
import threading
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Span

logger = get_logger(__name__)

# Model constants
DEFAULT_MODEL_NAME = "en_core_web_sm"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 32

# Financial entity patterns
TICKER_PATTERN: Final[str] = r"\$?[A-Z]{1,5}\b"
MONEY_PATTERN: Final[str] = r"\$?\d+(?:\.\d+)?(?:[KMBTkmbt])?(?:\s*(?:USD|EUR|GBP|dollars?|euros?|pounds?))?"
PERCENT_PATTERN: Final[str] = r"\d+(?:\.\d+)?(?:\s*)(?:%|percent|percentage|pct)"


@dataclass
class NamedEntity:
    """Represents a single named entity extracted from text.

    Attributes:
        text: The entity text as it appears in the document.
        label: Entity type label (PERSON, ORG, TICKER, MONEY, etc.).
        start: Character offset where entity starts in the document.
        end: Character offset where entity ends in the document.
        confidence: Confidence score for the entity (0.0 to 1.0).
    """

    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate entity fields."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        if self.start < 0:
            raise ValueError(f"start must be non-negative, got {self.start}")

        if self.end < self.start:
            raise ValueError(f"end must be >= start, got start={self.start}, end={self.end}")

        if not self.text:
            raise ValueError("Entity text cannot be empty")

        if not self.label:
            raise ValueError("Entity label cannot be empty")


@dataclass
class EntityExtractionResult:
    """Result of entity extraction on a text document.

    Attributes:
        text: Original input text that was analyzed.
        entities: List of extracted entities.
        entity_counts: Count of entities by label type.
    """

    text: str
    entities: list[NamedEntity]
    entity_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Populate entity counts if not provided."""
        if not self.entity_counts and self.entities:
            counter: Counter[str] = Counter(entity.label for entity in self.entities)
            self.entity_counts = dict(counter)


@dataclass
class NERConfig:
    """Configuration for Named Entity Recognition.

    Attributes:
        model_name: spaCy model name to use (e.g., "en_core_web_sm").
        include_custom_patterns: Whether to include custom financial patterns.
        confidence_threshold: Minimum confidence score to include entity.
        batch_size: Number of texts to process in parallel.
        enable_ner: Enable standard NER pipeline component.
        enable_entity_ruler: Enable entity ruler for custom patterns.
        cache_model: Whether to cache the loaded model in memory.
    """

    model_name: str = DEFAULT_MODEL_NAME
    include_custom_patterns: bool = True
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    batch_size: int = DEFAULT_BATCH_SIZE
    enable_ner: bool = True
    enable_entity_ruler: bool = True
    cache_model: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.confidence_threshold}"
            )

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if not self.model_name:
            raise ValueError("model_name cannot be empty")


class BaseEntityExtractor(ABC):
    """Abstract base class for entity extractors.

    This class defines the interface that all entity extractors must implement.
    """

    @abstractmethod
    def extract(self, text: str) -> EntityExtractionResult:
        """Extract entities from a single text.

        Args:
            text: Input text to analyze.

        Returns:
            EntityExtractionResult containing extracted entities.

        Raises:
            ValueError: If text is empty or invalid.
        """
        pass

    @abstractmethod
    def extract_batch(self, texts: list[str]) -> list[EntityExtractionResult]:
        """Extract entities from multiple texts in batch.

        Args:
            texts: List of input texts to analyze.

        Returns:
            List of EntityExtractionResult objects, one per input text.

        Raises:
            ValueError: If texts list is empty or contains invalid entries.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the underlying model."""
        pass


class SpaCyEntityExtractor(BaseEntityExtractor):
    """spaCy-based entity extractor for general text.

    This extractor uses spaCy's pre-trained NER models to extract standard
    entities like PERSON, ORG, GPE, DATE, etc. The model is lazy-loaded on
    first use to avoid unnecessary initialization overhead.

    Examples:
        >>> extractor = SpaCyEntityExtractor()
        >>> result = extractor.extract("Apple Inc. is located in Cupertino.")
        >>> print([(e.text, e.label) for e in result.entities])
        [('Apple Inc.', 'ORG'), ('Cupertino', 'GPE')]
    """

    _model_cache: dict[str, Language] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        config: NERConfig | None = None,
        model: Language | None = None,
    ) -> None:
        """Initialize the spaCy entity extractor.

        Args:
            config: Configuration for the extractor. If None, uses defaults.
            model: Pre-loaded spaCy model for dependency injection (testing).
                   If provided, model will not be loaded from disk.
        """
        self._config = config or NERConfig()
        self._model: Language | None = model
        self._custom_patterns_added = False

        logger.info(
            "spacy_extractor_initialized",
            model_name=self._config.model_name,
            include_custom_patterns=self._config.include_custom_patterns,
            confidence_threshold=self._config.confidence_threshold,
        )

    @property
    def model_name(self) -> str:
        """Get the name of the underlying model."""
        return self._config.model_name

    def _load_model(self) -> Language:
        """Load the spaCy model.

        Returns:
            Loaded spaCy Language model.

        Raises:
            RuntimeError: If model loading fails.
        """
        # Check cache first if caching is enabled
        if self._config.cache_model:
            cache_key = self._config.model_name
            with self._cache_lock:
                if cache_key in self._model_cache:
                    logger.info("model_loaded_from_cache", model=self._config.model_name)
                    return self._model_cache[cache_key]

        logger.info("loading_spacy_model", model=self._config.model_name)

        try:
            import spacy

            # Load the spaCy model
            nlp = spacy.load(self._config.model_name)

            # Disable unused pipeline components for performance
            disabled_components = []
            if not self._config.enable_ner and "ner" in nlp.pipe_names:
                disabled_components.append("ner")

            if disabled_components:
                nlp.disable_pipes(*disabled_components)
                logger.info("disabled_pipeline_components", components=disabled_components)

            logger.info(
                "spacy_model_loaded",
                model=self._config.model_name,
                pipeline=nlp.pipe_names,
            )

            # Cache the model if enabled
            if self._config.cache_model:
                cache_key = self._config.model_name
                with self._cache_lock:
                    self._model_cache[cache_key] = nlp

            return nlp

        except ImportError as e:
            error_msg = (
                "spacy not installed. Install with: pip install spacy && "
                f"python -m spacy download {self._config.model_name}"
            )
            logger.error("model_load_failed", error=str(e), reason="missing_dependencies")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            logger.error("model_load_failed", error=str(e), model=self._config.model_name)
            raise RuntimeError(
                f"Failed to load spaCy model {self._config.model_name}: {e}"
            ) from e

    def _ensure_model_loaded(self) -> Language:
        """Ensure the model is loaded, loading it if necessary.

        Returns:
            The loaded spaCy model.
        """
        if self._model is None:
            self._model = self._load_model()

            # Add custom patterns if configured
            if self._config.include_custom_patterns and not self._custom_patterns_added:
                self._add_custom_patterns()
                self._custom_patterns_added = True

        return self._model

    def _add_custom_patterns(self) -> None:
        """Add custom entity patterns to the model.

        This method is a no-op in the base class and can be overridden
        by subclasses to add domain-specific patterns.
        """
        logger.debug("no_custom_patterns_to_add")

    def _extract_entities_from_doc(self, doc: Doc) -> list[NamedEntity]:
        """Extract entities from a processed spaCy Doc.

        Args:
            doc: Processed spaCy document.

        Returns:
            List of extracted NamedEntity objects.
        """
        entities = []

        for ent in doc.ents:
            # Calculate confidence score
            # spaCy doesn't provide confidence scores directly, so we use 1.0
            # for all entities. Subclasses can override this behavior.
            confidence = self._get_entity_confidence(ent)

            # Filter by confidence threshold
            if confidence < self._config.confidence_threshold:
                continue

            entity = NamedEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=confidence,
            )
            entities.append(entity)

        return entities

    def _get_entity_confidence(self, _ent: Span) -> float:
        """Get confidence score for an entity.

        Base implementation returns 1.0 for all entities since spaCy
        doesn't provide confidence scores by default.

        Args:
            _ent: spaCy Span representing the entity (unused in base impl).

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        return 1.0

    def extract(self, text: str) -> EntityExtractionResult:
        """Extract entities from a single text.

        Args:
            text: Input text to analyze.

        Returns:
            EntityExtractionResult containing extracted entities.

        Raises:
            ValueError: If text is empty or invalid.
            RuntimeError: If extraction fails.

        Examples:
            >>> extractor = SpaCyEntityExtractor()
            >>> result = extractor.extract("Apple Inc. was founded in 1976.")
            >>> len(result.entities) > 0
            True
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        logger.debug("extracting_entities", text_length=len(text))

        try:
            # Ensure model is loaded
            nlp = self._ensure_model_loaded()

            # Process the text
            doc = nlp(text)

            # Extract entities
            entities = self._extract_entities_from_doc(doc)

            logger.debug(
                "entities_extracted",
                num_entities=len(entities),
            )

            return EntityExtractionResult(
                text=text,
                entities=entities,
            )

        except ValueError:
            raise
        except Exception as e:
            logger.error("entity_extraction_failed", error=str(e), text=text[:100])
            raise RuntimeError(f"Entity extraction failed: {e}") from e

    def extract_batch(self, texts: list[str]) -> list[EntityExtractionResult]:
        """Extract entities from multiple texts in batch.

        Args:
            texts: List of input texts to analyze.

        Returns:
            List of EntityExtractionResult objects, one per input text.

        Raises:
            ValueError: If texts list is empty.
            RuntimeError: If batch extraction fails.

        Examples:
            >>> extractor = SpaCyEntityExtractor()
            >>> texts = ["Apple Inc. is based in Cupertino.", "Microsoft is in Redmond."]
            >>> results = extractor.extract_batch(texts)
            >>> len(results)
            2
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        logger.info("extracting_batch", num_texts=len(texts), batch_size=self._config.batch_size)

        try:
            # Ensure model is loaded
            nlp = self._ensure_model_loaded()

            # Process texts in batch using spaCy's pipe for efficiency
            results = []

            for doc, original_text in zip(
                nlp.pipe(texts, batch_size=self._config.batch_size),
                texts,
                strict=True,
            ):
                # Extract entities from document
                entities = self._extract_entities_from_doc(doc)

                result = EntityExtractionResult(
                    text=original_text,
                    entities=entities,
                )
                results.append(result)

            logger.info(
                "batch_extraction_complete",
                total_texts=len(texts),
                total_entities=sum(len(r.entities) for r in results),
            )

            return results

        except ValueError:
            raise
        except Exception as e:
            logger.error("batch_extraction_failed", error=str(e), num_texts=len(texts))
            raise RuntimeError(f"Batch entity extraction failed: {e}") from e


class FinancialEntityExtractor(SpaCyEntityExtractor):
    """Entity extractor with custom financial entity patterns.

    This extractor extends SpaCyEntityExtractor with custom patterns for
    financial entities like ticker symbols, monetary amounts, and percentages.

    Financial entity types:
        - TICKER: Stock ticker symbols ($AAPL, MSFT)
        - MONEY: Monetary amounts ($1.5B, 50M USD)
        - PERCENT: Percentages (15%, 3.5 percent)

    Examples:
        >>> extractor = FinancialEntityExtractor()
        >>> text = "$AAPL rose 15% to $150.25, with market cap of $2.5T."
        >>> result = extractor.extract(text)
        >>> tickers = [e.text for e in result.entities if e.label == 'TICKER']
        >>> print(tickers)
        ['AAPL']
    """

    def __init__(
        self,
        config: NERConfig | None = None,
        model: Language | None = None,
    ) -> None:
        """Initialize the financial entity extractor.

        Args:
            config: Configuration for the extractor. If None, uses defaults.
            model: Pre-loaded spaCy model for dependency injection (testing).
        """
        super().__init__(config, model)
        logger.info("financial_entity_extractor_initialized")

    def _add_custom_patterns(self) -> None:
        """Add custom financial entity patterns to the model.

        This method adds patterns for:
        - TICKER: Stock ticker symbols
        - MONEY: Monetary amounts (enhanced from spaCy's default)
        - PERCENT: Percentage values
        """
        if self._model is None:
            logger.warning("cannot_add_patterns_model_not_loaded")
            return

        logger.info("adding_custom_financial_patterns")

        try:
            # Check if entity ruler exists, if not create one
            if "entity_ruler" not in self._model.pipe_names:
                ruler = self._model.add_pipe("entity_ruler", before="ner")
            else:
                ruler = self._model.get_pipe("entity_ruler")

            # Define patterns for financial entities
            patterns = [
                # Ticker patterns: $AAPL, AAPL
                {"label": "TICKER", "pattern": [{"TEXT": {"REGEX": r"\$[A-Z]{1,5}\b"}}]},
                {"label": "TICKER", "pattern": [{"TEXT": {"REGEX": r"^[A-Z]{2,5}$"}}]},
                # Money patterns: $1.5B, 50M USD, $100.25
                {
                    "label": "MONEY",
                    "pattern": [{"TEXT": {"REGEX": r"\$\d+(?:\.\d+)?[KMBTkmbt]?"}}],
                },
                {
                    "label": "MONEY",
                    "pattern": [
                        {"TEXT": {"REGEX": r"\d+(?:\.\d+)?[KMBTkmbt]?"}},
                        {"TEXT": {"IN": ["USD", "EUR", "GBP", "dollars", "euros", "pounds"]}},
                    ],
                },
                # Percent patterns: 15%, 3.5 percent
                {"label": "PERCENT", "pattern": [{"TEXT": {"REGEX": r"\d+(?:\.\d+)?%"}}]},
                {
                    "label": "PERCENT",
                    "pattern": [
                        {"TEXT": {"REGEX": r"\d+(?:\.\d+)?"}},
                        {"TEXT": {"IN": ["percent", "percentage", "pct"]}},
                    ],
                },
            ]

            ruler.add_patterns(patterns)  # type: ignore[attr-defined]

            logger.info("custom_financial_patterns_added", num_patterns=len(patterns))

        except Exception as e:
            logger.error("failed_to_add_custom_patterns", error=str(e))
            # Don't fail initialization if pattern addition fails
            # Fall back to standard NER


# Singleton cache for default extractor
_default_extractor: FinancialEntityExtractor | None = None
_extractor_lock = threading.Lock()


def get_entity_extractor(config: NERConfig | None = None) -> FinancialEntityExtractor:
    """Get an entity extractor instance.

    This factory function returns a singleton instance for the default
    configuration, and creates new instances for custom configurations.

    Args:
        config: Configuration for the extractor. If None, uses defaults
                and returns a cached singleton instance.

    Returns:
        FinancialEntityExtractor instance.

    Examples:
        >>> extractor = get_entity_extractor()
        >>> result = extractor.extract("$AAPL rose 15% today.")
    """
    global _default_extractor

    if config is None:
        # Return singleton instance for default config
        with _extractor_lock:
            if _default_extractor is None:
                _default_extractor = FinancialEntityExtractor()
                logger.debug("default_extractor_created")
            return _default_extractor
    else:
        # Create new instance for custom config
        logger.debug("creating_entity_extractor", model=config.model_name)
        return FinancialEntityExtractor(config)


def extract_entities(text: str) -> EntityExtractionResult:
    """Convenience function for extracting entities from text.

    This is a high-level convenience function that uses default configuration
    and caching for quick entity extraction.

    Args:
        text: Text to analyze.

    Returns:
        EntityExtractionResult containing extracted entities.

    Raises:
        ValueError: If text is empty.
        RuntimeError: If extraction fails.

    Examples:
        >>> result = extract_entities("Apple Inc. reported $1.5B revenue.")
        >>> print(f"Found {len(result.entities)} entities")
        Found 2 entities
    """
    extractor = get_entity_extractor()
    return extractor.extract(text)


def extract_tickers(text: str) -> list[str]:
    """Extract ticker symbols from text.

    This convenience function extracts only ticker symbols using a combination
    of regex patterns and NER results.

    Args:
        text: Text to analyze for ticker symbols.

    Returns:
        List of unique ticker symbols found in the text.

    Examples:
        >>> text = "Trading $AAPL and MSFT today. AAPL looks strong."
        >>> tickers = extract_tickers(text)
        >>> print(sorted(tickers))
        ['AAPL', 'MSFT']
    """
    if not text or not text.strip():
        return []

    # Use regex to find potential tickers
    ticker_pattern = re.compile(TICKER_PATTERN)
    matches = ticker_pattern.findall(text)

    # Clean up ticker symbols (remove $ prefix, deduplicate)
    tickers = set()
    for match in matches:
        ticker = match.lstrip("$").upper()
        # Filter out common false positives (1-letter tickers, lowercase)
        if len(ticker) >= 2 and ticker.isupper():
            tickers.add(ticker)

    logger.debug("tickers_extracted", num_tickers=len(tickers), tickers=list(tickers))

    return sorted(tickers)
