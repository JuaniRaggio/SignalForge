"""Text preprocessing pipeline for financial documents.

This module provides comprehensive text preprocessing capabilities tailored
for financial documents, including news articles, earnings reports, and
market commentary.

Key Features:
- Text cleaning and normalization
- Financial term standardization
- Sentence segmentation
- Tokenization
- Boilerplate removal
- Batch processing support

Examples:
    Basic text preprocessing:

    >>> from signalforge.nlp.preprocessing import TextPreprocessor, PreprocessingConfig
    >>>
    >>> preprocessor = TextPreprocessor()
    >>> text = "Apple Inc. (AAPL) reported revenue of $1.5M in Q1 2024."
    >>> cleaned = preprocessor.clean_text(text)
    >>> tokens = preprocessor.tokenize(cleaned)

    Document batch processing:

    >>> from signalforge.nlp.preprocessing import DocumentPreprocessor
    >>> config = PreprocessingConfig(lowercase=True, remove_stopwords=False)
    >>> doc_processor = DocumentPreprocessor()
    >>> texts = ["Market update...", "Earnings report..."]
    >>> results = doc_processor.process_batch(texts, config)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Final

from signalforge.core.logging import get_logger

logger = get_logger(__name__)

# Financial term normalization mappings
CURRENCY_PATTERNS: Final[dict[str, str]] = {
    r"\$\s*": "$",
    r"USD\b": "$",
    r"dollars?\b": "$",
    r"EUR\b": "EUR",
    r"euros?\b": "EUR",
    r"GBP\b": "GBP",
    r"pounds?\b": "GBP",
    r"CNY\b": "CNY",
    r"yuan\b": "CNY",
}

# Number abbreviation patterns (e.g., 1.5M -> 1500000)
NUMBER_ABBREVIATIONS: Final[dict[str, float]] = {
    "K": 1_000,
    "M": 1_000_000,
    "B": 1_000_000_000,
    "T": 1_000_000_000_000,
}

# Common boilerplate patterns in financial documents
BOILERPLATE_PATTERNS: Final[list[str]] = [
    r"(?i)forward[-\s]looking\s+statements?.*?(?:\n\n|\Z)",
    r"(?i)this\s+press\s+release\s+contains.*?(?:\n\n|\Z)",
    r"(?i)safe\s+harbor\s+statement.*?(?:\n\n|\Z)",
    r"(?i)about\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*:.*?(?:\n\n|\Z)",
    r"(?i)for\s+more\s+information.*?(?:\n\n|\Z)",
    r"(?i)contact\s*:.*?(?:\n\n|\Z)",
    r"(?i)investor\s+relations.*?(?:\n\n|\Z)",
    r"(?i)media\s+relations.*?(?:\n\n|\Z)",
]

# Maximum text length for single chunk processing
MAX_CHUNK_LENGTH: Final[int] = 100_000

# Sentence boundary patterns
SENTENCE_ENDINGS: Final[str] = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+"


@dataclass
class PreprocessingConfig:
    """Configuration options for text preprocessing.

    Attributes:
        lowercase: Convert text to lowercase.
        remove_stopwords: Remove common English stopwords.
        remove_punctuation: Remove punctuation marks (except financial symbols).
        preserve_tickers: Keep ticker symbols like $AAPL or AAPL.
        preserve_numbers: Keep numeric values and percentages.
        normalize_financial_terms: Standardize financial terminology.
        remove_boilerplate: Remove legal disclaimers and common headers.
        max_token_length: Maximum length for individual tokens.
        min_token_length: Minimum length for individual tokens.
    """

    lowercase: bool = True
    remove_stopwords: bool = False
    remove_punctuation: bool = False
    preserve_tickers: bool = True
    preserve_numbers: bool = True
    normalize_financial_terms: bool = True
    remove_boilerplate: bool = True
    max_token_length: int = 50
    min_token_length: int = 1

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_token_length <= 0:
            raise ValueError("max_token_length must be positive")
        if self.min_token_length < 0:
            raise ValueError("min_token_length cannot be negative")
        if self.min_token_length > self.max_token_length:
            raise ValueError("min_token_length cannot exceed max_token_length")


@dataclass
class ProcessedDocument:
    """Result of document preprocessing.

    Attributes:
        original_text: The original input text.
        cleaned_text: Text after cleaning and normalization.
        sentences: List of segmented sentences.
        tokens: List of tokens from the entire document.
        metadata: Additional metadata about the processing.
    """

    original_text: str
    cleaned_text: str
    sentences: list[str]
    tokens: list[str]
    metadata: dict[str, int | float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Populate metadata if not provided."""
        if not self.metadata:
            self.metadata = {
                "original_length": len(self.original_text),
                "cleaned_length": len(self.cleaned_text),
                "num_sentences": len(self.sentences),
                "num_tokens": len(self.tokens),
            }


class TextPreprocessor:
    """Text preprocessing utilities for financial documents.

    This class provides methods for cleaning, normalizing, and tokenizing
    financial text data. It handles financial-specific requirements such as
    preserving ticker symbols, normalizing currency notation, and handling
    numeric abbreviations.

    Examples:
        >>> preprocessor = TextPreprocessor()
        >>> text = "  Apple Inc. reported $1.5M in revenue.  "
        >>> cleaned = preprocessor.clean_text(text)
        >>> print(cleaned)
        'Apple Inc. reported $1.5M in revenue.'
    """

    def __init__(self) -> None:
        """Initialize the text preprocessor."""
        self._ticker_pattern = re.compile(r"\$?[A-Z]{1,5}\b")
        self._number_pattern = re.compile(r"\d+(?:\.\d+)?%?")
        self._whitespace_pattern = re.compile(r"\s+")
        self._url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        self._email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

    def clean_text(self, text: str) -> str:
        """Clean and normalize text.

        This method performs the following operations:
        - Unicode normalization (NFKC)
        - Whitespace normalization
        - URL and email removal
        - Control character removal

        Args:
            text: Raw input text.

        Returns:
            Cleaned and normalized text.

        Examples:
            >>> preprocessor = TextPreprocessor()
            >>> text = "  Hello\\n\\nWorld  "
            >>> preprocessor.clean_text(text)
            'Hello World'
        """
        if not text or not text.strip():
            logger.debug("clean_text called with empty or whitespace-only text")
            return ""

        # Normalize Unicode characters (NFKC: compatibility decomposition + canonical composition)
        text = unicodedata.normalize("NFKC", text)

        # Remove URLs
        text = self._url_pattern.sub(" ", text)

        # Remove email addresses
        text = self._email_pattern.sub(" ", text)

        # Remove control characters (except newlines and tabs)
        text = "".join(
            char for char in text if unicodedata.category(char)[0] != "C" or char in "\n\t"
        )

        # Normalize whitespace (preserve single spaces and newlines)
        text = self._whitespace_pattern.sub(" ", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        logger.debug(
            "text_cleaned",
            original_length=len(text),
            cleaned_length=len(text),
        )

        return text

    def normalize_financial_terms(self, text: str) -> str:
        """Normalize financial terminology to standard formats.

        This method standardizes:
        - Currency symbols and names (USD, $, dollars -> $)
        - Number abbreviations (1M -> 1000000)
        - Date formats
        - Common financial acronyms

        Args:
            text: Input text with financial terms.

        Returns:
            Text with normalized financial terminology.

        Examples:
            >>> preprocessor = TextPreprocessor()
            >>> text = "Revenue was 1.5M dollars"
            >>> preprocessor.normalize_financial_terms(text)
            'Revenue was 1500000 $'
        """
        if not text:
            return ""

        result = text

        # Normalize currency symbols
        for pattern, replacement in CURRENCY_PATTERNS.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        # Normalize number abbreviations (e.g., 1.5M -> 1500000)
        def replace_abbreviation(match: re.Match[str]) -> str:
            number_str = match.group(1)
            abbrev = match.group(2)
            try:
                number = float(number_str)
                multiplier = NUMBER_ABBREVIATIONS.get(abbrev.upper(), 1)
                result_val = (
                    int(number * multiplier)
                    if number * multiplier % 1 == 0
                    else number * multiplier
                )
                return str(result_val)
            except ValueError:
                return match.group(0)

        abbrev_pattern = r"(\d+(?:\.\d+)?)\s*([KMBT])\b"
        result = re.sub(abbrev_pattern, replace_abbreviation, result, flags=re.IGNORECASE)

        # Normalize common date formats to YYYY-MM-DD
        # Pattern: MM/DD/YYYY or DD/MM/YYYY
        date_pattern = r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"

        def normalize_date(match: re.Match[str]) -> str:
            part1, part2, year = match.group(1), match.group(2), match.group(3)
            # Assume MM/DD/YYYY format (US standard)
            month = part1.zfill(2)
            day = part2.zfill(2)
            return f"{year}-{month}-{day}"

        result = re.sub(date_pattern, normalize_date, result)

        logger.debug("financial_terms_normalized")

        return result

    def remove_boilerplate(self, text: str) -> str:
        """Remove common legal disclaimers and boilerplate text.

        This method removes:
        - Forward-looking statements
        - Safe harbor notices
        - Contact information sections
        - Standard legal disclaimers

        Args:
            text: Input text potentially containing boilerplate.

        Returns:
            Text with boilerplate sections removed.

        Examples:
            >>> preprocessor = TextPreprocessor()
            >>> text = "Revenue increased. Forward-looking statements: ..."
            >>> cleaned = preprocessor.remove_boilerplate(text)
            >>> "Forward-looking" in cleaned
            False
        """
        if not text:
            return ""

        result = text

        for pattern in BOILERPLATE_PATTERNS:
            result = re.sub(pattern, "", result, flags=re.MULTILINE)

        # Clean up multiple consecutive newlines
        result = re.sub(r"\n{3,}", "\n\n", result)

        logger.debug("boilerplate_removed")

        return result.strip()

    def extract_sentences(self, text: str) -> list[str]:
        """Segment text into sentences.

        Uses regex-based sentence boundary detection that handles common
        edge cases in financial text (e.g., abbreviations, decimal numbers).

        Args:
            text: Input text to segment.

        Returns:
            List of sentences.

        Examples:
            >>> preprocessor = TextPreprocessor()
            >>> text = "Revenue increased. Earnings were strong."
            >>> preprocessor.extract_sentences(text)
            ['Revenue increased.', 'Earnings were strong.']
        """
        if not text or not text.strip():
            return []

        # Split on sentence boundaries
        sentences = re.split(SENTENCE_ENDINGS, text)

        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                cleaned_sentences.append(sentence)

        logger.debug(
            "sentences_extracted",
            num_sentences=len(cleaned_sentences),
        )

        return cleaned_sentences

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words.

        Performs word-level tokenization while preserving:
        - Ticker symbols ($AAPL)
        - Numbers and percentages
        - Hyphenated words

        Args:
            text: Input text to tokenize.

        Returns:
            List of tokens.

        Examples:
            >>> preprocessor = TextPreprocessor()
            >>> text = "Apple's revenue was $100M."
            >>> preprocessor.tokenize(text)
            ['Apple', 's', 'revenue', 'was', '$', '100M', '.']
        """
        if not text or not text.strip():
            return []

        # Simple word tokenization using regex
        # Matches: words, numbers, punctuation, ticker symbols
        token_pattern = r"\$?[A-Za-z][A-Za-z0-9]*(?:['-][A-Za-z0-9]+)*|\d+(?:\.\d+)?%?|[^\w\s]"
        tokens = re.findall(token_pattern, text)

        logger.debug(
            "text_tokenized",
            num_tokens=len(tokens),
        )

        return tokens


class DocumentPreprocessor:
    """Batch document preprocessing with configurable options.

    This class provides high-level document processing capabilities with
    support for batch operations and chunking of very long documents.

    Examples:
        >>> preprocessor = DocumentPreprocessor()
        >>> config = PreprocessingConfig(lowercase=True)
        >>> doc = preprocessor.process_document("Sample text...", config)
        >>> print(doc.num_tokens)
    """

    def __init__(self) -> None:
        """Initialize the document preprocessor."""
        self._text_preprocessor = TextPreprocessor()

    def process_document(
        self,
        text: str,
        config: PreprocessingConfig,
    ) -> ProcessedDocument:
        """Process a single document with specified configuration.

        Args:
            text: Raw document text.
            config: Preprocessing configuration.

        Returns:
            ProcessedDocument containing all preprocessing results.

        Raises:
            ValueError: If text is empty after preprocessing.

        Examples:
            >>> preprocessor = DocumentPreprocessor()
            >>> config = PreprocessingConfig()
            >>> doc = preprocessor.process_document("Sample text", config)
            >>> assert doc.cleaned_text
        """
        if not text or not text.strip():
            logger.warning("process_document called with empty text")
            return ProcessedDocument(
                original_text=text,
                cleaned_text="",
                sentences=[],
                tokens=[],
                metadata={
                    "original_length": len(text) if text else 0,
                    "cleaned_length": 0,
                    "num_sentences": 0,
                    "num_tokens": 0,
                },
            )

        original_text = text

        # Handle very long texts by chunking
        if len(text) > MAX_CHUNK_LENGTH:
            logger.warning(
                "processing_large_document",
                length=len(text),
                max_length=MAX_CHUNK_LENGTH,
            )
            text = text[:MAX_CHUNK_LENGTH]

        # Step 1: Remove boilerplate (if configured)
        if config.remove_boilerplate:
            text = self._text_preprocessor.remove_boilerplate(text)

        # Step 2: Clean text
        text = self._text_preprocessor.clean_text(text)

        # Step 3: Normalize financial terms (if configured)
        if config.normalize_financial_terms:
            text = self._text_preprocessor.normalize_financial_terms(text)

        # Step 4: Extract sentences
        sentences = self._text_preprocessor.extract_sentences(text)

        # Step 5: Apply lowercase (if configured)
        if config.lowercase:
            text = text.lower()
            sentences = [s.lower() for s in sentences]

        # Step 6: Tokenize
        all_tokens = self._text_preprocessor.tokenize(text)

        # Step 7: Filter tokens based on configuration
        filtered_tokens = self._filter_tokens(all_tokens, config)

        logger.info(
            "document_processed",
            original_length=len(original_text),
            cleaned_length=len(text),
            num_sentences=len(sentences),
            num_tokens=len(filtered_tokens),
        )

        return ProcessedDocument(
            original_text=original_text,
            cleaned_text=text,
            sentences=sentences,
            tokens=filtered_tokens,
        )

    def process_batch(
        self,
        texts: list[str],
        config: PreprocessingConfig,
    ) -> list[ProcessedDocument]:
        """Process multiple documents in batch.

        Args:
            texts: List of raw document texts.
            config: Preprocessing configuration.

        Returns:
            List of ProcessedDocument objects.

        Examples:
            >>> preprocessor = DocumentPreprocessor()
            >>> config = PreprocessingConfig()
            >>> texts = ["Doc 1", "Doc 2", "Doc 3"]
            >>> results = preprocessor.process_batch(texts, config)
            >>> len(results) == 3
            True
        """
        if not texts:
            logger.warning("process_batch called with empty list")
            return []

        logger.info(
            "processing_batch",
            num_documents=len(texts),
        )

        results = []
        for idx, text in enumerate(texts):
            try:
                doc = self.process_document(text, config)
                results.append(doc)
            except Exception as e:
                logger.error(
                    "document_processing_failed",
                    index=idx,
                    error=str(e),
                )
                # Append empty document for failed processing
                results.append(
                    ProcessedDocument(
                        original_text=text,
                        cleaned_text="",
                        sentences=[],
                        tokens=[],
                        metadata={"error": str(e)},
                    )
                )

        logger.info(
            "batch_processed",
            total_documents=len(texts),
            successful=sum(1 for r in results if r.cleaned_text),
            failed=sum(1 for r in results if not r.cleaned_text),
        )

        return results

    def _filter_tokens(
        self,
        tokens: list[str],
        config: PreprocessingConfig,
    ) -> list[str]:
        """Filter tokens based on configuration.

        Args:
            tokens: List of raw tokens.
            config: Preprocessing configuration.

        Returns:
            Filtered list of tokens.
        """
        filtered = []

        for token in tokens:
            # Skip tokens based on configuration
            # Skip pure punctuation unless it's a financial symbol
            if (
                config.remove_punctuation
                and re.match(r"^[^\w\s]+$", token)
                and not (config.preserve_tickers and token.startswith("$"))
            ):
                continue

            # Filter by token length
            if len(token) < config.min_token_length or len(token) > config.max_token_length:
                continue

            # Skip stopwords if configured
            if config.remove_stopwords and token.lower() in ENGLISH_STOPWORDS:
                continue

            filtered.append(token)

        return filtered


# Common English stopwords (minimal set for optional filtering)
ENGLISH_STOPWORDS: Final[set[str]] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "will",
    "with",
}
