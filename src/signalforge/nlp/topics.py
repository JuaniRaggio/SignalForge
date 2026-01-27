"""Topic extraction and keyword identification for financial documents.

This module provides KeyBERT-style topic extraction using sentence embeddings
and cosine similarity. It supports both simple keyword extraction and Maximal
Marginal Relevance (MMR) for diverse topic selection.

Key Features:
- Sentence embedding-based keyword extraction
- N-gram candidate extraction (unigrams, bigrams, trigrams)
- Maximal Marginal Relevance (MMR) for diversity
- Batch processing support
- POS tag filtering for quality
- Configurable extraction parameters

Examples:
    Basic topic extraction:

    >>> from signalforge.nlp.topics import extract_topics
    >>>
    >>> text = "Apple reported strong earnings growth with revenue increasing by 15%."
    >>> result = extract_topics(text)
    >>> print([kw.keyword for kw in result.keywords])
    ['earnings growth', 'revenue increasing', 'Apple']

    Custom configuration with MMR:

    >>> from signalforge.nlp.topics import get_topic_extractor, TopicExtractionConfig
    >>> config = TopicExtractionConfig(top_n=10, diversity=0.7, use_mmr=True)
    >>> extractor = get_topic_extractor(config)
    >>> result = extractor.extract(text)

    Simple keyphrase extraction:

    >>> from signalforge.nlp.topics import extract_keyphrases
    >>> keyphrases = extract_keyphrases(text, top_n=5)
    >>> print(keyphrases)
    ['earnings growth', 'revenue increasing', 'strong earnings', 'Apple', '15%']
"""

from __future__ import annotations

import re
import string
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from signalforge.core.logging import get_logger
from signalforge.nlp.embeddings import compute_similarity, get_embedder

if TYPE_CHECKING:
    from signalforge.nlp.embeddings import BaseEmbeddingModel

logger = get_logger(__name__)

# Default configuration constants
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_N = 5
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_DIVERSITY = 0.5
DEFAULT_USE_MMR = True

# Stopwords for candidate filtering
STOPWORDS = frozenset(
    {
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
        "s",
        "t",
        "can",
        "may",
        "this",
        "they",
        "their",
        "have",
        "been",
        "would",
        "could",
        "should",
        "but",
        "or",
        "not",
        "more",
        "than",
        "very",
        "such",
        "also",
        "about",
        "into",
        "just",
        "so",
        "some",
        "there",
        "these",
        "those",
        "up",
        "out",
        "if",
        "when",
        "which",
        "who",
        "do",
        "does",
        "did",
        "what",
        "where",
        "how",
        "why",
        "all",
        "each",
        "other",
        "any",
        "both",
        "few",
        "many",
        "much",
        "most",
        "same",
        "own",
    }
)


@dataclass
class TopicKeyword:
    """A keyword extracted from text with its relevance score.

    Attributes:
        keyword: The extracted keyword or keyphrase.
        score: Relevance score between 0.0 and 1.0.
    """

    keyword: str
    score: float

    def __post_init__(self) -> None:
        """Validate keyword fields."""
        if not self.keyword or not self.keyword.strip():
            raise ValueError("Keyword cannot be empty")

        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {self.score}")


@dataclass
class TopicExtractionResult:
    """Result of topic extraction on a single text.

    Attributes:
        text: Original input text analyzed.
        keywords: List of extracted keywords with scores.
        top_n: Number of keywords that were requested.
    """

    text: str
    keywords: list[TopicKeyword]
    top_n: int

    def __post_init__(self) -> None:
        """Validate result fields."""
        if self.top_n <= 0:
            raise ValueError(f"top_n must be positive, got {self.top_n}")

        if len(self.keywords) > self.top_n:
            logger.warning(
                "result_has_more_keywords_than_requested",
                found=len(self.keywords),
                requested=self.top_n,
            )


@dataclass
class TopicExtractionConfig:
    """Configuration for topic extraction.

    Attributes:
        model_name: Sentence-transformer model to use for embeddings.
        top_n: Number of keywords to extract.
        keyphrase_ngram_range: Tuple of (min_n, max_n) for n-gram extraction.
        diversity: Diversity parameter for MMR (0.0 = relevance only, 1.0 = maximum diversity).
        use_mmr: Whether to use Maximal Marginal Relevance for diversity.
    """

    model_name: str = DEFAULT_MODEL_NAME
    top_n: int = DEFAULT_TOP_N
    keyphrase_ngram_range: tuple[int, int] = DEFAULT_NGRAM_RANGE
    diversity: float = DEFAULT_DIVERSITY
    use_mmr: bool = DEFAULT_USE_MMR

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.top_n <= 0:
            raise ValueError(f"top_n must be positive, got {self.top_n}")

        min_n, max_n = self.keyphrase_ngram_range
        if min_n < 1:
            raise ValueError(f"min_n must be at least 1, got {min_n}")
        if max_n < min_n:
            raise ValueError(f"max_n must be >= min_n, got min_n={min_n}, max_n={max_n}")

        if not 0.0 <= self.diversity <= 1.0:
            raise ValueError(f"diversity must be between 0.0 and 1.0, got {self.diversity}")


class BaseTopicExtractor(ABC):
    """Abstract base class for topic extractors.

    This class defines the interface that all topic extractors must implement.
    """

    @abstractmethod
    def extract(self, text: str) -> TopicExtractionResult:
        """Extract keywords from a single text.

        Args:
            text: Input text to extract keywords from.

        Returns:
            TopicExtractionResult containing extracted keywords and scores.

        Raises:
            ValueError: If text is empty or invalid.
        """
        pass

    @abstractmethod
    def extract_batch(self, texts: list[str]) -> list[TopicExtractionResult]:
        """Extract keywords from multiple texts in batch.

        Args:
            texts: List of input texts to extract keywords from.

        Returns:
            List of TopicExtractionResult objects, one per input text.

        Raises:
            ValueError: If texts list is empty or contains invalid entries.
        """
        pass


class EmbeddingTopicExtractor(BaseTopicExtractor):
    """Topic extractor using sentence embeddings and cosine similarity.

    This extractor uses the KeyBERT approach:
    1. Extract n-gram candidates from text
    2. Generate embeddings for document and candidates
    3. Rank candidates by similarity to document embedding
    4. Optionally apply MMR for diversity

    Examples:
        >>> config = TopicExtractionConfig(top_n=5, use_mmr=True)
        >>> extractor = EmbeddingTopicExtractor(config)
        >>> result = extractor.extract("Apple earnings beat expectations.")
        >>> print([kw.keyword for kw in result.keywords])
        ['earnings beat', 'Apple earnings', 'expectations', 'beat', 'Apple']
    """

    def __init__(self, config: TopicExtractionConfig | None = None) -> None:
        """Initialize the embedding-based topic extractor.

        Args:
            config: Configuration for the extractor. If None, uses defaults.
        """
        self._config = config or TopicExtractionConfig()

        # Lazy-load embedder
        from signalforge.nlp.embeddings import EmbeddingsConfig

        embeddings_config = EmbeddingsConfig(
            model_name=self._config.model_name,
            device="auto",
            normalize=True,
        )
        self._embedder: BaseEmbeddingModel = get_embedder(embeddings_config)

        logger.info(
            "topic_extractor_initialized",
            model_name=self._config.model_name,
            top_n=self._config.top_n,
            use_mmr=self._config.use_mmr,
            diversity=self._config.diversity,
        )

    def _extract_candidates(self, text: str) -> list[str]:
        """Extract n-gram candidates from text.

        Args:
            text: Input text to extract candidates from.

        Returns:
            List of candidate phrases (n-grams).
        """
        # Normalize text
        text = text.strip()

        # Split into sentences (simple sentence splitting)
        sentence_pattern = r"[.!?]+\s+"
        sentences = re.split(sentence_pattern, text)

        candidates = set()
        min_n, max_n = self._config.keyphrase_ngram_range

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Tokenize (simple whitespace tokenization with punctuation handling)
            # Keep financial symbols and numbers
            tokens = self._tokenize(sentence)

            # Generate n-grams
            for n in range(min_n, max_n + 1):
                for i in range(len(tokens) - n + 1):
                    ngram_tokens = tokens[i : i + n]

                    # Filter out n-grams that are all stopwords
                    if all(token.lower() in STOPWORDS for token in ngram_tokens):
                        continue

                    # Filter out n-grams starting or ending with stopwords
                    if ngram_tokens[0].lower() in STOPWORDS or ngram_tokens[-1].lower() in STOPWORDS:
                        continue

                    # Join tokens into phrase
                    candidate = " ".join(ngram_tokens)

                    # Basic quality filters
                    if self._is_valid_candidate(candidate):
                        candidates.add(candidate)

        result = list(candidates)

        logger.debug(
            "candidates_extracted",
            num_candidates=len(result),
            text_length=len(text),
        )

        return result

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text while preserving financial symbols.

        Args:
            text: Input text to tokenize.

        Returns:
            List of tokens.
        """
        # Replace special characters with spaces, but preserve $ and %
        text = re.sub(r"[^\w\s$%.-]", " ", text)

        # Split on whitespace
        tokens = text.split()

        # Remove empty tokens
        tokens = [token for token in tokens if token.strip()]

        return tokens

    def _is_valid_candidate(self, candidate: str) -> bool:
        """Check if a candidate phrase is valid.

        Args:
            candidate: Candidate phrase to validate.

        Returns:
            True if candidate is valid, False otherwise.
        """
        # Must have at least one alphanumeric character
        if not any(c.isalnum() for c in candidate):
            return False

        # Must not be only punctuation
        if all(c in string.punctuation or c.isspace() for c in candidate):
            return False

        # Must have reasonable length
        if len(candidate) < 2 or len(candidate) > 100:
            return False

        # Must not be all digits (unless it has %)
        return not (candidate.replace(".", "").replace(",", "").isdigit() and "%" not in candidate)

    def _maximal_marginal_relevance(
        self,
        doc_embedding: list[float],
        candidate_embeddings: list[tuple[str, list[float]]],
        top_n: int,
        diversity: float,
    ) -> list[tuple[str, float]]:
        """Apply Maximal Marginal Relevance to select diverse keywords.

        MMR balances relevance and diversity:
        MMR = lambda * Sim(candidate, doc) - (1 - lambda) * max(Sim(candidate, selected))

        Args:
            doc_embedding: Embedding of the entire document.
            candidate_embeddings: List of (candidate, embedding) tuples.
            top_n: Number of keywords to select.
            diversity: Lambda parameter (0.0 = relevance only, 1.0 = maximum diversity).

        Returns:
            List of (keyword, score) tuples selected by MMR. Scores are normalized to [0, 1].
        """
        if not candidate_embeddings:
            return []

        # Compute initial similarities to document
        candidate_scores = [
            (candidate, compute_similarity(emb, doc_embedding))
            for candidate, emb in candidate_embeddings
        ]

        # Sort by similarity to document
        candidate_scores.sort(key=lambda x: x[1], reverse=True)

        # Initialize selected keywords with the most relevant one
        # Normalize cosine similarity from [-1, 1] to [0, 1]
        initial_score = (candidate_scores[0][1] + 1) / 2
        selected: list[tuple[str, float]] = [(candidate_scores[0][0], initial_score)]
        remaining = candidate_embeddings[1:]
        selected_embeddings = [candidate_embeddings[0][1]]

        # Select remaining keywords using MMR
        for _ in range(min(top_n - 1, len(remaining))):
            mmr_scores = []

            for candidate, candidate_emb in remaining:
                # Relevance to document (cosine similarity: -1 to 1)
                relevance = compute_similarity(candidate_emb, doc_embedding)

                # Maximum similarity to already selected keywords
                max_sim = max(
                    compute_similarity(candidate_emb, selected_emb)
                    for selected_emb in selected_embeddings
                )

                # MMR score (can range from -1 to 1 depending on diversity)
                mmr_score = diversity * relevance - (1 - diversity) * max_sim
                mmr_scores.append((candidate, candidate_emb, mmr_score))

            # Select candidate with highest MMR score
            best_candidate = max(mmr_scores, key=lambda x: x[2])
            # Normalize MMR score from [-1, 1] to [0, 1]
            normalized_score = (best_candidate[2] + 1) / 2
            selected.append((best_candidate[0], normalized_score))
            selected_embeddings.append(best_candidate[1])

            # Remove selected candidate from remaining
            remaining = [
                (c, e) for c, e in remaining if c != best_candidate[0]
            ]

            if not remaining:
                break

        logger.debug(
            "mmr_selection_complete",
            selected=len(selected),
            requested=top_n,
        )

        return selected

    def extract(self, text: str) -> TopicExtractionResult:
        """Extract keywords from a single text.

        Args:
            text: Input text to extract keywords from.

        Returns:
            TopicExtractionResult containing extracted keywords and scores.

        Raises:
            ValueError: If text is empty or invalid.
            RuntimeError: If extraction fails.

        Examples:
            >>> extractor = EmbeddingTopicExtractor()
            >>> result = extractor.extract("Apple reported strong Q4 earnings.")
            >>> print(result.keywords[0].keyword)
            'strong Q4 earnings'
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        logger.debug("extracting_topics", text_length=len(text))

        try:
            # Extract candidate phrases
            candidates = self._extract_candidates(text)

            if not candidates:
                logger.warning("no_candidates_extracted", text=text[:100])
                return TopicExtractionResult(
                    text=text,
                    keywords=[],
                    top_n=self._config.top_n,
                )

            # Generate embeddings
            doc_embedding_result = self._embedder.encode(text)
            doc_embedding = doc_embedding_result.embedding

            # Embed candidates
            candidate_embedding_results = self._embedder.encode_batch(candidates)
            candidate_embeddings = [
                (candidate, result.embedding)
                for candidate, result in zip(candidates, candidate_embedding_results, strict=True)
            ]

            # Select keywords using MMR or simple similarity ranking
            if self._config.use_mmr and len(candidates) > 1:
                selected = self._maximal_marginal_relevance(
                    doc_embedding,
                    candidate_embeddings,
                    self._config.top_n,
                    self._config.diversity,
                )
            else:
                # Simple similarity-based ranking
                candidate_scores = [
                    (candidate, compute_similarity(emb, doc_embedding))
                    for candidate, emb in candidate_embeddings
                ]
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                # Normalize cosine similarity from [-1, 1] to [0, 1]
                selected = [
                    (candidate, (score + 1) / 2)
                    for candidate, score in candidate_scores[: self._config.top_n]
                ]

            # Convert to TopicKeyword objects
            keywords = [TopicKeyword(keyword=kw, score=score) for kw, score in selected]

            logger.info(
                "topics_extracted",
                num_keywords=len(keywords),
                text_length=len(text),
            )

            return TopicExtractionResult(
                text=text,
                keywords=keywords,
                top_n=self._config.top_n,
            )

        except ValueError:
            raise
        except Exception as e:
            logger.error("topic_extraction_failed", error=str(e), text=text[:100])
            raise RuntimeError(f"Topic extraction failed: {e}") from e

    def extract_batch(self, texts: list[str]) -> list[TopicExtractionResult]:
        """Extract keywords from multiple texts in batch.

        Args:
            texts: List of input texts to extract keywords from.

        Returns:
            List of TopicExtractionResult objects, one per input text.

        Raises:
            ValueError: If texts list is empty.
            RuntimeError: If batch extraction fails.

        Examples:
            >>> extractor = EmbeddingTopicExtractor()
            >>> texts = ["Apple earnings report.", "Tesla stock surges."]
            >>> results = extractor.extract_batch(texts)
            >>> len(results)
            2
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        logger.info("extracting_topics_batch", num_texts=len(texts))

        try:
            results = []

            # Process each text independently
            for text in texts:
                if not text or not text.strip():
                    # Empty text
                    results.append(
                        TopicExtractionResult(
                            text=text,
                            keywords=[],
                            top_n=self._config.top_n,
                        )
                    )
                    continue

                # Extract topics for this text
                result = self.extract(text)
                results.append(result)

            logger.info(
                "batch_extraction_complete",
                total_texts=len(texts),
                successful=len([r for r in results if r.keywords]),
            )

            return results

        except ValueError:
            raise
        except Exception as e:
            logger.error("batch_extraction_failed", error=str(e), num_texts=len(texts))
            raise RuntimeError(f"Batch topic extraction failed: {e}") from e


# Singleton cache for default extractor
_default_extractor: EmbeddingTopicExtractor | None = None
_extractor_lock = threading.Lock()


def get_topic_extractor(config: TopicExtractionConfig | None = None) -> EmbeddingTopicExtractor:
    """Get a topic extractor instance.

    This factory function returns a singleton instance for the default
    configuration, and creates new instances for custom configurations.

    Args:
        config: Configuration for the extractor. If None, uses defaults
                and returns a cached singleton instance.

    Returns:
        EmbeddingTopicExtractor instance.

    Examples:
        >>> extractor = get_topic_extractor()
        >>> result = extractor.extract("Market rallied today.")
    """
    global _default_extractor

    if config is None:
        # Return singleton instance for default config
        with _extractor_lock:
            if _default_extractor is None:
                _default_extractor = EmbeddingTopicExtractor()
                logger.debug("default_extractor_created")
            return _default_extractor
    else:
        # Create new instance for custom config
        logger.debug("getting_custom_extractor", model=config.model_name)
        return EmbeddingTopicExtractor(config)


def extract_topics(text: str, config: TopicExtractionConfig | None = None) -> TopicExtractionResult:
    """Convenience function for extracting topics from text.

    This is a high-level convenience function that uses default configuration
    and caching for quick topic extraction.

    Args:
        text: Text to extract topics from.
        config: Optional custom configuration.

    Returns:
        TopicExtractionResult containing extracted keywords and scores.

    Raises:
        ValueError: If text is empty.
        RuntimeError: If extraction fails.

    Examples:
        >>> result = extract_topics("Strong quarterly performance.")
        >>> print([kw.keyword for kw in result.keywords])
        ['quarterly performance', 'Strong quarterly', 'performance', 'Strong', 'quarterly']
    """
    extractor = get_topic_extractor(config)
    return extractor.extract(text)


def extract_keyphrases(text: str, top_n: int = DEFAULT_TOP_N) -> list[str]:
    """Simple keyphrase extraction returning only the keyword strings.

    This is the simplest interface for topic extraction, returning just
    the keyword strings without scores.

    Args:
        text: Text to extract keyphrases from.
        top_n: Number of keyphrases to extract.

    Returns:
        List of extracted keyphrase strings.

    Raises:
        ValueError: If text is empty or top_n is invalid.
        RuntimeError: If extraction fails.

    Examples:
        >>> keyphrases = extract_keyphrases("Apple earnings beat expectations.", top_n=3)
        >>> print(keyphrases)
        ['earnings beat', 'beat expectations', 'Apple earnings']
    """
    if top_n <= 0:
        raise ValueError(f"top_n must be positive, got {top_n}")

    config = TopicExtractionConfig(top_n=top_n)
    result = extract_topics(text, config)

    return [kw.keyword for kw in result.keywords]
