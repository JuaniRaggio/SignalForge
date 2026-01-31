"""Text embedding generation for financial documents using sentence-transformers.

This module provides text embedding capabilities for converting financial text
into dense vector representations. It supports multiple sentence-transformer models
with GPU acceleration and batch processing.

Key Features:
- Multiple pre-trained sentence-transformer models
- Lazy model loading for efficient resource usage
- GPU/CPU/MPS automatic device selection
- Batch processing with configurable batch sizes
- L2 normalization support
- Cosine similarity computation
- Thread-safe singleton pattern

Examples:
    Basic embedding generation:

    >>> from signalforge.nlp.embeddings import embed_text
    >>>
    >>> text = "Apple reported strong Q4 earnings, beating analyst expectations."
    >>> result = embed_text(text)
    >>> print(f"Embedding dimension: {result.dimension}")
    Embedding dimension: 384

    Batch processing with custom configuration:

    >>> from signalforge.nlp.embeddings import get_embedder, EmbeddingsConfig
    >>> config = EmbeddingsConfig(model_name="all-mpnet-base-v2", device="cuda")
    >>> embedder = get_embedder(config)
    >>> texts = ["Market rallied today...", "Recession fears mount..."]
    >>> results = embedder.encode_batch(texts)

    Computing similarity between texts:

    >>> from signalforge.nlp.embeddings import embed_text, compute_similarity
    >>> emb1 = embed_text("Apple stock rose today")
    >>> emb2 = embed_text("AAPL shares increased")
    >>> similarity = compute_similarity(emb1.embedding, emb2.embedding)
    >>> print(f"Similarity: {similarity:.3f}")
    Similarity: 0.876
"""

from __future__ import annotations

import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

# Model constants
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, 384 dimensions
SUPPORTED_MODELS: dict[str, dict[str, int | str]] = {
    "all-MiniLM-L6-v2": {"dimension": 384, "description": "Fast and efficient (384 dims)"},
    "all-mpnet-base-v2": {"dimension": 768, "description": "High quality (768 dims)"},
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "dimension": 384,
        "description": "Multilingual support (384 dims)",
    },
}

DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 512
MODEL_LOAD_TIMEOUT = 300  # seconds


@dataclass
class EmbeddingResult:
    """Result of embedding generation for a single text.

    Attributes:
        text: Original input text that was embedded.
        embedding: Dense vector representation of the text.
        model_name: Name of the model used to generate the embedding.
        dimension: Dimensionality of the embedding vector.
    """

    text: str
    embedding: list[float]
    model_name: str
    dimension: int

    def __post_init__(self) -> None:
        """Validate embedding result fields."""
        if self.dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {self.dimension}")

        if len(self.embedding) != self.dimension:
            raise ValueError(
                f"Embedding length ({len(self.embedding)}) does not match "
                f"dimension ({self.dimension})"
            )

        # Validate that embeddings contain valid float values
        for idx, value in enumerate(self.embedding):
            if not isinstance(value, (int, float)):
                raise ValueError(f"Embedding value at index {idx} is not numeric: {value}")
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"Embedding contains invalid value at index {idx}: {value}")


@dataclass
class EmbeddingsConfig:
    """Configuration for text embedding generation.

    Attributes:
        model_name: Sentence-transformer model identifier.
        device: Device to use for inference ("auto", "cpu", "cuda", "mps").
        normalize: Whether to L2 normalize embeddings (recommended for similarity).
        batch_size: Number of texts to process in parallel.
        max_length: Maximum token length (texts will be truncated).
        cache_model: Whether to cache the loaded model in memory.
    """

    model_name: str = DEFAULT_MODEL_NAME
    device: str = "auto"
    normalize: bool = True
    batch_size: int = DEFAULT_BATCH_SIZE
    max_length: int = DEFAULT_MAX_LENGTH
    cache_model: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")

        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError(
                f"device must be one of ['auto', 'cpu', 'cuda', 'mps'], got {self.device}"
            )


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models.

    This class defines the interface that all embedding models must implement.
    """

    @abstractmethod
    def encode(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Input text to encode.

        Returns:
            EmbeddingResult containing the text and its vector representation.

        Raises:
            ValueError: If text is empty or invalid.
        """
        pass

    @abstractmethod
    def encode_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts in batch.

        Args:
            texts: List of input texts to encode.

        Returns:
            List of EmbeddingResult objects, one per input text.

        Raises:
            ValueError: If texts list is empty or contains invalid entries.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the dimensionality of the embeddings produced by this model."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the underlying model."""
        pass


class SentenceTransformerEmbedder(BaseEmbeddingModel):
    """Sentence-transformer based text embedder.

    This embedder uses sentence-transformers library to generate dense vector
    representations of text. The model is lazy-loaded on first use and supports
    GPU acceleration.

    Examples:
        >>> embedder = SentenceTransformerEmbedder()
        >>> result = embedder.encode("Strong quarterly earnings beat expectations.")
        >>> print(result.dimension)
        384
    """

    _model_cache: dict[str, SentenceTransformer] = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: EmbeddingsConfig | None = None) -> None:
        """Initialize the sentence-transformer embedder.

        Args:
            config: Configuration for the embedder. If None, uses defaults.
        """
        self._config = config or EmbeddingsConfig()
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None

        logger.info(
            "embedder_initialized",
            model_name=self._config.model_name,
            device=self._config.device,
            batch_size=self._config.batch_size,
            normalize=self._config.normalize,
        )

    @property
    def model_name(self) -> str:
        """Get the name of the underlying model."""
        return self._config.model_name

    @property
    def dimension(self) -> int:
        """Get the dimensionality of the embeddings."""
        if self._dimension is None:
            # Ensure model is loaded to get dimension
            model = self._ensure_model_loaded()
            dim = model.get_sentence_embedding_dimension()
            if dim is None:
                raise RuntimeError("Model did not return embedding dimension")
            self._dimension = dim
        return self._dimension

    def _get_device(self) -> str:
        """Determine the appropriate device for inference.

        Returns:
            Device identifier ("cpu", "cuda", or "mps").
        """
        if self._config.device == "cpu":
            return "cpu"

        if self._config.device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    logger.info("device_selected", device="cuda", reason="auto_detect")
                    return "cuda"
                elif torch.backends.mps.is_available():
                    logger.info("device_selected", device="mps", reason="auto_detect")
                    return "mps"
            except Exception as e:
                logger.warning(
                    "gpu_detection_failed",
                    error=str(e),
                    fallback="cpu",
                )
            return "cpu"

        if self._config.device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning(
                        "cuda_not_available",
                        fallback="cpu",
                    )
                    return "cpu"
                return "cuda"
            except ImportError:
                logger.error("torch_not_installed", fallback="cpu")
                return "cpu"

        if self._config.device == "mps":
            try:
                import torch

                if not torch.backends.mps.is_available():
                    logger.warning(
                        "mps_not_available",
                        fallback="cpu",
                    )
                    return "cpu"
                return "mps"
            except ImportError:
                logger.error("torch_not_installed", fallback="cpu")
                return "cpu"

        return "cpu"

    def _load_model(self) -> SentenceTransformer:
        """Load the sentence-transformer model.

        Returns:
            Loaded SentenceTransformer instance.

        Raises:
            RuntimeError: If model loading fails.
        """
        # Check cache first if caching is enabled
        if self._config.cache_model:
            cache_key = f"{self._config.model_name}_{self._config.device}"
            with self._cache_lock:
                if cache_key in self._model_cache:
                    logger.info("model_loaded_from_cache", model=self._config.model_name)
                    return self._model_cache[cache_key]

        logger.info(
            "loading_model",
            model=self._config.model_name,
            device=self._config.device,
        )

        try:
            from sentence_transformers import SentenceTransformer

            device = self._get_device()

            # Load the sentence transformer model
            model = SentenceTransformer(
                self._config.model_name,
                device=device,
            )

            # Set maximum sequence length if specified
            if self._config.max_length > 0:
                model.max_seq_length = self._config.max_length

            logger.info(
                "model_loaded",
                model=self._config.model_name,
                device=device,
                dimension=model.get_sentence_embedding_dimension(),
            )

            # Cache the model if enabled
            if self._config.cache_model:
                cache_key = f"{self._config.model_name}_{self._config.device}"
                with self._cache_lock:
                    self._model_cache[cache_key] = model

            return model

        except ImportError as e:
            error_msg = (
                "sentence-transformers or torch not installed. "
                "Install with: pip install sentence-transformers torch"
            )
            logger.error("model_load_failed", error=str(e), reason="missing_dependencies")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            logger.error("model_load_failed", error=str(e), model=self._config.model_name)
            raise RuntimeError(f"Failed to load model {self._config.model_name}: {e}") from e

    def _ensure_model_loaded(self) -> SentenceTransformer:
        """Ensure the model is loaded, loading it if necessary.

        Returns:
            The loaded sentence transformer model.
        """
        if self._model is None:
            self._model = self._load_model()
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    def encode(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text.

        Args:
            text: Input text to encode.

        Returns:
            EmbeddingResult containing the text and its vector representation.

        Raises:
            ValueError: If text is empty or invalid.
            RuntimeError: If model inference fails.

        Examples:
            >>> embedder = SentenceTransformerEmbedder()
            >>> result = embedder.encode("Revenue exceeded expectations.")
            >>> len(result.embedding)
            384
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        logger.debug("encoding_text", text_length=len(text))

        try:
            # Ensure model is loaded
            model = self._ensure_model_loaded()

            # Generate embedding
            # sentence_transformers returns numpy array by default
            embedding_array = model.encode(
                text,
                normalize_embeddings=self._config.normalize,
                convert_to_numpy=True,
            )

            # Convert to list of floats
            embedding = embedding_array.tolist()

            logger.debug(
                "text_encoded",
                dimension=len(embedding),
                normalized=self._config.normalize,
            )

            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model_name=self._config.model_name,
                dimension=len(embedding),
            )

        except ValueError:
            raise
        except Exception as e:
            logger.error("encoding_failed", error=str(e), text=text[:100])
            raise RuntimeError(f"Text encoding failed: {e}") from e

    def encode_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts in batch.

        Args:
            texts: List of input texts to encode.

        Returns:
            List of EmbeddingResult objects, one per input text.

        Raises:
            ValueError: If texts list is empty.
            RuntimeError: If batch inference fails.

        Examples:
            >>> embedder = SentenceTransformerEmbedder()
            >>> texts = ["Profits soared.", "Sales declined."]
            >>> results = embedder.encode_batch(texts)
            >>> len(results)
            2
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        logger.info("encoding_batch", num_texts=len(texts), batch_size=self._config.batch_size)

        try:
            # Ensure model is loaded
            model = self._ensure_model_loaded()

            # Filter out empty texts and track indices
            non_empty_indices = [idx for idx, text in enumerate(texts) if text and text.strip()]
            non_empty_texts = [texts[idx] for idx in non_empty_indices]

            if not non_empty_texts:
                raise ValueError("All texts are empty")

            # Generate embeddings in batch
            # sentence_transformers handles batching internally
            embeddings_array = model.encode(
                non_empty_texts,
                batch_size=self._config.batch_size,
                normalize_embeddings=self._config.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            # Convert to list of lists
            embeddings = embeddings_array.tolist()

            # Create results mapping back to original indices
            result_map = {}
            for idx, embedding in zip(non_empty_indices, embeddings, strict=True):
                result_map[idx] = embedding

            # Build final results list maintaining original order
            results = []
            for idx, text in enumerate(texts):
                if idx in result_map:
                    results.append(
                        EmbeddingResult(
                            text=text,
                            embedding=result_map[idx],
                            model_name=self._config.model_name,
                            dimension=len(result_map[idx]),
                        )
                    )
                else:
                    # Empty text, create zero embedding
                    dimension = self.dimension
                    results.append(
                        EmbeddingResult(
                            text=text,
                            embedding=[0.0] * dimension,
                            model_name=self._config.model_name,
                            dimension=dimension,
                        )
                    )

            logger.info(
                "batch_encoding_complete",
                total_texts=len(texts),
                successful=len(result_map),
                empty=len(texts) - len(result_map),
            )

            return results

        except ValueError:
            raise
        except Exception as e:
            logger.error("batch_encoding_failed", error=str(e), num_texts=len(texts))
            raise RuntimeError(f"Batch encoding failed: {e}") from e

    def cleanup(self) -> None:
        """Clean up GPU memory if used.

        This method should be called when the embedder is no longer needed
        to free up GPU resources.
        """
        if self._model is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("gpu_memory_cleared")
            except Exception as e:
                logger.warning("gpu_cleanup_failed", error=str(e))

        self._model = None
        self._dimension = None


# Singleton cache for default embedder
_default_embedder: SentenceTransformerEmbedder | None = None
_embedder_lock = threading.Lock()


def get_embedder(config: EmbeddingsConfig | None = None) -> SentenceTransformerEmbedder:
    """Get a sentence-transformer embedder instance.

    This factory function returns a singleton instance for the default
    configuration, and creates new instances for custom configurations.

    Args:
        config: Configuration for the embedder. If None, uses defaults
                and returns a cached singleton instance.

    Returns:
        SentenceTransformerEmbedder instance.

    Examples:
        >>> embedder = get_embedder()
        >>> result = embedder.encode("Market rallied today.")
    """
    global _default_embedder

    if config is None:
        # Return singleton instance for default config
        with _embedder_lock:
            if _default_embedder is None:
                _default_embedder = SentenceTransformerEmbedder()
                logger.debug("default_embedder_created")
            return _default_embedder
    else:
        # Create new instance for custom config
        logger.debug("getting_custom_embedder", model=config.model_name)
        return SentenceTransformerEmbedder(config)


def embed_text(text: str) -> EmbeddingResult:
    """Convenience function for generating text embeddings.

    This is a high-level convenience function that uses default configuration
    and caching for quick embedding generation.

    Args:
        text: Text to encode into an embedding.

    Returns:
        EmbeddingResult containing the text and its vector representation.

    Raises:
        ValueError: If text is empty.
        RuntimeError: If encoding fails.

    Examples:
        >>> result = embed_text("Strong quarterly performance.")
        >>> print(f"Dimension: {result.dimension}")
        Dimension: 384
    """
    embedder = get_embedder()
    return embedder.encode(text)


def embed_texts(texts: list[str]) -> list[EmbeddingResult]:
    """Convenience function for batch embedding generation.

    This is a high-level convenience function that uses default configuration
    and caching for quick batch embedding generation.

    Args:
        texts: List of texts to encode into embeddings.

    Returns:
        List of EmbeddingResult objects, one per input text.

    Raises:
        ValueError: If texts list is empty.
        RuntimeError: If encoding fails.

    Examples:
        >>> texts = ["Revenue increased.", "Profits declined."]
        >>> results = embed_texts(texts)
        >>> len(results)
        2
    """
    embedder = get_embedder()
    return embedder.encode_batch(texts)


def compute_similarity(emb1: list[float], emb2: list[float]) -> float:
    """Compute cosine similarity between two embeddings.

    Cosine similarity measures the cosine of the angle between two vectors,
    ranging from -1 (opposite) to 1 (identical). For normalized embeddings,
    this is equivalent to the dot product.

    Args:
        emb1: First embedding vector.
        emb2: Second embedding vector.

    Returns:
        Cosine similarity score between -1.0 and 1.0.

    Raises:
        ValueError: If embeddings have different dimensions or are empty.

    Examples:
        >>> emb1 = [0.5, 0.5, 0.0]
        >>> emb2 = [0.6, 0.4, 0.0]
        >>> similarity = compute_similarity(emb1, emb2)
        >>> print(f"Similarity: {similarity:.3f}")
        Similarity: 0.985
    """
    if not emb1 or not emb2:
        raise ValueError("Embeddings cannot be empty")

    if len(emb1) != len(emb2):
        raise ValueError(f"Embeddings must have same dimension, got {len(emb1)} and {len(emb2)}")

    # Compute dot product
    dot_product = sum(a * b for a, b in zip(emb1, emb2, strict=True))

    # Compute magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in emb1))
    magnitude2 = math.sqrt(sum(b * b for b in emb2))

    # Handle zero vectors
    if magnitude1 == 0.0 or magnitude2 == 0.0:
        logger.warning("compute_similarity_zero_vector", mag1=magnitude1, mag2=magnitude2)
        return 0.0

    # Compute cosine similarity
    similarity = dot_product / (magnitude1 * magnitude2)

    # Clamp to [-1, 1] to handle floating point errors
    similarity = max(-1.0, min(1.0, similarity))

    logger.debug("similarity_computed", similarity=similarity)

    return similarity
