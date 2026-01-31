"""Sentiment analysis for financial text using FinBERT.

This module provides sentiment analysis capabilities specifically tuned for
financial text using the FinBERT model (ProsusAI/finbert). It includes both
single-text and batch processing with GPU acceleration support.

Key Features:
- FinBERT-based sentiment classification
- Lazy model loading for efficient resource usage
- GPU/CPU automatic device selection
- Batch processing with configurable batch sizes
- Integration with text preprocessing pipeline
- Calibrated confidence scores

Examples:
    Basic sentiment analysis:

    >>> from signalforge.nlp.sentiment import analyze_financial_text
    >>>
    >>> text = "Apple reported strong Q4 earnings, beating analyst expectations."
    >>> result = analyze_financial_text(text)
    >>> print(f"Sentiment: {result.label} (confidence: {result.confidence:.2f})")
    Sentiment: positive (confidence: 0.89)

    Batch processing with custom configuration:

    >>> from signalforge.nlp.sentiment import get_sentiment_analyzer, SentimentConfig
    >>> config = SentimentConfig(device="cuda", batch_size=32)
    >>> analyzer = get_sentiment_analyzer(config)
    >>> texts = ["Market rallied today...", "Recession fears mount..."]
    >>> results = analyzer.analyze_batch(texts)
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

from signalforge.core.logging import get_logger
from signalforge.nlp.preprocessing import PreprocessingConfig, TextPreprocessor

if TYPE_CHECKING:
    from transformers import Pipeline

logger = get_logger(__name__)

# Sentiment labels
SentimentLabel = Literal["positive", "negative", "neutral"]

# Model constants
DEFAULT_MODEL_NAME = "ProsusAI/finbert"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16
MODEL_LOAD_TIMEOUT = 300  # seconds


@dataclass
class SentimentResult:
    """Result of sentiment analysis on a single text.

    Attributes:
        text: Original input text analyzed.
        label: Predicted sentiment label (positive, negative, neutral).
        confidence: Confidence score for the predicted label (0.0 to 1.0).
        scores: Raw scores for all sentiment classes.
    """

    text: str
    label: SentimentLabel
    confidence: float
    scores: dict[str, float]

    def __post_init__(self) -> None:
        """Validate sentiment result fields."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        if self.label not in ("positive", "negative", "neutral"):
            raise ValueError(f"Invalid label: {self.label}")

        # Validate scores
        expected_labels = {"positive", "negative", "neutral"}
        if set(self.scores.keys()) != expected_labels:
            raise ValueError(
                f"Scores must contain exactly {expected_labels}, got {set(self.scores.keys())}"
            )

        for label, score in self.scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Score for {label} must be between 0.0 and 1.0, got {score}")


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis.

    Attributes:
        model_name: HuggingFace model identifier.
        device: Device to use for inference ("auto", "cpu", "cuda", "mps").
        batch_size: Number of texts to process in parallel.
        max_length: Maximum token length (texts will be truncated).
        cache_model: Whether to cache the loaded model in memory.
        preprocess_text: Whether to preprocess text before analysis.
        preprocessing_config: Configuration for text preprocessing.
        temperature: Temperature for calibrated confidence scores (1.0 = no scaling).
    """

    model_name: str = DEFAULT_MODEL_NAME
    device: str = "auto"
    batch_size: int = DEFAULT_BATCH_SIZE
    max_length: int = DEFAULT_MAX_LENGTH
    cache_model: bool = True
    preprocess_text: bool = False
    preprocessing_config: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    temperature: float = 1.0

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

        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")


class BaseSentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers.

    This class defines the interface that all sentiment analyzers must implement.
    """

    @abstractmethod
    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text.

        Args:
            text: Input text to analyze.

        Returns:
            SentimentResult containing label, confidence, and scores.

        Raises:
            ValueError: If text is empty or invalid.
        """
        pass

    @abstractmethod
    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment of multiple texts in batch.

        Args:
            texts: List of input texts to analyze.

        Returns:
            List of SentimentResult objects, one per input text.

        Raises:
            ValueError: If texts list is empty or contains invalid entries.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the underlying model."""
        pass


class FinBERTSentimentAnalyzer(BaseSentimentAnalyzer):
    """FinBERT-based sentiment analyzer for financial text.

    This analyzer uses the ProsusAI/finbert model fine-tuned on financial
    text for sentiment classification. The model is lazy-loaded on first use
    and supports GPU acceleration.

    Examples:
        >>> analyzer = FinBERTSentimentAnalyzer()
        >>> result = analyzer.analyze("Strong quarterly earnings beat expectations.")
        >>> print(result.label)
        positive
    """

    _model_cache: dict[str, Pipeline] = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: SentimentConfig | None = None) -> None:
        """Initialize the FinBERT sentiment analyzer.

        Args:
            config: Configuration for the analyzer. If None, uses defaults.
        """
        self._config = config or SentimentConfig()
        self._pipeline: Pipeline | None = None
        self._preprocessor: TextPreprocessor | None = None

        if self._config.preprocess_text:
            self._preprocessor = TextPreprocessor()

        logger.info(
            "finbert_analyzer_initialized",
            model_name=self._config.model_name,
            device=self._config.device,
            batch_size=self._config.batch_size,
            preprocess_text=self._config.preprocess_text,
        )

    @property
    def model_name(self) -> str:
        """Get the name of the underlying model."""
        return self._config.model_name

    def _get_device(self) -> int | str:
        """Determine the appropriate device for inference.

        Returns:
            Device identifier (-1 for CPU, 0 for GPU).
        """
        if self._config.device == "cpu":
            return -1

        if self._config.device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    logger.info("device_selected", device="cuda", reason="auto_detect")
                    return 0
                elif torch.backends.mps.is_available():
                    logger.info("device_selected", device="mps", reason="auto_detect")
                    return "mps"
            except Exception as e:
                logger.warning(
                    "gpu_detection_failed",
                    error=str(e),
                    fallback="cpu",
                )
            return -1

        if self._config.device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    logger.warning(
                        "cuda_not_available",
                        fallback="cpu",
                    )
                    return -1
                return 0
            except ImportError:
                logger.error("torch_not_installed", fallback="cpu")
                return -1

        if self._config.device == "mps":
            try:
                import torch

                if not torch.backends.mps.is_available():
                    logger.warning(
                        "mps_not_available",
                        fallback="cpu",
                    )
                    return -1
                return "mps"
            except ImportError:
                logger.error("torch_not_installed", fallback="cpu")
                return -1

        return -1

    def _load_model(self) -> Pipeline:
        """Load the FinBERT model and create inference pipeline.

        Returns:
            HuggingFace pipeline for sentiment analysis.

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
            from transformers import pipeline

            device = self._get_device()

            # Load the sentiment analysis pipeline
            # Note: transformers pipeline has complex overloaded types
            sentiment_pipeline = pipeline(
                task="sentiment-analysis",
                model=self._config.model_name,
                device=device,
                max_length=self._config.max_length,
                truncation=True,
            )  # type: ignore[call-overload]

            logger.info(
                "model_loaded",
                model=self._config.model_name,
                device=device,
            )

            # Cache the model if enabled
            if self._config.cache_model:
                cache_key = f"{self._config.model_name}_{self._config.device}"
                with self._cache_lock:
                    self._model_cache[cache_key] = sentiment_pipeline

            return sentiment_pipeline  # type: ignore[no-any-return]

        except ImportError as e:
            error_msg = (
                "transformers or torch not installed. Install with: pip install transformers torch"
            )
            logger.error("model_load_failed", error=str(e), reason="missing_dependencies")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            logger.error("model_load_failed", error=str(e), model=self._config.model_name)
            raise RuntimeError(f"Failed to load model {self._config.model_name}: {e}") from e

    def _ensure_model_loaded(self) -> Pipeline:
        """Ensure the model is loaded, loading it if necessary.

        Returns:
            The loaded sentiment analysis pipeline.
        """
        if self._pipeline is None:
            self._pipeline = self._load_model()
        return self._pipeline

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text if preprocessing is enabled.

        Args:
            text: Raw input text.

        Returns:
            Preprocessed text.
        """
        if not self._config.preprocess_text or self._preprocessor is None:
            return text

        # Clean text
        cleaned = self._preprocessor.clean_text(text)

        # Optionally normalize financial terms
        if self._config.preprocessing_config.normalize_financial_terms:
            cleaned = self._preprocessor.normalize_financial_terms(cleaned)

        return cleaned

    def _parse_model_output(self, text: str, output: dict[str, str | float]) -> SentimentResult:
        """Parse raw model output into SentimentResult.

        Args:
            text: Original input text.
            output: Raw output from the model pipeline.

        Returns:
            Parsed SentimentResult.
        """
        # FinBERT outputs labels like "positive", "negative", "neutral"
        label_map = {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
        }

        raw_label = str(output.get("label", "")).lower()
        label = cast(SentimentLabel, label_map.get(raw_label, "neutral"))
        confidence = float(output.get("score", 0.0))

        # Apply temperature scaling if configured
        if self._config.temperature != 1.0:
            import math

            # Convert confidence to logit, scale, and convert back
            logit = math.log(confidence / (1 - confidence + 1e-10))
            scaled_logit = logit / self._config.temperature
            confidence = 1 / (1 + math.exp(-scaled_logit))

        # Create scores dict (we only have confidence for predicted label)
        # For a complete implementation, we'd need all class probabilities
        scores = {
            "positive": confidence if label == "positive" else (1 - confidence) / 2,
            "negative": confidence if label == "negative" else (1 - confidence) / 2,
            "neutral": confidence if label == "neutral" else (1 - confidence) / 2,
        }

        # Normalize scores to sum to 1.0
        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}

        return SentimentResult(
            text=text,
            label=label,
            confidence=confidence,
            scores=scores,
        )

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text.

        Args:
            text: Input text to analyze.

        Returns:
            SentimentResult containing label, confidence, and scores.

        Raises:
            ValueError: If text is empty or invalid.
            RuntimeError: If model inference fails.

        Examples:
            >>> analyzer = FinBERTSentimentAnalyzer()
            >>> result = analyzer.analyze("Revenue exceeded expectations.")
            >>> result.label
            'positive'
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        logger.debug("analyzing_sentiment", text_length=len(text))

        try:
            # Preprocess if configured
            processed_text = self._preprocess_text(text)

            if not processed_text.strip():
                logger.warning("text_empty_after_preprocessing", original_text=text[:100])
                # Return neutral sentiment for empty text
                return SentimentResult(
                    text=text,
                    label="neutral",
                    confidence=1.0,
                    scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                )

            # Ensure model is loaded
            pipeline = self._ensure_model_loaded()

            # Perform inference
            result = pipeline(processed_text[: self._config.max_length])[0]

            # Parse and return result
            sentiment_result = self._parse_model_output(text, result)

            logger.debug(
                "sentiment_analyzed",
                label=sentiment_result.label,
                confidence=sentiment_result.confidence,
            )

            return sentiment_result

        except ValueError:
            raise
        except Exception as e:
            logger.error("sentiment_analysis_failed", error=str(e), text=text[:100])
            raise RuntimeError(f"Sentiment analysis failed: {e}") from e

    def analyze_batch(self, texts: list[str]) -> list[SentimentResult]:
        """Analyze sentiment of multiple texts in batch.

        Args:
            texts: List of input texts to analyze.

        Returns:
            List of SentimentResult objects, one per input text.

        Raises:
            ValueError: If texts list is empty.
            RuntimeError: If batch inference fails.

        Examples:
            >>> analyzer = FinBERTSentimentAnalyzer()
            >>> texts = ["Profits soared.", "Sales declined."]
            >>> results = analyzer.analyze_batch(texts)
            >>> len(results)
            2
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        logger.info("analyzing_batch", num_texts=len(texts), batch_size=self._config.batch_size)

        try:
            # Preprocess texts if configured
            processed_texts = [self._preprocess_text(text) for text in texts]

            # Ensure model is loaded
            pipeline = self._ensure_model_loaded()

            # Process in batches
            results = []
            for i in range(0, len(processed_texts), self._config.batch_size):
                batch = processed_texts[i : i + self._config.batch_size]
                original_batch = texts[i : i + self._config.batch_size]

                # Truncate texts to max length
                truncated_batch = [text[: self._config.max_length] for text in batch]

                # Filter out empty texts
                non_empty_indices = [
                    idx for idx, text in enumerate(truncated_batch) if text.strip()
                ]
                non_empty_texts = [truncated_batch[idx] for idx in non_empty_indices]

                if non_empty_texts:
                    # Perform batch inference
                    batch_results = pipeline(non_empty_texts)

                    # Map results back to original positions
                    result_map = dict(zip(non_empty_indices, batch_results, strict=True))

                    # Create SentimentResult objects
                    for idx, original_text in enumerate(original_batch):
                        if idx in result_map:
                            sentiment_result = self._parse_model_output(
                                original_text, result_map[idx]
                            )
                        else:
                            # Empty text, return neutral
                            sentiment_result = SentimentResult(
                                text=original_text,
                                label="neutral",
                                confidence=1.0,
                                scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                            )
                        results.append(sentiment_result)
                else:
                    # All texts in batch were empty
                    for original_text in original_batch:
                        results.append(
                            SentimentResult(
                                text=original_text,
                                label="neutral",
                                confidence=1.0,
                                scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                            )
                        )

                logger.debug(
                    "batch_processed",
                    batch_num=i // self._config.batch_size + 1,
                    batch_size=len(batch),
                )

            logger.info(
                "batch_analysis_complete",
                total_texts=len(texts),
                successful=len([r for r in results if r.label is not None]),
            )

            return results

        except ValueError:
            raise
        except Exception as e:
            logger.error("batch_analysis_failed", error=str(e), num_texts=len(texts))
            raise RuntimeError(f"Batch sentiment analysis failed: {e}") from e

    def cleanup(self) -> None:
        """Clean up GPU memory if used.

        This method should be called when the analyzer is no longer needed
        to free up GPU resources.
        """
        if self._pipeline is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("gpu_memory_cleared")
            except Exception as e:
                logger.warning("gpu_cleanup_failed", error=str(e))

        self._pipeline = None


# Singleton cache for default analyzer
_default_analyzer: FinBERTSentimentAnalyzer | None = None
_analyzer_lock = threading.Lock()


def get_sentiment_analyzer(config: SentimentConfig | None = None) -> FinBERTSentimentAnalyzer:
    """Get a sentiment analyzer instance.

    This factory function returns a singleton instance for the default
    configuration, and creates new instances for custom configurations.

    Args:
        config: Configuration for the analyzer. If None, uses defaults
                and returns a cached singleton instance.

    Returns:
        FinBERTSentimentAnalyzer instance.

    Examples:
        >>> analyzer = get_sentiment_analyzer()
        >>> result = analyzer.analyze("Market rallied today.")
    """
    global _default_analyzer

    if config is None:
        # Return singleton instance for default config
        with _analyzer_lock:
            if _default_analyzer is None:
                _default_analyzer = FinBERTSentimentAnalyzer()
                logger.debug("default_analyzer_created")
            return _default_analyzer
    else:
        # Create new instance for custom config
        logger.debug("getting_sentiment_analyzer", model=config.model_name)
        return FinBERTSentimentAnalyzer(config)


def analyze_financial_text(text: str) -> SentimentResult:
    """Convenience function for analyzing sentiment of financial text.

    This is a high-level convenience function that uses default configuration
    and caching for quick sentiment analysis.

    Args:
        text: Financial text to analyze.

    Returns:
        SentimentResult containing sentiment label, confidence, and scores.

    Raises:
        ValueError: If text is empty.
        RuntimeError: If analysis fails.

    Examples:
        >>> result = analyze_financial_text("Strong quarterly performance.")
        >>> print(f"{result.label}: {result.confidence:.2f}")
        positive: 0.92
    """
    analyzer = get_sentiment_analyzer()
    return analyzer.analyze(text)
