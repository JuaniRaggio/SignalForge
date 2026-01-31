"""GICS sector classification for financial documents.

This module provides sector classification capabilities using the Global Industry
Classification Standard (GICS) with 11 primary sectors. It uses embedding-based
similarity to classify documents based on pre-defined sector descriptions.

Key Features:
- GICS 11-sector taxonomy
- Embedding-based similarity classification
- Batch processing support
- Confidence scoring for predictions
- Top-K sector predictions

Examples:
    Basic sector classification:

    >>> from signalforge.nlp.sector_classifier import classify_sector
    >>>
    >>> text = "Apple released new iPhone with improved AI capabilities"
    >>> prediction = classify_sector(text)
    >>> print(f"Sector: {prediction.sector}, Confidence: {prediction.confidence:.2f}")
    Sector: Information Technology, Confidence: 0.85

    Batch classification with custom configuration:

    >>> from signalforge.nlp.sector_classifier import get_sector_classifier, SectorClassifierConfig
    >>> config = SectorClassifierConfig(top_k=3, similarity_threshold=0.2)
    >>> classifier = get_sector_classifier(config)
    >>> texts = ["Banking merger announced", "Oil prices surge"]
    >>> predictions = classifier.classify_batch(texts)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Final

from signalforge.core.logging import get_logger
from signalforge.nlp.embeddings import (
    EmbeddingsConfig,
    compute_similarity,
    get_embedder,
)

logger = get_logger(__name__)

# GICS Sector definitions
GICS_SECTORS: Final[list[str]] = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
]

# Pre-defined sector descriptions for embedding similarity
# Each sector has multiple reference descriptions with keywords and phrases
SECTOR_DESCRIPTIONS: Final[dict[str, list[str]]] = {
    "Energy": [
        "oil gas drilling exploration petroleum energy utilities fossil fuel",
        "renewable energy solar wind power electricity generation",
        "oil refineries petroleum products natural gas pipelines",
        "energy equipment services drilling offshore onshore",
        "coal mining natural resources power generation",
    ],
    "Materials": [
        "chemicals metals mining construction materials commodities",
        "steel aluminum copper precious metals gold silver",
        "paper forest products packaging industrial materials",
        "chemical manufacturing specialty chemicals fertilizers",
        "mining operations mineral extraction raw materials",
    ],
    "Industrials": [
        "manufacturing machinery equipment industrial production transportation",
        "aerospace defense airlines railroads trucking logistics",
        "construction engineering infrastructure building materials",
        "industrial machinery electrical equipment automation",
        "professional services consulting human resources employment",
    ],
    "Consumer Discretionary": [
        "retail consumer products automobiles entertainment luxury goods",
        "ecommerce online shopping marketplace retail stores",
        "automotive vehicles cars trucks manufacturing",
        "hotels restaurants leisure tourism hospitality",
        "media entertainment streaming content production",
    ],
    "Consumer Staples": [
        "food beverage tobacco household products supermarkets",
        "packaged foods beverages soft drinks alcoholic drinks",
        "personal care products cosmetics hygiene items",
        "grocery stores food retail convenience stores",
        "tobacco cigarettes agricultural products farming",
    ],
    "Health Care": [
        "pharmaceuticals biotechnology medical devices healthcare services",
        "drug development clinical trials FDA approval therapies",
        "hospitals healthcare facilities medical services",
        "health insurance managed care pharmacy benefits",
        "medical equipment surgical instruments diagnostics",
    ],
    "Financials": [
        "banking insurance investment trading loans credit",
        "commercial banks retail banking lending mortgages",
        "investment banking capital markets securities trading",
        "insurance companies property casualty life insurance",
        "asset management wealth management investment funds",
    ],
    "Information Technology": [
        "software hardware cloud computing AI machine learning",
        "semiconductors chips processors technology manufacturing",
        "enterprise software SaaS applications business software",
        "IT services consulting systems integration cybersecurity",
        "technology hardware computers smartphones electronics",
    ],
    "Communication Services": [
        "telecommunications wireless mobile broadband internet services",
        "media broadcasting entertainment content streaming",
        "social media platforms digital advertising marketing",
        "telecom infrastructure networks fiber optic 5G",
        "cable satellite providers video on demand",
    ],
    "Utilities": [
        "electric utilities water gas distribution regulated utilities",
        "power generation transmission distribution electricity",
        "natural gas utilities pipeline distribution",
        "water utilities municipal water wastewater treatment",
        "renewable energy utilities solar wind hydroelectric",
    ],
    "Real Estate": [
        "real estate investment trusts REITs property management",
        "commercial real estate office buildings retail properties",
        "residential properties apartments housing developments",
        "real estate development construction property sales",
        "property management leasing real estate services",
    ],
}


@dataclass
class SectorPrediction:
    """Result of sector classification for a document.

    Attributes:
        text: Original input text that was classified.
        sector: Primary predicted sector (highest confidence).
        confidence: Confidence score for the primary sector (0.0 to 1.0).
        all_scores: Dictionary mapping all sectors to their scores.
        metadata: Additional metadata about the classification.
    """

    text: str
    sector: str
    confidence: float
    all_scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate sector prediction fields."""
        if self.sector not in GICS_SECTORS:
            raise ValueError(f"Invalid sector: {self.sector}. Must be one of {GICS_SECTORS}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        if not self.all_scores:
            raise ValueError("all_scores dictionary cannot be empty")

        # Validate all scores are in valid range
        for sector, score in self.all_scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Score for sector {sector} must be between 0.0 and 1.0, got {score}"
                )

    def get_top_k_sectors(self, k: int = 3) -> list[tuple[str, float]]:
        """Get top K sectors by confidence score.

        Args:
            k: Number of top sectors to return.

        Returns:
            List of (sector, score) tuples sorted by score descending.

        Examples:
            >>> prediction = SectorPrediction(...)
            >>> top_3 = prediction.get_top_k_sectors(3)
            >>> print(top_3)
            [('Information Technology', 0.85), ('Communication Services', 0.72), ...]
        """
        sorted_sectors = sorted(
            self.all_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_sectors[:k]


@dataclass
class SectorClassifierConfig:
    """Configuration for sector classification.

    Attributes:
        model_name: Sentence-transformer model for embeddings.
        similarity_threshold: Minimum similarity score to consider (0.0 to 1.0).
        top_k: Number of top sectors to include in prediction metadata.
        normalize_embeddings: Whether to normalize embeddings for similarity.
        cache_embeddings: Whether to cache sector description embeddings.
    """

    model_name: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.3
    top_k: int = 3
    normalize_embeddings: bool = True
    cache_embeddings: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}"
            )

        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")

        if self.top_k > len(GICS_SECTORS):
            logger.warning(
                "top_k_exceeds_sectors",
                top_k=self.top_k,
                num_sectors=len(GICS_SECTORS),
                adjusted_to=len(GICS_SECTORS),
            )
            self.top_k = len(GICS_SECTORS)


class BaseSectorClassifier(ABC):
    """Abstract base class for sector classifiers.

    This class defines the interface that all sector classifiers must implement.
    """

    @abstractmethod
    def classify(self, text: str) -> SectorPrediction:
        """Classify a single text into a GICS sector.

        Args:
            text: Input text to classify.

        Returns:
            SectorPrediction containing the predicted sector and scores.

        Raises:
            ValueError: If text is empty or invalid.
        """
        pass

    @abstractmethod
    def classify_batch(self, texts: list[str]) -> list[SectorPrediction]:
        """Classify multiple texts into GICS sectors.

        Args:
            texts: List of input texts to classify.

        Returns:
            List of SectorPrediction objects, one per input text.

        Raises:
            ValueError: If texts list is empty.
        """
        pass


class EmbeddingSectorClassifier(BaseSectorClassifier):
    """Embedding-based sector classifier using cosine similarity.

    This classifier uses sentence embeddings to compute similarity between
    the input text and pre-defined sector descriptions. The sector with the
    highest average similarity score is selected as the prediction.

    Examples:
        >>> classifier = EmbeddingSectorClassifier()
        >>> text = "Microsoft Azure cloud services revenue increased"
        >>> prediction = classifier.classify(text)
        >>> print(prediction.sector)
        'Information Technology'
    """

    def __init__(self, config: SectorClassifierConfig | None = None) -> None:
        """Initialize the embedding-based sector classifier.

        Args:
            config: Configuration for the classifier. If None, uses defaults.
        """
        self._config = config or SectorClassifierConfig()

        # Create embedder with matching configuration
        embeddings_config = EmbeddingsConfig(
            model_name=self._config.model_name,
            normalize=self._config.normalize_embeddings,
        )
        self._embedder = get_embedder(embeddings_config)

        # Cache for sector description embeddings
        self._sector_embeddings: dict[str, list[list[float]]] = {}

        logger.info(
            "sector_classifier_initialized",
            model=self._config.model_name,
            threshold=self._config.similarity_threshold,
            top_k=self._config.top_k,
        )

    def _ensure_sector_embeddings_cached(self) -> None:
        """Ensure sector description embeddings are computed and cached."""
        if self._sector_embeddings and self._config.cache_embeddings:
            return

        logger.info("computing_sector_embeddings", num_sectors=len(GICS_SECTORS))

        for sector in GICS_SECTORS:
            descriptions = SECTOR_DESCRIPTIONS[sector]
            embeddings = []

            for description in descriptions:
                result = self._embedder.encode(description)
                embeddings.append(result.embedding)

            self._sector_embeddings[sector] = embeddings

        logger.info(
            "sector_embeddings_cached",
            num_sectors=len(self._sector_embeddings),
            total_descriptions=sum(len(embs) for embs in self._sector_embeddings.values()),
        )

    def _compute_sector_scores(self, text_embedding: list[float]) -> dict[str, float]:
        """Compute similarity scores for all sectors.

        Args:
            text_embedding: Embedding vector of the input text.

        Returns:
            Dictionary mapping sectors to similarity scores.
        """
        self._ensure_sector_embeddings_cached()

        sector_scores = {}

        for sector, sector_embeddings in self._sector_embeddings.items():
            # Compute similarity with each description and take the average
            similarities = [
                compute_similarity(text_embedding, sector_emb) for sector_emb in sector_embeddings
            ]

            # Average similarity across all descriptions
            avg_similarity = sum(similarities) / len(similarities)

            # Normalize to 0-1 range (cosine similarity is -1 to 1)
            normalized_score = (avg_similarity + 1.0) / 2.0

            sector_scores[sector] = normalized_score

        return sector_scores

    def classify(self, text: str) -> SectorPrediction:
        """Classify a single text into a GICS sector.

        Args:
            text: Input text to classify.

        Returns:
            SectorPrediction containing the predicted sector and scores.

        Raises:
            ValueError: If text is empty or invalid.

        Examples:
            >>> classifier = EmbeddingSectorClassifier()
            >>> text = "Goldman Sachs investment banking revenue grew"
            >>> prediction = classifier.classify(text)
            >>> prediction.sector
            'Financials'
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        logger.debug("classifying_text", text_length=len(text))

        # Generate embedding for input text
        text_result = self._embedder.encode(text)

        # Compute similarity scores with all sectors
        sector_scores = self._compute_sector_scores(text_result.embedding)

        # Find sector with highest score
        primary_sector = max(sector_scores.items(), key=lambda x: x[1])
        sector_name, confidence = primary_sector

        # Check if confidence meets threshold
        if confidence < self._config.similarity_threshold:
            logger.warning(
                "low_confidence_prediction",
                sector=sector_name,
                confidence=confidence,
                threshold=self._config.similarity_threshold,
            )

        # Get top K sectors for metadata
        top_k_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)[
            : self._config.top_k
        ]

        logger.info(
            "text_classified",
            sector=sector_name,
            confidence=confidence,
            top_k=[f"{s}:{c:.3f}" for s, c in top_k_sectors],
        )

        return SectorPrediction(
            text=text,
            sector=sector_name,
            confidence=confidence,
            all_scores=sector_scores,
            metadata={
                "model": self._config.model_name,
                "top_k_sectors": dict(top_k_sectors),
                "threshold_met": confidence >= self._config.similarity_threshold,
            },
        )

    def classify_batch(self, texts: list[str]) -> list[SectorPrediction]:
        """Classify multiple texts into GICS sectors.

        Args:
            texts: List of input texts to classify.

        Returns:
            List of SectorPrediction objects, one per input text.

        Raises:
            ValueError: If texts list is empty.

        Examples:
            >>> classifier = EmbeddingSectorClassifier()
            >>> texts = ["Banking profits rise", "Oil production increases"]
            >>> predictions = classifier.classify_batch(texts)
            >>> [p.sector for p in predictions]
            ['Financials', 'Energy']
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        logger.info("classifying_batch", num_texts=len(texts))

        # Ensure sector embeddings are cached
        self._ensure_sector_embeddings_cached()

        # Generate embeddings for all input texts
        text_results = self._embedder.encode_batch(texts)

        # Classify each text
        predictions = []
        for text, text_result in zip(texts, text_results, strict=True):
            if not text or not text.strip():
                logger.warning("skipping_empty_text_in_batch")
                # Create a low-confidence prediction for empty text
                # Create all_scores with 0.0 for all sectors
                all_scores = {sector: 0.0 for sector in GICS_SECTORS}  # noqa: C420
                predictions.append(
                    SectorPrediction(
                        text=text,
                        sector=GICS_SECTORS[0],  # Default to first sector
                        confidence=0.0,
                        all_scores=all_scores,
                        metadata={"error": "Empty text"},
                    )
                )
                continue

            # Compute scores for this text
            sector_scores = self._compute_sector_scores(text_result.embedding)

            # Find primary sector
            primary_sector = max(sector_scores.items(), key=lambda x: x[1])
            sector_name, confidence = primary_sector

            # Get top K sectors
            top_k_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)[
                : self._config.top_k
            ]

            predictions.append(
                SectorPrediction(
                    text=text,
                    sector=sector_name,
                    confidence=confidence,
                    all_scores=sector_scores,
                    metadata={
                        "model": self._config.model_name,
                        "top_k_sectors": dict(top_k_sectors),
                        "threshold_met": confidence >= self._config.similarity_threshold,
                    },
                )
            )

        logger.info(
            "batch_classification_complete",
            total_texts=len(texts),
            avg_confidence=sum(p.confidence for p in predictions) / len(predictions),
            below_threshold=sum(
                1 for p in predictions if p.confidence < self._config.similarity_threshold
            ),
        )

        return predictions


# Singleton cache for default classifier
_default_classifier: EmbeddingSectorClassifier | None = None


def get_sector_classifier(
    config: SectorClassifierConfig | None = None,
) -> EmbeddingSectorClassifier:
    """Get a sector classifier instance.

    This factory function returns a singleton instance for the default
    configuration, and creates new instances for custom configurations.

    Args:
        config: Configuration for the classifier. If None, uses defaults
                and returns a cached singleton instance.

    Returns:
        EmbeddingSectorClassifier instance.

    Examples:
        >>> classifier = get_sector_classifier()
        >>> prediction = classifier.classify("Technology company launches AI product")
    """
    global _default_classifier

    if config is None:
        if _default_classifier is None:
            _default_classifier = EmbeddingSectorClassifier()
            logger.debug("default_sector_classifier_created")
        return _default_classifier
    else:
        logger.debug("creating_custom_sector_classifier", model=config.model_name)
        return EmbeddingSectorClassifier(config)


def classify_sector(text: str) -> SectorPrediction:
    """Convenience function for classifying text into GICS sectors.

    This is a high-level convenience function that uses default configuration
    and caching for quick sector classification.

    Args:
        text: Text to classify into a sector.

    Returns:
        SectorPrediction containing the predicted sector and scores.

    Raises:
        ValueError: If text is empty.

    Examples:
        >>> prediction = classify_sector("Pharmaceutical company develops new drug")
        >>> print(f"Sector: {prediction.sector}")
        Sector: Health Care
    """
    classifier = get_sector_classifier()
    return classifier.classify(text)


def get_all_sectors() -> list[str]:
    """Get list of all GICS sectors.

    Returns:
        List of all 11 GICS sector names.

    Examples:
        >>> sectors = get_all_sectors()
        >>> len(sectors)
        11
        >>> 'Energy' in sectors
        True
    """
    return GICS_SECTORS.copy()
