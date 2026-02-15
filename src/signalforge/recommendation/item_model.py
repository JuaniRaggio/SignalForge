"""Item model for representing signals as recommendable items.

This module provides the core data structures and logic for representing trading
signals as items that can be recommended to users. It handles:
- Feature extraction from signal data
- Embedding generation for signals
- Similarity computation between signals
- Finding similar signals

Key Features:
- Rich signal feature representation
- Vector embeddings for similarity search
- Cosine similarity computation
- Top-K similar item retrieval

Examples:
    Creating a signal item:

    >>> from signalforge.recommendation.item_model import ItemModel, SignalFeatures
    >>> features = SignalFeatures(
    ...     symbol="AAPL",
    ...     sector="Technology",
    ...     volatility=0.25,
    ...     expected_return=0.05,
    ...     holding_period=5,
    ...     risk_level="medium",
    ...     sentiment_score=0.7,
    ...     regime="bull"
    ... )
    >>> item_model = ItemModel()
    >>> signal_item = item_model.create_signal_item(
    ...     signal_id="sig_001",
    ...     features=features
    ... )

    Finding similar signals:

    >>> similar = item_model.find_similar(signal_item, candidates, top_k=5)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from signalforge.core.logging import get_logger

logger = get_logger(__name__)

RiskLevel = Literal["low", "medium", "high"]
RegimeType = Literal["bull", "bear", "neutral", "volatile"]


@dataclass
class SignalFeatures:
    """Features that characterize a trading signal.

    Attributes:
        symbol: Stock ticker symbol.
        sector: Market sector classification.
        volatility: Historical volatility measure (0.0 to 1.0+).
        expected_return: Expected return percentage (can be negative).
        holding_period: Recommended holding period in days.
        risk_level: Categorical risk assessment (low, medium, high).
        sentiment_score: Sentiment score from NLP analysis (-1.0 to 1.0).
        regime: Current market regime classification.
    """

    symbol: str
    sector: str
    volatility: float
    expected_return: float
    holding_period: int
    risk_level: RiskLevel
    sentiment_score: float
    regime: RegimeType

    def __post_init__(self) -> None:
        """Validate signal features."""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")

        if self.volatility < 0.0:
            raise ValueError(f"Volatility must be non-negative, got {self.volatility}")

        if self.holding_period <= 0:
            raise ValueError(f"Holding period must be positive, got {self.holding_period}")

        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError(
                f"Sentiment score must be between -1.0 and 1.0, got {self.sentiment_score}"
            )

        if self.risk_level not in ("low", "medium", "high"):
            raise ValueError(f"Invalid risk level: {self.risk_level}")

        if self.regime not in ("bull", "bear", "neutral", "volatile"):
            raise ValueError(f"Invalid regime: {self.regime}")


@dataclass
class SignalItem:
    """A signal represented as a recommendable item.

    Attributes:
        signal_id: Unique identifier for the signal.
        features: Feature representation of the signal.
        embedding: Vector embedding of the signal (for similarity search).
        created_at: Timestamp when the signal was created.
    """

    signal_id: str
    features: SignalFeatures
    embedding: NDArray[np.float64]
    created_at: datetime

    def __post_init__(self) -> None:
        """Validate signal item."""
        if not self.signal_id:
            raise ValueError("Signal ID cannot be empty")

        if len(self.embedding) == 0:
            raise ValueError("Embedding cannot be empty")

        # Validate embedding contains valid values
        if np.any(np.isnan(self.embedding)) or np.any(np.isinf(self.embedding)):
            raise ValueError("Embedding contains invalid values (NaN or Inf)")


class ItemModel:
    """Model for creating and managing signal items.

    This class handles the conversion of raw signal data into structured
    SignalItem objects with feature vectors and embeddings suitable for
    recommendation algorithms.

    Examples:
        >>> item_model = ItemModel()
        >>> signal_data = {
        ...     "symbol": "AAPL",
        ...     "sector": "Technology",
        ...     "volatility": 0.25,
        ...     "expected_return": 0.05,
        ...     "holding_period": 5,
        ...     "risk_level": "medium",
        ...     "sentiment_score": 0.7,
        ...     "regime": "bull"
        ... }
        >>> features = item_model.extract_features(signal_data)
        >>> embedding = item_model.create_embedding(features)
    """

    def __init__(self, embedding_dim: int = 32) -> None:
        """Initialize the item model.

        Args:
            embedding_dim: Dimensionality of the embedding vectors.
        """
        if embedding_dim <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {embedding_dim}")

        self._embedding_dim = embedding_dim
        logger.info("item_model_initialized", embedding_dim=embedding_dim)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimensionality."""
        return self._embedding_dim

    def extract_features(self, signal_data: dict[str, str | float | int]) -> SignalFeatures:
        """Extract structured features from raw signal data.

        Args:
            signal_data: Dictionary containing signal attributes.

        Returns:
            SignalFeatures object with validated features.

        Raises:
            ValueError: If required fields are missing or invalid.
            KeyError: If required fields are not present in signal_data.

        Examples:
            >>> item_model = ItemModel()
            >>> data = {
            ...     "symbol": "AAPL",
            ...     "sector": "Technology",
            ...     "volatility": 0.25,
            ...     "expected_return": 0.05,
            ...     "holding_period": 5,
            ...     "risk_level": "medium",
            ...     "sentiment_score": 0.7,
            ...     "regime": "bull"
            ... }
            >>> features = item_model.extract_features(data)
        """
        required_fields = [
            "symbol",
            "sector",
            "volatility",
            "expected_return",
            "holding_period",
            "risk_level",
            "sentiment_score",
            "regime",
        ]

        missing_fields = [field for field in required_fields if field not in signal_data]
        if missing_fields:
            raise KeyError(f"Missing required fields: {missing_fields}")

        try:
            features = SignalFeatures(
                symbol=str(signal_data["symbol"]),
                sector=str(signal_data["sector"]),
                volatility=float(signal_data["volatility"]),
                expected_return=float(signal_data["expected_return"]),
                holding_period=int(signal_data["holding_period"]),
                risk_level=str(signal_data["risk_level"]),  # type: ignore[arg-type]
                sentiment_score=float(signal_data["sentiment_score"]),
                regime=str(signal_data["regime"]),  # type: ignore[arg-type]
            )

            logger.debug(
                "features_extracted",
                symbol=features.symbol,
                sector=features.sector,
                risk_level=features.risk_level,
            )

            return features

        except (ValueError, TypeError) as e:
            logger.error("feature_extraction_failed", error=str(e), signal_data=signal_data)
            raise ValueError(f"Failed to extract features: {e}") from e

    def create_embedding(self, features: SignalFeatures) -> NDArray[np.float64]:
        """Create a vector embedding from signal features.

        This method converts structured features into a dense vector representation
        suitable for similarity computations. The embedding combines:
        - Numerical features (volatility, return, holding period)
        - Categorical features (risk level, regime) via one-hot encoding
        - Sentiment score

        Args:
            features: SignalFeatures to encode.

        Returns:
            Normalized embedding vector as numpy array.

        Examples:
            >>> item_model = ItemModel(embedding_dim=32)
            >>> features = SignalFeatures(
            ...     symbol="AAPL",
            ...     sector="Technology",
            ...     volatility=0.25,
            ...     expected_return=0.05,
            ...     holding_period=5,
            ...     risk_level="medium",
            ...     sentiment_score=0.7,
            ...     regime="bull"
            ... )
            >>> embedding = item_model.create_embedding(features)
            >>> embedding.shape
            (32,)
        """
        logger.debug("creating_embedding", symbol=features.symbol)

        # Numerical features (normalized)
        volatility_norm = min(features.volatility, 1.0)  # Cap at 1.0
        return_norm = np.tanh(features.expected_return)  # Squash to [-1, 1]
        holding_norm = min(features.holding_period / 30.0, 1.0)  # Normalize to ~month

        # Risk level encoding (one-hot)
        risk_encoding = {"low": [1.0, 0.0, 0.0], "medium": [0.0, 1.0, 0.0], "high": [0.0, 0.0, 1.0]}
        risk_vec = risk_encoding[features.risk_level]

        # Regime encoding (one-hot)
        regime_encoding = {
            "bull": [1.0, 0.0, 0.0, 0.0],
            "bear": [0.0, 1.0, 0.0, 0.0],
            "neutral": [0.0, 0.0, 1.0, 0.0],
            "volatile": [0.0, 0.0, 0.0, 1.0],
        }
        regime_vec = regime_encoding[features.regime]

        # Combine all features
        feature_vector = np.array(
            [
                volatility_norm,
                return_norm,
                holding_norm,
                features.sentiment_score,
                *risk_vec,
                *regime_vec,
            ],
            dtype=np.float64,
        )

        # Pad or truncate to target dimension
        if len(feature_vector) < self._embedding_dim:
            # Pad with zeros
            padding = np.zeros(self._embedding_dim - len(feature_vector), dtype=np.float64)
            feature_vector = np.concatenate([feature_vector, padding])
        elif len(feature_vector) > self._embedding_dim:
            # Truncate
            feature_vector = feature_vector[: self._embedding_dim]

        # L2 normalize the embedding
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm

        logger.debug(
            "embedding_created",
            dimension=len(feature_vector),
            norm=float(np.linalg.norm(feature_vector)),
        )

        return feature_vector

    def calculate_similarity(self, item1: SignalItem, item2: SignalItem) -> float:
        """Calculate cosine similarity between two signal items.

        Args:
            item1: First signal item.
            item2: Second signal item.

        Returns:
            Cosine similarity score between -1.0 and 1.0.

        Raises:
            ValueError: If embeddings have different dimensions.

        Examples:
            >>> item_model = ItemModel()
            >>> similarity = item_model.calculate_similarity(signal1, signal2)
            >>> print(f"Similarity: {similarity:.3f}")
            Similarity: 0.847
        """
        if len(item1.embedding) != len(item2.embedding):
            raise ValueError(
                f"Embedding dimension mismatch: {len(item1.embedding)} vs {len(item2.embedding)}"
            )

        # Compute cosine similarity (dot product for normalized vectors)
        similarity = float(np.dot(item1.embedding, item2.embedding))

        # Clamp to [-1, 1] to handle floating point errors
        similarity = max(-1.0, min(1.0, similarity))

        logger.debug(
            "similarity_calculated",
            item1=item1.signal_id,
            item2=item2.signal_id,
            similarity=similarity,
        )

        return similarity

    def find_similar(
        self, item: SignalItem, candidates: list[SignalItem], top_k: int = 10
    ) -> list[tuple[SignalItem, float]]:
        """Find the most similar signals to a given signal.

        Args:
            item: Query signal item.
            candidates: List of candidate signal items to compare against.
            top_k: Number of top similar items to return.

        Returns:
            List of (signal_item, similarity_score) tuples, sorted by similarity
            in descending order.

        Raises:
            ValueError: If top_k is non-positive or candidates list is empty.

        Examples:
            >>> item_model = ItemModel()
            >>> similar_signals = item_model.find_similar(
            ...     query_signal,
            ...     all_signals,
            ...     top_k=5
            ... )
            >>> for signal, score in similar_signals:
            ...     print(f"{signal.signal_id}: {score:.3f}")
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if not candidates:
            raise ValueError("Candidates list cannot be empty")

        logger.info(
            "finding_similar_signals",
            query_signal=item.signal_id,
            num_candidates=len(candidates),
            top_k=top_k,
        )

        # Calculate similarities for all candidates
        similarities: list[tuple[SignalItem, float]] = []
        for candidate in candidates:
            # Skip self-comparison
            if candidate.signal_id == item.signal_id:
                continue

            try:
                similarity = self.calculate_similarity(item, candidate)
                similarities.append((candidate, similarity))
            except Exception as e:
                logger.warning(
                    "similarity_calculation_failed",
                    query=item.signal_id,
                    candidate=candidate.signal_id,
                    error=str(e),
                )
                continue

        # Sort by similarity (descending) and take top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_similar = similarities[:top_k]

        logger.info(
            "similar_signals_found",
            query_signal=item.signal_id,
            num_found=len(top_similar),
            top_similarity=top_similar[0][1] if top_similar else 0.0,
        )

        return top_similar

    def create_signal_item(
        self, signal_id: str, features: SignalFeatures, created_at: datetime | None = None
    ) -> SignalItem:
        """Create a complete SignalItem from features.

        This is a convenience method that combines feature extraction and
        embedding creation into a single step.

        Args:
            signal_id: Unique identifier for the signal.
            features: Signal features.
            created_at: Timestamp for signal creation. Defaults to now.

        Returns:
            Complete SignalItem with embedding.

        Examples:
            >>> item_model = ItemModel()
            >>> features = SignalFeatures(...)
            >>> signal_item = item_model.create_signal_item("sig_001", features)
        """
        if created_at is None:
            created_at = datetime.utcnow()

        embedding = self.create_embedding(features)

        signal_item = SignalItem(
            signal_id=signal_id, features=features, embedding=embedding, created_at=created_at
        )

        logger.info(
            "signal_item_created",
            signal_id=signal_id,
            symbol=features.symbol,
            sector=features.sector,
        )

        return signal_item
