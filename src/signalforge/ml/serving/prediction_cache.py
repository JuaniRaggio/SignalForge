"""Redis-based prediction caching for performance optimization.

This module provides intelligent caching for ML predictions using Redis,
reducing latency for repeated requests and improving overall throughput.

Features:
- Configurable TTL per prediction type
- Cache key generation from features
- Hash-based feature fingerprinting
- Automatic cache invalidation
- Cache hit/miss metrics

Key Classes:
    PredictionCache: Redis-based cache for predictions

Examples:
    Basic usage:

    >>> from signalforge.ml.serving import PredictionCache
    >>> from signalforge.core.redis import get_redis
    >>>
    >>> redis = await get_redis()
    >>> cache = PredictionCache(redis, default_ttl=300)
    >>>
    >>> # Check cache
    >>> cached = await cache.get_cached_prediction("AAPL:5:model_v1")
    >>> if cached:
    ...     print("Cache hit!")
    >>>
    >>> # Cache new prediction
    >>> await cache.cache_prediction(
    ...     "AAPL:5:model_v1",
    ...     prediction_result,
    ...     ttl=300
    ... )

    With feature hashing:

    >>> import polars as pl
    >>> features = pl.DataFrame({"rsi_14": [65.0], "macd": [0.5]})
    >>> cache_key = cache.generate_cache_key(
    ...     ticker="AAPL",
    ...     horizon=5,
    ...     features=features
    ... )
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import polars as pl
from redis.asyncio import Redis

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class PredictionCache:
    """Redis-based cache for ML predictions.

    This class provides intelligent caching for predictions, reducing latency
    for repeated requests. Cache keys are generated from ticker, horizon, and
    feature hash to ensure consistent results.

    Attributes:
        redis: Redis client instance
        default_ttl: Default time-to-live in seconds
        key_prefix: Prefix for all cache keys
    """

    def __init__(
        self,
        redis: Redis,
        default_ttl: int = 300,
        key_prefix: str = "pred:",
    ) -> None:
        """Initialize prediction cache.

        Args:
            redis: Redis client instance
            default_ttl: Default TTL in seconds (default: 300 = 5 minutes)
            key_prefix: Prefix for cache keys (default: "pred:")

        Raises:
            ValueError: If default_ttl is invalid
        """
        if default_ttl < 1:
            raise ValueError("default_ttl must be >= 1")

        self.redis = redis
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix

        # Metrics
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info(
            "prediction_cache_initialized",
            default_ttl=default_ttl,
            key_prefix=key_prefix,
        )

    async def get_cached_prediction(
        self,
        cache_key: str,
    ) -> dict[str, Any] | None:
        """Retrieve cached prediction if available.

        Args:
            cache_key: Cache key to lookup

        Returns:
            Cached prediction data or None if not found/expired

        Examples:
            >>> cached = await cache.get_cached_prediction("AAPL:5:abc123")
            >>> if cached:
            ...     print(f"Prediction: {cached['prediction']}")
        """
        full_key = f"{self.key_prefix}{cache_key}"

        try:
            cached_data = await self.redis.get(full_key)

            if cached_data is None:
                self._cache_misses += 1
                logger.debug("cache_miss", cache_key=cache_key)
                return None

            # Parse JSON
            result = json.loads(cached_data)
            self._cache_hits += 1

            logger.debug(
                "cache_hit",
                cache_key=cache_key,
                hit_rate=self.get_hit_rate(),
            )

            return result  # type: ignore[no-any-return]

        except Exception as e:
            logger.warning(
                "cache_get_failed",
                cache_key=cache_key,
                error=str(e),
            )
            self._cache_misses += 1
            return None

    async def cache_prediction(
        self,
        cache_key: str,
        prediction_data: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """Cache a prediction result.

        Args:
            cache_key: Cache key
            prediction_data: Prediction data to cache (must be JSON-serializable)
            ttl: Time-to-live in seconds (uses default_ttl if None)

        Examples:
            >>> await cache.cache_prediction(
            ...     "AAPL:5:abc123",
            ...     {"prediction": 150.5, "confidence": 0.85},
            ...     ttl=600
            ... )
        """
        full_key = f"{self.key_prefix}{cache_key}"
        ttl_value = ttl if ttl is not None else self.default_ttl

        try:
            # Serialize to JSON
            data_json = json.dumps(prediction_data)

            # Store with TTL
            await self.redis.setex(full_key, ttl_value, data_json)

            logger.debug(
                "prediction_cached",
                cache_key=cache_key,
                ttl=ttl_value,
            )

        except Exception as e:
            logger.error(
                "cache_set_failed",
                cache_key=cache_key,
                error=str(e),
            )

    def generate_cache_key(
        self,
        ticker: str,
        horizon: int,
        features: pl.DataFrame | None = None,
        model_id: str | None = None,
    ) -> str:
        """Generate cache key from prediction parameters.

        The cache key includes a hash of the feature values to ensure
        that predictions are only reused when features are identical.

        Args:
            ticker: Stock ticker symbol
            horizon: Prediction horizon in days
            features: Feature DataFrame (optional, for hash)
            model_id: Model identifier (optional)

        Returns:
            Cache key string

        Examples:
            >>> features = pl.DataFrame({"rsi_14": [65.0], "macd": [0.5]})
            >>> key = cache.generate_cache_key("AAPL", 5, features, "model_v1")
            >>> print(key)
            'AAPL:5:model_v1:a1b2c3d4'
        """
        parts = [ticker, str(horizon)]

        if model_id:
            parts.append(model_id)

        # Add features hash if provided
        if features is not None and features.height > 0:
            features_hash = self._hash_features(features)
            parts.append(features_hash)

        return ":".join(parts)

    def _hash_features(self, features: pl.DataFrame) -> str:
        """Generate hash from feature values.

        Uses MD5 hash of serialized feature values for fast comparison.

        Args:
            features: Feature DataFrame

        Returns:
            Hexadecimal hash string (first 8 characters)
        """
        try:
            # Convert to JSON for consistent serialization
            # Use only the last row for single predictions
            if features.height > 0:
                feature_row = features.tail(1)
                feature_dict = feature_row.to_dicts()[0]

                # Sort keys for consistent hashing
                sorted_items = sorted(feature_dict.items())
                feature_str = json.dumps(sorted_items)

                # Generate MD5 hash
                hash_obj = hashlib.md5(feature_str.encode())
                return hash_obj.hexdigest()[:8]
            return "empty"

        except Exception as e:
            logger.warning(
                "feature_hashing_failed",
                error=str(e),
            )
            return "unknown"

    async def invalidate(self, cache_key: str) -> None:
        """Invalidate a cached prediction.

        Args:
            cache_key: Cache key to invalidate

        Examples:
            >>> await cache.invalidate("AAPL:5:model_v1")
        """
        full_key = f"{self.key_prefix}{cache_key}"

        try:
            await self.redis.delete(full_key)
            logger.debug("cache_invalidated", cache_key=cache_key)

        except Exception as e:
            logger.error(
                "cache_invalidation_failed",
                cache_key=cache_key,
                error=str(e),
            )

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all cache keys matching a pattern.

        Args:
            pattern: Redis key pattern (e.g., "AAPL:*")

        Returns:
            Number of keys deleted

        Examples:
            >>> # Invalidate all AAPL predictions
            >>> count = await cache.invalidate_pattern("AAPL:*")
            >>> print(f"Invalidated {count} keys")
        """
        full_pattern = f"{self.key_prefix}{pattern}"

        try:
            # Find matching keys
            keys = []
            async for key in self.redis.scan_iter(match=full_pattern):
                keys.append(key)

            # Delete all matching keys
            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info(
                    "cache_pattern_invalidated",
                    pattern=pattern,
                    count=deleted,
                )
                return int(deleted)

            return 0

        except Exception as e:
            logger.error(
                "cache_pattern_invalidation_failed",
                pattern=pattern,
                error=str(e),
            )
            return 0

    async def clear_all(self) -> int:
        """Clear all cached predictions.

        Returns:
            Number of keys deleted

        Examples:
            >>> count = await cache.clear_all()
            >>> print(f"Cleared {count} predictions")
        """
        return await self.invalidate_pattern("*")

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate as a percentage (0-100)

        Examples:
            >>> hit_rate = cache.get_hit_rate()
            >>> print(f"Cache hit rate: {hit_rate:.1f}%")
        """
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return (self._cache_hits / total) * 100

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics

        Examples:
            >>> stats = cache.get_stats()
            >>> print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
        """
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "total_requests": self._cache_hits + self._cache_misses,
            "hit_rate_pct": self.get_hit_rate(),
            "default_ttl": self.default_ttl,
        }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("cache_stats_reset")


__all__ = ["PredictionCache"]
