"""Model routing and A/B testing for production deployments.

This module provides intelligent routing between multiple model versions
for A/B testing, canary deployments, and gradual rollouts.

Features:
- Traffic splitting across model versions
- Per-version metrics tracking
- Automatic rollback on degradation
- Canary deployment support
- Shadow mode for testing

Key Classes:
    TrafficConfig: Traffic split configuration
    ModelRouter: Routes requests to model versions

Examples:
    Basic A/B testing:

    >>> from signalforge.ml.serving import ModelRouter
    >>> from signalforge.ml.inference import ModelRegistry
    >>>
    >>> registry = ModelRegistry()
    >>> router = ModelRouter(registry)
    >>>
    >>> # Split traffic 80/20 between v1 and v2
    >>> await router.set_traffic_split("lstm_model", {
    ...     "v1.0": 0.8,
    ...     "v2.0": 0.2
    ... })
    >>>
    >>> # Route prediction request
    >>> model = await router.route_request("lstm_model", request_id="123")

    Canary deployment:

    >>> # Start with 5% on new version
    >>> await router.set_traffic_split("lstm_model", {
    ...     "v1.0": 0.95,
    ...     "v2.0": 0.05
    ... })
    >>>
    >>> # Monitor metrics...
    >>> stats = router.get_version_stats("lstm_model", "v2.0")
    >>>
    >>> # Gradually increase if metrics are good
    >>> await router.set_traffic_split("lstm_model", {
    ...     "v1.0": 0.50,
    ...     "v2.0": 0.50
    ... })
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from signalforge.core.logging import get_logger

if TYPE_CHECKING:
    from signalforge.ml.inference.model_registry import ModelRegistry
    from signalforge.ml.models.base import BasePredictor

logger = get_logger(__name__)


@dataclass
class TrafficConfig:
    """Traffic split configuration for a model.

    Attributes:
        model_name: Name of the model
        version_weights: Mapping of version to traffic weight (0.0-1.0)
        updated_at: When this config was last updated
        shadow_version: Optional version running in shadow mode
    """

    model_name: str
    version_weights: dict[str, float]
    updated_at: float = field(default_factory=time.time)
    shadow_version: str | None = None

    def __post_init__(self) -> None:
        """Validate traffic configuration."""
        # Validate weights sum to 1.0
        total = sum(self.version_weights.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(
                f"version_weights must sum to 1.0, got {total}"
            )

        # Validate individual weights
        for version, weight in self.version_weights.items():
            if not (0.0 <= weight <= 1.0):
                raise ValueError(
                    f"Weight for {version} must be 0.0-1.0, got {weight}"
                )


@dataclass
class VersionStats:
    """Performance statistics for a model version.

    Attributes:
        version: Version identifier
        request_count: Total requests routed to this version
        total_latency_ms: Cumulative latency in milliseconds
        error_count: Number of errors
        last_error: Last error message
        created_at: When tracking started
    """

    version: str
    request_count: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0
    last_error: str | None = None
    created_at: float = field(default_factory=time.time)

    def get_avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count

    def get_error_rate(self) -> float:
        """Calculate error rate as percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100


class ModelRouter:
    """Routes prediction requests to model versions for A/B testing.

    This class implements intelligent routing between multiple model versions,
    enabling A/B testing, canary deployments, and gradual rollouts.

    Attributes:
        registry: Model registry containing all versions
        traffic_configs: Traffic split configurations per model
        version_stats: Performance metrics per version
    """

    def __init__(self, registry: ModelRegistry) -> None:
        """Initialize model router.

        Args:
            registry: Model registry instance
        """
        self.registry = registry
        self.traffic_configs: dict[str, TrafficConfig] = {}
        self.version_stats: dict[str, dict[str, VersionStats]] = {}

        logger.info("model_router_initialized")

    async def set_traffic_split(
        self,
        model_name: str,
        version_weights: dict[str, float],
        shadow_version: str | None = None,
    ) -> None:
        """Configure traffic split for a model.

        Args:
            model_name: Name of the model
            version_weights: Mapping of version ID to traffic weight
            shadow_version: Optional version to run in shadow mode

        Raises:
            ValueError: If weights don't sum to 1.0 or versions don't exist

        Examples:
            >>> # 80/20 split
            >>> await router.set_traffic_split("lstm_model", {
            ...     "abc-123": 0.8,
            ...     "def-456": 0.2
            ... })
            >>>
            >>> # Canary with shadow mode
            >>> await router.set_traffic_split(
            ...     "lstm_model",
            ...     {"abc-123": 0.95, "def-456": 0.05},
            ...     shadow_version="ghi-789"
            ... )
        """
        # Validate versions exist in registry
        for version_id in version_weights:
            try:
                self.registry.get(version_id)
            except KeyError:
                raise ValueError(f"Version not found in registry: {version_id}")

        # Create traffic config
        config = TrafficConfig(
            model_name=model_name,
            version_weights=version_weights,
            shadow_version=shadow_version,
        )

        self.traffic_configs[model_name] = config

        # Initialize stats for new versions
        if model_name not in self.version_stats:
            self.version_stats[model_name] = {}

        for version_id in version_weights:
            if version_id not in self.version_stats[model_name]:
                self.version_stats[model_name][version_id] = VersionStats(
                    version=version_id
                )

        # Initialize shadow version stats if specified
        if shadow_version and shadow_version not in self.version_stats[model_name]:
            self.version_stats[model_name][shadow_version] = VersionStats(
                version=shadow_version
            )

        logger.info(
            "traffic_split_configured",
            model_name=model_name,
            version_weights=version_weights,
            shadow_version=shadow_version,
        )

    async def route_request(
        self,
        model_name: str,
        request_id: str,
    ) -> BasePredictor:
        """Route a request to the appropriate model version.

        Uses consistent hashing based on request_id to ensure the same
        request always routes to the same version (for determinism).

        Args:
            model_name: Name of the model
            request_id: Unique request identifier

        Returns:
            Selected model instance

        Raises:
            ValueError: If no traffic config exists for model
            KeyError: If model version not found

        Examples:
            >>> model = await router.route_request("lstm_model", "req_123")
            >>> predictions = model.predict(features)
        """
        if model_name not in self.traffic_configs:
            raise ValueError(
                f"No traffic config for model: {model_name}. "
                "Call set_traffic_split() first."
            )

        config = self.traffic_configs[model_name]

        # Select version using consistent hashing
        version_id = self._select_version(request_id, config.version_weights)

        # Load model from registry
        model = self.registry.get(version_id)

        logger.debug(
            "request_routed",
            model_name=model_name,
            version_id=version_id,
            request_id=request_id,
        )

        return model

    def _select_version(
        self,
        request_id: str,
        version_weights: dict[str, float],
    ) -> str:
        """Select version using consistent hashing.

        This ensures that the same request_id always maps to the same version,
        providing deterministic routing which is important for debugging and
        reproducibility.

        Args:
            request_id: Request identifier
            version_weights: Version weights

        Returns:
            Selected version ID
        """
        # Hash request ID to 0-1 range
        hash_obj = hashlib.md5(request_id.encode())
        hash_value = int(hash_obj.hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0

        # Select version based on cumulative weights
        cumulative = 0.0
        for version_id, weight in sorted(version_weights.items()):
            cumulative += weight
            if normalized_hash <= cumulative:
                return version_id

        # Fallback to last version (shouldn't happen if weights sum to 1.0)
        return list(version_weights.keys())[-1]

    def record_request(
        self,
        model_name: str,
        version_id: str,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        """Record metrics for a request.

        Args:
            model_name: Model name
            version_id: Version that handled the request
            latency_ms: Request latency in milliseconds
            error: Error message if request failed

        Examples:
            >>> start = time.perf_counter()
            >>> try:
            ...     result = model.predict(features)
            ...     latency_ms = (time.perf_counter() - start) * 1000
            ...     router.record_request("lstm_model", version_id, latency_ms)
            ... except Exception as e:
            ...     latency_ms = (time.perf_counter() - start) * 1000
            ...     router.record_request("lstm_model", version_id, latency_ms, str(e))
        """
        if model_name not in self.version_stats:
            self.version_stats[model_name] = {}

        if version_id not in self.version_stats[model_name]:
            self.version_stats[model_name][version_id] = VersionStats(
                version=version_id
            )

        stats = self.version_stats[model_name][version_id]
        stats.request_count += 1
        stats.total_latency_ms += latency_ms

        if error:
            stats.error_count += 1
            stats.last_error = error

    def get_version_stats(
        self,
        model_name: str,
        version_id: str,
    ) -> dict[str, Any]:
        """Get performance statistics for a version.

        Args:
            model_name: Model name
            version_id: Version identifier

        Returns:
            Dictionary with performance metrics

        Examples:
            >>> stats = router.get_version_stats("lstm_model", "abc-123")
            >>> print(f"Avg latency: {stats['avg_latency_ms']:.2f}ms")
            >>> print(f"Error rate: {stats['error_rate_pct']:.2f}%")
        """
        if (
            model_name not in self.version_stats
            or version_id not in self.version_stats[model_name]
        ):
            return {
                "version": version_id,
                "request_count": 0,
                "avg_latency_ms": 0.0,
                "error_rate_pct": 0.0,
                "error_count": 0,
            }

        stats = self.version_stats[model_name][version_id]

        return {
            "version": version_id,
            "request_count": stats.request_count,
            "avg_latency_ms": stats.get_avg_latency_ms(),
            "error_rate_pct": stats.get_error_rate(),
            "error_count": stats.error_count,
            "last_error": stats.last_error,
            "uptime_seconds": time.time() - stats.created_at,
        }

    def get_all_stats(self, model_name: str) -> dict[str, dict[str, Any]]:
        """Get statistics for all versions of a model.

        Args:
            model_name: Model name

        Returns:
            Dictionary mapping version IDs to their stats

        Examples:
            >>> all_stats = router.get_all_stats("lstm_model")
            >>> for version, stats in all_stats.items():
            ...     print(f"{version}: {stats['avg_latency_ms']:.2f}ms")
        """
        if model_name not in self.version_stats:
            return {}

        return {
            version_id: self.get_version_stats(model_name, version_id)
            for version_id in self.version_stats[model_name]
        }

    def should_rollback(
        self,
        model_name: str,
        version_id: str,
        error_rate_threshold: float = 5.0,
        min_requests: int = 100,
    ) -> bool:
        """Check if a version should be rolled back due to errors.

        Args:
            model_name: Model name
            version_id: Version to check
            error_rate_threshold: Max acceptable error rate percentage
            min_requests: Minimum requests before checking rollback

        Returns:
            True if version should be rolled back

        Examples:
            >>> if router.should_rollback("lstm_model", "new_version"):
            ...     logger.warning("Rolling back due to high error rate")
            ...     await router.set_traffic_split("lstm_model", {"old_version": 1.0})
        """
        stats_dict = self.get_version_stats(model_name, version_id)

        # Need minimum requests to make decision
        if stats_dict["request_count"] < min_requests:
            return False

        # Check error rate
        if stats_dict["error_rate_pct"] > error_rate_threshold:
            logger.warning(
                "rollback_recommended",
                model_name=model_name,
                version_id=version_id,
                error_rate_pct=stats_dict["error_rate_pct"],
                threshold=error_rate_threshold,
            )
            return True

        return False

    def reset_stats(self, model_name: str) -> None:
        """Reset statistics for a model.

        Args:
            model_name: Model name

        Examples:
            >>> router.reset_stats("lstm_model")
        """
        if model_name in self.version_stats:
            self.version_stats[model_name] = {}
            logger.info("model_stats_reset", model_name=model_name)


__all__ = [
    "TrafficConfig",
    "VersionStats",
    "ModelRouter",
]
