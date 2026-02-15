"""Auto-scaling policies based on system metrics and load.

This module provides intelligent auto-scaling policies that monitor system
resources and trigger scaling actions based on configurable thresholds.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import psutil  # type: ignore[import-untyped]

from signalforge.core.logging import LoggerMixin


class ScalingDirection(str, Enum):
    """Direction of scaling action."""

    UP = "up"
    DOWN = "down"
    NONE = "none"


class MetricType(str, Enum):
    """Types of metrics to monitor."""

    CPU = "cpu"
    MEMORY = "memory"
    QUEUE_DEPTH = "queue_depth"
    REQUEST_RATE = "request_rate"
    ERROR_RATE = "error_rate"


@dataclass
class ScalingThreshold:
    """Threshold configuration for triggering scaling."""

    metric: MetricType
    scale_up_threshold: float
    scale_down_threshold: float
    evaluation_periods: int = 3
    cooldown_seconds: int = 300


@dataclass
class ScalingMetrics:
    """Current system metrics for scaling decisions."""

    cpu_percent: float
    memory_percent: float
    queue_depth: int
    request_rate: float
    error_rate: float
    timestamp: datetime


@dataclass
class ScalingEvent:
    """Record of a scaling event."""

    direction: ScalingDirection
    metric: MetricType
    metric_value: float
    threshold: float
    timestamp: datetime
    success: bool
    reason: str


class ScalingPolicy(LoggerMixin):
    """Manages auto-scaling policies based on system metrics.

    Features:
    - Multi-metric monitoring (CPU, memory, queue depth, etc.)
    - Configurable thresholds and evaluation periods
    - Cooldown periods to prevent flapping
    - Historical metrics tracking
    - Scaling event logging
    """

    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        default_cooldown: int = 300,
    ) -> None:
        """Initialize scaling policy manager.

        Args:
            min_instances: Minimum number of instances to maintain
            max_instances: Maximum number of instances allowed
            default_cooldown: Default cooldown period in seconds
        """
        self._min_instances = min_instances
        self._max_instances = max_instances
        self._default_cooldown = default_cooldown
        self._current_instances = min_instances

        # Thresholds by metric
        self._thresholds: dict[MetricType, ScalingThreshold] = {}

        # Metrics history
        self._metrics_history: list[ScalingMetrics] = []
        self._max_metrics_history = 1000

        # Scaling events history
        self._events_history: list[ScalingEvent] = []
        self._max_events_history = 100

        # Last scaling event time per metric
        self._last_scale_time: dict[MetricType, datetime] = {}

        # Callback for scaling actions
        self._scale_callback: Callable[[ScalingDirection, int], bool] | None = None

        # Queue depth tracker (external source)
        self._queue_depth_provider: Callable[[], int] | None = None

        # Request/error rate trackers (external sources)
        self._request_rate_provider: Callable[[], float] | None = None
        self._error_rate_provider: Callable[[], float] | None = None

        self.logger.info(
            "scaling_policy_initialized",
            min_instances=min_instances,
            max_instances=max_instances,
        )

    def add_threshold(self, threshold: ScalingThreshold) -> None:
        """Add or update a scaling threshold.

        Args:
            threshold: Threshold configuration
        """
        self._thresholds[threshold.metric] = threshold
        self.logger.info(
            "scaling_threshold_added",
            metric=threshold.metric.value,
            scale_up=threshold.scale_up_threshold,
            scale_down=threshold.scale_down_threshold,
        )

    def set_scale_callback(
        self,
        callback: Callable[[ScalingDirection, int], bool],
    ) -> None:
        """Set callback function for scaling actions.

        Args:
            callback: Function that performs scaling action
        """
        self._scale_callback = callback

    def set_queue_depth_provider(self, provider: Callable[[], int]) -> None:
        """Set provider for queue depth metric.

        Args:
            provider: Function that returns current queue depth
        """
        self._queue_depth_provider = provider

    def set_request_rate_provider(self, provider: Callable[[], float]) -> None:
        """Set provider for request rate metric.

        Args:
            provider: Function that returns current request rate
        """
        self._request_rate_provider = provider

    def set_error_rate_provider(self, provider: Callable[[], float]) -> None:
        """Set provider for error rate metric.

        Args:
            provider: Function that returns current error rate
        """
        self._error_rate_provider = provider

    async def collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics.

        Returns:
            Current metrics snapshot
        """
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Queue depth (if provider is set)
        queue_depth = 0
        if self._queue_depth_provider:
            queue_depth = self._queue_depth_provider()

        # Request rate (if provider is set)
        request_rate = 0.0
        if self._request_rate_provider:
            request_rate = self._request_rate_provider()

        # Error rate (if provider is set)
        error_rate = 0.0
        if self._error_rate_provider:
            error_rate = self._error_rate_provider()

        metrics = ScalingMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            queue_depth=queue_depth,
            request_rate=request_rate,
            error_rate=error_rate,
            timestamp=datetime.now(UTC),
        )

        # Store in history
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._max_metrics_history:
            self._metrics_history = self._metrics_history[-self._max_metrics_history :]

        return metrics

    def evaluate_metric(
        self,
        metric: MetricType,
        current_metrics: ScalingMetrics,
    ) -> ScalingDirection:
        """Evaluate a metric against its threshold.

        Args:
            metric: Metric to evaluate
            current_metrics: Current system metrics

        Returns:
            Scaling direction decision
        """
        if metric not in self._thresholds:
            return ScalingDirection.NONE

        threshold = self._thresholds[metric]

        # Get current value for this metric
        if metric == MetricType.CPU:
            current_value = current_metrics.cpu_percent
        elif metric == MetricType.MEMORY:
            current_value = current_metrics.memory_percent
        elif metric == MetricType.QUEUE_DEPTH:
            current_value = float(current_metrics.queue_depth)
        elif metric == MetricType.REQUEST_RATE:
            current_value = current_metrics.request_rate
        elif metric == MetricType.ERROR_RATE:
            current_value = current_metrics.error_rate
        else:
            return ScalingDirection.NONE

        # Check if in cooldown period
        if metric in self._last_scale_time:
            cooldown = timedelta(seconds=threshold.cooldown_seconds)
            if datetime.now(UTC) - self._last_scale_time[metric] < cooldown:
                self.logger.debug(
                    "metric_in_cooldown",
                    metric=metric.value,
                )
                return ScalingDirection.NONE

        # Check against thresholds
        if current_value >= threshold.scale_up_threshold:
            return ScalingDirection.UP
        elif current_value <= threshold.scale_down_threshold:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.NONE

    async def evaluate_all_metrics(self) -> tuple[ScalingDirection, MetricType | None]:
        """Evaluate all configured metrics and determine scaling action.

        Returns:
            Tuple of (scaling_direction, triggering_metric)
        """
        # Collect current metrics
        metrics = await self.collect_metrics()

        # Evaluate each metric
        scale_up_triggers: list[MetricType] = []
        scale_down_triggers: list[MetricType] = []

        for metric in self._thresholds:
            direction = self.evaluate_metric(metric, metrics)

            if direction == ScalingDirection.UP:
                scale_up_triggers.append(metric)
            elif direction == ScalingDirection.DOWN:
                scale_down_triggers.append(metric)

        # Scale up has priority over scale down
        if scale_up_triggers:
            # Use metric with highest priority (CPU > Memory > Queue > Request > Error)
            priority_order = [
                MetricType.CPU,
                MetricType.MEMORY,
                MetricType.QUEUE_DEPTH,
                MetricType.REQUEST_RATE,
                MetricType.ERROR_RATE,
            ]

            for metric in priority_order:
                if metric in scale_up_triggers:
                    return (ScalingDirection.UP, metric)

        elif scale_down_triggers and len(scale_down_triggers) == len(self._thresholds):
            # Only scale down if ALL metrics are below threshold
            return (ScalingDirection.DOWN, scale_down_triggers[0])

        return (ScalingDirection.NONE, None)

    async def execute_scaling(
        self,
        direction: ScalingDirection,
        metric: MetricType,
        metrics: ScalingMetrics,
    ) -> bool:
        """Execute a scaling action.

        Args:
            direction: Direction to scale
            metric: Metric that triggered scaling
            metrics: Current metrics snapshot

        Returns:
            True if scaling was successful
        """
        if direction == ScalingDirection.NONE:
            return True

        # Calculate target instances
        if direction == ScalingDirection.UP:
            target_instances = min(self._current_instances + 1, self._max_instances)
        else:
            target_instances = max(self._current_instances - 1, self._min_instances)

        # Check if we're already at limits
        if target_instances == self._current_instances:
            self.logger.warning(
                "scaling_at_limit",
                direction=direction.value,
                current_instances=self._current_instances,
            )
            return False

        # Get metric value and threshold
        threshold = self._thresholds[metric]
        if metric == MetricType.CPU:
            metric_value = metrics.cpu_percent
            threshold_value = (
                threshold.scale_up_threshold
                if direction == ScalingDirection.UP
                else threshold.scale_down_threshold
            )
        elif metric == MetricType.MEMORY:
            metric_value = metrics.memory_percent
            threshold_value = (
                threshold.scale_up_threshold
                if direction == ScalingDirection.UP
                else threshold.scale_down_threshold
            )
        else:
            metric_value = 0.0
            threshold_value = 0.0

        # Execute scaling via callback
        success = False
        if self._scale_callback:
            try:
                success = self._scale_callback(direction, target_instances)
            except Exception as e:
                self.logger.error(
                    "scaling_callback_failed",
                    error=str(e),
                )
                success = False

        if success:
            self._current_instances = target_instances
            self._last_scale_time[metric] = datetime.now(UTC)

        # Record event
        event = ScalingEvent(
            direction=direction,
            metric=metric,
            metric_value=metric_value,
            threshold=threshold_value,
            timestamp=datetime.now(UTC),
            success=success,
            reason=f"Metric {metric.value} triggered scaling {direction.value}",
        )

        self._events_history.append(event)
        if len(self._events_history) > self._max_events_history:
            self._events_history = self._events_history[-self._max_events_history :]

        self.logger.info(
            "scaling_executed",
            direction=direction.value,
            metric=metric.value,
            metric_value=metric_value,
            target_instances=target_instances,
            success=success,
        )

        return success

    async def auto_scale(self) -> bool:
        """Evaluate metrics and perform auto-scaling if needed.

        Returns:
            True if scaling was performed
        """
        # Evaluate all metrics
        direction, metric = await self.evaluate_all_metrics()

        if direction == ScalingDirection.NONE or metric is None:
            return False

        # Collect metrics for scaling execution
        metrics = self._metrics_history[-1] if self._metrics_history else await self.collect_metrics()

        # Execute scaling
        return await self.execute_scaling(direction, metric, metrics)

    def get_scaling_summary(
        self,
        lookback_minutes: int = 60,
    ) -> dict[str, Any]:
        """Get summary of scaling events over a time window.

        Args:
            lookback_minutes: How many minutes of history to analyze

        Returns:
            Dictionary with summary statistics
        """
        cutoff = datetime.now(UTC) - timedelta(minutes=lookback_minutes)
        recent_events = [e for e in self._events_history if e.timestamp >= cutoff]

        if not recent_events:
            return {
                "period_minutes": lookback_minutes,
                "total_events": 0,
                "current_instances": self._current_instances,
            }

        scale_ups = sum(1 for e in recent_events if e.direction == ScalingDirection.UP)
        scale_downs = sum(1 for e in recent_events if e.direction == ScalingDirection.DOWN)
        successful = sum(1 for e in recent_events if e.success)

        return {
            "period_minutes": lookback_minutes,
            "total_events": len(recent_events),
            "scale_ups": scale_ups,
            "scale_downs": scale_downs,
            "successful_events": successful,
            "failed_events": len(recent_events) - successful,
            "current_instances": self._current_instances,
            "min_instances": self._min_instances,
            "max_instances": self._max_instances,
        }

    @property
    def current_instances(self) -> int:
        """Get current number of instances."""
        return self._current_instances

    @property
    def thresholds(self) -> dict[MetricType, ScalingThreshold]:
        """Get all configured thresholds."""
        return self._thresholds.copy()
