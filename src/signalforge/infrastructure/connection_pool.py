"""Connection pool management with dynamic sizing and health monitoring.

This module provides advanced connection pool management with:
- Dynamic pool size adjustment based on load
- Connection health checks and automatic recycling
- Graceful connection draining during shutdown
- Detailed pool utilization metrics
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from signalforge.core.logging import LoggerMixin


@dataclass
class PoolMetrics:
    """Metrics for connection pool monitoring."""

    size: int
    checked_in: int
    checked_out: int
    overflow: int
    total_connections: int
    utilization_percent: float
    timestamp: datetime


class ConnectionPoolManager(LoggerMixin):
    """Manages database connection pools with dynamic sizing and monitoring.

    Features:
    - Dynamic pool size adjustment based on utilization
    - Periodic health checks for all connections
    - Graceful connection draining
    - Comprehensive metrics collection
    """

    def __init__(
        self,
        database_url: str,
        min_pool_size: int = 5,
        max_pool_size: int = 20,
        max_overflow: int = 10,
        pool_recycle: int = 3600,
        pool_timeout: int = 30,
        pool_pre_ping: bool = True,
        echo: bool = False,
        enable_metrics: bool = True,
    ) -> None:
        """Initialize connection pool manager.

        Args:
            database_url: Database connection URL
            min_pool_size: Minimum number of connections to maintain
            max_pool_size: Maximum number of pooled connections
            max_overflow: Maximum number of connections beyond pool_size
            pool_recycle: Recycle connections after this many seconds
            pool_timeout: Timeout for getting connection from pool
            pool_pre_ping: Test connections before using them
            echo: Whether to log SQL statements
            enable_metrics: Whether to collect pool metrics
        """
        self._database_url = database_url
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._max_overflow = max_overflow
        self._pool_recycle = pool_recycle
        self._pool_timeout = pool_timeout
        self._pool_pre_ping = pool_pre_ping
        self._echo = echo
        self._enable_metrics = enable_metrics

        # Current pool size (starts at minimum)
        self._current_pool_size = min_pool_size

        # Engine instance
        self._engine: AsyncEngine | None = None

        # Metrics storage
        self._metrics_history: list[PoolMetrics] = []
        self._max_metrics_history = 1000

        # Connection tracking
        self._total_connections_created = 0
        self._total_checkouts = 0
        self._total_checkins = 0

        # Shutdown flag
        self._is_shutting_down = False

        self.logger.info(
            "connection_pool_manager_initialized",
            min_pool_size=min_pool_size,
            max_pool_size=max_pool_size,
            max_overflow=max_overflow,
        )

    async def initialize(self) -> None:
        """Initialize the connection pool and set up monitoring."""
        # SQLite doesn't support pool_size/max_overflow with StaticPool
        is_sqlite = "sqlite" in self._database_url.lower()

        if is_sqlite:
            from sqlalchemy.pool import StaticPool

            self._engine = create_async_engine(
                self._database_url,
                echo=self._echo,
                poolclass=StaticPool,
            )
        else:
            self._engine = create_async_engine(
                self._database_url,
                echo=self._echo,
                pool_size=self._current_pool_size,
                max_overflow=self._max_overflow,
                pool_recycle=self._pool_recycle,
                pool_timeout=self._pool_timeout,
                pool_pre_ping=self._pool_pre_ping,
            )

        # Set up connection pool event listeners
        if self._enable_metrics:
            self._setup_pool_listeners()

        self.logger.info("connection_pool_initialized")

    def _setup_pool_listeners(self) -> None:
        """Set up SQLAlchemy pool event listeners for metrics."""
        if not self._engine:
            return

        pool = self._engine.pool

        @event.listens_for(pool, "connect")
        def receive_connect(
            dbapi_conn: Any,  # noqa: ARG001
            connection_record: Any,  # noqa: ARG001
        ) -> None:
            """Track connection creation."""
            self._total_connections_created += 1
            self.logger.debug(
                "pool_connection_created",
                total_created=self._total_connections_created,
            )

        @event.listens_for(pool, "checkout")
        def receive_checkout(
            dbapi_conn: Any,  # noqa: ARG001
            connection_record: Any,  # noqa: ARG001
            connection_proxy: Any,  # noqa: ARG001
        ) -> None:
            """Track connection checkout."""
            self._total_checkouts += 1

        @event.listens_for(pool, "checkin")
        def receive_checkin(
            dbapi_conn: Any,  # noqa: ARG001
            connection_record: Any,  # noqa: ARG001
        ) -> None:
            """Track connection checkin."""
            self._total_checkins += 1

    def get_current_metrics(self) -> PoolMetrics | None:
        """Get current pool metrics.

        Returns:
            Current pool metrics, or None if pool not initialized
        """
        if not self._engine or not self._engine.pool:
            return None

        pool = self._engine.pool

        # StaticPool (used by SQLite) doesn't have size(), checkedin(), etc.
        # Return basic metrics for StaticPool
        from sqlalchemy.pool import StaticPool

        if isinstance(pool, StaticPool):
            metrics = PoolMetrics(
                size=1,
                checked_in=1,
                checked_out=0,
                overflow=0,
                total_connections=1,
                utilization_percent=0.0,
                timestamp=datetime.now(UTC),
            )
        else:
            # Calculate metrics for QueuePool and similar pools
            size = pool.size()  # type: ignore[attr-defined]
            checked_in = pool.checkedin()  # type: ignore[attr-defined]
            checked_out = pool.checkedout()  # type: ignore[attr-defined]
            overflow = pool.overflow()  # type: ignore[attr-defined]
            total_connections = size + overflow
            utilization = (checked_out / size * 100) if size > 0 else 0.0

            metrics = PoolMetrics(
                size=size,
                checked_in=checked_in,
                checked_out=checked_out,
                overflow=overflow,
                total_connections=total_connections,
                utilization_percent=utilization,
                timestamp=datetime.now(UTC),
            )

        # Store in history
        if self._enable_metrics:
            self._metrics_history.append(metrics)
            # Trim history if needed
            if len(self._metrics_history) > self._max_metrics_history:
                self._metrics_history = self._metrics_history[-self._max_metrics_history :]

        return metrics

    async def adjust_pool_size(self, target_size: int) -> bool:
        """Dynamically adjust pool size.

        Args:
            target_size: Desired pool size (clamped to min/max bounds)

        Returns:
            True if adjustment was successful
        """
        # SQLite with StaticPool doesn't support pool size adjustment
        is_sqlite = "sqlite" in self._database_url.lower()
        if is_sqlite:
            self.logger.debug(
                "pool_size_adjustment_skipped",
                reason="SQLite with StaticPool doesn't support pool size adjustment",
            )
            return True

        # Clamp to bounds
        target_size = max(self._min_pool_size, min(target_size, self._max_pool_size))

        if target_size == self._current_pool_size:
            return True

        self.logger.info(
            "adjusting_pool_size",
            current_size=self._current_pool_size,
            target_size=target_size,
        )

        try:
            # Dispose current engine
            if self._engine:
                await self._engine.dispose()

            # Create new engine with adjusted pool size
            self._engine = create_async_engine(
                self._database_url,
                echo=self._echo,
                pool_size=target_size,
                max_overflow=self._max_overflow,
                pool_recycle=self._pool_recycle,
                pool_timeout=self._pool_timeout,
                pool_pre_ping=self._pool_pre_ping,
            )

            if self._enable_metrics:
                self._setup_pool_listeners()

            self._current_pool_size = target_size

            self.logger.info(
                "pool_size_adjusted",
                new_size=target_size,
            )
            return True
        except Exception as e:
            self.logger.error(
                "pool_size_adjustment_failed",
                error=str(e),
            )
            return False

    async def auto_scale(
        self,
        high_threshold: float = 80.0,
        low_threshold: float = 30.0,
        scale_up_factor: float = 1.5,
        scale_down_factor: float = 0.75,
    ) -> bool:
        """Automatically scale pool based on utilization.

        Args:
            high_threshold: Utilization % to trigger scale-up
            low_threshold: Utilization % to trigger scale-down
            scale_up_factor: Multiplier for scaling up
            scale_down_factor: Multiplier for scaling down

        Returns:
            True if scaling was performed
        """
        metrics = self.get_current_metrics()
        if not metrics:
            return False

        utilization = metrics.utilization_percent

        if utilization >= high_threshold:
            # Scale up
            target_size = int(self._current_pool_size * scale_up_factor)
            self.logger.info(
                "auto_scaling_up",
                utilization=utilization,
                threshold=high_threshold,
                target_size=target_size,
            )
            return await self.adjust_pool_size(target_size)
        elif utilization <= low_threshold:
            # Scale down
            target_size = int(self._current_pool_size * scale_down_factor)
            self.logger.info(
                "auto_scaling_down",
                utilization=utilization,
                threshold=low_threshold,
                target_size=target_size,
            )
            return await self.adjust_pool_size(target_size)

        return False

    async def health_check(self) -> bool:
        """Check if connection pool is healthy.

        Returns:
            True if pool is healthy
        """
        if not self._engine:
            return False

        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            self.logger.error(
                "pool_health_check_failed",
                error=str(e),
            )
            return False

    async def drain_connections(self, timeout: float = 30.0) -> bool:
        """Gracefully drain all connections from the pool.

        Args:
            timeout: Maximum time to wait for connections to drain

        Returns:
            True if all connections were drained within timeout
        """
        if not self._engine:
            return True

        self._is_shutting_down = True
        self.logger.info("draining_connections", timeout=timeout)

        start_time = datetime.now(UTC)
        deadline = start_time + timedelta(seconds=timeout)

        while datetime.now(UTC) < deadline:
            metrics = self.get_current_metrics()
            if metrics and metrics.checked_out == 0:
                self.logger.info(
                    "connections_drained",
                    elapsed_seconds=(datetime.now(UTC) - start_time).total_seconds(),
                )
                return True

            await asyncio.sleep(0.5)

        # Timeout reached
        metrics = self.get_current_metrics()
        self.logger.warning(
            "connection_drain_timeout",
            remaining_connections=metrics.checked_out if metrics else "unknown",
        )
        return False

    async def dispose(self) -> None:
        """Dispose of the connection pool and cleanup resources."""
        if not self._engine:
            return

        # Try graceful drain first
        await self.drain_connections(timeout=10.0)

        # Dispose engine
        await self._engine.dispose()
        self._engine = None

        self.logger.info("connection_pool_disposed")

    def get_metrics_summary(self, lookback_minutes: int = 5) -> dict[str, Any]:
        """Get summary statistics for pool metrics over a time window.

        Args:
            lookback_minutes: How many minutes of history to analyze

        Returns:
            Dictionary with summary statistics
        """
        if not self._metrics_history:
            return {}

        cutoff = datetime.now(UTC) - timedelta(minutes=lookback_minutes)
        recent_metrics = [m for m in self._metrics_history if m.timestamp >= cutoff]

        if not recent_metrics:
            return {}

        utilizations = [m.utilization_percent for m in recent_metrics]
        checked_out = [m.checked_out for m in recent_metrics]
        overflows = [m.overflow for m in recent_metrics]

        return {
            "period_minutes": lookback_minutes,
            "sample_count": len(recent_metrics),
            "avg_utilization": sum(utilizations) / len(utilizations),
            "max_utilization": max(utilizations),
            "min_utilization": min(utilizations),
            "avg_checked_out": sum(checked_out) / len(checked_out),
            "max_checked_out": max(checked_out),
            "avg_overflow": sum(overflows) / len(overflows),
            "max_overflow": max(overflows),
            "current_pool_size": self._current_pool_size,
            "total_connections_created": self._total_connections_created,
            "total_checkouts": self._total_checkouts,
            "total_checkins": self._total_checkins,
        }

    @property
    def engine(self) -> AsyncEngine | None:
        """Get the managed engine instance."""
        return self._engine

    @property
    def is_initialized(self) -> bool:
        """Check if pool is initialized."""
        return self._engine is not None

    @property
    def is_shutting_down(self) -> bool:
        """Check if pool is in shutdown mode."""
        return self._is_shutting_down
